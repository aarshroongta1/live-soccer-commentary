"""Agent identity decisions, researcher handoff and validation-only code guards."""

from datetime import date

import pytest
from pydantic import ValidationError

from commentary.llm.fake import ScriptedBackend
from commentary.recorded_demo import Observation, ObservedActor
from commentary.recorded_research import ResearchBrief, ResearchDraft, ResearchSquad, SquadPlayer
from commentary.replay.identity import IdentityAction, IdentityProposal, identify_players
from commentary.replay.preparation import prepare_research


def squad(**updates):
    data = dict(
        id="england-squad",
        side="home",
        team="England",
        players=[
            SquadPlayer(name="Harry Kane", number=9),
            SquadPlayer(name="Bukayo Saka", number=17),
        ],
        source_url="https://example.org/pre-match-squad",
        source_date=date(2022, 11, 14),
    )
    return ResearchSquad.model_validate({**data, **updates})


def brief(**updates):
    return ResearchBrief(
        fixture="England v France", as_of=date(2022, 12, 10), facts=[], squads=[squad(**updates)]
    )


def seen(identifier="shot", kind="play"):
    return Observation(
        id=identifier,
        at_s=1,
        kind=kind,
        description="Number 9 strikes the ball.",
        actor=ObservedActor(
            side="home", shirt_number=9, identity_source="shirt_number", action_role="shooter"
        ),
    )


def propose(name="Harry Kane", observation_id="shot", squad_id="england-squad"):
    return IdentityAction(
        kind="identify",
        identities=[
            IdentityProposal(observation_id=observation_id, squad_id=squad_id, player_name=name)
        ],
    )


async def identify(backend, observations=None, research=None):
    return await identify_players(
        backend,
        model="offline",
        observations=observations or [seen()],
        research=research or brief(),
        pack=None,
    )


async def test_researcher_squad_reaches_observer_tool_with_its_citation() -> None:
    import json

    backend = ScriptedBackend()
    backend.queue("replay_observer_identity", [IdentityAction(kind="research_roster"), propose()])
    observations, audit, _ = await identify(backend)
    assert observations[0].actor.resolved_name == "Harry Kane"
    first = json.loads(backend.calls[0].blocks[0]["text"])
    assert "Harry Kane" not in json.dumps(first)
    second = json.loads(backend.calls[1].blocks[0]["text"])
    assert second["tool_result"]["squads"][0]["source_url"] == squad().source_url
    assert audit[0]["action"]["kind"] == "research_roster"


async def test_agent_can_abstain_even_when_number_has_one_match() -> None:
    backend = ScriptedBackend()
    backend.always("replay_observer_identity", IdentityAction(kind="abstain"))
    observations, _, _ = await identify(backend)
    assert observations[0].actor is None
    assert len(backend.calls) == 1


@pytest.mark.parametrize(
    "proposal",
    [
        propose("Bukayo Saka"),
        propose(squad_id="unknown"),
        propose(observation_id="future"),
        propose("Invented Player"),
    ],
)
async def test_wrong_proposal_is_rejected_without_substituting_correct_name(proposal) -> None:
    backend = ScriptedBackend()
    backend.queue("replay_observer_identity", [IdentityAction(kind="research_roster"), proposal])
    observations, audit, _ = await identify(backend)
    assert observations[0].actor is None
    assert audit[-1]["rejected"]


async def test_even_correct_name_requires_the_agent_to_retrieve_context() -> None:
    backend = ScriptedBackend()
    backend.always("replay_observer_identity", propose())
    observations, _, _ = await identify(backend)
    assert observations[0].actor is None


async def test_duplicate_roster_number_and_wrong_team_are_rejected() -> None:
    for research in [
        brief(
            players=[
                SquadPlayer(name="Harry Kane", number=9),
                SquadPlayer(name="Other Player", number=9),
            ]
        ),
        brief(side="away"),
    ]:
        backend = ScriptedBackend()
        backend.queue(
            "replay_observer_identity", [IdentityAction(kind="research_roster"), propose()]
        )
        observations, _, _ = await identify(backend, research=research)
        assert observations[0].actor is None


async def test_celebrant_name_cannot_fill_unnamed_goal() -> None:
    backend = ScriptedBackend()
    backend.queue(
        "replay_observer_identity",
        [
            IdentityAction(kind="research_roster"),
            IdentityAction(
                kind="identify",
                identities=[
                    IdentityProposal(
                        observation_id="goal", squad_id="england-squad", player_name="Harry Kane"
                    ),
                    IdentityProposal(
                        observation_id="celebration",
                        squad_id="england-squad",
                        player_name="Harry Kane",
                    ),
                ],
            ),
        ],
    )
    goal = Observation(id="goal", at_s=0, kind="goal", description="Ball is inside the net.")
    celebration = seen("celebration", "celebration")
    observations, _, _ = await identify(backend, observations=[goal, celebration])
    assert observations[0].actor is None
    assert observations[1].actor.resolved_name == "Harry Kane"


async def test_tool_loops_and_failures_cannot_discard_visual_events() -> None:
    backend = ScriptedBackend()
    backend.always("replay_observer_identity", IdentityAction(kind="research_roster"))
    observations, _, _ = await identify(backend)
    assert observations[0].description == seen().description
    assert observations[0].actor is None
    assert len(backend.calls) == 2
    observations, audit, _ = await identify(ScriptedBackend())
    assert observations[0].description == seen().description
    assert audit[0]["error"] == "LLMError"


def test_future_squad_and_bad_sources_are_rejected() -> None:
    with pytest.raises(ValidationError, match="after as_of"):
        brief(source_date=date(2022, 12, 18))
    with pytest.raises(ValidationError, match="HTTP"):
        brief(source_url="file:///squad.json")
    with pytest.raises(ValidationError, match="unique"):
        brief(players=[SquadPlayer(name="Harry Kane", number=9)] * 2)


async def test_research_preparation_preserves_model_generated_squads() -> None:
    backend = ScriptedBackend()
    backend.extra_params = {}  # Search-capable test adapter; network is blocked by conftest.
    backend.always("replay_researcher", ResearchDraft(facts=[], squads=[squad()]))
    prepared = await prepare_research(
        backend, fixture="England v France", as_of=date(2022, 12, 10), model="offline"
    )
    assert prepared.brief.squads == [squad()]
    assert prepared.audit["model_called"]


async def test_research_team_context_reaches_vision_without_exposing_names() -> None:
    import json

    from commentary.recorded_demo import Frame
    from commentary.replay.preparation import ReplayObserver, ReplayObserverResult, team_names

    research = brief(kit_colours="white and navy")
    backend = ScriptedBackend()
    backend.always("replay_observer", ReplayObserverResult(observations=[]))
    await ReplayObserver(backend, model="offline", research=research)([Frame(0, b"test")], 0)
    payload = json.loads(backend.calls[0].blocks[0]["text"])
    assert payload["pack"] == {
        "teams": {
            "home": {
                "name": "England",
                "kit_colours": "white and navy",
            }
        }
    }
    assert "Harry Kane" not in json.dumps(payload)
    assert team_names(None, research) == {"home": "England"}
