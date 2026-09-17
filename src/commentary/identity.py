"""Conservative player resolution for structured action beats.

The caller can see a shirt better than it can remember a squad.  This module
therefore uses the pack to *resolve* a read, rather than using the pack to
guess a player.  A role can rank the plausible squad members, but it never
promotes one of them to a name on its own.  That distinction keeps a useful
``right-back crosses`` fact when the number cannot be read.

Unlike caller inference, continuity is stateful.  It is deliberately applied
by the runtime after concurrent model calls have been put back in cursor
order; applying it when a request finishes would let a slower older request
inherit a newer player's identity.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

from commentary.schemas import (
    Action,
    ActionBeat,
    CallerLine,
    Event,
    IdentitySource,
    KnowledgePack,
    Player,
    PlayerIdentity,
    Scene,
    Side,
    Sighting,
    Trigger,
)

# A number or a graphic has to be reasonably legible before it is allowed to
# turn into a roster name.  The action remains useful below this threshold.
VERIFIED_IDENTITY_CONFIDENCE = 0.70
# This is intentionally shorter than the old spoken-name carry: it bridges a
# receive, turn, and delivery, not an entire attacking spell.
CONTINUITY_WINDOW_S = 8.0

_TURNOVERS = frozenset({Event.INTERCEPTION, Event.CLEARANCE, Event.TACKLE})
_HANDOFFS = frozenset({Action.PASS, Action.LAYOFF, Action.CROSS})


@dataclass(frozen=True)
class RankedPlayer:
    """A pack player and the deterministic evidence score behind their rank."""

    side: Side
    player: Player
    score: int


@dataclass(frozen=True)
class _HeldIdentity:
    identity: PlayerIdentity
    ts: float


def rank_players(
    pack: KnowledgePack | None,
    identity: PlayerIdentity,
    *,
    fallback_side: Side = Side.UNKNOWN,
) -> list[RankedPlayer]:
    """Rank pack candidates without turning a likely role into a claimed name.

    Number and readable-name evidence dominate.  Side and position only break
    ties and make the resulting ordering explainable in tests and traces.
    Contradictory name/number combinations have no candidates at all.
    """
    if pack is None:
        return []
    side = identity.side if identity.side is not Side.UNKNOWN else fallback_side
    candidates: list[RankedPlayer] = []
    for candidate_side in (Side.HOME, Side.AWAY):
        if side is not Side.UNKNOWN and candidate_side is not side:
            continue
        sheet = pack.team(candidate_side)
        if sheet is None:
            continue
        for player in sheet.squad:
            number_match = identity.number is not None and player.number == identity.number
            name_match = bool(identity.name and _same_player_name(identity.name, player.name))
            role_match = bool(identity.role and _role_matches(identity.role, player.position))
            # Two positive, incompatible assertions are a bad shirt read, not
            # a reason to choose whichever half happens to fit the roster.
            if identity.number is not None and identity.name and not (number_match and name_match):
                continue
            score = 0
            if number_match:
                score += 100
            if name_match:
                score += 100
            if side is not Side.UNKNOWN:
                score += 20
            if role_match:
                score += 10
            if number_match or name_match or role_match:
                candidates.append(RankedPlayer(candidate_side, player, score))
    return sorted(candidates, key=lambda item: (-item.score, item.player.name.casefold()))


def resolve_explicit_identity(
    identity: PlayerIdentity | None,
    pack: KnowledgePack | None,
    *,
    fallback_side: Side = Side.UNKNOWN,
) -> tuple[PlayerIdentity | None, bool]:
    """Canonicalise a well-supported name/number, returning ``(value, bad)``.

    ``bad`` means the visible identity contradicts the pack.  Callers use it
    to stop an in-flight carry, while retaining the action and any role the
    image actually supports.
    """
    if identity is None:
        return None, False
    # A pack validates and canonicalises a read; it does not make an
    # unsupported identity false.  Matches without a pack still retain what
    # the picture reported, but cannot bootstrap continuity from it.
    if pack is None:
        return identity, False

    side = identity.side if identity.side is not Side.UNKNOWN else fallback_side
    proposed = identity.model_copy(update={"side": side})
    ranked = rank_players(pack, proposed, fallback_side=side)
    has_read = proposed.number is not None or bool(proposed.name)
    if has_read and not ranked:
        # Do not leak a roster-invalid name into later stages.  The visible
        # number/role remains useful, but cannot bootstrap a carry.
        return proposed.model_copy(update={"name": None}), True
    source_supports_name = (
        proposed.number is not None and proposed.source is IdentitySource.SHIRT_NUMBER
    ) or (
        proposed.name is not None
        and proposed.source
        in {
            IdentitySource.SHIRT_NAME,
            IdentitySource.GRAPHIC,
            IdentitySource.MATCH_STATE,
        }
    )
    if (
        not ranked
        or proposed.confidence < VERIFIED_IDENTITY_CONFIDENCE
        or not source_supports_name
        or not _decisive(ranked)
    ):
        # Keep a visible number or role as a useful fallback, but never let a
        # weak/ambiguous name pass itself off as a verified roster identity.
        return proposed.model_copy(update={"name": None}), False

    winner = ranked[0]
    return (
        proposed.model_copy(
            update={
                "name": winner.player.name,
                "number": winner.player.number,
                "side": winner.side,
            }
        ),
        False,
    )


class IdentityContinuity:
    """Resolve action identities and retain one verified ball player briefly."""

    def __init__(
        self, pack: KnowledgePack | None, *, window_s: float = CONTINUITY_WINDOW_S
    ) -> None:
        self.pack = pack
        self.window_s = window_s
        self._held: _HeldIdentity | None = None
        self._last_resolved_ts = float("-inf")

    def resolve(
        self,
        line: CallerLine,
        ts: float,
        triggers: Iterable[Trigger] = (),
        *,
        cut_timestamps: Iterable[float] | None = None,
    ) -> CallerLine:
        """Return a line with canonical/carry identities, advancing state once.

        The caller's form is still observed when it chose silence: a silent
        pass to a verified receiver is exactly the evidence a later cross may
        need.  Replays and non-live scenes never bridge live possession.
        """
        cuts = tuple(cut_timestamps) if cut_timestamps is not None else ()
        triggers = tuple(triggers)
        if self.pack is None:
            self.clear()
            return line
        if line.scene is not Scene.LIVE_PLAY:
            self.clear()
        elif cut_timestamps is None and Trigger.CAMERA_CUT in triggers:
            # Direct callers without the runtime's timestamped cut history
            # still get the safe answer.  Runtime callers always pass cuts.
            self.clear()
        elif self._crosses_cut(ts, cuts):
            self.clear()

        # An explicit side change or a defensive action ends the old player’s
        # possession before this observation can inherit it.
        if line.event in _TURNOVERS or self._line_changes_side(line):
            self.clear()

        resolved: list[ActionBeat] = []
        for beat in line.actions:
            beat_ts = beat.video_ts if beat.video_ts is not None else ts
            if self._crosses_cut(beat_ts, cuts):
                self.clear()
            actor_input = _support_from_sightings(beat.actor, line.sightings)
            target_input = _support_from_sightings(beat.target, line.sightings)
            actor, actor_bad = resolve_explicit_identity(
                actor_input, self.pack, fallback_side=line.side
            )
            target, _target_bad = resolve_explicit_identity(
                target_input, self.pack, fallback_side=line.side
            )
            if actor_bad or self._contradicts_held(actor):
                self.clear()

            if line.scene is Scene.LIVE_PLAY and not actor_bad and actor is not None:
                actor = self._carry_into(actor, beat_ts)
            elif line.scene is Scene.LIVE_PLAY and not actor_bad and actor is None:
                actor = self._carried_actor(None, beat_ts)

            resolved_beat = beat.model_copy(update={"actor": actor, "target": target})
            resolved.append(resolved_beat)
            if line.scene is Scene.LIVE_PLAY:
                self._advance(resolved_beat, beat_ts)

        self._last_resolved_ts = max(self._last_resolved_ts, ts)
        return line.model_copy(update={"actions": resolved})

    def clear(self) -> None:
        self._held = None

    def _crosses_cut(self, ts: float, cuts: Iterable[float]) -> bool:
        return any(self._last_resolved_ts < cut <= ts for cut in cuts)

    def _line_changes_side(self, line: CallerLine) -> bool:
        held = self._held
        return bool(
            held
            and line.side is not Side.UNKNOWN
            and held.identity.side is not Side.UNKNOWN
            and line.side is not held.identity.side
        )

    def _contradicts_held(self, identity: PlayerIdentity | None) -> bool:
        held = self._held
        if identity is None or held is None or not (identity.name or identity.number is not None):
            return False
        prior = held.identity
        if (
            identity.side is not Side.UNKNOWN
            and prior.side is not Side.UNKNOWN
            and identity.side is not prior.side
        ):
            return True
        if (
            identity.number is not None
            and prior.number is not None
            and identity.number != prior.number
        ):
            return True
        return bool(
            identity.name
            and prior.name
            and not _same_player_name(identity.name, prior.name)
        )

    def _carry_into(self, identity: PlayerIdentity, ts: float) -> PlayerIdentity:
        if identity.name is not None or identity.number is not None:
            return identity
        carried = self._carried_actor(identity, ts)
        return identity if carried is None else carried

    def _carried_actor(self, observed: PlayerIdentity | None, ts: float) -> PlayerIdentity | None:
        held = self._held
        if held is None or ts < held.ts or ts - held.ts > self.window_s:
            return None
        if observed is not None and observed.side not in (Side.UNKNOWN, held.identity.side):
            return observed
        role = observed.role if observed is not None and observed.role else held.identity.role
        side = (
            observed.side
            if observed is not None and observed.side is not Side.UNKNOWN
            else held.identity.side
        )
        return held.identity.model_copy(
            update={
                "role": role,
                "side": side,
                "source": IdentitySource.POSSESSION_CONTINUITY,
                "evidence": f"continuous {side.value} possession since {held.ts:.1f}s",
            }
        )

    def _advance(self, beat: ActionBeat, ts: float) -> None:
        # A named pass recipient is the next ball player.  An unnamed target
        # cannot justify carrying the passer through a completed pass.
        next_identity = beat.target if beat.action in _HANDOFFS else beat.actor
        if _is_verified(next_identity):
            assert next_identity is not None
            held = self._held
            if (
                next_identity is not None
                and next_identity.source is IdentitySource.POSSESSION_CONTINUITY
                and held is not None
                and held.identity.name
                and _same_player_name(next_identity.name or "", held.identity.name)
            ):
                # An inherited name does not become fresh evidence merely
                # because another observation used it.  Otherwise a chain of
                # anonymous forms could carry one player forever.
                return
            self._held = _HeldIdentity(next_identity, ts)
        elif beat.action is Action.PASS:
            self.clear()


def _is_verified(identity: PlayerIdentity | None) -> bool:
    return bool(
        identity
        and identity.name
        and identity.confidence >= VERIFIED_IDENTITY_CONFIDENCE
        and identity.source
        in {
            IdentitySource.SHIRT_NUMBER,
            IdentitySource.SHIRT_NAME,
            IdentitySource.GRAPHIC,
            IdentitySource.MATCH_STATE,
            IdentitySource.POSSESSION_CONTINUITY,
        }
    )


def _decisive(ranked: list[RankedPlayer]) -> bool:
    """Whether the leading pack candidate beats every alternative."""
    return len(ranked) == 1 or ranked[0].score > ranked[1].score


def _support_from_sightings(
    identity: PlayerIdentity | None, sightings: list[Sighting]
) -> PlayerIdentity | None:
    """Add a matching line-level shirt read without guessing which body it is.

    A sighting only helps an action identity that already shares its readable
    number or name.  A lone sighting elsewhere in the frame may be a defender
    or the goalkeeper, so it is never assigned to an anonymous actor.
    """
    if identity is None or not sightings:
        return identity
    matching = [
        sighting
        for sighting in sightings
        if (
            identity.number is not None
            and sighting.number == identity.number
            and _same_side_or_unknown(sighting.side, identity.side)
        )
        or (
            identity.name
            and sighting.name
            and _same_player_name(identity.name, sighting.name)
            and _same_side_or_unknown(sighting.side, identity.side)
        )
    ]
    if len(matching) != 1:
        return identity
    sighting = matching[0]
    return identity.model_copy(
        update={
            "name": sighting.name or identity.name,
            "number": sighting.number if sighting.number is not None else identity.number,
            "side": sighting.side if sighting.side is not Side.UNKNOWN else identity.side,
            "source": (
                IdentitySource.SHIRT_NUMBER
                if sighting.number is not None
                else IdentitySource.SHIRT_NAME
            ),
        }
    )


def _same_player_name(left: str, right: str) -> bool:
    left_words = _name_words(left)
    right_words = _name_words(right)
    return bool(
        left_words
        and right_words
        and (left_words == right_words or left_words[-1] == right_words[-1])
    )


def _same_side_or_unknown(left: Side, right: Side) -> bool:
    return left is Side.UNKNOWN or right is Side.UNKNOWN or left is right


def _name_words(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = "".join(char for char in normalized if not unicodedata.combining(char))
    return tuple(re.findall(r"[a-z0-9]+", ascii_value.casefold()))


def _role_matches(role: str, position: str | None) -> bool:
    if not position:
        return False
    wanted = _role_tokens(role)
    actual = _role_tokens(position)
    if not wanted or not actual:
        return False
    if wanted == actual:
        return True
    # Broad position groups are only tiebreakers: a right-back matches a
    # right wing-back, but not a generic right midfielder.
    groups = (
        ("goalkeeper", "gk"),
        ("back", "defender"),
        ("midfield", "midfielder"),
        ("forward", "striker", "winger"),
    )
    same_group = any(
        any(word in wanted for word in group) and any(word in actual for word in group)
        for group in groups
    )
    wanted_side = {"left", "right"} & wanted
    return same_group and (not wanted_side or wanted_side == ({"left", "right"} & actual))


def _role_tokens(value: str) -> set[str]:
    compact = re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
    aliases = {
        "rb": "right back",
        "lb": "left back",
        "cb": "center back",
        "gk": "goalkeeper",
        "dm": "defensive midfield",
        "cm": "center midfield",
        "am": "attacking midfield",
        "rw": "right wing",
        "lw": "left wing",
        "cf": "center forward",
        "st": "striker",
    }
    compact = aliases.get(compact, compact).replace("centre", "center")
    words = set(compact.split())
    if "back" in words:
        words.add("defender")
    if "midfield" in words:
        words.add("midfielder")
    return words
