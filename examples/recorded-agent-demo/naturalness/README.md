# Listening candidate

The user rejected the earlier prepared penalty demo for unnatural writing, voice
delivery and timing. Passing factual and timing checks did not establish quality.

The reproduction showed 4–13-word caps, a mandatory single-sentence instruction,
immediate analyst handoffs, isolated speech requests, and no stadium audio track.
The first two new Mini drafts and a stronger-model draft were still generic;
their generated text and costs are retained in [evaluation.json](evaluation.json).

The candidate separates the caller's reaction from the analyst's explanation,
allows varied phrases within the available slot, leaves 1.2 seconds after an
important call before the next turn, and runs a reviewer with at most one rewrite.
Speech uses the same custom voices and Flash model, with same-voice textual
context between requests. It does not change the voice settings or add ambience.
These are hypotheses for listening review, not a claim that voice quality is fixed.

The final [plan](candidate-plan.json) is model-written without manual word edits.
Its [trace](dialogue-trace.json) retains the review decisions. GPT-5.4 wrote the
deliberate quality candidates; Mini handled the earlier drafts and reviews.
All four text passes together cost $0.05472875; one four-line speech render used
87 credits. No training, new footage, research or recognition calls were needed.

Locally, compare the original and candidate at
`http://127.0.0.1:3000/commentary-comparison.html`.
The candidate video is ignored by Git, like the existing demo media. It has not
replaced the main demo. The source clip has no audio track; silent gaps will
still lack stadium atmosphere. An ambience track and an expressive speech-model
comparison remain separate audio work, rather than being hidden by a script score.
