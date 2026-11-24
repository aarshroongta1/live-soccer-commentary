# Learning an editorial preference

Share: https://www.aarshroongta.com/demo/trajectory

The recorded demo has two steps: Review and Compare. On the first clip, Aarsh
kept the existing Ballard commentary because it “uses relevant facts about the
player's career and how the team has been performing to enrich the commentary”.
That actual review was saved in Trajectory and read back before reuse.

On different footage, both model runs receive identical six frames, identity
context and three UEFA background facts. The baseline describes the action. The
revised run includes Messi's 100th UEFA appearance. This demonstrates a change
consistent with the stated preference, not a proven quality improvement. There
is no human judgment between the final takes yet. The baseline also incorrectly
calls Messi's chip a low finish, so neither factual accuracy nor general quality
is established by this example.

Simply including the original positive feedback did not add background context.
That first attempt is retained in the public evidence. One revision distilled the
review into a reusable preference and applied it with an explicit, application-
authored instruction to favor one relevant supplied fact. No repeated search for
a winning output followed. The policy extraction and new visual generation ran
through Trajectory, with native traces verified before public export. Trajectory
stores the actual feedback and traces and supplies inference; the application
implements the feedback-to-instruction connection. No weights were trained and
no native Continual Grading feature is claimed.

The inferred preference adds “concise”, which the original reason did not
explicitly request. The extra instruction to select background is also a design
choice, so this is an adaptive development example rather than an isolated test
of the user's feedback. Background facts were prepared after the first review.
The source UEFA page was published before the match but updated afterward; an
archived historical version was not verified. One case cannot establish general
improvement, timing performance, or reliable preference learning.

## Record a 60-second walkthrough

1. Open the share link, play the first clip using “Replay the line”, and show the
   saved Keep decision and the user's reason.
2. Point to the preference inferred from that review.
3. Open Compare, play the Messi clip, and listen to the baseline and revised take.
   Highlight the career milestone newly included in the revised take.
4. Say: “Raw feedback alone didn't transfer here. Making the preference explicit
   changed the next commentary. These are real Trajectory runs, with no training.”

The public demo replays the verified run. Visitor judgments are browser-local;
its API cannot spend credits or modify Aarsh's Trajectory account. Evidence is
linked in the demo. The connected owner session is
http://127.0.0.1:3010/demo/editor .

## Reproduce and verify

- `scripts/prepare_commentary_editor.py`: offline source footage/frame preparation.
- `scripts/commentary_editor.py`: real review capture and initial comparison.
- `scripts/finish_commentary_learning.py`: one bounded policy revision; completed
  invocations reuse saved state and incomplete runs require inspection.
- `scripts/export_commentary_editor.py`: read-only native verification and export.

Run SDK workers with the installed `trajectory-sdk` Python environment. Raw
responses and input hashes are retained in gitignored
`runs/commentary-editor/session.json`. Public evidence excludes credentials,
reviewer email, request IDs and inline images. Tests use a fake client and no
paid API calls:

```sh
.venv/bin/python -m pytest tests/test_commentary_editor.py tests/test_trajectory_preference_spec.py -q
```
