# Demo walkthrough

## The 30-second explanation

“I built a recorded football commentary prototype. A vision model turns sampled
frames into timestamped observations. A caller agent reacts to the play, and an
analyst agent adds an explanation. They share evidence, but have different prompts
and tool permissions. Python controls turn-taking and checks references and timing.
I evaluated it on 15 short development clips, found both vision and coordination
failures, and kept the failed runs so I could inspect what went wrong.”

These are existing models, not a model you trained. The featured videos are
prepared before playback, not live commentary. Speak about the implementation and
experiments you can explain; use the code walkthrough below to check that understanding.

## A two-minute demonstration

1. Open `/demo` and play the 20-second England–France example. Point out the caller
   reacting and the analyst following. The generated words are unedited.
2. For the implementation, **How I built it** opens the GitHub README. For the
   comparison, open `/demo/engineering/runs` and switch to **Before: reviewer enabled**. At the goal, the caller is missing.
   Open the explanation: the reviewer rejected even “Kane scores!” twice.
3. Switch back to **After: reviewer disabled**. The caller announces the goal.
   Deterministic citation, permission and timing checks still run. This one pair
   used 17 versus 9 calls, but also started later: 1s versus 3s. It is a concrete
   case study, not proof of general quality or cost improvement.
4. Show Arsenal or Argentina. These use fresh vision and generated dialogue, with
   disclosed source cuts and light edits. Open **See the exact light edits**.
   Arsenal keeps every retained word; Argentina removes one shirt-number phrase.
5. Finish with the main limitation: if the observer confidently sees the wrong
   event, multiple writers can repeat it. Valid JSON and citations do not prove truth.

The three downloadable MP4s are silent, captioned clips. No API keys or paid calls
are needed to play them. The older heavily rewritten showcases are not featured.

## Four pieces of code to understand

| Open this | What to follow |
| --- | --- |
| `scripts/prepare_agent_demo.py`, `run` | Load the source, team context and budget; run vision, then generate dialogue; save the plan and trace. |
| `src/commentary/replay/recorded.py`, `observe_recorded` | Sampled images become typed observations. This recorded path sees the whole short sequence before playback. |
| `src/commentary/replay/agents.py`, `_payload` and `_build_graph` | A role receives evidence and recent lines. It chooses speak, a permitted context tool, or abstain. A tool result routes back to the same role; the reviewer is optional. |
| `src/commentary/replay/recorded.py`, `prepare_dialogue` | Code selects turns, restricts evidence by supporting timestamps, computes word budgets, checks proposed lines, and records rejections. It does not write the dialogue. |

A separate causal replay implementation has asyncio deadlines and stale-result
handling. Keep that distinction clear: those controls are not evidence that the
featured recorded videos achieve real-time spoken latency.

## Questions to answer in your own words

- **Why agents?** Separate caller/analyst objectives, evidence permissions and tools
  make decisions inspectable. More agents also add cost and failure paths. The
  evaluation did not prove they outperform one writer.
- **What does LangGraph do?** It routes a role's bounded action loop through its
  tools and optional review. It does not control the video clock or verify pixels.
- **What does Pydantic do?** It checks the shape and structural rules for evidence,
  actions and lines. It cannot tell whether a ball actually crossed the line.
- **What did evaluation change?** Short-window word budgets and permission checks
  were tightened. Focused review then found an over-rejecting critic; disabling that
  optional step recovered the caller’s goal turn in the retained example.
- **What was manually edited?** The page links the exact edit records. No football
  action, player name, statistic or explanation was added to the new featured text.
- **Would a stronger model fix it?** Not reliably in these checks. A fresh stronger
  observer attempt also failed schema validation, and the false-goal investigation
  previously reproduced an error with a larger model.
- **What would you improve next?** Independently verify uncertain high-impact events
  and review more outputs against footage before claiming general accuracy.

## Resume bullet

Built a Python/LangGraph football commentary prototype with vision-grounded caller
and analyst agents, bounded context tools, and timestamped evidence checks;
evaluated 15 development clips and retained reproducible failure traces.

Do not claim a trained model, production deployment, measured live speech latency,
benchmark accuracy, or a suite-wide quality gain. If asked about AI assistance,
describe it candidly and explain the actual design and evaluation decisions.
