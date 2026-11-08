# Continuity comparison

The user found the previous candidate better but heard disconnected sentences.
The previous render left 4.17 seconds between the penalty award and goal, then
2.15 seconds between the caller and analyst. Each phrase was a separate speech
request despite textual context being supplied.

This candidate keeps the earlier attack/decision words and analyst recording.
A Mini caller turn adds the recognized run-up at 9 seconds, followed by a connected
goal turn. Their bounded reviewer traces are retained in [trace.json](trace.json).
The caller's four phrases are synthesized as one performance. Provider character
[alignment](alignment.json) identifies the between-phrase cuts; the renderer
places these phrases on the original event cues. No words are manually rewritten.
The analyst is placed 250 ms after the measured caller recording ends, respecting
the availability of all cited evidence. [Measured placements](render-report.md)
show no truncated speech, overlaps or early event announcements.

The scheduler also now tries the caller when a short slot cannot accommodate
the analyst, rather than skipping the source frontier entirely. That skip had
prevented the run-up observation from being called.

This tests shared prosody and handoff timing, not a different voice model. It
retains purposeful pauses during setup and after analysis; stadium ambience is
still absent. Grouping speech does not make it continuous over the entire video.
Subjective improvement remains for listening review.

The two completed Mini turns/reviews report $0.0066885 combined. An earlier attempt
hit its $0.01 model reservation cap before saving its audit; its exact usage was
not retained. Model spend for this comparison is therefore below $0.017, rather
than claiming the known subtotal is complete. One timestamped caller request used
44 voice credits. The analyst audio is reused without another paid request.

Reproduce speech using `scripts/synthesize_demo.py --continuous` with a caller-only
copy of [candidate-plan.json](candidate-plan.json). Merge that manifest with the
saved unchanged analyst audio, then render the full plan with `--response-gap 0.25`.
Alternatively synthesize both blocks from the full plan (this also charges for
the analyst). Both tools default to the same Flash model and saved voice settings.
Raw audio, the provider response including audio, and render sidecars remain
locally under `runs/recorded-agents/penalty-continuity-v2/`.

The local listening page is `/commentary-comparison.html#continuity`; media is
ignored by Git and no main demo asset has been replaced.
