# Demo presentation

Reference: Aarsh's live website and the local personal-website/src/index.css,
reviewed 10 November 2026. Keep Space Grotesk, the left navigation rail, blue
links, restrained dividers and an optional dark theme. The default matches the
website's light presentation. Avoid a fake terminal or simulated agent activity.

Tokens: paper #ffffff, ink #16161a, body #26262b, muted #62626c,
rule #e7e7e9, accent #2547d0. Dark mode uses the site's #0f1011 base,
#ededf0 ink and #8fa9ff accent. Space Grotesk for reading; existing Geist Mono
only for actual timestamps, model IDs and saved run metadata.

Layout:

    name / project rail   title + short description
    demo                  clip selection
    pipeline              recorded footage | transcript / evidence / run
    evaluation            provenance + optional before/after findings
                          pipeline and evaluation below

The actual footage is the focus. The inspector uses real saved data and makes
clear that preparation happened before playback. Clip provenance remains visible.
The reviewer comparison is a saved development example, never a live agent demo.

Brief review: a dark green terminal dashboard would not match the reference.
Use the website's white/blue presentation, with functional developer tooling
(timestamps, inspectable evidence, downloadable runs), rather than decorative code.
No new model calls, changes to generated commentary, or public deployment.

## Recruiter-facing refinement

User feedback: the workbench had too much text and backend detail for recruiters.
The main route now has only a title/sentence, clip choices and a full-width video,
with short provenance and recorded/silent labels. No rail, stack tags, inspector,
notes, metrics, evaluation or comparison controls appear there. The existing
workbench is preserved at /demo/engineering through “How I built it”.
