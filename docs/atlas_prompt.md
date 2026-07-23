# PROMPT: Build a "System Atlas" for this project

Build me an interactive system atlas for this project: a single
self-contained HTML file that presents the entire system as a
clickable drill-down map, from a pipeline overview all the way
down to the actual commented source code. I will use it to walk
coworkers through the system. Follow this specification exactly.

## The artifact

ONE file (e.g. `docs/system_map.html`), zero dependencies, pure
HTML/CSS/vanilla JS + inline SVG. It must open from disk, offline,
on a locked-down machine. No CDNs, no frameworks, no build step at
view time.

## The generator (this is the important part)

Do NOT hand-write the HTML with pasted code. Write a generator
script (e.g. `scripts/build_system_map.py`) that:

1. Holds the atlas as a data structure: domains -> modules ->
   components, where each component names a real symbol
   (function/class/method) in a real source file.
2. Extracts each symbol's ACTUAL source at build time by parsing
   the file (Python: use `ast` with `lineno`/`end_lineno`;
   truncate snippets over ~90 lines with a "... N more lines"
   marker). The map must carry live code, never paraphrase.
3. FAILS LOUDLY if any referenced symbol has moved or vanished
   ("MAP DRIFT: `X` not found in Y — update the generator").
   The build refusing is the map's accuracy guarantee.
4. Embeds the data as JSON in a `<script type="application/json">`
   tag (escape `</` as `<\/`), stamps the page with the project's
   version/build info if it has any, and writes the single file.
5. Ends with a structural self-check: re-open the output, parse
   the embedded JSON, assert every component carries code and
   every interactive function marker is present.

Rebuilding after any code change is the maintenance ritual; say so
in the page footer.

## The three layers

**Layer 1 — the pipeline hub.** NOT a radial/solar-system layout.
Order the domains as the DATA'S JOURNEY, left to right: input
enters at the far left (label it, e.g. "an English paragraph ->"),
numbered stage nodes (01, 02, ...) connected by arrows, output
exits at the far right (e.g. "-> a measured verdict"). Each stage
node shows a 2-3 word stage name inside and a one-line
"what is accomplished here" note beside it (alternate notes
above/below the flow line to avoid crowding). Cross-cutting
domains that touch every stage (quality gates, interfaces,
infrastructure) are NOT stages: render them as full-width
clickable RAILS beneath the pipeline with faint vertical ties to
each stage.

**Layer 2 — domain view.** Clicking a stage or rail shows that
domain's modules as cards on connecting curves, keeping the
domain's color. Each card: module name, component count, filename.
Breadcrumbs at top-right navigate back (atlas / domain / module).

**Layer 3 — module explorer.** Two-panel: component chips on the
left (accent-striped in the domain color, active state), and on
the right a code box: header (path + symbol), a short blurb
paragraph, then the highlighted source. Clicking the first chip
auto-loads.

## Content rules

- 6-9 domains total; 2-4 modules per domain; 2-3 components per
  module (~25-40 components overall). Choose the MONEY symbols:
  the functions that embody the system's key decisions, not
  boilerplate.
- Every blurb tells WHY the code exists, ideally citing a real
  incident, number, or decision from the project's history
  ("built after X broke", "this closed the Y bug"). The blurbs
  are the tour script.
- Each domain gets: a color, a 1-2 sentence blurb (shown in
  layer 2), a `stage` label, and a `note` (the accomplished-here
  line, may contain one \n for two lines).

## Visual identity

Match the project's existing design language if it has one;
otherwise: light neutral background (#EDF0F3-ish), white panels,
1px hairline borders, ink #17222C, dim #5C6B78, monospace-forward
type (ui-monospace stack) for all labels/code, a distinct color
per domain (muted, professional — no neon), minimal motion (one
0.16s fade, respect prefers-reduced-motion). Syntax highlighting
via a small regex pass: comments green-italic, strings muted red,
keywords blue-bold. Header: project wordmark + "system atlas ·
click nodes to descend" + version stamp. Footer: domain/module/
component counts + the rebuild command.

## Process requirements

- Read any available frontend-design guidance before writing UI.
- Validate by running: execute the generator, then run the
  structural self-check, then fix whatever fails. Never deliver
  unexecuted.
- Deliver the generator AND the built HTML, with a short install
  block (where to put them, the rebuild command, git add/commit).
- If the project has smoke tests, note that the generator's
  drift-refusal makes it a de facto accuracy check; suggest
  adding it to the release ritual.

## Reference implementation

This pattern was first built for the synthkit project
(farmingfarmer/synthkit, `scripts/build_system_map.py` +
`docs/system_map.html`): 8 domains (6 pipeline stages + 2 rails),
34 ast-extracted components, 90 KB single file. Consult it if
available; otherwise this spec is complete.
