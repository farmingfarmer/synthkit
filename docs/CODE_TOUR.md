# A tour of the code, for the deep dive

Written for a reader who asked to see the back end, not the
slides. Every claim here is a pointer into a file; nothing is
described that cannot be opened and read in the same sitting.

## The shape of the system

Two halves, meeting in the middle:

- **The measuring half** learns a k-anonymous blueprint from a
  real extract and generates from the blueprint alone.
  `synthkit/discover.py` (what explains each column, confirmed on
  held-out patients), `synthkit/blueprint.py` (what may be
  published - the privacy decisions live here), and
  `synthkit/generate.py` (drawing new patients from the contract,
  never from a record).
- **The examining half** turns a spec into a tiered exam with a
  planted answer and grades any solver against the ceiling that
  answer implies. `synthkit/tablespec.py`, `synthkit/campaign.py`,
  `synthkit/autosolver.py`, `synthkit/semisynth.py` (the planting
  - effects declared in standard deviations, intercept SOLVED by
  bisection rather than centered).
- **The bridge** (`synthkit/bridge.py`) carries measured marginals
  and correlations from the first half into the second, and states
  on the artifact what did NOT cross. Outcomes never cross -
  nobody knows the answer in real data, which is the entire reason
  the planting step exists.

## The one-computation rule

Anywhere two surfaces show the same number, they call the same
module - because two implementations of one idea is how the two
sides of a codebase come to disagree:

- `synthkit/gate.py` - the eight release criteria, their
  explanations, and the what-now guidance. Consumed by the bench's
  Verdict station, `scripts/m0_gate.py`, and the sign-off page.
  The three cannot disagree.
- `synthkit/curvecheck.py` - shape and interaction-surface
  fidelity. The pipeline records it, the gate reads it, the
  dashboard draws from the identical functions.
- `synthkit/sets.py` - what counts as scaffolding, one predicate,
  called by both halves.
- `scripts/fidelity_deck.py` `build_deck` - the dashboard the
  bench serves and the file the team receives are one build.

## The measurement discipline

`CONVENTIONS.md` is the project's lab notebook: every rule in it
was paid for by a specific failure, recorded with its numbers.
The load-bearing habits:

- **Nothing ships on one seed.** `scripts/pair_fidelity_sweep.py`
  and `scripts/repro_mechanisms.py` are the gates; a change to
  generation is measured against them across seeds, with
  inversions held at zero absolutely. Changes have been built,
  measured worse, and reverted WITH their numbers - the reverts
  are in the git history on purpose.
- **Checks must be able to fail.** New checks are run against the
  old code (or a perturbed one) and watched go red before they are
  believed. The suite count is stated in `docs/WINDOWS.md` for
  both a checkout and a zipball, and the two differ for a stated
  reason.
- **A refusal is a sentence, not a stack.** Every CLI failure path
  names the file, the cause, and the next command.

## Where the artifacts come from

Every run directory is self-describing: `blueprint.json` (the
k-anonymous contract), `fidelity.json` (every comparison, plus the
generation report), `findings.txt` (the run's own narration),
`provenance.json` (the build the run is tied to). The dashboard,
the report card (`scripts/report_card.py`) and the sign-off page
(`scripts/signoff.py`) are assembled FROM those files and
recompute nothing.

## Reading order for a first sitting

1. `synthkit/gate.py` - small, and the house style is all there:
   the criteria, the explanations, the guidance, one module.
2. `synthkit/semisynth.py` `plant` - the answer key, and why
   effects are in standard deviations.
3. `synthkit/generate.py` from `_order` - cycle-breaking, the
   refinement sweeps, and the synchronized final pass, each with
   the measurement that justified it in the comment above it.
4. `scripts/repro_mechanisms.py` - how a real-data failure becomes
   a fixture with a gate.
5. `CONVENTIONS.md` end to end, when there is an hour: it is the
   project's memory, and the code makes more sense after it.
