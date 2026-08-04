# synthkit

Synthetic datasets with a signed answer key — for testing data
tools, models, and vendor claims against truth you planted.

Describe a dataset in plain English. synthkit compiles it to a
strict spec (LLM-assisted, never trusted, human-gated), plants
realistic mess and outcome signal deterministically, and keeps a
cell-level ledger of every corruption. Then it turns your goal
into a tiered campaign, trains its own blinded baseline, and
reports any solver's numbers next to the theoretical ceiling —
with confidence intervals and sample-size prescriptions.

```
ceiling 0.678 / synthkit baseline 0.667 / vendor 0.520  [FAIL]
```

## Two ways to get a dataset

**From a description.** Write what you need in plain English;
synthkit compiles it to a reviewable recipe, plants the truth, and
renders the data.

**From an existing extract.** Point it at real tables and it
measures them — distributions, missingness, contamination, and the
relationships between fields, including nonlinear ones a
correlation cannot represent — then hands back a recipe where
every number is editable. Real data teaches PARAMETERS; generation
never touches a record, k-anonymity counts PATIENTS rather than
rows, and a privacy scorecard verifies that no synthetic record
sits closer to a real person than real people sit to each other.

```
python scripts/phase2_pipeline.py --src DIR -o OUT --engine both
```

Seven stages, one command: wrangle, label, diagnose, profile,
compile, generate, score. Add `--transcribe` to render the
structured facts into messy clinical notes with a truth ledger, so
extraction can be graded by the KIND of mess that defeated it.

## Patients, not rows

Real records have people in them. Someone whose blood pressure runs
high at one visit tends to run high at the next; a patient with
four visits is a different thing from four patients with one. Any
analysis that groups by patient — which is most clinical analysis —
only behaves correctly when the data has that structure.

```
python scripts/phase2_pipeline.py --src DIR -o OUT --engine condnet --hierarchical
```

Each synthetic person gets fixed traits that stay fixed, a visit
count drawn from the real distribution, and values that drift from
one visit to the next. The steadiness is calibrated against the
source: measured visit-to-visit correlation 0.67 against a source
of 0.78, where sampling rows independently gives 0.03. Overshooting
is treated as a failure too — synthetic patients steadier than real
ones would flatter every model tested on them.

## Free text, and grading what reads it

Most clinical value is locked in prose, and an exam made only of
columns cannot test for it. Add `--transcribe` and the structured
facts are written into notes the way clinicians write — shorthand,
denials, hedges, findings carried forward, transposed digits — with
a hidden ledger of what each sentence actually asserts.

That ledger makes it possible to grade a reader by the KIND of mess
that beat it. On 800 generated notes, two readers that both look
competent fail in opposite directions: a keyword matcher scores
perfect recall and falls for every negated mention, while a
clause-scoped reader fixes negation entirely and then misses every
scope trap — a sentence like "no improvement in heart failure",
where the negation does not reach the finding.

Nothing here learns language from real notes. Facts are authored
and rendered, so no phrasing can be traced to a patient.

## Privacy, attacked rather than asserted

Architecture arguments are not measurements. An adversary is handed
the published model and asked which patients it was built from;
0.5 is a coin flip.

```
python scripts/end_to_end.py --patients 500 --epsilon 1.0 --report
```

`--epsilon` adds calibrated noise covering every published
quantity — conditional tables, transition tables, the visit-count
histogram, the bin edges and the steadiness targets — with each
person's contribution capped so the sensitivity is bounded.

Both the attack and the cost of a budget depend strongly on cohort
size — **at a fixed width of roughly 30 columns**, which is the
condition under which every number below was measured:

| patients | attack | structure kept at ε=1 |
|---------:|-------:|----------------------:|
|       92 |  0.619 |                   11% |
|      250 |  0.478 |                   67% |
|      600 |  0.501 |                   69% |
|    1,500 |  0.502 |                   87% |
|    4,000 |  0.509 |                   95% |

At that width, above roughly 250 patients membership stops being
recoverable, above roughly 600 a formal ε=1 guarantee becomes
affordable, and at 4,000 it is nearly free. The noise protects one
person's contribution, so it overwhelms a small cohort and rounds to
nothing in a large one.

**The width condition is load-bearing and the table does not
generalise without it.** Records get more identifying as columns are
added, so a cohort size that puts the attack at chance in 30 columns
need not do so in 88. Measured on an 88-column extract, 800 patients
returned 0.600 — MARGINAL — with no budget set at all, where this
table would predict roughly 0.501. Read the rows as "patients at ~30
columns", never as "patients".

Cohort size also sets how precisely the attack itself can be
measured, and that is a separate question from how large the attack
is. The audit scores cohort members against the held-out remainder,
so its precision follows the PRODUCT of the two and is best at a
50/50 split — not at the largest possible holdout. Against a
5,467-patient source: 250 patients gives ±0.037 on the attack
estimate, 800 gives ±0.022, 1,500 gives ±0.017, and 2,733 gives
±0.015. A cohort of 800 is a weaker instrument than one of 4,000,
which is the opposite of the intuition that a bigger holdout buys
more confidence.

## One command, end to end

Every capability above has its own tests. This takes a single
cohort the whole way and reports the lot, which makes it both the
deliverable and the integration test — a feature that works alone
and breaks in company shows up here and nowhere else.

```
python scripts/end_to_end.py --patients 500 -o docs/e2e --report
```

```
patients learned from              250
real relationships found           2
visit-to-visit correlation         0.723 vs source 0.775
fidelity checks passed             24/26
privacy                            0 exact matches, PASS
strongest membership attack        0.506  (PASS)
ceiling (known exactly)            0.910
our model reading the notes        0.897
our model, notes withheld          0.713  (reading worth +0.184)
```

## How much data does it need?

Measured against planted truth rather than asserted
(`scripts/power_sweep.py`): roughly 400 patients to recover a
simple monotone relationship, 800 for a nonlinear one, and 3,200
for an interaction between two factors — with zero noise columns
ever adopted as structure at any cohort size.

## Where to start reading

`docs/synthkit_presentation.html` — an interactive walkthrough of
what this is, what it does and where it goes. Self-contained; open
it in a browser. Start here if you have fifteen minutes and no
context.


`docs/system_map.html` is a generated, self-contained map of the
whole system: hub to domains to modules to live source, with a
plain-English walkthrough at every level. Open it in a browser; it
needs nothing installed.

## Why

Evaluating a data-quality tool, an extraction model, or a vendor
pitch requires data where you KNOW the truth. Real data never
comes with an answer key; synthetic data usually comes without
realistic mess or planted signal. synthkit generates both at
once: every missing value, typo, format drift, outlier, wrong
value, duplicate row, and orphaned foreign key is deliberate,
seeded, and ledgered — so scoring is exact, reproducible, and
arguable in a meeting.

## Capabilities

- **Tables**: arbitrary schemas from English or JSON; nine
  distribution kinds (mixtures, weighted categoricals, sequences,
  date ranges); rules for date ordering (`days_from` couples
  gaps to a column) and derived numerics; seven mess operators,
  mutually exclusive per cell, all ledgered.
- **Outcomes**: logistic (AUROC ceiling from generating
  probabilities) and linear (R^2 ceiling from declared noise);
  `target_prevalence` / `target_range` make the English
  enforceable.
- **Documents**: clinical-note-style corpora with planted
  elements, difficulty tiers, style axes, and distractor traps;
  rendered via local LLM with verify-and-retry, or a
  deterministic stub.
- **Relational**: multi-table specs with seeded foreign keys;
  join mess includes random orphans, near-key orphans (one
  transposition from real), and format drift — the precision
  trap for naive integrity checkers.
- **Campaigns**: goal (clean / predict / regress / extract) ->
  three-tier ladder with bars; blinded train/test splits;
  interval-aware three-way verdicts (DECISIVE / INCONCLUSIVE
  with an executable sample-size prescription); append-only
  trial history and an arm-comparison table.
- **Baselines**: mess-tolerant feature encoder + stdlib
  logistic/ridge baselines; heuristic reference cleaner; every
  vendor number is reported against the baseline and the
  ceiling.
- **LLM-as-vendor**: any LLM becomes a testable extractor —
  blind prompts, JSON salvage, majority voting, reliability
  stats, and a named intervention seam for measuring what
  prompt-hardening buys.
- **Semantic lint**: probe-based checks that the spec means what
  you meant — realized prevalence vs. declared target, derived
  magnitudes, rule coupling, dropped mess clauses, unloaded
  distractor traps.
- **Interfaces**: a CLI for everything; `synthkit gui` — a
  five-station local calibration bench with a live build
  fingerprint; a standalone self-validating notebook (zero
  imports beyond stdlib) for SageMaker or any Jupyter.
- **Backends**: Ollama (local), AWS Bedrock (Converse API,
  injectable client), Anthropic API, deterministic stub.

## Quickstart

```bash
pip install -e .
python scripts/smoke_tables.py        # any suite; 13 suites, 261+ checks
synthkit gui                          # the bench
synthkit table-compile -d "a 400-row inpatient encounter extract..." \
    -o spec.json --backend ollama     # English -> spec (+ auto-lint)
synthkit campaign-compile --goal predict --spec spec.json \
    --outcome readmitted --bars auroc=0.65 -o campaigns/c1
synthkit campaign-run campaigns/c1 --solver mypkg.mod:solver
synthkit showdown campaigns/c1 --solver mypkg.mod:solver
synthkit trials campaigns/c1
```

## The working demonstration

`docs/first_vendor_trial.md` is a complete study conducted with
synthkit in one day: a local LLM's clinically dangerous failure
mode characterized (26% discontinued-medication confusion,
decisive at n=50+), a mechanism probe, a named prompt
intervention (26% -> ~4%, recall intact), a cross-model
replication (the gap is endemic to small models; the treatment
transfers), an instrument-hardening episode (temperature-0
default after sampling variance flipped a bar-edge verdict), and
a confirmatory run at the sample size the instrument itself
prescribed. Every number regenerates from a spec and a seed.

## Design laws

1. Nothing from an LLM is trusted: compiled specs are drafts
   behind a human gate; rendered documents are verified against
   their blueprints; vendor output is salvaged and counted.
2. Everything is deterministic: cell-level seeds, layered table
   seeds, temperature-0 vendor evaluation. Same spec, same data.
3. Every corruption is ledgered. Scoring is against the ledger,
   never against heuristics.
4. Verdicts know their uncertainty: intervals on every rate,
   INCONCLUSIVE when the data cannot resolve the bar, and a
   prescription for how much generated data would.
5. Every live failure becomes a permanent fixture. The smoke
   suites are the project's memory.

## Layout

```
synthkit/       the library (stdlib-only core)
scripts/        smoke suites + notebook generator
notebooks/      synthkit_standalone.ipynb (self-contained),
                synthkit_demo.ipynb (guided tour)
docs/           the vendor trial, benchmark prompt, notes
```

Personal R&D. Built and validated entirely on personal
hardware and time.
