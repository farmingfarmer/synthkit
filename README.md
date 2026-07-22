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

Personal R&D by Alex Marunycz. Built and validated entirely on
personal hardware and time.
