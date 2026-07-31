# Phase 2, end to end: from a real extract to a graded exam

The complete loop, as run on a development machine.
Substitute `--src` with the
real extract folder when the governance gate is cleared; nothing
else changes.

## 1. Run the pipeline

    python scripts/phase2_pipeline.py --src data/mimic_omop \
        -o data/phase2_run

Seven stages: wrangle -> label -> diagnose -> profile -> compile ->
generate -> score. It ends with a fidelity and privacy scorecard
and a draft spec in which every number is a dial.

Expect, on the mimic set: 72/72 fidelity checks passed, privacy
PASS, and a null diagnostic reporting *no learnable structure* —
correct, because a profiled spec carries realistic SHAPE but no
planted causality. That is the gap the next step closes.

## 2. Author the planted truth

The draft lands with `outcomes: []` by design. Densities were
measured; causes must be stated:

    python scripts/author_outcome.py \
        --draft data/phase2_run/draft_spec.json \
        --profile data/phase2_run/profile.json \
        -o data/phase2_run/final_spec.json \
        --note-from active_drugs --note-elements 8 \
        --note-weight linaclotide=1.2 \
        --note-weight iopamidol=0.9 \
        --outcome readmit_planted \
        --coef age_at_visit=0.018 \
        --coef active_drug_count=0.22 \
        --coef glucose=0.004 \
        --drop returned_within_30d --drop is_last_visit \
        --prevalence 0.06,0.14 --calibrate --report

What each part does:

- `--note-from active_drugs` turns the deferred medication list
  into a free-text note column. Each frequent drug becomes an
  element whose DENSITY is the rate measured in the source.
- `--note-weight X=w` states that drug X genuinely raises risk by
  w. Anything left unstated defaults to 0.0 — it appears in the
  text as a decoy carrying no signal.
- A denial trap is planted automatically on the strongest element
  ("denies taking any linaclotide"), guarded by `excludes` so it
  appears only in patients WITHOUT the risk. Keyword matching gets
  it backwards; reading does not.
- `--coef` states tabular causes.
- `--calibrate` solves the intercept so realized prevalence lands
  in the declared band. The coefficients are causal claims and are
  never touched; the intercept is only a base rate.

## 3. The result

Generated at 2,500 rows, the planted spec produced:

    realized prevalence                 9.6%   (declared 6-14%)
    ceiling (planted truth)             0.782
    note-reading model (ours)           0.735
    text-blind model (vendor)           0.658
    value of reading the notes         +0.078

The ceiling is exact — it is the AUROC of the planted
probabilities themselves. And the note-reading model rediscovered
the signal from labels alone; its strongest learned text feature
was the phrase "on linaclotide", which is precisely what was
planted.

## What this demonstrates

Data whose SHAPE came from a real extract — distributions,
categorical weights, missingness, timing, contamination rates —
carrying an outcome whose CAUSES are known exactly. That
combination is what neither pure synthetic data nor real data can
offer on its own: realistic enough to be persuasive, planted
enough to be gradeable.

The privacy property is unchanged by any of this. Generation
consumes parameters, never records; the scorecard verifies zero
exact matches and confirms synthetic records sit no closer to real
records than real records sit to each other.
