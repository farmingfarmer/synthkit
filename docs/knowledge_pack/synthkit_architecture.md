---
id: synthkit-architecture
type: architecture
name: synthkit architecture
part_of: synthkit
updated: 2026-07-28
tags: [pipeline, domains, modules]
---

# synthkit architecture — six stages, two rails

Pipeline (left to right): Spec Layer -> Truth Planning ->
Rendering -> Evaluation & Ceilings -> Campaigns & Verdicts ->
Solvers & Vendors. Rails alongside every stage: Quality Gates,
Interfaces & Backends. 43 atlas-documented components.

## Spec Layer (compiler.py, tablespec.py, spec.py)
English -> formal spec. LLM drafts (one repair round with
validator problems fed back verbatim), validator teaches every
problem at once, HUMAN GATE approves. TableSpec: columns with
distributions (normal/lognormal/mixture/categorical/sequence/
date_range...), mess rates per column, rules (date_after with
days_from; derived), logistic/linear outcomes with
target_prevalence. Note columns (ctype "note"): elements
(id/phrasings/density/weight), distractors (with `excludes` —
denial traps only in patients WITHOUT the risk), fillers;
outcome coefficients reference "notecol.element_id".

## Truth Planning (tableplan.py, planner.py, relational.py)
Answer key BEFORE data. _cell_rng: sha256(master:row:column)
per-cell RNG — column-independence law. plan_table: clean rows,
outcome probabilities stored (the exact AUROC ceiling), then
deliberate mess with a signed ledger. _build_note: seeded
phrase draws per patient; presence feeds the logit, so the
ceiling INCLUDES text signal by construction. note_truth
ledgered per row.

## Rendering (renderer.py, write_table)
Deterministic stub writer by default; optional LLM writers with
verify-retry-deterministic-fallback per note (nothing an LLM
writes is trusted).

## Evaluation & Ceilings (tableeval.py, evaluator.py, mlmetrics.py)
Exact grading vs ledger and planted truth. Wilson intervals on
every rate; auroc_interval; required_n prescriptions.

## Campaigns & Verdicts (campaign.py)
compile_campaign(goal, spec, bars, outcome) -> three difficulty
tiers (gentle / as-specified / adversarial; predict goals:
strong-signal / as-specified / weak-signal). Blinded training
(shifted seed population), PASS/FAIL/INCONCLUSIVE verdicts on
intervals, append-only trial records.

## Solvers & Vendors (autosolver.py, llmvendor.py)
FeatureEncoder (type-sniffing through mess; text columns
skipped, avg len > 40; identifier-like columns excluded —
n>=20 & uniqueness>0.5). autoclean. TextMiner: label-driven
uni/bigram mining, lift*sqrt(df), affirmed vs negated presence
as SEPARATE features. autosolver_hybrid: tabular + mined text
pseudo-columns into LogisticBaseline. _publish_fit -> LAST_FIT:
real named coefficients for transparency. run_showdown:
ceiling/baseline/vendor per tier, selectable baseline,
[bar:] verdict split from baseline comparison. LLMExtractor:
blind prompts, JSON salvage (_first_balanced_array linear
scanner), majority voting, malformed-call reliability counting,
--llm-extra-system intervention seam.

## Rails
Gates: validate (lawful) -> semantic lint L1-L6/D1-D5 probe-
based (kept promises) -> statistics (honest uncertainty).
Interfaces: gui.py five-station bench; cli.py; backends stub /
ollama / openai-compatible (openai_compat.py — llama.cpp,
LM Studio, vLLM, llamafile; protocol NOT the company) /
bedrock.py / anthropic. GUI extras: api_export downloads,
_showdown_report plain-English report with winner cards and
model coefficients.
