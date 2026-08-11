---
id: synthkit_architecture
display_name: synthkit architecture
type: knowledge
status: active
owner: project-owner
tech_stack: [python-stdlib, faiss-free, llama.cpp, ollama, bedrock]
related: [synthkit]
---

# synthkit architecture

synthkit's pipeline runs six stages with two rails alongside:
Spec Layer -> Truth Planning -> Rendering -> Evaluation &
Ceilings -> Campaigns & Verdicts -> Solvers & Vendors, with
Quality Gates and Interfaces & Backends beside every stage. 43
components are documented in the interactive atlas.

## Spec layer

Plain English becomes a formal, fingerprinted JSON recipe: an
LLM drafts, a validator lists every problem in teaching
language, and a human gate approves — nothing is created from
an unapproved recipe. Table specs declare per-column
distributions (normal, lognormal, mixtures, categoricals,
sequences, date ranges), per-column mess rates, cross-field
rules (dates coupled by length of stay; charges derived from
days), and logistic or linear outcomes with declared target
prevalence. Note columns declare planted elements (phrase
variants, density, outcome weight), distractors (denial traps
carry an excludes rule so they appear only in patients WITHOUT
the risk), and neutral fillers.

## Truth planning

The answer key exists before the data. Every cell has its own
seeded random stream (hash of recipe seed, row, and column), so
editing one column never reshuffles another. Outcomes are
computed from planted causes and the true probabilities are
stored — that is why the AUROC ceiling is exact. Clinical notes
are assembled per patient from seeded phrase draws, and phrase
presence feeds the outcome logit directly, so the ceiling
includes text signal by construction. Deliberate mess (missing
values, typos, mixed formats, outliers, wrong values,
duplicates) is applied afterward with every corruption signed
into a ledger.

## Rendering, evaluation, campaigns

Rendering writes messy data, answer key, and ledger; optional
LLM writers phrase the notes with per-note verification,
retry, and deterministic fallback. Evaluation grades exactly
against planted truth with Wilson intervals on every rate and
required-sample-size prescriptions when inconclusive. Campaigns
compile one exam at three difficulties (predict:
strong-signal / as-specified / weak-signal; extraction:
gentle / as-specified / adversarial), train contestants on a
blinded shifted-seed population, and file append-only trial
records.

## Learning from real data (Phase 2)

A wrangler collapses six OMOP tables into one tidy row per visit,
normalising join keys that arrive as both integers and
leading-zero text. A profiler measures each column as parameters —
never records — with k-anonymity counted in patients, percentile
clamping instead of true extremes, and multiple-comparison
correction before any relationship is believed. A conditional
dependency network learns the joint distribution: columns are
discretised, parent sets are found by conditional mutual
information, and conditional tables are sampled ancestrally, so
nonlinear and interaction structure survives. Resolution is solved
from the effective sample size. Every table is a dial via a
geometric tilt toward or away from its marginal. A fidelity and
privacy scorecard grades the result on marginals, missingness,
correlations, dependence shape, interactions, exact matches and
nearest-neighbour distances — with tolerances derived from
sampling rather than fixed thresholds.

## Notes and extraction

A transcriber renders structured facts into messy clinical prose
under a nine-part corruption taxonomy, declaring per fact whether
it lives in the columns, the note, both, or both-while-disagreeing
— and ledgers what each sentence actually asserts. Extraction is
graded per corruption type. A language model can be seated in the
chair through a narrow present/current/certain contract, blind to
the traps, with unusable replies counted separately.

## Solvers and vendors

The built-in challenger standardizes messy tables (excluding
identifier-like columns — names once snuck in wearing risk
weights), mines note phrases from training labels alone
(keeping negated mentions as separate features from affirmed
ones), and trains a transparent logistic regression whose real
named coefficients are published for display. Any vendor model
plugs into the identical seat as a function of (training rows,
training labels, test rows) returning scores; LLM vendors get
blind prompting, linear-time JSON salvage, majority voting,
and malformed-call reliability counting. Showdowns report
ceiling / baseline / vendor per tier with a selectable
baseline. Backends: deterministic stub, Ollama, any
OpenAI-COMPATIBLE local server (llama.cpp, LM Studio, vLLM —
a protocol name, not the company; nothing goes to ChatGPT),
AWS Bedrock, and Anthropic.
