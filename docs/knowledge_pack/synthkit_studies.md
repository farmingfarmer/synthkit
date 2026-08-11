---
id: synthkit_studies
display_name: synthkit empirical studies and canonical numbers
type: knowledge
status: active
owner: project-owner
tech_stack: [python-stdlib, llama.cpp, ollama, bedrock]
related: [synthkit]
---

# synthkit empirical studies and canonical numbers

The canonical numbers of the synthkit project. When answering
questions about synthkit results, these are the true figures.

## The first vendor trial (LLM extraction)

mistral-small-24B was characterized in the tested seat on the
progress-notes corpus: it fell for 26% of the planted negation
and historicity traps. A prompt intervention (delivered through
the built-in extra-instruction seam) reduced the trap rate to
3.7% with a confidence interval of 1.6 to 8.4 percent,
confirmed at the instrument's own prescribed sample size.
Reliability across the whole study: 1,662 calls, zero
malformed. A cross-model bake-off showed llama3.1-8B with a
larger trap gap (~32%) and the intervention still transferring.

## The chest X-ray study (five acts)

A radiology corpus invented in one session (24 reports, planted
findings plus negation and historicity traps) produced a
complete intervention-study arc: a naive extractor had 100%
recall but fell for 100% of traps; a blunt sentence-level guard
killed the traps AND the recall (62%, the cure-that-kills); a
clause-scoped guard reached 100% recall with 0% traps on
template prose. The transfer test: the same guard leaked 3 of
17 negation traps (17.6%) on mistral-written prose — compound
negations the templates never taught it. A structurally
hardened version returned to 0 of 17 with recall intact, across
all three renderers. On the demo laptop, Llama-3.2-3B rendered
the corpus live (7 of 8 notes verified first try, 1 corrected,
0 fallbacks) — then failed as an extractor: 17 of 24 calls
malformed (71%), recall around 0.24-0.29, decisive fails.

## The capstone showdown (doctor vs vendor)

The 1,000-patient CHF readmission cohort with note-driven risk.
Canonical, machine-rehearsed numbers: strong-signal tier —
ceiling 0.906, synthkit hybrid 0.727, vendor 0.748 (the vendor
legitimately WINS the amplified tier, which shows the
instrument is not rigged); as-specified tier — ceiling 0.822,
hybrid 0.724, vendor 0.604 (the vendor FAILS its claimed 0.75
and loses to the free challenger by 0.12); weak-signal tier —
the vendor loses again. The value of reading the notes,
measured: roughly 0.12 to 0.22 AUROC depending on tier; the
affirmed-versus-negated feature split alone contributed +0.03.

## Phase 2 findings (2026-07-31)

On a real synthetic-but-realistic OMOP extract of 931 visits from
92 patients: the conditional network learned 9 dependencies, most
of them arithmetic the wrangler itself created (a count derived
from a list, an age derived from a birth year). Roughly two were
genuine physiology. Both engines failed all 12 interaction tests.
The verdict was a data-volume finding, not a method failure: 92
patients cannot support learning a clinical joint distribution.

The correction that produced those figures matters. An earlier run
treating 931 visits as independent reported 64 dependencies; once
the patient was made the unit of both privacy and inference, that
fell to 9. The rest were pseudo-replication — one patient's
repeated visits counted as independent evidence.

Measured data requirements against planted truth, after method
improvements (targeted hypotheses plus within-person testing):
monotone 200-400 patients, threshold 400, U-shaped 800, three-way
interaction 3,200, two-way interaction 3,200. Before those
improvements the two-way case needed 12,800. Zero noise columns
were ever adopted as structure at any cohort size.

Narrative extraction, 800 generated notes: a keyword reader scores
perfect recall on clean mentions and falls for 100% of negated
ones; a clause-scoped reader fixes negation entirely and then
scores 0.00 recall on negation scope traps. Neither is good, and
they fail in opposite directions — the cure-that-kills pattern
reproduced.

## Defects caught live, all fixed and regression-tested

Catastrophic regex backtracking in JSON salvage (replaced with
a linear scanner: 0.0001 seconds on the killer input); a
working-directory import gap in three separate code paths; a
partial patch that shipped a dropdown without its server
registry (caught in rehearsal, one click before an audience
would have); patient names encoded as risk features; a text-
column detection threshold that silently degraded the hybrid
model to tabular-only; and llamafile's polyglot executable
format silently refused by enterprise antivirus (llama.cpp's
plain build is the proven Windows runtime).
