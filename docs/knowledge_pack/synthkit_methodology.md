---
id: synthkit_methodology
display_name: synthkit methodology and transferable principles
type: knowledge
status: active
owner: project-owner
tech_stack: [python-stdlib, faiss-free, llama.cpp, ollama, bedrock]
related: [synthkit]
---

# synthkit methodology and transferable principles

Eleven principles, each earned by a real incident in the
synthkit project, each transferable beyond it.

## Evaluation principles

Planted truth over found truth: grading is only exact when the
answer key exists before the data; synthetic populations with
planted causality yield a KNOWN performance ceiling, so scores
are judged against the possible rather than the imagined.
Uncertainty honesty: every rate carries a Wilson interval,
verdicts are PASS / FAIL / INCONCLUSIVE computed on intervals,
and INCONCLUSIVE ships with the exact additional sample size
that would settle it — validated live when a prescribed re-run
confirmed a trap rate at the tighter interval. The
free-challenger floor: never compare a vendor to nothing; a
cheap transparent logistic baseline reframes procurement as
"beat the model we get for free, on data where we know the
answers."

## LLM-handling principles

Assisted, never trusted: LLMs draft, validators teach, humans
gate, verifiers check prose fact-by-fact with retry then
deterministic fallback; every LLM failure becomes curriculum
or a counted statistic, never a crash. Transfer-test across
renderers: an intervention validated against one renderer's
phrasings can be flattered by them — synthkit's negation guard
scored 0% traps on template prose, leaked 17.6% on
mistral-written prose, and 0% again after structural hardening,
across three renderers of the same planted truth. Role
asymmetry in small models: generation and disciplined
structured extraction are different capabilities — Llama-3.2-3B
rendered clinical prose well (7 of 8 verified first try) yet
failed the extraction protocol on 17 of 24 calls (71%
malformed), while mistral-24B went 0 malformed in 1,662 calls.
Linear-time salvage: nested-quantifier regexes go exponential
on almost-matching LLM rambles (one such bug caused three live
incidents); salvage must be linear string-aware scanning that
counts hopeless output as malformed in microseconds.

## Modeling and craft principles

Negation is signal: clinical risk is often phrased negatively
("no home support"), so affirmed and negated mentions are
separate learned features and denial traps appear only where
the denial is true. Identifiers are memorization bait:
mostly-unique string columns (names, MRNs) are excluded from
features after a transparency view caught a patient name
wearing a risk weight. Provenance leaks: interface labels must
describe requirements, not development history — fresh-eyes
review finds what test suites cannot, and each finding becomes
a pinned check. Verification-first delivery: verify the edit,
not just the compile — partial patch states with green
syntax checks happened live, twice, and produced process rules
(assert per replace, save before print, size-check every
download: failed transfers always presented as tiny files
where megabytes belonged).
