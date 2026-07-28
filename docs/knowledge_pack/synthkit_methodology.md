---
id: synthkit-methodology
type: methodology
name: synthkit methodology and design principles
part_of: synthkit
updated: 2026-07-28
tags: [principles, evaluation-methodology, llm-safety-patterns]
---

# synthkit methodology — transferable principles

Each principle below was earned by an incident in this project
and generalizes beyond it.

## Planted truth over found truth
Grading is only exact when the answer key exists before the
data. Synthetic populations with planted causality yield a
KNOWN performance ceiling; scores are judged against the
possible, not the imagined.

## Assisted, never trusted (LLM in the loop)
LLMs draft; validators teach; humans gate; verifiers check
prose fact-by-fact with retry then deterministic fallback.
Every LLM failure becomes curriculum (compiler prompt chapters)
or a counted statistic (malformed calls) — never a crash.

## Uncertainty honesty
Every rate carries a Wilson interval; verdicts are
PASS/FAIL/INCONCLUSIVE computed on intervals; INCONCLUSIVE
ships with required_n — the price of certainty as a number.
Validated live: a prescribed re-run confirmed a trap-rate at
the tighter interval.

## The free-challenger floor
Never compare a vendor to nothing. A cheap, transparent,
reproducible baseline (logistic regression, stdlib) reframes
procurement: beat the model we get for free on data where we
know the answers.

## Transfer testing across renderers (distribution shift)
An intervention validated against one renderer's phrasings can
be flattered by them. The xray negation guard scored 0% on stub
prose, leaked 17.6% on mistral prose ("No pneumothorax or
pleural effusion is identified" — compound negation), 0% after
structural hardening — across THREE renderers. Lesson:
re-render the same planted truth with a different pen to keep
tests honest.

## Role asymmetry in small models
Generation and disciplined structured extraction are different
capabilities. Llama-3.2-3B rendered clinical prose well (7/1/0
verified) but failed the extraction protocol on 17/24 calls
(71% malformed), while mistral-24B went 0/1,662. "Writes
clinical text" != "can be trusted to extract from it."

## Negation is signal, not noise
Clinical risk is often phrased negatively ("no home support").
The miner keeps affirmed and negated mentions as SEPARATE
features and lets labels decide the sign (+0.03 AUROC). Denial
traps (`excludes`) appear only in patients WITHOUT the risk —
punishing keyword matching.

## Identifiers are memorization bait
Mostly-unique string columns (names, MRNs) must be excluded
from features; transparency views caught "patient_name = jules
kim" wearing a risk weight. Heuristic: n>=20 present and
uniqueness>0.5 -> not a feature.

## Linear-time salvage of adversarial LLM output
Nested-quantifier regexes over almost-matching LLM rambles go
exponential (one bug -> three live incidents: frozen ticker,
idle server, Ctrl-C traceback). Salvage must be linear
string-aware scanning; hopeless output counts as malformed in
microseconds.

## Provenance leaks in UI
Labels must describe requirements, not development history
("Ollama app (only if installed)" not "mistral-24B on a Mac";
"OpenAI-compatible" disambiguated as protocol-not-company).
Fresh-eyes review finds what test suites cannot; each finding
becomes a pinned check.

## Verification-first delivery discipline
Verify the EDIT, not just the compile: patch scripts that die
mid-run leave partial states with green py_compile; a script
that prints without saving loses edits silently. Both happened
live; both now have process rules (assert per replace, save
before print, marker checks after write).

## Size-check every download
Failed syncs always presented as tiny files where megabytes
belonged (9-byte llamafile 404, 106-byte private-repo JSON,
twice). `dir <file>` before extraction is a mandatory gate.
