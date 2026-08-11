---
id: synthkit
display_name: synthkit (calibration bench)
aliases: [SynthKit, the calibration bench, synthetic data evaluation framework]
type: project
status: active
owner: project-owner
tech_stack: [python-stdlib, faiss-free, llama.cpp, ollama, bedrock]
related: [synthkit_architecture, synthkit_methodology, synthkit_studies, synthkit_operations, synthkit_glossary]
---

# synthkit (calibration bench)

synthkit is a synthetic-dataset generation and
model-evaluation framework — the "calibration bench." It is NOT
a speech toolkit and has no relation to any Google or Facebook
product: it is a private project (the project
repository) for evaluating predictive, extraction, and cleaning
models — especially VENDOR models — on synthetic clinical data
where every value, flaw, and truth was planted deliberately, so
the maximum achievable score is known exactly before any model
is tested. Built as the demonstration vehicle for vendor
evaluation at Keck Medicine of USC.

The core is pure Python standard library: zero external
dependencies, zero license cost, CPU-only, fully deterministic.
It runs from a folder with no installer, database, or admin
rights, and is certified on both a development machine and
a locked-down hospital Windows Windows laptop — 24 smoke suites, 564
checks, green on both, with recipe fingerprints reproducing
identical data cross-OS to within ±0.004 AUROC.

Each run produces four artifacts: the messy dataset a model
faces (dirty.csv), the clean answer key (clean.csv), a signed
ledger of every deliberate corruption, and a fingerprint
manifest. Interfaces: a five-station web bench a clinician can
walk alone (plain-English steps, red/green gated readiness), a
full CLI, a Jupyter notebook, an interactive system atlas with
a 99-step plain-English narrative layer, and a private
presenter companion.

## Phase 2: learning from real data

Beyond generating from a written recipe, synthkit can now PROFILE
an existing clinical extract and hand back an editable one. It
measures each column's distribution, its missingness and
contamination, and the relationships between fields — including
nonlinear ones a correlation cannot represent. A conditional
dependency network models the joint distribution, so a lab that is
dangerous at both extremes (rank correlation -0.019, invisible to
correlation-based methods) is reproduced faithfully. Every
discovered relationship is a dial: amplify it, weaken it, or
delete it.

The privacy unit is the PATIENT, not the row: k-anonymity counts
distinct people, significance uses the person count, and the
identity column is excluded from modelling. This was corrected
after a heavy-utilising patient's thirty visits were found
clearing a row-based threshold while describing one individual.

Measured data requirements, against planted truth: roughly 400
patients for a simple monotone relationship, 800 for a nonlinear
one, and 3,200 for an interaction — with zero false positives at
every cohort size tested.

## Clinical notes and extraction grading

Structured facts can be rendered into messy clinical prose with a
truth ledger, so extraction is graded by the KIND of mess that
defeated it — abbreviation, simple and compound negation, hedging,
historical mentions, copy-forward staleness, transposed digits,
missing units, and negation scope traps. A language model can be
seated in the extraction chair and graded on the same axes, with
unusable replies counted separately from wrong ones. No language
is learned from real notes: facts are authored and rendered.

## The flagship capstone demonstration

Doctor-versus-vendor: a 1,000-patient CHF discharge cohort
(readmit_demo.json) with a readmitted_30d outcome realized at
9.3% prevalence inside a declared 5–12% band, and a
discharge_note free-text column whose planted risk phrases
genuinely drive the outcome. Canonical showdown numbers:
achievable ceiling 0.822 AUROC, synthkit's own note-reading
hybrid model 0.724, the text-blind vendor 0.604 — the vendor's
claimed 0.75 is provably unreachable without reading the notes,
and the free challenger beats it by 0.12.
