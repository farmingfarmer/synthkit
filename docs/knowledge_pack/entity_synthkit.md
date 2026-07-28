---
id: synthkit
type: system
name: synthkit
aliases: [synthkit calibration bench, the calibration bench]
status: active
updated: 2026-07-28
version: v0.9.x (fingerprint-stamped builds; 324 checks / 16 suites)
owner: Alex (farmingfarmer)
repos: [github.com/farmingfarmer/synthkit (private)]
machines:
  canonical: M3 Max MacBook Pro (venv ~/ai_copilot/.venv)
  deployment: locked-down Windows ThinkPad NE1002766 (no admin,
    venv %USERPROFILE%\dev\verbatim\.venv, repo %USERPROFILE%\dev\synthkit)
tags: [synthetic-data, model-evaluation, clinical-ml, benchmark,
       vendor-evaluation, keck]
relations:
  demonstrates_for: Keck Medicine of USC (vendor-evaluation demo)
  distinct_from: [Mnemo, Orrery, Task Ledger, Loom, Smart Problem List]
---

# synthkit — the entity

synthkit is a synthetic-dataset generation and model-evaluation
framework ("calibration bench"). Its purpose: evaluate any
predictive/extraction/cleaning model — especially vendor models —
on synthetic data where every value, flaw, and truth was planted
deliberately, so the maximum achievable score is KNOWN exactly.

## Core identity facts
- Pure Python standard library in the core. Zero external
  dependencies, zero license cost, CPU-only, deterministic.
- Runs from a folder: no installer, no database, no admin
  rights. Certified on a locked-down hospital Windows laptop.
- Everything is reproducible: a spec (recipe) is fingerprinted;
  same fingerprint => byte-identical data forever, cross-OS
  (verified Mac vs Windows to ±0.004 AUROC on the capstone).
- Test net: 16 smoke suites, 324 checks, green on both machines.
- Interfaces: five-station web bench (clinician-walkable, plain
  English, gated readiness), full CLI, Jupyter notebook, system
  atlas (docs/system_map.html, v2 with 99-step plain-English
  narrative layer), presenter companion (private teleprompter).

## What it produces per run
- dirty.csv (the messy dataset a model faces), clean.csv (the
  answer key), ledger.json (every deliberate corruption,
  signed), manifest (fingerprints). Document corpora produce
  per-doc .txt + blueprints.

## Flagship demonstration (the capstone)
Doctor-vs-vendor scenario: a 1,000-patient CHF discharge cohort
(readmit_demo.json, seed 20260727), outcome readmitted_30d at
9.3% realized prevalence inside a declared 5–12% band, with a
discharge_note free-text column whose planted phrases drive the
outcome. Measured spine: ceiling 0.822 / synthkit hybrid 0.724 /
text-blind vendor 0.604. The vendor's claimed AUROC >= 0.75 is
provably unreachable without reading the notes.
