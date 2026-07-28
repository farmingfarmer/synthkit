---
id: synthkit-studies
type: empirical-record
name: synthkit empirical studies and canonical numbers
part_of: synthkit
updated: 2026-07-28
tags: [results, benchmarks, canonical-numbers]
---

# synthkit — the empirical record (canonical numbers)

## The first vendor trial (mistral-small-24B, extraction)
Characterized falling for 26% of negation/historicity traps on
the progress-notes corpus. Mechanism probed; intervention via
--llm-extra-system reduced trap rate to 3.7% [1.6–8.4];
confirmed at the instrument's own required_n. Reliability:
1,662 calls, 0 malformed. Cross-model bake-off: llama3.1:8b
showed a larger trap gap (~32%) with the intervention still
transferring.

## The xray study (docs/xray_study.md, five acts)
Chest X-ray corpus invented in one session (xray.json, seed 7,
24 docs; negation + historicity traps). Act 1 stub arc: naive
100% recall/100% trap fp; blunt 62% recall (cure-that-kills);
careful 100%/0%. Act 2 transfer: careful leaks 3/17 (17.6%) on
mistral prose. Act 3: extract_careful_v2 structural negation ->
0/17 with recall 100, stub regression clean; arc 0% -> 17.6% ->
0%. Act 4: Llama-3.2-3B renders live on the ThinkPad (8 docs:
7 first try / 1 retry / 0 fallbacks); v2 holds 0/0. Act 5: the
3B in the witness stand: 24 calls, 17 malformed (71%), recall
~0.24–0.29 DECISIVE fails — role asymmetry measured.

## The capstone (doctor vs vendor, hybrid tables)
readmit_demo.json: 1,000 CHF patients, 15 fields incl. bimodal
EF, zero-inflated priors, lognormal creatinine, date rule,
derived charges, full mess menu, discharge_note with 4 weighted
elements + excludes-guarded denial traps + historical trap.
Prevalence 9.3% (declared [5%,12%]; lint L1 INFO in-band).
Canonical showdown (ThinkPad-rehearsed, cross-OS ±0.004):
strong-signal ceiling 0.906 / hybrid 0.727 / vendor 0.748
(vendor legitimately wins the amplified tier — instrument not
rigged); as-specified 0.822 / 0.724 / 0.604 (vendor FAILS its
0.75 claim; hybrid beats it by 0.12); weak-signal vendor loses.
Text-mining value measured: affirmed/negated split +0.03; the
note signal worth ~0.12–0.22 AUROC depending on tier.

## Live-caught defects (all fixed + fixtured)
- Catastrophic regex backtracking in JSON salvage (3B rambles);
  linear scanner, 0.0001s on killer input, timing-bounded tests.
- cwd-import gap in THREE locations (CLI loaders, GUI solver
  resolution) — console scripts lack cwd on sys.path.
- Partial-edit shipping (registry entry + dropdown option lost
  to a dead patch script) — caught in rehearsal, now a smoke.
- Identifier features (patient names with risk weights).
- Text-column detection threshold 60 -> 40 (short notes missed;
  hybrid silently degraded to tabular).
- launchd-analog: GUI job re-attach after refresh still pending
  (frozen-ticker incidents; findings ledger).
- llamafile APE format silently refused by enterprise AV;
  llama.cpp win-cpu-x64 is the proven Windows runtime.
