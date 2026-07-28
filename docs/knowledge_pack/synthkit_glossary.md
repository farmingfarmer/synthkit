---
id: synthkit-glossary
type: glossary
name: synthkit glossary
part_of: synthkit
updated: 2026-07-28
tags: [terms, definitions]
---

# synthkit glossary

- **spec / recipe**: the fingerprinted JSON contract describing
  a dataset (columns, distributions, mess, rules, outcomes,
  note plans). Same fingerprint => same data forever.
- **blueprint**: per-record planned truth (clean values, planted
  elements, probabilities) — exists before any artifact.
- **ledger**: signed record of every deliberate corruption
  (cell, op, truth). Makes "handles messy data" gradeable.
- **note column**: a free-text table column whose planted
  elements feed the outcome logit (hybrid tables).
- **element / distractor / filler**: planted signal phrase with
  weight+density / trap phrase (negation via `excludes`, or
  historical) / neutral sentence.
- **ceiling**: the exact best achievable score, known from
  planted probabilities (AUROC) or noiseless signal (R^2).
- **campaign / ladder / tier**: one exam compiled at three
  difficulties; predict tiers: strong-signal, as-specified,
  weak-signal; extract tiers: gentle, as-specified, adversarial.
- **bars**: declared pass marks (auroc, gap_max, recall,
  trap_max, fix_rate, detect_rate, rmse_max).
- **verdict**: PASS / FAIL / INCONCLUSIVE computed on Wilson or
  AUROC intervals; INCONCLUSIVE carries required_n.
- **showdown**: ceiling / baseline / vendor per tier; selectable
  baseline; [bar:] tag separates threshold from comparison.
- **autosolver / autosolver_hybrid**: stdlib logistic baseline;
  hybrid adds TextMiner note features. LAST_FIT publishes named
  coefficients (transparency).
- **vendor chair / witness stand**: the tested seat — any
  callable(train_rows, train_labels, test_rows)->scores, incl.
  LLMs via LLMExtractor with reliability counting.
- **malformed**: a tested LLM reply from which no valid answer
  could be salvaged; counted, never crashing.
- **semantic lint**: probe-generated sample measured against the
  spec's promises (L1 prevalence in-band, etc.).
- **the bench**: the five-station GUI (Describe / Spec / Data /
  Campaign / Showdown) with gated readiness and the
  plain-English final report.
- **atlas**: docs/system_map.html — generated drill-down map
  (hub -> domains -> modules -> live source), v2 with a
  99-step narrative layer and NARRATIVE-drift refusal.
- **openai backend**: the OpenAI-COMPATIBLE protocol door
  (llama.cpp / LM Studio / vLLM) — a message format, not the
  company; default 127.0.0.1:8080/v1.
