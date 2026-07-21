# First real vendor trial - mistral-small3.1 as clinical extractor
2026-07-20. Extract campaign, reference vertical, 16 docs/tier.
Single-sample: recall 1.000 all tiers; trap fp 0.10 -> 0.286.
Voted x3:      recall 1.000 all tiers; trap fp 0.30 / 0.143;
               9/144 voted out; 0 malformed across 192 calls.
Reading: recall solved; discontinued-med confusion ~15-25%
per distractor, partly systematic (survives voting); n=10
denominators too small - see precision run.

Precision run (40 docs/tier, 120 calls): trap fp 0.263 / 0.258 -
tiers CONVERGE at ~26% per distractor, density-independent.
Revision: no pressure effect; each discontinued med is an
independent ~26% misread. Recall 1.000 across all 312 docs of
the day. 0 malformed in 312 calls.
Character sheet: recall solved; medication-status discrimination
is the deficit, in the clinically dangerous direction.

INTERVENTION (docs/intervention_status.txt, same 120 docs):
trap fp 0.263/0.258 -> 0.053/0.065. Four-to-five-fold reduction.
Recall held 1.000; 0 malformed; tiers converge again.
Reading: the 26% was an instruction gap, not representational -
status discipline is teachable by prompt. Residual ~6% is the
honest floor for prompt-only hardening on mistral-small.
Full arc: characterize -> hypothesize -> treat -> re-measure,
all on frozen data. This is the product's working demonstration.

2x2 BAKE-OFF (frozen campaign, 480 calls, temp 0.2):
  arm               tier2-fp  tier3-fp  recall     cleared
  mistral-control    0.263     0.226    1.000       1/3
  mistral-hardened   0.105     0.032    1.000       1/3*
  llama8b-control    0.316     0.323    0.957-0.986 1/3
  llama8b-hardened   0.105     0.129    1.000       1/3
Findings: (1) status confusion is ENDEMIC to small models
(26-32% both controls), not mistral-specific. (2) The same
treatment transfers: 2.5-5x reduction on both. (3) Mistral
dominates llama8b: recall, discipline, hardened residual.
(4) *Verdict instability: mistral-hardened cleared 3/3
yesterday, 1/3 today (tier2 0.105 vs 0.053) - temp-0.2
sampling variance flips bar-edge verdicts. Fixed: vendor
evaluation now temperature-0 by default (V1.3).
