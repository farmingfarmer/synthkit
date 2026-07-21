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
