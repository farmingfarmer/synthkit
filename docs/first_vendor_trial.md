# First real vendor trial - mistral-small3.1 as clinical extractor
2026-07-20. Extract campaign, reference vertical, 16 docs/tier.
Single-sample: recall 1.000 all tiers; trap fp 0.10 -> 0.286.
Voted x3:      recall 1.000 all tiers; trap fp 0.30 / 0.143;
               9/144 voted out; 0 malformed across 192 calls.
Reading: recall solved; discontinued-med confusion ~15-25%
per distractor, partly systematic (survives voting); n=10
denominators too small - see precision run.
