# RUNBOX — demo prep, plus the latent challenger's first real read

Updated 2026-09-24 (night). Two workstreams share this pull: the
demo prep (unchanged below) and the new latent challenger - the
competing autoencoder generator you proposed, built and measured
today. On fixtures it holds the interactions the blueprint loses
(XOR nearly exact, the colinear family 63/63 close, driver shares
transferring 35/65 -> 42/58 where the blueprint flips) and it
PASSED the membership attack at the fixture (worst 0.546, with a
weights-leak adversary and a republish control that reads 1.000
FAIL). It also has its own costs, stated: 2-4 inversions at full
width where the blueprint holds zero, no within-patient dynamics,
and it trains on records - every run says "a MEASUREMENT, not a
release."

## 1. Pull once

Zipball per WINDOWS.md. Expect **69 suites, 1934 checks, ALL
GREEN** (new suite: smoke_latent). Then `pip install -e .`,
relaunch the bench, wordmark check - as before.

## 2. The latent challenger reads the real 300-patient sample

The named triple is the REAL attribution flip - span_days driven
by procedure_count vs active_drug_count, where the blueprint's
SHAP read 74/26 flipping to 22/78:

```bat
python scripts\latent_challenger.py csv %USERPROFILE%\dev\tidy_live300.csv --group-by person_id --seeds 0,1,2 --shares span_days=procedure_count,active_drug_count
```

**Expect:** a settings echo, then one line per seed - sign /
close / INVERTED / nn-ratio / shares src vs gen. A few minutes
per seed. Nothing is written to disk without --out, and any
--out output is real-derived and STAYS on this machine.

**Send back:** the full output. The three numbers that decide
the next step: the shares drift (does the driver ordering hold
on real data), the INVERTED count, and the nn-ratio (under ~0.8
means the memorization tripwire is pulling and the k-screen
needs tightening before anything else).

## 3. Demo prep - the second-seed check (still pending)

Know before the room whether the rehearsal's inversion
reproduces on a different 300-patient draw:

```bat
python -m synthkit.cli sample %USERPROFILE%\dev\tidy_visits.csv --group-by person_id --patients 300 --seed 11 -o %USERPROFILE%\dev\tidy_live300b.csv
```

```bat
python -m synthkit.cli fit --src %USERPROFILE%\dev\tidy_live300b.csv --out %USERPROFILE%\dev\live300b_rehearsal --group-by person_id --generate --exclude conditions,procedures,drug_routes
```

```bat
python -m synthkit.cli gate %USERPROFILE%\dev\live300b_rehearsal
```

**Send back:** the gate's INVERTED line either way - it decides
which in-room sentence is true. Then one out-loud rehearsal, and
freeze.
