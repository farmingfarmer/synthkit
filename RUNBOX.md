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

Zipball per WINDOWS.md. Expect **69 suites, 1935 checks, ALL
GREEN** (new suite: smoke_latent). Then `pip install -e .`,
relaunch the bench, wordmark check - as before.

## 2. The head-to-head: new engine vs old, one command, one table

`compare` trains the latent challenger on the real 300-patient
sample AND measures your existing blueprint run's generated.csv
against the SAME source with the SAME metrics - source shares
first, then one row per engine. The named triple is the REAL
attribution flip (span_days driven by procedure_count vs
active_drug_count, where the blueprint's SHAP read 74/26
flipping to 22/78):

```bat
python scripts\latent_challenger.py compare %USERPROFILE%\dev\tidy_live300.csv --group-by person_id --blueprint-run %USERPROFILE%\dev\live300_rehearsal --seeds 0,1,2 --shares span_days=procedure_count,active_drug_count
```

**Expect:** a settings echo; a note naming the columns dropped
because the blueprint run excluded them (conditions, procedures,
drug_routes - all sides are restricted to shared columns so the
metrics match); then the table - `source ... shares`, one
`blueprint` row, three `latent seed N` rows, each with sign /
close / INVERTED / shares, the latent rows also carrying the
nn-ratio memorization tripwire. A few minutes per latent seed.
Nothing is written without --out; --out output is real-derived
and STAYS on this machine.

**Send back:** the whole table. The four numbers that decide the
next step: whose shares sit closer to the source's, the INVERTED
counts side by side (the blueprint's guarantee vs the latent's
known 2-4-at-width cost), close side by side, and the nn-ratio
(under ~0.8 means the tripwire is pulling and the k-screen
tightens before anything else).

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
