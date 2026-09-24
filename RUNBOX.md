# RUNBOX — one final pull, then the second-seed check

Updated 2026-09-23 (night). Your rehearsal landed and it was
worth every minute: fit 329s, and the gate caught a real
inversion on the 300-patient sample - span_days ~
diastolic_blood_pressure_invasive, source -0.139, generated
+0.188 - the EXACT pair the new fragility flag named first
during the fit. The demo now scripts that as its centerpiece
(SPEAKER.html has the words). Two things remain.

## 1. One final pull (the bench gained the exclude field)

The Fit station could not carry `--exclude` until this build -
the room's run must use the rehearsed flags THROUGH the bench,
so this pull is required. Zipball per WINDOWS.md; expect
**68 suites, 1924 checks, ALL GREEN**. Then:

```bat
pip install -e .
```

```bat
python -m synthkit.cli version
```

Run version AFTER the install (your rehearsal ran it before,
which is why it showed a stale id). Relaunch the bench,
hard-refresh, wordmark must match this version line.

## 2. The second-seed check (the gate's own guidance)

Know before the room whether the inversion reproduces - a
reproducible inversion is a defect to report, a single-seed one
is a stop for that file only. Different sample seed, everything
else identical:

```bat
python -m synthkit.cli sample %USERPROFILE%\dev\tidy_visits.csv --group-by person_id --patients 300 --seed 11 -o %USERPROFILE%\dev\tidy_live300b.csv
```

```bat
python -m synthkit.cli fit --src %USERPROFILE%\dev\tidy_live300b.csv --out %USERPROFILE%\dev\live300b_rehearsal --group-by person_id --generate --exclude conditions,procedures,drug_routes
```

```bat
python -m synthkit.cli gate %USERPROFILE%\dev\live300b_rehearsal
```

**Expect:** about 5-6 minutes for the fit. Read the INVERTED
line either way.

**Send back:** the gate output. If seed 11 also inverts that
pair, the in-room line changes from "single-seed, this file
stops" to "reproducible at this cohort size, it is on the
fidelity list by name" - both are honest, but you want to know
which one is true before someone asks.

Demo day itself: NO pull, no reinstall. The build you verify
tonight is the build you demo.
