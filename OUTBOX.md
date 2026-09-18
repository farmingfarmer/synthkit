# OUTBOX — next step: the 5x scale measurement

Updated 2026-09-17. This file is overwritten each time there are
new commands — whatever it shows now IS the next step. Run on the
data machine, one command per line; each block has a copy button
on the repository page.

The last open sprint item — the "does 5x degrade it" answer on
real data. No pull is needed: the current build runs this as-is.
Total machine time about 40 minutes, mostly unattended.

## 1 — the 5x fit (about 40 minutes; start it and walk away)

```bat
python -m synthkit.cli fit --src %USERPROFILE%\dev\tidy_visits.csv --out %USERPROFILE%\dev\run_5x --group-by person_id --lags --patients 4000 --generate
```

Same source and flags as the seed runs, but generating 4,000
patients (5x the 800). The first line must say
`build: <sha> (from archive)`.

## 2 — the compare deck (about a minute)

```bat
python scripts\fidelity_deck.py --src %USERPROFILE%\dev\tidy_visits.csv --run %USERPROFILE%\dev\run_seed11b -o %USERPROFILE%\dev\exam\deck_5x.html --group-by person_id --compare "5x=%USERPROFILE%\dev\run_5x"
```

## 3 — open it

```bat
start %USERPROFILE%\dev\exam\deck_5x.html
```

**Expect:** the full dashboard for run_seed11b plus a scale table
near the top — 1x beside 5x — answering whether expansion degrades
direction, close, inversions, coverage, and the gate. On the
clinic rehearsal both scales read identically; the real extract's
answer is the point of the run.

**Send back:** a screenshot of the scale-comparison table (and the
5x gate line if it differs from 1x). That row is a ready-made
exhibit for the Wednesday deep dive.
