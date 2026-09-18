# OUTBOX — next step: the 5x compare deck (two commands)

Updated 2026-09-18. The 5x fit is DONE and measured - 288,631 rows
for 4,000 patients, fidelity within seed noise of 1x (direction
92.3% vs 93.9%, close 77.8% vs 76.5%, center 31/33, every spread
miss bound-explained). What remains is the one-table exhibit.

## 1 — build the compare deck (about a minute)

```bat
python scripts\fidelity_deck.py --src %USERPROFILE%\dev\tidy_visits.csv --run %USERPROFILE%\dev\run_seed11b -o %USERPROFILE%\dev\exam\deck_5x.html --group-by person_id --compare "5x=%USERPROFILE%\dev\run_5x"
```

## 2 — open it

```bat
start %USERPROFILE%\dev\exam\deck_5x.html
```

**Expect:** the scale table near the top - 1x beside 5x, one row
per run, across direction, close, inverted, coverage and the gate.

**Send back:** a screenshot of that table. It is the "does 5x
degrade it" answer in one exhibit, ready for the Wednesday deep
dive - and with it, every sprint item is complete.
