# OUTBOX — two new commands worth a run

Updated 2026-09-18 (evening). Three goals moved today: the exam
loop is ONE command now (goal 7), a structured PHI scrub exists
with a two-way gate (goal 2), and relationship dials landed
earlier (goal 4). Roadmap says ~77% overall.

## 0. Pull as usual

Zipball as in WINDOWS.md section 1. The net now expects
**68 suites, 1911 checks, ALL GREEN** on a zipball (a new suite,
`smoke_scrub`, joined the net).

## 1. The PHI scrub, read against the real extract

```bat
python -m synthkit.cli scrub %USERPROFILE%\dev\tidy_visits.csv --group-by person_id
```

**Expect:** one line per column - `PHI`, `OUT OF SCOPE`, or
counted in the `clear` total. On this extract everything should
read clear (it was de-identified upstream), `person_id` explained
as the group key. **An exit code of 1 with PHI lines is the
command WORKING** - it refuses to stay quiet about undropped
findings - not a crash.

**Send back:** the full output, whatever it says. This is goal
2's "read against the real extract's own headers" item.

## 2. Optional — the whole exam loop, one command

The five-command loop from E2E_EVAL.md is now one command:

```bat
python -m synthkit.cli exam %USERPROFILE%\dev\run_seed11b --effect age_at_visit=0.8 --effect span_days=-0.5 -o %USERPROFILE%\dev\exam2
```

**Expect:** six stage echoes - bridge, plant, ladder, baseline,
showdown, report card - ending in `exam -> ...\exam2` with
`report_card.html` inside. Same numbers as the September 11 run,
about 15 minutes, and this time nothing to typo.

**Send back:** nothing needed unless a stage refuses.
