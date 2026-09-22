# OUTBOX — pull, then re-read the scrub

Updated 2026-09-22. Your two runs both landed. The exam loop
reproduced the September 11 numbers exactly with nothing to typo
- that chapter is closed. And the scrub's first real read did its
job on the DETECTOR: of the five non-clear lines, one was a true
catch and three were scrub faults, now fixed.

What your output taught it:

- `visit_id` was a TRUE catch - 55,428 distinct over 800 patients
  is a per-row identifier - but the line said "one per person"
  beside numbers that say 69 per person. It states the measured
  ratio now.
- `visit_start_date` / `visit_end_date` were FALSE positives: a
  date wears the digits-and-hyphens shape the identifier rule
  matched, and 4,692 distinct values cleared its bar. Dates are
  never identifiers now - the pipeline models them and generates
  invented dates.
- `active_drugs` is a semicolon-joined SET column, not free text.
  Set columns are judged by their TOKENS now - which also means a
  set of email addresses gets caught, where the joined string
  matches no pattern.

## 1. Pull as usual

Zipball per WINDOWS.md section 1. The net now expects
**68 suites, 1915 checks, ALL GREEN** on a zipball.

## 2. Re-read the scrub

```bat
python -m synthkit.cli scrub %USERPROFILE%\dev\tidy_visits.csv --group-by person_id
```

**Expect:** `PHI visit_id` (about one per row), `OUT OF SCOPE
person_id` (group key), and both dates plus `active_drugs` now in
the clear count - 42 of 44 clear. Exit code 1 is still the
command working: it refuses to stay quiet about the undropped
visit_id.

**Send back:** the full output. If anything besides visit_id is
flagged, that is the next detector lesson.
