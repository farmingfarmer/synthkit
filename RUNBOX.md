# RUNBOX — nothing pending

Updated 2026-09-22. This file carries the CURRENT data-machine CLI
commands, one fenced block each (GitHub's copy button works per
block). Outgoing communications live elsewhere; this file is only
ever commands.

The scrub re-read landed exactly as predicted: `PHI visit_id`
(about one per row), `OUT OF SCOPE person_id`, and 42 of 44
clear. Goal 2's real-extract read is closed. No commands are
required right now.

## Reference — the scrub read, for a future re-run

```bat
python -m synthkit.cli scrub %USERPROFILE%\dev\tidy_visits.csv --group-by person_id
```

**Expect:** PHI visit_id, OUT OF SCOPE person_id, clear: 42 of 44.
Exit code 1 is the command working - it refuses to stay quiet
about the undropped visit_id.
