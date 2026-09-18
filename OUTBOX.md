# OUTBOX — nothing required; one optional polish before Wednesday

Updated 2026-09-18. The sprint plan is fully delivered - all eight
items, both optional runs included. No commands are required.

## Optional — rebuild the 5x deck with the corrected caption

The scale table's caption now states WHICH comparison it makes
(yours pairs two independent fits - the stronger claim, and the
caption should say so under the exhibit). If you want the
corrected wording for Wednesday: pull as usual (the net expects
**67 suites, 1897 checks, ALL GREEN** on a zipball), then:

```bat
python scripts\fidelity_deck.py --src %USERPROFILE%\dev\tidy_visits.csv --run %USERPROFILE%\dev\run_seed11b -o %USERPROFILE%\dev\exam\deck_5x.html --group-by person_id --compare "5x=%USERPROFILE%\dev\run_5x"
```

**Expect:** the same table, with the caption now reading
"independent fits ... the STRONGER form of no-degradation."

Otherwise: rehearse `docs/demo_script_v2.md` once at your leisure,
and Wednesday's deep dive has everything it needs.
