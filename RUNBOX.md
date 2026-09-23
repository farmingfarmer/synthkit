# RUNBOX — demo-day prep (about 25 minutes, run tonight or early)

Updated 2026-09-22. The test kit's START_HERE was re-executed
VERBATIM from a kit folder today and every command now works as
written - including its FIRST one, which was broken (`types`
demanded --out; a look no longer demands a destination). The gate
and sign-off are `synthkit gate` / `synthkit signoff` verbs now,
so the kit never depends on where scripts/ sits.

## 1. Pull as usual, run the net

Zipball per WINDOWS.md section 1. Expect **68 suites, 1918
checks, ALL GREEN** on a zipball.

## 2. Reinstall and relaunch the bench (the stale-serve ritual)

```bat
pip install -e .
```

Then close any terminal already running the bench, and launch:

```bat
python -m synthkit.cli gui
```

It serves at http://127.0.0.1:8377 and opens the browser.
Hard-refresh (Ctrl+F5) and confirm the build id in the wordmark
matches `python -m synthkit.cli version` - it should read the
same id the fit log printed. A UI change that cannot be seen is
indistinguishable from one never made.

## 3. Give every station food

```bat
python scripts\make_demo_clinic.py -o %USERPROFILE%\dev\demo_clinic
```

```bat
python -m synthkit.cli fit --src %USERPROFILE%\dev\demo_clinic\demo_clinic.csv --out %USERPROFILE%\dev\demo_run --group-by person_id --generate
```

**Expect:** about two minutes; gate MET on all 8 criteria
(`python -m synthkit.cli gate %USERPROFILE%\dev\demo_run`).

## 4. Mint the two report cards for Act 4 (FAIL card + clean card)

The clean card already exists from your exam2 run. The FAIL card
- the miscalibrated-bar story - is one command, demanding 0.83
where the as-specified ceiling is about 0.757:

```bat
python -m synthkit.cli exam %USERPROFILE%\dev\run_seed11b --effect age_at_visit=0.8 --effect span_days=-0.5 --bars auroc=0.83 -o %USERPROFILE%\dev\exam_failcard
```

**Expect:** the showdown reports bar FAILs, and
`exam_failcard\report_card.html` calls out the miscalibrated bar
BY NAME - "a bar above its own ceiling, not a failed solver."
Open it beside `exam2\report_card.html` in tabs for Act 4.

## 5. Optional - SHAP for the "drivers, attributed" section

```bat
pip install shap
```

Without it the dashboard states the omission and names this
command; with it you get the attribution bars for Act 1.

## 6. After the commands - the staging (no more terminals after this)

Two more artifacts, then it is tabs and a rehearsal:

```bat
python -m synthkit.cli signoff %USERPROFILE%\dev\run_seed11b -o %USERPROFILE%\dev\exam\signoff_real.html
```

**Expect:** one line, "wrote ... - gate NOT MET" naming close -
that is correct and Act 2 embraces it: the gate reads honestly on
real data and says what now.

```bat
python -m synthkit.cli exam %USERPROFILE%\dev\demo_run --effect stress_score=0.9 --effect sleep_hours=-0.5 -o %USERPROFILE%\dev\demo_exam_backup
```

**Expect:** about a minute, all six stages. This is the BACKUP of
Act 4's live command - if the live run hiccups, you open this
directory and say "we pre-ran this exact command."

Then, per docs/demo_script_v2.md "Before the room fills":

- Browser tabs: the bench (build id checked), exam2's clean
  report card, exam_failcard's FAIL card, signoff_real.html.
- Bench stations pre-pointed: VIEW Dashboard at the demo clinic
  fit; 03 Verdict at run_seed11b (the honest NOT MET).
- Editor tabs: CONVENTIONS.md and synthkit/gate.py.
- One full rehearsal of the script, out loud, ~30 minutes.
- Then FREEZE: no pull tomorrow morning, no reinstall - the
  build you rehearsed is the build you demo.

Kits, whenever convenient tonight:

```bat
python scripts\make_testkit.py -o %USERPROFILE%\dev\team_kit
```

Zip that folder once and share it however you message the team;
every recipient also needs the repository itself (zipball is
fine) and `pip install -e .` on Python 3.10+, because the kit
deliberately carries no code.

**Send back:** nothing unless something refuses. The walkthrough
is docs/demo_script_v2.md; kits assemble with
`python scripts\make_testkit.py -o A_FOLDER_YOU_CHOOSE` - one
folder per teammate, or one zipped folder shared, their machines
need only Python 3.10+ and `pip install -e .` from the repo.
