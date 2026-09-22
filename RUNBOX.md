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

Then stop any running `synthkit gui`, relaunch it, hard-refresh
the browser, and confirm the build id in the wordmark matches
`synthkit version`. A UI change that cannot be seen is
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

**Send back:** nothing unless something refuses. The walkthrough
is docs/demo_script_v2.md; kits assemble with
`python scripts\make_testkit.py -o A_FOLDER_YOU_CHOOSE` - one
folder per teammate, or one zipped folder shared, their machines
need only Python 3.10+ and `pip install -e .` from the repo.
