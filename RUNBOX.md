# RUNBOX — the live-fit act, updated after the full sweep

Updated 2026-09-23 (evening; supersedes the morning version -
the rehearsal fit command gained a load-bearing --exclude, and
the net count moved). Pull expects **68 suites, 1923 checks,
ALL GREEN** on a zipball.

One more expectation for the rehearsal, measured on the
extract-shaped fixture: at 300 patients the gate will likely
read NOT MET with `set token shares` failing hard (every
published active_drugs token near one third of its source share
- a known, recorded solve defect at small-cohort scale, not a
sampler fault). If the rehearsal shows that, the honest demo
line is scripted in SPEAKER.html; if it bothers you, add
active_drugs to the --exclude list and the live act runs with
no set columns.

Original prep notes below. The demo gains a LIVE act: the bench fitting
a 300-patient whole-patient sample of the real extract in real
time, in the room. `synthkit sample` is new in this build, so one
more pull is needed - after that, freeze again.

## 1. Pull once more

Zipball per WINDOWS.md section 1. Expect **68 suites, 1920
checks, ALL GREEN** on a zipball. Then:

```bat
pip install -e .
```

Relaunch the bench and confirm the wordmark matches:

```bat
python -m synthkit.cli version
```

## 2. Build the live-fit sample (seconds)

```bat
python -m synthkit.cli sample %USERPROFILE%\dev\tidy_visits.csv --group-by person_id --patients 300 -o %USERPROFILE%\dev\tidy_live300.csv
```

**Expect:** the echo of settings, then "wrote ... rows for 300
patient(s), whole patients only." Patients travel whole - a row
sample would break the visit structure everything is measured on.

## 3. Rehearse the live fit ONCE, for time and verdict

```bat
python -m synthkit.cli fit --src %USERPROFILE%\dev\tidy_live300.csv --out %USERPROFILE%\dev\live300_rehearsal --group-by person_id --generate --exclude conditions,procedures,drug_routes
```

The `--exclude` is load-bearing: set-token expansion owns the
search budget (the fixture's 85 columns became 143 searched, and
the un-excluded 300-patient fit blew past ten minutes on the dev
machine). Excluding the three heaviest set columns keeps
active_drugs as the set showcase and cuts the search by roughly
a third. Use the SAME flags in the bench's Fit station in the
room.

**Expect:** the full narration. WRITE DOWN two numbers into
SPEAKER.html's two rehearsal blanks: the total minutes, and what
the gate says:

```bat
python -m synthkit.cli gate %USERPROFILE%\dev\live300_rehearsal
```

If the fit ran longer than about 4 minutes, rebuild the sample
at 150 patients (same command, --patients 150, a new -o name)
and rehearse that instead.

In the ROOM, Act 1 repeats this exact fit through the bench's
02 Fit station - point --src at the sample with the full literal
path and pick a NEW out directory live.

**Send back:** the rehearsal minutes and the gate verdict, so the
speaker notes match the room.
