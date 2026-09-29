"""Assemble the team test kit - a folder a teammate runs and
tries to break, with no real data anywhere near it.

    python scripts/make_testkit.py -o OUTDIR

Writes into OUTDIR: the clinic dataset (invented, with a printed
answer key), START_HERE.md (every command from types-check to
sign-off, copy-pasteable on Windows and Unix), and
TRY_TO_BREAK.md (a challenge list - anything that breaks becomes
a check, and the kit gets harder). The kit assumes synthkit is
installed (`pip install -e .` from the repository); it carries no
code of its own, so a kit never goes stale against the install.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

START_HERE = """# synthkit test kit - start here

Everything below runs on the INVENTED clinic dataset in this
folder. No real patient data is involved anywhere; the answer key
in `answer_key.txt` lists the nine patterns that were planted, so
you can judge every claim the tool makes against known truth.

Prerequisite: synthkit installed - from the repository root,
`pip install -e .` (Python 3.10+). Optional: `pip install shap`
for the driver-attribution section of the dashboard.

Commands are shown Unix-style with forward slashes; on Windows
use backslashes and the same words.

## 1. The two-second look (always first)

    python -m synthkit.cli types --src clinic/demo_clinic.csv --group-by person_id

Every column, how it was read, and whether anonymization would
destroy it - before anything expensive runs. Then the PHI check:

    python -m synthkit.cli scrub clinic/demo_clinic.csv --group-by person_id

The clinic data should come back CLEAR - it was invented without
PHI on purpose - and a long-text column would be reported OUT OF
SCOPE by name, because this scrub reads structured fields only
and says so rather than pretending otherwise.

## 2. Fit and generate (a few minutes - watch it narrate)

    python -m synthkit.cli fit --src clinic/demo_clinic.csv --out run1 --group-by person_id --generate

Watch the narration: search with a live countdown, the contract,
generation, self-grading. Then read the verdict:

    python -m synthkit.cli gate run1

Expect the gate MET on all criteria - and read one criterion's
explanation to see how you would check it yourself.

## 3. The bench

    python -m synthkit.cli gui

Point 01 Source / VIEW Dashboard at `clinic/demo_clinic.csv` and
`run1` (full literal paths - browser fields do not expand
environment variables). In the Dashboard: press and hold any
histogram; hold a correlation heatmap; find the planted threshold
in the pattern cards and check it against `answer_key.txt`.

## 4. The exam loop (about three minutes, one command)

    python -m synthkit.cli exam run1 --effect stress_score=0.9 --effect sleep_hours=-0.5 -o exam

That is bridge -> plant -> ladder -> baseline -> showdown ->
report card, each stage echoing what it did. The showdown prints
ceiling / baseline / vendor per tier - the ceiling is computable
only because the answer was planted, which is the product's
entire thesis in one number. Every artifact lands in `exam/`.

The stages also exist as separate commands (`bridge`, `plant`,
`campaign-compile`, `campaign-run`, `showdown`) when you want to
vary one of them; `python -m synthkit.cli exam --help` names the
dials (effects, prevalence, bars, solvers).

## 5. The NEURAL engine (the newer one)

There are two generators. Everything above used the RULES
engine, which writes down what it found as a published recipe
and builds from that alone. The NEURAL engine trains a small
autoencoder on the records themselves and samples what it
learned. On the real extract it keeps more of the structure -
notably effects that only exist when two columns act together,
where the rules engine carries none - and it is the one being
developed now.

    python -m synthkit.cli gui

Click **GEN Neural** in the left rail, then fill four boxes:

    source CSV path          clinic/demo_clinic.csv   (full path)
    output directory         neural1                  (full path)
    patient / entity column  person_id
    seed                     0

Press **Generate**, wait for "written in NNNs", then press
**Review the output**. The review leads with whether a patient
still looks like a person, then how many columns a model cannot
tell apart from the real ones, then the combination effects,
then every column drawn twice.

That output directory also works in **03 Verdict** and **VIEW
Dashboard** - same directory, same boxes.

Read the closing section of the review page once. It says what
the page does NOT measure, which is safety: the neural engine
learns from records, and that is a different posture from the
rules engine.

## 6. YOUR OWN DATA

Everything above works on any tidy CSV - one row per event, a
column identifying the entity. Point it at something of yours:

    python -m synthkit.cli types --src YOUR.csv --group-by YOUR_ID_COLUMN

Read how every column was typed before anything else. A column
read as the wrong type is the most expensive fault this tool
has, because it is silent.

    python -m synthkit.cli scrub YOUR.csv --group-by YOUR_ID_COLUMN

Then fit it, or run it through the Neural station above.

**What has been measured about other shapes:** both engines were
swept across 23 dataset shapes a customer could plausibly hand
over - cross-sectional, no time column, numeric-only,
categorical-only, high-cardinality codes, 120 rows, 60 columns,
constant and all-empty columns, duplicate rows, unicode, one row
per entity, one entity holding half the rows, free text, mixed
types in one column, 92% sparse, extreme skew, boolean-ish
spellings, and a flat table with no entity column at all. All 23
come through without a crash, a lost column, or a column
collapsed to a single value. If YOUR data breaks one of them,
that is exactly the kind of finding this kit is for.

## 7. The artifacts

    python -m synthkit.cli signoff run1 -o signoff.html

`exam/report_card.html` was already written by the exam command.

Open both in a browser. The report card judges the BAR as well
as the solver; the sign-off is the one page a decision-maker
signs, limitations stated.

## What to read when something says NO

Every refusal here is a sentence naming the file, the cause, and
the next command - if you ever see a raw stack trace instead,
that is a bug worth reporting on its own.

Then open `TRY_TO_BREAK.md`.
"""

TRY_TO_BREAK = """# Try to break it

Anything you break becomes a check, and the kit gets harder.
Report what you did, what you expected, and what happened -
screenshots welcome. Some starting points, roughly in order of
mischief:

0a. Point the Neural station at a CSV of your own with a shape
   nothing here has: one row per entity, no entity column at
   all, a column of codes that are all unique, 200 rows, 300
   columns. All 23 shapes we tried come through clean; find the
   24th.
0. Add a column of phone numbers to the clinic CSV under an
   innocent header (`fax_pref`, say) and run `scrub` - it must
   still be caught, because values decide and headers only
   assist. Then add a column NAMED `ssn` holding lab values - it
   must come back clear. Break either direction and you have
   found a real bug.
1. Feed `types` a file it should struggle with: currencies with
   symbols, times like 14:32, percent signs, a column that is
   half numbers and half words. Does the two-second look tell
   the truth about every column?
2. Point `fit --src` at a FOLDER instead of a file, a file that
   does not exist, and a CSV with one row. Sentences or stacks?
3. Point `exam` at a directory that holds no blueprint.json,
   then delete `manifest.json` from a campaign folder and run
   showdown. Then point `--json-out` at a folder wearing a
   filename.
4. Plant an effect on a column that does not exist, and then an
   exam where EVERY effect is refused.
5. Set a campaign bar above 1.0, then run the exam. Read what
   the report card says about the bar - it should blame the
   bar, not the solver.
6. Run the same fit twice with the same seed - the outputs
   should be identical. Then change only the seed and read what
   moves; nothing here believes a single-seed number, and
   neither should you.
7. Open the dashboard on a run whose generated.csv you deleted.
8. Edit blueprint.json by hand - set a coverage dial to 2.0 or
   rename a column - and regenerate. Dials must report requested
   AGAINST achieved; an unknown name must be an error, never a
   shrug.
9. Hand `fit` a dataset with one patient holding half the rows,
   or a column that is 95% empty, and read whether the outputs
   say so honestly.
10. Read `answer_key.txt`, then hunt each planted pattern in the
    dashboard. Every one you cannot find is a finding - either
    the tool lost it (report it) or the tool refused it for
    privacy and said so (check the findings file for the
    refusal).

House rule worth knowing while you hunt: the numbers this tool
prints are measured, never assumed - so when your experiment and
the tool disagree, one of you has a bug, and the tool would
genuinely like to know which.
"""


def main():
    ap = argparse.ArgumentParser(
        description="Assemble the team test kit.")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    clinic = out / "clinic"
    p = subprocess.run(
        [sys.executable, str(ROOT / "scripts" /
                             "make_demo_clinic.py"),
         "-o", str(clinic)],
        capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit("STOPPED: the clinic builder failed:\n"
                 + (p.stdout + p.stderr)[-1500:])
    (out / "answer_key.txt").write_text(p.stdout,
                                        encoding="utf-8")
    (out / "START_HERE.md").write_text(START_HERE,
                                       encoding="utf-8")
    (out / "TRY_TO_BREAK.md").write_text(TRY_TO_BREAK,
                                         encoding="utf-8")
    print("test kit -> {}".format(out))
    print("  clinic/demo_clinic.csv  (invented data)")
    print("  answer_key.txt          (the nine planted patterns)")
    print("  START_HERE.md           (every command, in order)")
    print("  TRY_TO_BREAK.md         (the challenge list)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
