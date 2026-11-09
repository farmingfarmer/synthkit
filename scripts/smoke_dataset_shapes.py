"""Smoke: the DATASET shapes that have historically FOUND faults.

NAMED `dataset_shapes` BECAUSE `smoke_shapes` WAS TAKEN, and I
found that out by overwriting it. `smoke_shapes.py` is about
EFFECT CURVE shapes - rising, U-shaped, threshold, saturating -
and this is about the shape of a whole dataset. This project
already records the rule: read a file before writing over it, or
pick a name the repo does not already use. I ran `ls` on the
target, it printed EXISTS, and the heredoc that destroyed it was
queued in the SAME command - so the check ran and its answer
could not stop anything. A guard whose result arrives too late
to act on is not a guard.

WHY THIS EXISTS. The 23-shape sweep is the evidence behind "it
should handle your data", and it is run BY HAND. CI runs the net
and not the sweep, so for months that claim was verified only
when somebody remembered to run a script - and on 2026-11-08 a
re-run came back with a CRASH on `wide + 92% sparse` against a
recorded "23 of 23 clean": scikit-learn 1.9 refuses any all-NaN
feature column, and discovery's split can empty a sparse one. The
claim had gone stale the day the data machine's libraries moved,
and nothing in the net could see it.

THIS IS A SUBSET AND SAYS SO. Eight shapes, chosen because each
has found a real defect here, at ~35 seconds total. The full
twenty-three remain the hand-run sweep (`shape_sweep.py`), and
the count is asserted so a reader can see how much of it this
covers.

A CRASH-ONLY CHECK WOULD PASS ON A USELESS FILE. Each shape must
also come back with the operator's own columns and without a
column collapsed to a single value that the SOURCE did not have -
which is the fault class this sweep found on the neural engine
(a code, a note and a date all folded into `__other__`).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import shape_sweep as SS                        # noqa: E402
from synthkit import blueprint as B             # noqa: E402
from synthkit.discover import discover          # noqa: E402
from synthkit.generate import generate          # noqa: E402

PASS = FAIL = 0

# Each of these has found a real defect in this project. Named
# rather than sampled, because "a few shapes" is not a statement
# anybody can check.
WHY = {
    "wide + 92% sparse":
        "scikit-learn 1.9 refuses an all-NaN feature and "
        "discovery's split can empty a sparse column",
    "all-missing column":
        "a column with no present value is neither numeric nor "
        "categorical, and the neural engine emitted the STRING "
        "'nan' on every row of one",
    "constant + empty column":
        "a constant column has no spread to divide by",
    "imbalanced groups":
        "a count's mean lives in the entities the k rule will "
        "not publish - one entity holding half the rows",
    "high-cardinality code":
        "every level below the k floor folds to __other__, and "
        "publishing nothing is not the only option",
    "one row per person":
        "no within-patient structure exists, so every "
        "constant-within-patient test is trivially true",
    "NO group column":
        "flat data must come back flat rather than carrying an "
        "invented person_id and visit_number",
    "mixed types in a column":
        "a column read as the wrong type fails SILENTLY",
}


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:3d}  {}".format(PASS, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def main():
    check("the subset NAMES shapes that exist in the sweep ({} of "
          "{} shapes) - a renamed shape must fail loudly here "
          "rather than silently skipping"
          .format(len(WHY), len(SS.SHAPES)),
          all(k in SS.SHAPES for k in WHY))
    check("...and it is a SUBSET, stated: the full {} remain the "
          "hand-run sweep, and a reader can see how much of it "
          "this covers".format(len(SS.SHAPES)),
          0 < len(WHY) < len(SS.SHAPES))

    for name in sorted(WHY):
        fn = SS.SHAPES[name]
        r = np.random.RandomState(0)
        df = fn(r)
        gb = "person_id" if "person_id" in df.columns else None
        err = None
        g = None
        try:
            cat = discover(df, group_by=gb, seed=1)
            bp = B.build(df, cat, group_by=gb)
            npat = int(df[gb].nunique()) if gb else len(df)
            g = generate(bp, n_patients=npat, seed=5)
        except Exception as exc:
            err = "{}: {}".format(type(exc).__name__, exc)
        check("{} completes discover -> build -> generate ({})"
              .format(name, err or "ok, {} rows".format(len(g))),
              err is None)
        if err is not None:
            continue

        # EVERY COLUMN THE OPERATOR GAVE US COMES BACK. A file
        # missing one of their columns is a different file.
        _missing = [c for c in df.columns if c not in g.columns
                    and c != gb]
        check("...and every column the operator gave us comes "
              "back ({} missing)".format(_missing or "none"),
              not _missing)

        # AND A CRASH-ONLY CHECK WOULD PASS ON A USELESS FILE.
        # A column the source varies must not arrive as a single
        # value - that is the shape the neural sweep found on a
        # code, a note and a date, all folded to __other__.
        _flat = []
        for c in df.columns:
            if c == gb or c not in g.columns:
                continue
            if df[c].nunique(dropna=True) < 2:
                continue        # the SOURCE was constant
            if g[c].nunique(dropna=True) < 2:
                _flat.append(c)
        check("...and no column the source VARIES arrives as a "
              "single value ({})".format(_flat or "none"),
              not _flat)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
