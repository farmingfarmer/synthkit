"""Smoke: reading whatever file the operator actually has.

WHY THIS EXISTS. The fitted path read CSV and nothing else, so
an operator with an .xlsx export or a .jsonl dump had to convert
it first - and the error when they did not was pandas' own, a
parser complaint about a binary file that says nothing about
what to do next.

THE READER IS PART OF THE SUSPECT PATH. This project has
recorded the wrong-type fault four times and the
empty-versus-missing fault twice, and both of those are reader
faults. So the load-bearing check here is not "xlsx opens" - it
is that EVERY FORMAT PRODUCES THE SAME FRAME. A second reader
that disagrees with the first about what a blank cell is would
put the folded-empties fault back exactly where it was fixed,
and it would be invisible from inside, because the run would be
self-consistently wrong.
"""
import io
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from synthkit.intake import (                        # noqa: E402
    FORMATS, IntakeError, describe_support, read_table)

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:3d}  {}".format(PASS, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def gated(label, available, cond):
    """Pass with a NOTE when the backing package is absent.

    Printed, never silent - the same rule `smoke_buildid` uses
    for git. A skipped check that says nothing reads as a check
    that passed.
    """
    if not available:
        print("     NOTE: {} - SKIPPED, the package that backs "
              "it is not installed".format(label))
        return
    check(label, cond)


def fixture():
    """Carries the cells that have gone wrong before.

    A BLANK, an INTEGRAL FLOAT and a UNICODE value. The blank is
    the empty-versus-missing fault; the integral float is how a
    workbook turns the integer 3 into "3.0", which is a
    different VALUE in a file of strings; the unicode is in the
    23-shape sweep for a reason.
    """
    rng = np.random.default_rng(4)
    n, per = 40, 4
    pid = np.repeat(["P{:02d}".format(i) for i in range(n)], per)
    N = n * per
    return pd.DataFrame({
        "person_id": pid,
        "visit": np.tile(np.arange(per), n),
        "num": np.round(rng.normal(50, 10, N), 2),
        "whole": rng.integers(0, 5, N),
        "blank": [""] * N,
        "uni": ["Ünicode"] * N,
        "cat": rng.choice(["a", "b", "c"], N),
    })


def main():
    df = fixture()
    have_xl = True
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        have_xl = False
    try:
        import pyarrow  # noqa: F401
        have_pq = True
    except ImportError:
        have_pq = False

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        df.to_csv(d / "s.csv", index=False)
        df.to_json(d / "s.jsonl", orient="records", lines=True)
        df.to_json(d / "s.json", orient="records")
        df.to_csv(d / "s.tsv", index=False, sep="\t")
        if have_xl:
            with pd.ExcelWriter(d / "s.xlsx") as w:
                df.to_excel(w, sheet_name="visits", index=False)
                df.to_excel(w, sheet_name="notes", index=False)
        if have_pq:
            df.to_parquet(d / "s.parquet", index=False)

        base, note = read_table(d / "s.csv")
        check("a CSV reads, and the reader SAYS how it read it "
              "- a reader that does not say what it did is the "
              "silent-wrong-type fault waiting to happen: {}"
              .format(note),
              base.shape == df.shape and "delimited text" in note)
        check("...with every column a STRING, because the "
              "typing pass decides what a column is and a "
              "reader that guesses makes that decision "
              "silently",
              all(base[c].map(type).eq(str).all()
                  for c in base.columns))
        check("...and a blank cell is \"\" and NOT missing - "
              "present-and-empty is a fact about a set column, "
              "and folding it into missing erased 36% of the "
              "medication-free visits on the real extract once",
              (base["blank"] == "").all()
              and base["blank"].notna().all())

        # THE LOAD-BEARING CHECK: every format agrees.
        others = [("jsonl", d / "s.jsonl"), ("json", d / "s.json"),
                  ("tsv", d / "s.tsv")]
        if have_xl:
            others.append(("xlsx", d / "s.xlsx"))
        if have_pq:
            others.append(("parquet", d / "s.parquet"))
        for name, path in others:
            got, _n = read_table(path)
            same_cols = list(got.columns) == list(base.columns)
            same_vals = same_cols and got.equals(base)
            check("{} produces the IDENTICAL frame to the CSV "
                  "- a second reader that disagrees about a "
                  "blank or a number puts the folded-empties "
                  "fault back, and it is invisible from inside "
                  "because the run is self-consistently wrong"
                  .format(name),
                  same_vals)

        check("an integral number does not gain a decimal tail "
              "- a workbook renders the integer 3 as 3.0, which "
              "is a different VALUE in a file of strings",
              set(base["whole"]) == {str(int(v))
                                     for v in df["whole"]})

        # ---- REFUSALS, EACH NAMING WHAT WOULD FIX IT
        (d / "s.xyz").write_text("a,b\n1,2\n", encoding="utf-8")
        try:
            read_table(d / "s.xyz")
            bad = ""
        except IntakeError as exc:
            bad = str(exc)
        check("an unreadable extension is REFUSED by name and "
              "the refusal LISTS what can be read - 'could not "
              "read as CSV' on a .xlsx is a true sentence that "
              "helps nobody",
              ".xyz" in bad and ".csv" in bad and ".jsonl" in bad)

        try:
            read_table(d / "nope.csv")
            miss = ""
        except IntakeError as exc:
            miss = str(exc)
        check("...and a file that is not there says so, rather "
              "than failing inside a parser",
              "does not exist" in miss)

        if not have_pq:
            (d / "s.parquet").write_text("x", encoding="utf-8")
        try:
            read_table(d / "s.parquet")
            dep = "READ-OK"
        except IntakeError as exc:
            dep = str(exc)
        gated("a format whose package is absent is refused with "
              "the INSTALL that would fix it", not have_pq,
              "pip install pyarrow" in dep)

        # ---- A WORKBOOK HAS SHEETS AND A CSV DOES NOT
        if have_xl:
            _f, n1 = read_table(d / "s.xlsx")
            check("an Excel read NAMES the sheet it took and "
                  "what else was in the book - reading the "
                  "first tab silently is how an operator gets "
                  "a run over their notes: {}".format(n1),
                  "'visits'" in n1 and "notes" in n1)
            _f2, n2 = read_table(d / "s.xlsx", sheet="notes")
            check("...and a named sheet is honoured",
                  "'notes'" in n2)
            try:
                read_table(d / "s.xlsx", sheet="absent")
                sh = ""
            except IntakeError as exc:
                sh = str(exc)
            check("...and an absent sheet is refused LISTING "
                  "the real ones, so the next attempt can "
                  "succeed",
                  "absent" in sh and "visits" in sh
                  and "notes" in sh)
        else:
            print("     NOTE: openpyxl is not installed - the "
                  "three Excel checks were SKIPPED rather than "
                  "passing on a reader that never ran")

        # ---- IT REACHES THE CLI, not only the module
        for name in ("s.csv", "s.jsonl"):
            r = subprocess.run(
                [sys.executable, "-m", "synthkit.cli", "fit",
                 "--src", str(d / name), "--out",
                 str(d / ("o_" + name.replace(".", "_"))),
                 "--group-by", "person_id", "--types-only"],
                capture_output=True, text=True, timeout=900)
            check("`synthkit fit` reads {} end to end and "
                  "prints how - a capability wired to nothing "
                  "is this repo's named silent-failure class"
                  .format(name),
                  r.returncode == 0 and "read " in r.stdout)

        # ---- AND THE SUPPORT TABLE TELLS THE TRUTH
        sup = "\n".join(describe_support())
        check("the support table lists every format the reader "
              "knows, and marks the ones that need a package "
              "that is absent rather than claiming them",
              all(e in sup for e in FORMATS)
              and ("NO - pip install" in sup or have_pq))

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
