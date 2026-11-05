"""Smoke: the permutation check, which used to be done by hand.

WHY IT MATTERS. Every claim the repair and the output dials
make rests on one property - the file written holds the same
values as the file read, reordered. That is what leaves the
k-anonymous bounds, the point masses and the whole privacy
posture exactly as published, and what lets a cipher undo the
change. It was verified by typing commands at a machine.

THE LOAD-BEARING CHECK IS THE NEGATIVE ONE. A permutation
checker that passes everything is worse than none, because the
sentence it licenses is a privacy claim. So the suite feeds it
a file with ONE VALUE CHANGED and demands a refusal that names
the column, and a RE-RENDERED float - 22.29 written back as
22.289999999999999, which this project has actually caught a
dial doing - and demands that too.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import verify_permutation as VP                        # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:3d}  {}".format(PASS, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def main():
    import pandas as pd
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        base = pd.DataFrame({
            "pid": ["P1", "P2", "P3", "P4"],
            "v": ["1.5", "2.5", "3.5", "4.5"],
            "w": ["10", "20", "30", "40"],
            "flat": ["k", "k", "k", "k"]})
        a = d / "a.csv"
        base.to_csv(a, index=False)

        # ---- A GENUINE REARRANGEMENT of one column
        perm = base.copy()
        perm["v"] = ["4.5", "3.5", "2.5", "1.5"]
        b = d / "b.csv"
        perm.to_csv(b, index=False)
        ok, lines = VP.compare(a, b)
        txt = "\n".join(lines)
        check("a reordered column PASSES, and the verdict says "
              "the released values are exactly the published "
              "ones in a different order",
              ok and "PURE REARRANGEMENT" in txt)
        check("...and it names the column that moved rather "
              "than reporting a bare pass",
              "CHANGED          v" in txt)
        check("...and reports every untouched column as "
              "identical, so a change nobody asked for would "
              "show",
              "OTHERS-IDENTICAL True" in txt)

        # ---- ONE VALUE CHANGED is not a rearrangement
        bad = base.copy()
        bad["v"] = ["1.5", "2.5", "3.5", "9.9"]
        c = d / "c.csv"
        bad.to_csv(c, index=False)
        ok2, l2 = VP.compare(a, c)
        t2 = "\n".join(l2)
        check("ONE value changed is REFUSED - the check that "
              "makes the privacy sentence worth saying, since "
              "a checker that passes everything licenses a "
              "claim about disclosure",
              not ok2 and "NOT a permutation" in t2)
        check("...and the refusal NAMES the value that is not "
              "in the original, so the step that introduced it "
              "can be found rather than the column re-read",
              "'9.9'" in t2)

        # ---- A RE-RENDERED FLOAT is not a rearrangement
        # either, and this one has actually happened here: a
        # dial assigned numeric values back into the frame and
        # wrote 22.289999999999999 where the original held
        # 22.29. The arrangement was right and the file was
        # still not a permutation of the other.
        rr = base.copy()
        rr["v"] = ["4.5", "3.5", "2.5", "1.4999999999999998"]
        e = d / "e.csv"
        rr.to_csv(e, index=False)
        ok3, l3 = VP.compare(a, e)
        check("a RE-RENDERED float is refused too - the "
              "arrangement can be right while the file is "
              "still not a permutation, which is a fault this "
              "project has already shipped once",
              not ok3)

        # ---- A ROW COUNT CHANGE stops before anything else
        short = base.iloc[:3]
        f = d / "f.csv"
        short.to_csv(f, index=False)
        ok4, l4 = VP.compare(a, f)
        t4 = "\n".join(l4)
        check("a file that LOST a row stops immediately, "
              "because nothing below would mean what it says",
              not ok4 and "ROWS-SAME        False" in t4
              and "MULTISET-SAME" not in t4)

        # ---- A CONSTANT COLUMN CANNOT SHOW A REORDERING
        flat = base.copy()
        flat["flat"] = ["k", "k", "k", "k"]
        flat["v"] = ["4.5", "3.5", "2.5", "1.5"]
        g = d / "g.csv"
        flat.to_csv(g, index=False)
        ok5, l5 = VP.compare(a, g)
        check("...and identical files PASS the permutation "
              "question while saying plainly that nothing "
              "changed - a pass and a no-op must not read the "
              "same",
              VP.compare(a, a)[0]
              and "IDENTICAL" in "\n".join(VP.compare(a, a)[1]))

        # ---- THE CONSTANT-COLUMN BRANCH HAS A CASE
        cst = base.copy()
        cst["flat"] = ["k", "k", "k", "j"]
        h0 = d / "h0.csv"
        cst.to_csv(h0, index=False)
        cst2 = cst.copy()
        cst2["flat"] = ["j", "k", "k", "k"]
        h1 = d / "h1.csv"
        cst2.to_csv(h1, index=False)
        okc, lc = VP.compare(h0, h1)
        check("a reordering of a two-level column is still a "
              "reordering",
              okc)
        # a genuinely constant column that "changed" cannot
        # happen as a permutation, so the UNORDERABLE branch is
        # reached by comparing a constant column against a
        # constant column of a different value - which is NOT a
        # permutation and must be refused, not excused.
        k0 = base.copy()
        k0["flat"] = ["k", "k", "k", "k"]
        k1 = base.copy()
        k1["flat"] = ["j", "j", "j", "j"]
        p0, p1 = d / "k0.csv", d / "k1.csv"
        k0.to_csv(p0, index=False)
        k1.to_csv(p1, index=False)
        okk, lk = VP.compare(p0, p1)
        check("...while a constant column replaced by a "
              "DIFFERENT constant is refused rather than "
              "excused as unorderable - the branch that could "
              "not show a reordering must not become a branch "
              "that waves one through",
              not okk)

        # ---- THROUGH THE PROCESS BOUNDARY
        r = subprocess.run(
            [sys.executable,
             str(ROOT / "scripts" / "verify_permutation.py"),
             str(a), str(b)],
            capture_output=True, text=True, timeout=600)
        check("the CLI exits 0 on a rearrangement, so it can "
              "gate a pipeline step rather than needing a human "
              "to read it",
              r.returncode == 0
              and "PURE REARRANGEMENT" in r.stdout)
        r2 = subprocess.run(
            [sys.executable,
             str(ROOT / "scripts" / "verify_permutation.py"),
             str(a), str(c)],
            capture_output=True, text=True, timeout=600)
        check("...and NON-ZERO when it is not one",
              r2.returncode == 1)
        r3 = subprocess.run(
            [sys.executable,
             str(ROOT / "scripts" / "verify_permutation.py")],
            capture_output=True, text=True, timeout=600)
        check("no arguments prints the usage rather than "
              "running something",
              r3.returncode == 2 and "usage:" in r3.stdout)
        r4 = subprocess.run(
            [sys.executable,
             str(ROOT / "scripts" / "verify_permutation.py"),
             str(d / "nope.csv"), str(b)],
            capture_output=True, text=True, timeout=600)
        check("...and a missing file is a sentence, not a "
              "stack",
              r4.returncode == 1 and "does not exist" in r4.stdout)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
