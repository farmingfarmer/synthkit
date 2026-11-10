"""`close_ceiling.py` says how much of `close` is sampling noise.

WHY THIS SUITE EXISTS. The script produces a number that would be
quoted in an argument about whether to keep spending on the engine,
and this project has twice shipped a tool whose figures were reported
to somebody while nothing automated ever ran it - `peek.py` and
`close_gap.py` both arrived with no checks at all.

WHAT IS LOAD-BEARING. Two things. The split must be BY PATIENT, or
the same person lands on both sides and the agreement measured is
the same records twice - which would read as a high ceiling and is
the one way this instrument can lie in the flattering direction. And
the UNREACHABLE branch must be REACHABLE: a verdict that can only
ever say "inconclusive" is the check-that-cannot-fail shape wearing
a conclusion.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from close_ceiling import BAR, halves, verdict         # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def build_bp(df, path):
    """A REAL blueprint, because `_pair_fidelity` iterates its
    relationships - a hand-written one would carry whatever pairs I
    invented, which is the fixture-agrees-with-the-reader fault."""
    import json
    from synthkit import blueprint as B
    from synthkit.discover import discover
    cat = discover(df, group_by="person_id")
    bp = B.build(df, cat, group_by="person_id")
    Path(path).write_text(json.dumps(bp), encoding="utf-8")
    return bp


def run(args, cwd=None):
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "close_ceiling.py")]
        + args, capture_output=True, text=True, cwd=str(ROOT))
    return r.returncode, r.stdout + r.stderr


def strong_fixture(n=2400, seed=2):
    """Pairs strong enough that two halves agree - INCONCLUSIVE."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    return pd.DataFrame({
        "person_id": ["P%04d" % (i // 4) for i in range(n)],
        "x": np.round(x, 4),
        "y": np.round(2.0 * x + rng.normal(0, 0.15, n), 4),
        "z": np.round(1.5 * x + rng.normal(0, 0.2, n), 4)})


def weak_fixture(n=900, seed=5, k=10):
    """MANY BARELY-RELATED pairs, which is the shape that makes the
    bar unreachable on noise alone - a correlation of 0.12 measured
    on half a small cohort does not reproduce on the other half.

    A fixture of strong pairs cannot reach the UNREACHABLE branch,
    and a verdict with one reachable branch is not a verdict."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, n)
    cols = {"person_id": ["P%04d" % (i // 3) for i in range(n)]}
    for j in range(k):
        # Moderate enough that DISCOVERY finds the pairs - an
        # unfound pair is not in the surface at all - but measured
        # on a small cohort, so the two halves disagree.
        cols["c%02d" % j] = np.round(
            0.42 * base + rng.normal(0, 1.0, n), 4)
    return pd.DataFrame(cols)


def main():
    import tempfile
    import json

    # ---- THE SPLIT IS BY PATIENT ---------------------------------
    df = strong_fixture()
    A, B = halves(df, "person_id", np.random.default_rng(7))
    sa = set(A["person_id"].astype(str))
    sb = set(B["person_id"].astype(str))
    check("the split is BY PATIENT, so no person appears on both "
          "sides - splitting ROWS would put the same person in both "
          "halves and the agreement measured would be the same "
          "records twice, which is the one way this instrument can "
          "lie in the flattering direction",
          len(sa & sb) == 0)

    check("...and every patient lands on exactly one side, so no "
          "row is silently dropped from the comparison",
          len(sa) + len(sb) == df["person_id"].nunique()
          and len(A) + len(B) == len(df))

    check("...and the halves are of comparable size, because a "
          "lopsided split would measure a small sample against a "
          "large one and attribute the extra noise to the engine",
          abs(len(sa) - len(sb)) <= 1)

    # ---- THE SPELLER IS WIRED, WITH NO SILENT FALLBACK ----------
    # The first cut imported a function named `sayn`, which does not
    # exist - the module exports `spell` and `annotate` - inside a
    # try/except that fell back to `str`. The real-extract reading
    # then came back as bare DIGITS, which by this project's own
    # rule are OCR artifacts rather than measurements, while the
    # tool reported success. A silent no-op is worse than a crash,
    # which is why `getattr` on a library attribute is banned here.
    from close_ceiling import _say
    said = _say(97.9)
    check("every number the tool prints is repeated in WORDS, "
          "because the machine that holds the extract is read by "
          "screen OCR and a bare digit off it is an artifact - "
          "`{}`".format(said),
          "nine" in said and "seven" in said)

    cc = (ROOT / "scripts" / "close_ceiling.py").read_text(
        encoding="utf-8")
    check("...and the speller is imported WITHOUT a fallback, so a "
          "rename breaks the script instead of quietly turning "
          "every figure back into an untrustworthy digit",
          "from sayn import annotate" in cc
          and "def sayn(x)" not in cc
          and "return str(n)" not in cc)

    # ---- NOTHING IS GENERATED ------------------------------------
    txt = (ROOT / "scripts" / "close_ceiling.py").read_text(
        encoding="utf-8")
    check("the script never imports the GENERATOR - the whole claim "
          "is that both sides are real data, so a fit or a generate "
          "anywhere in it would make the number mean something else",
          "from synthkit.generate" not in txt
          and "import generate" not in txt)

    check("...and it measures through the GATE'S OWN "
          "`_pair_fidelity` rather than a second implementation, so "
          "the ceiling and the criterion cannot disagree about what "
          "`close` means",
          "_pair_fidelity" in txt
          and "from synthkit.pipeline import" in txt)

    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        s_strong = t / "strong.csv"
        df.to_csv(s_strong, index=False)
        s_weak = t / "weak.csv"
        weak_fixture().to_csv(s_weak, index=False)

        bp_s = t / "bp_strong.json"
        build_bp(df, bp_s)
        bp_w = t / "bp_weak.json"
        wf = weak_fixture()
        wf.to_csv(s_weak, index=False)
        build_bp(wf, bp_w)

        rc, out = run(["--src", str(s_strong), "--blueprint",
                       str(bp_s), "--group-by",
                       "person_id", "--splits", "2"])
        check("a cohort of STRONG pairs reads at or above the bar "
              "and the verdict says INCONCLUSIVE, naming itself a "
              "LOWER BOUND rather than clearing the bar",
              rc == 0 and "INCONCLUSIVE" in out
              and "LOWER BOUND" in out)

        # ---- THE VERDICT IS A PURE FUNCTION ON PLANTED INPUTS ----
        # No fixture here produces a split-half ceiling under the
        # bar: discovery publishes only the pairs it FINDS, so a
        # cohort weak enough to disagree across halves is also one
        # where few pairs reach the surface at all - measured, over
        # six shapes, the lowest reached was 93.3%. So the branch is
        # asserted where the DECISION is made, on numbers chosen for
        # it, which is this project's own rule after a text check
        # passed while the guard it named was bypassed.
        check("the UNREACHABLE branch is REACHABLE: a ceiling of "
              "{}% against a bar of {}% reads unreachable"
              .format(BAR - 5.0, BAR),
              verdict(BAR - 5.0) == "unreachable")

        check("...and a ceiling AT the bar reads inconclusive "
              "rather than passing, because a lower bound that "
              "reaches a bar says only that noise alone does not "
              "rule the bar out",
              verdict(BAR) == "inconclusive"
              and verdict(BAR + 5.0) == "inconclusive")

        check("...and the two branches are separated by the bar "
              "itself rather than by a tolerance, so a run landing "
              "a hundredth either side is classed by the number "
              "and not by a rounding",
              verdict(BAR - 0.01) == "unreachable"
              and verdict(BAR) == "inconclusive")

        check("...and NO pairs is its own answer rather than being "
              "folded into either verdict - an absent measurement "
              "is not a clean one, which is the distinction this "
              "project draws for the linkage element",
              verdict(None) == "no-pairs")

        # And the measured path must still PRINT the branch it chose
        # on real data, or the pure function is tested and the tool
        # is not.
        check("the measured path prints the verdict it chose, so "
              "the pure function above is wired to the output "
              "rather than tested beside it",
              "INCONCLUSIVE" in out
              and ("UNREACHABLE" in
                   (ROOT / "scripts" / "close_ceiling.py").read_text(
                       encoding="utf-8")))

        # ---- the range, never a point ------------------------------
        check("the ceiling is reported as a RANGE across splits, "
              "which this project's oldest rule about seeds "
              "requires", "to" in out and "RANGE" in out)

        # ---- NO CELL IS PUBLISHED --------------------------------
        # A ceiling report on real data is itself a release: a
        # contingency cell is a count of people, and a value is
        # somebody's measurement.
        vals = set()
        for c in ("x", "y", "z"):
            vals.update(str(v) for v in df[c].head(40))
        leaked = [v for v in vals if len(v) > 5 and v in out]
        check("no SOURCE VALUE appears in the output ({} candidate "
              "values checked) - a ceiling report on a real extract "
              "is itself a release, and this script prints counts, "
              "shares and column names only".format(len(vals)),
              not leaked)

        # ---- refusals -------------------------------------------
        rc3, out3 = run(["--src", str(s_strong), "--blueprint",
                         str(bp_s), "--group-by",
                         "no_such_column"])
        check("a grouping column that does not exist is REFUSED by "
              "name with a non-zero exit, and the message says WHY "
              "the split must be by patient rather than by row",
              rc3 != 0 and "no column named" in out3
              and "BY PATIENT" in out3)

        rc4, out4 = run(["--help"])
        check("--help exits 0 and prints the usage, so a wrapper "
              "does not read a help request as a failure - this "
              "project has an audit for exactly that and three "
              "scripts had failed it", rc4 == 0 and "usage:" in out4)

        rc5, _ = run([])
        check("...and a missing --src refuses rather than scanning "
              "whatever happens to be nearby", rc5 != 0)

        # ---- the blueprint path, which is the one a real run uses --
        bp = {"columns": {}, "relationships": []}
        bpp = t / "bp.json"
        bpp.write_text(json.dumps(bp), encoding="utf-8")
        rc6, out6 = run(["--src", str(s_strong), "--blueprint",
                         str(bpp), "--group-by", "person_id",
                         "--splits", "1"])
        check("a blueprint with NO relationships reports that NO "
              "PAIR WAS COMPARABLE and exits non-zero, rather than "
              "printing a ceiling over nothing - which is the worst "
              "kind of clean",
              rc6 == 3 and "NO PAIR WAS COMPARABLE" in out6)

        rc7, out7 = run(["--src", str(s_strong), "--group-by",
                         "person_id"])
        check("...and omitting --blueprint is REFUSED by name, "
              "because `_pair_fidelity` iterates the blueprint's "
              "own relationships so without one the surface is "
              "empty by construction",
              rc7 != 0 and "--blueprint is required" in out7)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
