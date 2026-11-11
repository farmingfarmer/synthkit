"""`k_price.py` separates the privacy price from the engine's.

WHY THIS SUITE EXISTS. The number this produces decides whether more
fidelity work on spread is worth doing at all - if the engine is
already at the rule's ceiling, the only remaining move is to change
the privacy rule, which is a governance decision. A tool whose output
redirects a week of work gets checks, which this project has twice
had to learn the hard way (`peek.py` and `close_gap.py` both shipped
with none).

WHAT IS LOAD-BEARING. That BOTH branches of the verdict are
reachable: a tool that can only ever say "there is headroom" is the
check-that-cannot-fail shape wearing a conclusion. And that the
arithmetic is asserted on a PLANTED frame where the answer is known
by construction, rather than on whatever a fixture happens to give.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from k_price import bounds_of, price                    # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def bp_with(lo, hi, name="x"):
    return {"columns": {name: {
        "kind": "numeric",
        "marginal": {"type": "quantiles",
                     "v": [lo, (lo + hi) / 2.0, hi],
                     "q": [0.0, 0.5, 1.0]}}}}


def run(args):
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "k_price.py")]
        + args, capture_output=True, text=True, cwd=str(ROOT))
    return r.returncode, r.stdout + r.stderr


def main():
    # ---- THE BOUNDS ARE THE TWO THE ENGINE PINS TO ------------
    check("the published floor and cap are v[0] and v[-1] of the "
          "quantile grid - the same two numbers generation pins "
          "values back inside, each the mean of the k most extreme "
          "patients rather than one person's reading",
          bounds_of(bp_with(-3.0, 7.0)["columns"]["x"]) == (-3.0,
                                                            7.0))

    check("...and a column with no quantile marginal has no "
          "published bound, so it is skipped rather than guessed "
          "at",
          bounds_of({"marginal": {"type": "list"}}) is None
          and bounds_of({}) is None)

    # ---- THE ARITHMETIC, ON A PLANTED FRAME -------------------
    # A heavy tail whose extremes sit outside the bound: the
    # ceiling is sd(clipped)/sd(source) by construction, so it can
    # be computed here independently of the tool.
    rng = np.random.default_rng(5)
    base = np.concatenate([rng.normal(0, 1, 2000),
                           rng.normal(0, 1, 40) * 12 + 40])
    src = pd.DataFrame({"x": base})
    lo, hi = -3.0, 7.0
    want_ceiling = float(src["x"].clip(lo, hi).std()
                         / src["x"].std())
    # a generator that fills the reachable range exactly
    gen = pd.DataFrame({"x": src["x"].clip(lo, hi).to_numpy()})
    rows = price(src, gen, bp_with(lo, hi))
    check("the CEILING is sd(source clipped to the bounds) over "
          "sd(source) - computed here independently and matching "
          "to four decimals ({:.4f})".format(want_ceiling),
          len(rows) == 1
          and abs(rows[0]["ceiling"] - want_ceiling) < 1e-4)

    check("...and the heavy tail makes that ceiling BITE ({:.3f}), "
          "so this frame exercises the thing the measure is about - "
          "a gaussian column would read ~1.00 and test nothing"
          .format(rows[0]["ceiling"]),
          rows[0]["ceiling"] < 0.75)

    check("...and a generator that fills the reachable range "
          "exactly reads DELIVERED ~ 1.00, which is what 'the "
          "engine is at the rule's ceiling' means",
          abs(rows[0]["delivered"] - 1.0) < 0.02)

    # ---- AND THE OTHER BRANCH MUST BE REACHABLE ---------------
    half = pd.DataFrame({"x": src["x"].clip(lo, hi).to_numpy()
                         * 0.5})
    rows2 = price(src, half, bp_with(lo, hi))
    check("a generator producing half the reachable spread reads "
          "DELIVERED ~ 0.50, so the headroom branch is reachable - "
          "a verdict with one reachable branch is the "
          "check-that-cannot-fail shape wearing a conclusion",
          abs(rows2[0]["delivered"] - 0.5) < 0.02)

    check("...and the two readings are separated by the measure "
          "rather than by the fixture: same source, same bounds, "
          "ceiling identical ({:.4f} both), delivered 1.00 against "
          "0.50".format(rows[0]["ceiling"]),
          abs(rows[0]["ceiling"] - rows2[0]["ceiling"]) < 1e-9
          and rows[0]["delivered"] > 1.8 * rows2[0]["delivered"])

    # A MATHEMATICAL INVARIANT, asserted because a violation
    # would mean the measure is wrong rather than the engine.
    # Clipping is a CONTRACTION - |clip(x)-clip(y)| <= |x-y| - and
    # the variance of a contraction cannot exceed the original, so
    # the ceiling can never be above 1. The first real-extract read
    # appeared to show 2.869 for one column, which would have been
    # impossible; it was the screen bridge rendering "0." as "2.",
    # and the words channel settled it at 0.869. A check is cheaper
    # than that round trip.
    for _bl, _bh in ((-3.0, 7.0), (-0.5, 0.5), (-50.0, 50.0)):
        _r = price(src, pd.DataFrame({"x": src["x"].to_numpy()}),
                   bp_with(_bl, _bh))
        check("the ceiling is never above 1.0 at bounds "
              "[{}, {}] ({:.4f}) - clipping is a contraction, so "
              "the variance it leaves cannot exceed the original, "
              "and a reading above one would mean the MEASURE is "
              "wrong rather than the engine"
              .format(_bl, _bh, _r[0]["ceiling"]),
              _r[0]["ceiling"] <= 1.0 + 1e-9)

    check("...and the share of rows sitting OUTSIDE the published "
          "bound is reported, because that is what tells a reader "
          "the shortfall is the privacy rule rather than a sampler "
          "fault", rows[0]["rows_outside_bounds"] > 0.0)

    # ---- THE CLI, END TO END ----------------------------------
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        rd = t / "run"
        rd.mkdir()
        s_csv = t / "src.csv"
        src.to_csv(s_csv, index=False)
        gen.to_csv(rd / "generated.csv", index=False)
        (rd / "blueprint.json").write_text(
            json.dumps(bp_with(lo, hi)), encoding="utf-8")
        (rd / "provenance.json").write_text(
            json.dumps({"source": {"rows_read": len(src)}}),
            encoding="utf-8")
        rc, out = run([str(rd), str(s_csv)])
        check("the CLI reports the ceiling and the delivered share "
              "and names which one the rule owns",
              rc == 0 and "ceiling" in out and "delivered" in out
              and "AT THE RULE'S CEILING" in out)

        # A POINT READING OF A NOISY STATISTIC INVITES THE
        # CONCLUSION IT CANNOT SUPPORT. `delivered` on a heavy-
        # tailed column rests on the top 1% of rows and ran 0.421
        # to 0.801 over eight generate seeds on ONE blueprint -
        # wider than most columns' distance to 1.00. The warning
        # must travel with EVERY reading, including a healthy one,
        # because the reading that gets quoted is the flattering
        # one.
        check("...and every reading carries that it is ONE DRAW, "
              "with the measured seed range beside it - this ran "
              "on a column AT the ceiling, so the warning is not "
              "conditional on a bad verdict",
              "ONE DRAW" in out and "0.421" in out
              and "0.801" in out)

        check("...and it names what DOES establish a shortfall, "
              "rather than only warning - a caveat a reader cannot "
              "act on is the fault this project records for a "
              "count with no total beside it",
              "PAIRED arm" in out and "range" in out)

        # NO VALUE MAY BE PUBLISHED. A report about a real extract
        # is itself a release, which is why the deck suppresses
        # sparse bins; this prints sds, ratios and names only.
        vals = {("%.6f" % v) for v in src["x"].head(40)}
        leaked = [v for v in vals if v in out]
        check("...and no source VALUE appears in the output ({} "
              "candidates checked)".format(len(vals)), not leaked)

        # THE SOURCE MUST BE THE ONE THAT PRODUCED THE RUN. A
        # session here was lost to measuring a generated file
        # against an extract that never produced it, and every
        # number looked reasonable.
        small = t / "small.csv"
        src.head(20).to_csv(small, index=False)
        rc2, out2 = run([str(rd), str(small)])
        check("a source whose row count does not match what the "
              "run recorded is REFUSED - a run measured against an "
              "extract that never produced it looks entirely "
              "reasonable",
              rc2 == 2 and "Refusing" in out2)

        rc3, out3 = run(["--help"])
        check("--help exits 0 with usage", rc3 == 0
              and "usage:" in out3)
        rc4, _ = run([])
        check("...and a missing argument refuses", rc4 == 2)

        empty = t / "empty"
        empty.mkdir()
        (empty / "blueprint.json").write_text(
            json.dumps({"columns": {}}), encoding="utf-8")
        gen.to_csv(empty / "generated.csv", index=False)
        rc5, out5 = run([str(empty), str(s_csv)])
        check("a blueprint with no bounded numeric column says so "
              "rather than printing a ceiling over nothing - an "
              "absent measurement is not a clean one",
              rc5 == 3 and "no k price" in out5.lower())

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
