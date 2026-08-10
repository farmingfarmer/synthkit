"""Smoke: the effect curve says what a parent DOES, not just that it
matters.

Four shapes are planted whose names a reader would use - rising,
U-shaped, threshold, saturating - and the extractor has to come back
with those names. A curve that merely exists proves nothing; the test
is whether the description would let somebody who has never seen the
data say something true about it.

The turning points are planted at known values too, so a threshold
reported in the wrong place fails rather than passes for being
approximately shaped right.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit.discover import discover                # noqa: E402
from synthkit.shapes import describe, effect_curve    # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def build(n_pat=300, n_vis=8, seed=9):
    r = np.random.RandomState(seed)
    rows = []
    for p in range(n_pat):
        for _v in range(n_vis):
            xr = r.uniform(0, 10)         # rising
            xu = r.uniform(-3, 3)         # U
            xt = r.uniform(0, 100)        # threshold at 60
            xs = r.uniform(0, 20)         # saturating
            xf = r.uniform(0, 5)          # flat / unrelated
            xa, xb = r.uniform(0, 1), r.uniform(0, 1)
            rows.append({
                "person_id": "P{:04d}".format(p),
                # a pure interaction: no main effect on either side
                "x_a": round(xa, 4),
                "x_b": round(xb, 4),
                "y_xor": round((1.0 if (xa > 0.5) != (xb > 0.5)
                                else 0.0) + r.normal(0, 0.06), 4),
                "x_rise": round(xr, 4),
                "y_rise": round(2.0 * xr + r.normal(0, 0.6), 4),
                "x_u": round(xu, 4),
                "y_u": round(xu * xu + r.normal(0, 0.4), 4),
                "x_thr": round(xt, 4),
                "y_thr": round((10.0 if xt > 60 else 0.0)
                               + r.normal(0, 0.5), 4),
                "x_sat": round(xs, 4),
                "y_sat": round(10.0 * (1.0 - np.exp(-xs / 1.5))
                               + r.normal(0, 0.25), 4),
                "x_flat": round(xf, 4),
            })
    return pd.DataFrame(rows)


def curve_for(cat, child, parent):
    for cl in cat["claims"]:
        if cl["child"] != child:
            continue
        for p in cl["predictors"]:
            if p["column"] == parent:
                return p.get("effect")
    return None


def main():
    df = build()
    cat = discover(df, group_by="person_id", seed=4)

    rise = curve_for(cat, "y_rise", "x_rise")
    check("a rising relationship is found AND carries a curve, not "
          "just an importance number",
          rise is not None and len(rise["grid"]) > 5)
    check("...and is NAMED as increasing",
          rise and rise["shape"] == "increasing" and rise["monotone"])
    check("...and the sentence names both columns, so it reads "
          "without the schema beside it",
          rise and "x_rise" in rise["description"]
          and "y_rise" in rise["description"])

    u = curve_for(cat, "y_u", "x_u")
    check("a U-shape is named U-shaped rather than reported as a weak "
          "linear effect - a correlation would call this nothing",
          u is not None and u["shape"] == "u-shaped")
    check("...and its lowest point is located near the planted "
          "minimum of 0",
          u is not None and abs(u["turning_point"]) < 0.8)

    thr = curve_for(cat, "y_thr", "x_thr")
    check("a threshold is named a threshold - most of the change "
          "happening at one point is a different fact from a slope",
          thr is not None and thr["shape"] == "threshold")
    check("...and it is located near the planted cut of 60, so a "
          "wrongly placed threshold FAILS rather than passing for "
          "being roughly the right shape",
          thr is not None and 50 <= thr["turning_point"] <= 70)

    sat = curve_for(cat, "y_sat", "x_sat")
    check("a saturating curve is distinguished from a plain rising "
          "one, because 'more buys little past here' is the "
          "actionable part",
          sat is not None and sat["shape"] == "saturating")

    check("a column unrelated to everything contributes no curve "
          "anywhere",
          all(p["column"] != "x_flat"
              for cl in cat["claims"] for p in cl["predictors"]))

    # the grid must not leave the observed range
    for name, cur in (("x_rise", rise), ("x_thr", thr)):
        if not cur:
            continue
        obs = df[name].astype(float)
        check("the {} grid stays inside the observed range - a curve "
              "must never claim territory the data did not "
              "cover".format(name),
              min(cur["grid"]) >= obs.min()
              and max(cur["grid"]) <= obs.max())

    # effect size is reported and ordered sensibly
    check("the effect size is reported, so two relationships can be "
          "compared by how much they actually move the child",
          rise and thr and rise["effect_size"] > 0
          and thr["effect_size"] > 0)

    # a flat response must be CALLED flat, not given a shape
    flatc = {"grid": [0.0, 1.0, 2.0, 3.0],
             "response": [5.0, 5.001, 4.999, 5.0],
             "grid_kind": "numeric", "class_of_interest": None}
    d = describe(flatc, "p", "c", child_spread=2.0)
    check("a response that barely moves is called FLAT rather than "
          "given a direction - noise with a shape name attached is "
          "the failure this whole layer exists to prevent",
          d["shape"] == "flat")

    # ---- a relationship carried by ABSENCE ----------------------
    # The real extract produced `is_last_visit <- days_to_next_visit`
    # explaining 100% of it with a FLAT curve - an internal
    # contradiction, because days_to_next_visit is missing on exactly
    # the last visit. The value never mattered; being measured did.
    pres = {"grid": [1.0, 2.0, 3.0, 4.0],
            "response": [5.0, 5.01, 4.99, 5.0],
            "grid_kind": "numeric", "class_of_interest": None,
            "response_when_missing": 30.0}
    dp = describe(pres, "p", "c", child_spread=6.0)
    check("a flat curve whose response JUMPS when the parent is "
          "absent is named presence-only, not flat - the value does "
          "nothing and being measured does everything",
          dp["shape"] == "presence-only"
          and "whether" in dp["description"])
    pres2 = dict(pres, response_when_missing=5.0)
    check("...and a flat curve that does NOT jump on absence is still "
          "just flat, so the new reading is not handed out for free",
          describe(pres2, "p", "c", child_spread=6.0)["shape"]
          == "flat")

    # ---- the interaction, which no single curve can show ----
    # A one-parent curve for a pure exclusive-or is correctly flat,
    # and two flat curves under a relationship scoring 0.95 is a true
    # and useless answer. Reporting the joint surface is the only way
    # this system explains the pattern it was built to find.
    xor = None
    for cl in cat["claims"]:
        if cl["child"] == "y_xor":
            xor = cl
    check("the interaction is FOUND at all", xor is not None
          and xor["skill"] > 0.5)
    it = (xor or {}).get("interaction")
    check("...and carries a JOINT surface, because neither "
          "one-parent curve can express it",
          it is not None and len(it.get("response") or []) >= 3)
    check("...named as OPPOSING - high when exactly one factor is "
          "high, which is what an exclusive-or is",
          it is not None and it["pattern"] == "opposing")
    check("...with the corners a reader can check by hand: the two "
          "disagreeing corners high, the two agreeing corners low",
          it is not None
          and min(it["corners"]["low_high"],
                  it["corners"]["high_low"])
          > max(it["corners"]["low_low"],
                it["corners"]["high_high"]) + 0.3)
    check("...and the surface is reported BECAUSE it travels further "
          "than either parent alone - a measurement, not a guess "
          "about which shape name survived the noise",
          it is not None and it["beyond_single"] > 3.0)

    lin = [c for c in cat["claims"] if c["child"] == "y_rise"]
    check("a plainly ADDITIVE relationship reports no interaction, so "
          "the surface is not offered for everything with two parents",
          lin and lin[0].get("interaction") is None)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
