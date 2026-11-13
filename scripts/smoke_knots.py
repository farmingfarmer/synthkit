"""The inverse CDF is interpolated LINEARLY between published knots.

That assumes uniform density in between, and this project already
records the assumption carrying the whole error across a top
segment. It does the same wherever mass is concentrated NEAR a
knot: on an spo2-shaped column every published quantile lands
exactly and the band below the median still comes out 0.10 light,
because the mass that really sits just under a knot is spread
evenly across the segment instead.

`EXTRA_KNOTS` adds evenly spaced interior quantiles. It is a DIAL
rather than a default because every added knot publishes one more
quantile VALUE of the real column.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from synthkit import blueprint as B                 # noqa: E402
import synthkit.blueprint as BP                     # noqa: E402
from synthkit.discover import discover              # noqa: E402
from synthkit.generate import generate              # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def frame(seed=11, n_pat=500, n_vis=8):
    """Bounded above, with a pile at the ceiling - the spo2 shape."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_pat):
        lvl = rng.normal(0, 1)
        for v in range(n_vis):
            drv = lvl + rng.normal(0, 0.6)
            if rng.random() < 0.34 + 0.08 * drv:
                s = 100
            else:
                s = int(np.clip(
                    round(98.6 - abs(rng.gamma(1.4, 2.2)) - 0.4 * drv),
                    70, 99))
            rows.append({"person_id": "P%04d" % p,
                         "visit_number": v + 1,
                         "drv": round(float(drv), 4), "spo2": s})
    return pd.DataFrame(rows)


def bands(x):
    e = []
    for q in list(np.quantile(x, [0.1, 0.25, 0.5, 0.75, 0.9])) + \
            [float(x.max())]:
        q = float(q)
        if not e or q > e[-1]:
            e.append(q)
    return e


def shares(x, e):
    o, lo = [], -np.inf
    for q in e:
        o.append(float(np.mean((x > lo) & (x <= q))))
        lo = q
    o.append(float(np.mean(x > lo)))
    return o


def main():
    df = frame()
    src = df["spo2"].to_numpy(float)
    cat = discover(df, group_by="person_id", targets=["spo2"])
    e = bands(src)
    ss = shares(src, e)

    check("the dial is OFF by default - every added knot publishes "
          "one more quantile value of the real column, so this is "
          "a disclosure choice and not ours to make silently",
          BP.EXTRA_KNOTS == 0)

    got = {}
    for n in (0, 9):
        _p = BP.EXTRA_KNOTS
        try:
            BP.EXTRA_KNOTS = n
            bp = B.build(df, cat, group_by="person_id")
            g = generate(bp, n_patients=500, seed=3)
            g = g[0] if isinstance(g, tuple) else g
        finally:
            BP.EXTRA_KNOTS = _p
        v = pd.to_numeric(g["spo2"], errors="coerce").dropna()
        got[n] = {"gap": max(abs(a - b) for a, b in
                             zip(ss, shares(v.to_numpy(float), e))),
                  "knots": len(bp["columns"]["spo2"]["marginal"]["q"]),
                  "ceil": float((v == src.max()).mean()),
                  "bp": bp}

    check("the published grid actually GAINS knots, so the dial is "
          "not read and ignored",
          got[9]["knots"] > got[0]["knots"])
    check("...and the shipping grid still carries its own eleven, "
          "so the extra ones are added rather than replacing them",
          got[0]["knots"] == 11)
    check("mass concentrated near a knot is misplaced by the "
          "linear interpolation - the fault, reproduced before it "
          "is fixed",
          got[0]["gap"] > 0.05)
    check("...and extra knots close most of it, measured against "
          "the SOURCE rather than against the grid",
          got[9]["gap"] < 0.4 * got[0]["gap"])
    check("...and a point mass at the ceiling comes closer for "
          "free, because enough knots represent a spike as "
          "repeated values",
          abs(got[9]["ceil"] - float((src == src.max()).mean()))
          < abs(got[0]["ceil"] - float((src == src.max()).mean())))

    # THE BOUNDS ARE THE PRIVACY PROMISE AND MUST NOT MOVE. Extra
    # knots are INTERIOR; the outermost pair is still the mean of
    # the k most extreme patients.
    m0 = got[0]["bp"]["columns"]["spo2"]["marginal"]
    m9 = got[9]["bp"]["columns"]["spo2"]["marginal"]
    check("the published BOUNDS are unchanged - the extra knots "
          "are interior, and the outermost pair is still the mean "
          "of the k most extreme patients",
          abs(float(m9["v"][0]) - float(m0["v"][0])) < 1e-9
          and abs(float(m9["v"][-1]) - float(m0["v"][-1])) < 1e-9)
    check("...and the k the bounds were screened at is unchanged, "
          "so this is a resolution dial and not a privacy one",
          m9.get("bounds_are_k_anonymous")
          == m0.get("bounds_are_k_anonymous"))
    # THE CLAMP NEEDS A FIXTURE THAT REACHES IT. On the spo2
    # shape no raw quantile ever exceeds the k-anonymous bound, so
    # removing the clamp broke nothing and the check was
    # decoration. The shape that binds is the one the k rule
    # exists for: a FEW patients seen MANY times, so a row
    # quantile sits far above the mean of the k most extreme
    # PATIENTS.
    heavy = []
    for p in range(495):
        for v in range(4):
            heavy.append({"person_id": "Q%04d" % p,
                          "visit_number": v + 1, "hv": 1})
    for p in range(5):
        for v in range(200):
            heavy.append({"person_id": "H%04d" % p,
                          "visit_number": v + 1, "hv": 1000})
    hdf = pd.DataFrame(heavy)
    hcat = discover(hdf, group_by="person_id", targets=["hv"])
    _p = BP.EXTRA_KNOTS
    try:
        BP.EXTRA_KNOTS = 9
        hbp = B.build(hdf, hcat, group_by="person_id")
    finally:
        BP.EXTRA_KNOTS = _p
    hm = hbp["columns"]["hv"]["marginal"]
    _raw = float(np.quantile(hdf["hv"].to_numpy(float), 0.99))
    check("the heavy fixture REACHES the clamp - a raw interior "
          "quantile sits above the k-anonymous high bound, which "
          "the spo2 shape never does",
          _raw > float(hm["v"][-1]) + 1e-9)
    check("...and every interior knot is still held INSIDE the "
          "bounds, which is what stops a published curve reaching "
          "past what k patients support",
          all(float(hm["v"][0]) - 1e-9 <= float(x)
              <= float(hm["v"][-1]) + 1e-9 for x in hm["v"]))
    check("...and the quantile list stays sorted and unique, so a "
          "duplicate knot cannot make a zero-width segment",
          all(float(a) < float(b) for a, b in
              zip(m9["q"], m9["q"][1:])))

    from synthkit.pipeline import build_parser as _bpk
    check("the CLI exposes the dial, so a real arm can be run "
          "without editing the module",
          any("--extra-knots" in o.option_strings
              for o in _bpk()._actions))
    check("...and its help states the DISCLOSURE cost, because a "
          "resolution gain that publishes more of the real column "
          "is not a free one",
          any("--extra-knots" in o.option_strings
              and "publishes one more quantile value"
              in (o.help or "") for o in _bpk()._actions))
    check("...and states that it is NOT monotone, so nobody "
          "cranks it expecting more",
          any("--extra-knots" in o.option_strings
              and "NOT monotone" in (o.help or "")
              for o in _bpk()._actions))

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
