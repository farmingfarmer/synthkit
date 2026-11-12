"""A count whose MODE is its own minimum gets nothing from zero_share.

`zero_share` exists because a continuous model cannot reproduce a
POINT MASS: drawn as a relationship child the value arrives as a
curve prediction plus roughly symmetric noise, and rounding cannot
land a spike. It only knows about ZERO.

`procedure_quantity` on the real extract has NO zeros at all - its
mass sits on the value ONE, 52.8% of rows - so nothing fired. A fix
that only knows about zero cannot see a count whose mode is one.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from synthkit import blueprint as B                 # noqa: E402
from synthkit import generate as G                  # noqa: E402
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


def frame(seed=7, n_pat=500, n_vis=8, floor=1):
    """The real column's shape: no zeros, a point mass at the
    minimum, a body, and a heavy tail."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_pat):
        lvl = rng.normal(0, 1)
        for v in range(n_vis):
            drv = lvl + rng.normal(0, 0.6)
            u = rng.random()
            if u < 0.53 - 0.12 * drv:
                q = floor
            elif u < 0.95:
                q = int(floor + 1 + rng.gamma(1.6, 3.0))
            else:
                q = int(floor + 24 + rng.gamma(1.2, 60.0))
            rows.append({"person_id": "P%04d" % p,
                         "visit_number": v + 1,
                         "drv": round(float(drv), 4), "pq": q})
    return pd.DataFrame(rows)


def bands(src):
    e = []
    for q in list(np.quantile(src, [0.5, 0.75, 0.9, 0.99])) + \
            [float(src.max())]:
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
    src = df["pq"].to_numpy(float)
    cat = discover(df, group_by="person_id", targets=["pq"])
    bp = B.build(df, cat, group_by="person_id")
    m = bp["columns"]["pq"]["marginal"]

    check("the default is OFF - this changes which values a "
          "shipped column carries and the real arm decides it",
          G.FLOOR_SHARE is False)

    check("the blueprint PUBLISHES the mass at a non-zero floor, "
          "so a reader can see it even where nothing consumes it",
          m.get("floor_share") is not None)
    # GUARDED: a mutation that stops publishing the field made
    # this check RAISE on the missing key, so the suite CRASHED
    # after one red instead of reporting two - which this project
    # records as indistinguishable from a check that never ran.
    check("...matching the source's own share at that value, "
          "because a published aggregate that is not the measured "
          "one is worse than none",
          m.get("floor_share") is not None
          and abs(float(m["floor_share"])
                  - float((src == src.min()).mean())) < 0.01)
    check("...and `zero_share` is ABSENT here, which is the gap: a "
          "column with no zeros at all gets nothing from it",
          m.get("zero_share") is None)

    e = bands(src)
    ss = shares(src, e)
    out = {}
    for flag in (False, True):
        _p = G.FLOOR_SHARE
        try:
            G.FLOOR_SHARE = flag
            g = generate(bp, n_patients=500, seed=3)
            g = g[0] if isinstance(g, tuple) else g
        finally:
            G.FLOOR_SHARE = _p
        v = pd.to_numeric(g["pq"], errors="coerce").dropna().to_numpy(float)
        out[flag] = {"mode": float((v == src.min()).mean()),
                     "gap": max(abs(a - b) for a, b in
                                zip(ss, shares(v, e))),
                     "mean": float(v.mean()),
                     "min": float(v.min())}

    check("without it the point mass is WRONG - the engine puts "
          "more rows on the mode than the source does, which "
          "rounding cannot fix",
          abs(out[False]["mode"] - float((src == src.min()).mean()))
          > 0.015)
    check("...and with it the mass is EXACT, because the rows are "
          "counted out rather than hoped for",
          abs(out[True]["mode"] - float((src == src.min()).mean()))
          < 0.002)
    check("...which improves the whole column against the SOURCE, "
          "the quantity a recipient actually holds",
          out[True]["gap"] < out[False]["gap"])
    check("...without moving the column's centre, since the mass "
          "is PLACED by rank rather than drawn again",
          abs(out[True]["mean"] - out[False]["mean"])
          < 0.08 * out[False]["mean"])
    check("...and nothing lands below the published floor",
          out[True]["min"] >= src.min() - 1e-9)

    # A COLUMN WITH ZEROS IS zero_share's, NOT THIS. Two fields
    # aimed at one column is how the halves come to disagree.
    dfz = frame(floor=0)
    catz = discover(dfz, group_by="person_id", targets=["pq"])
    bpz = B.build(dfz, catz, group_by="person_id")
    mz = bpz["columns"]["pq"]["marginal"]
    check("a column whose floor IS zero publishes `zero_share` and "
          "NOT `floor_share` - one mass, one field, so the two "
          "cannot both claim the same rows",
          mz.get("zero_share") is not None
          and mz.get("floor_share") is None)

    # INERT WHERE THE FIELD IS ABSENT, which is every blueprint
    # written before this existed.
    import hashlib
    bp2 = B.build(df, cat, group_by="person_id")
    # EVERY column, not just the one under test. Stripping it from
    # `pq` alone left `visit_number` carrying a floor_share of its
    # own - a within-patient counter has minimum 1 on every patient
    # - so the flag still changed the output and the inertness
    # check went red against correct code. The flag reaches EVERY
    # integral column whose minimum is not zero, which is worth
    # knowing on its own and is why it waits for a real arm.
    _stripped = 0
    for _spec in (bp2.get("columns") or {}).values():
        _mm = (_spec or {}).get("marginal") or {}
        if _mm.pop("floor_share", None) is not None:
            _stripped += 1
    check("more than one column publishes a non-zero floor mass, "
          "so this flag is not a one-column change",
          _stripped >= 2)

    def dig(flag):
        _p = G.FLOOR_SHARE
        try:
            G.FLOOR_SHARE = flag
            g = generate(bp2, n_patients=500, seed=3)
            g = g[0] if isinstance(g, tuple) else g
        finally:
            G.FLOOR_SHARE = _p
        return hashlib.md5(g.to_csv(index=False).encode()).hexdigest()

    check("a blueprint PREDATING the field generates identically "
          "with the flag on - an old artifact must still load, and "
          "absent is not the same fact as zero",
          dig(False) == dig(True))

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
