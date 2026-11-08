"""Does modelling a patient's course over time move the gate?

WHAT THIS SETTLES. `dynamics tracked 17/87` sat on the sign-off
page for weeks with "structural - the engine is row-level and
models no patient trajectory" beside it. It is no longer
structural: the latent deviation can carry an AR(1), and this
measures whether doing so moves the criterion, across seeds and
across shapes, against a control arm with the mechanism off.

THE RHO IS SOLVED, AND THE OBJECTIVE TOOK THREE TRIES. The gate
grades `within_lag1` - the pooled lag-1 with the between-patient
share DIVIDED OUT, which is the patient's own course rather than
their level. Solving against the two measurements it is DERIVED
from is not the same thing:

    objective                       solved rho   dynamics
    pooled lag-1                       0.520        2/4
    pooled lag-1 and icc               0.500        2/4
    within_lag1, what the gate reads   0.900        4/4

`(lag1 - icc) / (1 - icc)` amplifies both errors as icc
approaches one, so a solver can drive each measurement to within
0.03 of source and move the graded number the wrong way. The
working objective lands essentially on the measured latent
deviation autocorrelation (0.892), which retroactively says that
measurement was right and the first two objectives were steering
away from it.

THE COUNTER COLUMN WAS BLOCKING THE GATE. `visit_number` IS the
within-patient index - lag-1 0.98 by construction - and the
engine was drawing it as an ordinary number, so one graded column
could never pass however good the trajectory was. It is
reproduced BY POSITION now: exact, and not modelled at all.

Run it:
    python scripts/dyn_sweep.py                  # 3 seeds
    python scripts/dyn_sweep.py --seeds 5
"""

import argparse
import json
import os as _os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# The within-patient bar the criterion uses, restated so a reader
# does not have to go and find it.
DYN_TOL = 0.15


def fixture(kind, seed=0):
    """A longitudinal frame with a KNOWN within-patient course.

    `wide` is the shape the solver was developed against: 600
    patients, 8 visits each, a strong course. `narrow` is the
    shape that caught a fixed rho overshooting - fewer patients,
    a weaker course, and one column with almost none. `flat` has
    one row per patient, so the criterion must stay silent and
    the mechanism must be inert.
    """
    rng = np.random.default_rng(seed)
    if kind == "flat":
        n = 500
        return pd.DataFrame({
            "person_id": ["P{:04d}".format(i) for i in range(n)],
            "systolic": rng.normal(126, 14, n).round(1),
            "severity": rng.normal(3.1, 0.9, n).round(2),
            "diastolic": rng.normal(78, 9, n).round(1)})
    if kind == "narrow":
        pats, vis, rhos, icc = 320, 6, (0.45, 0.70, 0.05), 0.55
    else:
        pats, vis, rhos, icc = 600, 8, (0.80, 0.70, 0.60), 0.60
    # THE COLUMNS MUST BE RELATED, or the sweep cannot see what
    # the trajectory COSTS. The first version gave each column
    # its own independent AR, so discovery found no
    # relationships at all and `close` read 0/0 on every arm -
    # a control column with nothing in it. Each column now
    # carries a SHARED factor as well as its own, so the pairs
    # are discoverable and the criteria that must not move have
    # something to move.
    rows = []
    for p in range(pats):
        anchors = rng.normal(size=len(rhos) + 1)
        dev = rng.normal(size=len(rhos) + 1)
        for v in range(vis):
            for j, rho in enumerate(rhos + (0.75,)):
                dev[j] = (rho * dev[j]
                          + np.sqrt(1.0 - rho * rho)
                          * rng.normal())
            z = (np.sqrt(icc) * anchors
                 + np.sqrt(1.0 - icc) * dev)
            sh = z[len(rhos)]
            rows.append({
                "person_id": "P{:04d}".format(p),
                "visit_number": v + 1,
                "systolic": round(126 + 14 * (
                    0.75 * sh + 0.66 * z[0]), 1),
                "severity": round(3.1 + 0.9 * (
                    0.70 * sh + 0.71 * z[1]), 2),
                "diastolic": round(78 + 9 * (
                    0.65 * sh + 0.76 * z[2]), 1)})
    return pd.DataFrame(rows)


def run(src, out, traj, seed, group="person_id"):
    cmd = [sys.executable,
           _os.path.join(ROOT, "scripts", "latent_challenger.py"),
           "csv", src, "--group-by", group,
           "--seeds", str(seed), "--out", out]
    if traj:
        cmd += ["--trajectory", "ar"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    f = _os.path.join(out, "fidelity.json")
    if r.returncode or not _os.path.exists(f):
        return None, r.stdout + r.stderr
    return json.load(open(f)), r.stdout


def read(fid):
    s = fid.get("summary") or {}
    rel = fid.get("relationships") or {}
    inv = rel.get("inverted")
    return {
        "dyn_ok": s.get("dynamics_ok"),
        "dyn_cmp": s.get("dynamics_compared"),
        "close": rel.get("close"), "pairs": rel.get("compared"),
        "sign": rel.get("sign_kept"),
        "inv": len(inv) if isinstance(inv, list) else inv}


def verdict(off, ar):
    """Is the `ar` arm worse than `off` on this arm?

    A PURE FUNCTION over the two readings, so it can be checked
    against planted inputs. The inline version could only be
    exercised by running the whole sweep, which is the same
    reason `dial_matrix` factored out `classify_move`: a verdict
    nothing can test is a verdict nobody has tested.

    Returns a list of reasons, empty when `ar` is acceptable.
    """
    bad = []
    if off is None or ar is None:
        return ["crash"]
    # A fixture with nothing to grade cannot be worse at
    # anything - the criterion is silent on cross-sectional
    # data, and reading that as a pass or a fail would both be
    # wrong.
    if (off.get("dyn_cmp") or 0) and (
            (ar.get("dyn_ok") or 0) < (off.get("dyn_ok") or 0)):
        bad.append("dynamics")
    # INVERTED is the one absolute criterion, so a new inversion
    # is disqualifying however the dynamics moved.
    if (ar.get("inv") or 0) > (off.get("inv") or 0):
        bad.append("inverted")
    # And a trajectory that buys its course by losing
    # relationships has not improved the output. Graded on the
    # COUNT, because the denominator can differ between arms.
    if (off.get("pairs") or 0) and (
            (ar.get("close") or 0) < (off.get("close") or 0)):
        bad.append("close")
    return bad


def main():
    ap = argparse.ArgumentParser(
        description="Does trajectory modelling move the dynamics "
                    "criterion?")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--kinds", default="wide,narrow,flat")
    a = ap.parse_args()

    print("DYNAMICS AGAINST A CONTROL ARM, PER SEED")
    print("`off` is the shipping row-level engine; `ar` carries "
          "an AR(1) on the latent deviation")
    print("with the rho SOLVED against the source's own "
          "within-patient AR. Bar {:.2f}.".format(DYN_TOL))
    print()
    print("{:<8} {:>5} | {:>9} {:>9} | {:>9} {:>9} | {:>5}"
          .format("fixture", "seed", "dyn off", "dyn ar",
                  "close off", "close ar", "inv"))
    worse = []
    for kind in a.kinds.split(","):
        for seed in range(a.seeds):
            with tempfile.TemporaryDirectory() as td:
                src = _os.path.join(td, "s.csv")
                fixture(kind, seed).to_csv(src, index=False)
                fo, lo = run(src, _os.path.join(td, "off"),
                             False, seed)
                fa, la = run(src, _os.path.join(td, "ar"),
                             True, seed)
            if fo is None or fa is None:
                print("{:<8} {:>5} | RUN FAILED".format(kind, seed))
                worse.append((kind, seed, "crash"))
                continue
            o, v = read(fo), read(fa)
            print("{:<8} {:>5} | {:>9} {:>9} | {:>9} {:>9} | {:>5}"
                  .format(kind, seed,
                          "{}/{}".format(o["dyn_ok"], o["dyn_cmp"]),
                          "{}/{}".format(v["dyn_ok"], v["dyn_cmp"]),
                          "{}/{}".format(o["close"], o["pairs"]),
                          "{}/{}".format(v["close"], v["pairs"]),
                          str(v["inv"])))
            for why in verdict(o, v):
                worse.append((kind, seed, why))
    print()
    if worse:
        print("ARMS WHERE `ar` IS WORSE: {}".format(worse))
        print("A mechanism that helps on average and hurts on a "
              "seed is not ready to be a default -")
        print("this project has twice shipped a 'gain' a control "
              "arm later overturned.")
    else:
        print("`ar` is never worse than `off` on any arm, and the "
              "flat fixture grades zero columns, which")
        print("is the criterion staying silent on data with no "
              "course to lose rather than failing on absence.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
