"""A two-valued CHILD saturates, and only the two-valued DRIVER case
was ever handled.

THE MECHANISM. `_informative_sets` rearranges the drawn sets so a
token indicator lands where its parents say, and `_place` builds the
ordering target as `lam * sys + sqrt(1 - lam^2) * noise` - so that
target correlates with the systematic at exactly `lam`. Rank-matching
a BINARY child onto it can reach only `ceiling(p)` of that, where
`ceiling(p) = phi(z)/sqrt(p(1-p))` is the point-biserial bound for the
token's own share. Every routed indicator therefore landed short by a
factor of how common the token is - 0.78 at a 0.35 share, 0.58 at
0.10 - and because the factor is a smooth function of the share it
looked like ordinary fade rather than a mechanism.

generate.py already carried this solve for the case where the PARENT
is two-valued, with a comment naming the saturation. The CHILD being
two-valued is the mirror image, it is universal for token indicators,
and it was unhandled.

WHAT IS ASSERTED HERE is the mechanism and the guard, not a fidelity
claim: the plain blend must track the PREDICTED ceiling (two
independent routes to one number), the tier must close most of that
gap, it must not lose a sign the control arm keeps, and it must be
OFF by default - because three fixture seeds are not the standard a
default meets in this project, and the real-extract arms decide it.
"""
import json
import subprocess
import sys
import inspect
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from synthkit.generate import generate, _informative_sets  # noqa: E402

PASS = FAIL = 0
GRID = [-2.0, -1.0, 0.0, 1.0, 2.0]
SKILL = 0.30
TARGET = float(np.sqrt(SKILL))


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def ceiling(p):
    """phi(z)/sqrt(p(1-p)), computed WITHOUT scipy so the check does
    not depend on a package the engine does not import here."""
    if not (0.0 < p < 1.0):
        return 1.0
    # inverse normal CDF, Acklam's rational approximation - plenty
    # for a bound quoted to three decimals.
    q = 1.0 - p
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if q < pl:
        r = np.sqrt(-2 * np.log(q))
        z = (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r
              + c[4]) * r + c[5]) / \
            ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1)
    elif q <= ph:
        r = q - 0.5
        s = r * r
        z = (((((a[0] * s + a[1]) * s + a[2]) * s + a[3]) * s
              + a[4]) * s + a[5]) * r / \
            (((((b[0] * s + b[1]) * s + b[2]) * s + b[3]) * s
              + b[4]) * s + 1)
    else:
        r = np.sqrt(-2 * np.log(1 - q))
        z = -(((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r
               + c[4]) * r + c[5]) / \
            ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1)
    phi = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    return float(phi / np.sqrt(p * (1 - p)))


def blueprint(shares):
    """A set column whose every token is a routed CHILD of one
    continuous column, at SEVERAL shares - the ceiling is a function
    of the share, so a single-share fixture could not tell this
    mechanism from a flat shortfall."""
    num = {"kind": "numeric", "level": "visit", "coverage": 1.0,
           "dials": {}, "marginal": {
               "type": "quantiles", "mean": 0.0, "integral": False,
               "q": [0.0, 0.25, 0.5, 0.75, 1.0],
               "v": [-2.5, -0.7, 0.0, 0.7, 2.5]}}
    setc = {"kind": "set", "level": "visit", "coverage": 1.0,
            "dials": {}, "marginal": {
                "type": "list", "separator": ";",
                "tokens": [{"value": "t%d" % i, "p": p}
                           for i, p in enumerate(shares)],
                "set_size": {"v": [1, 2, 3], "p": [0.4, 0.4, 0.2]}}}
    ind = {"kind": "numeric", "level": "visit", "coverage": 1.0,
           "dials": {}, "derived_from": {"column": "sets"},
           "marginal": {"type": "quantiles", "mean": 0.3,
                        "integral": False, "q": [0.0, 1.0],
                        "v": [0.0, 1.0]}}
    cols = {"sev": num, "sets": setc}
    rels = []
    for i, p in enumerate(shares):
        cols["sets__has__t%d" % i] = dict(ind)
        rels.append({
            "child": "sets__has__t%d" % i, "parents": ["sev"],
            "dials": {"strength": None},
            "evidence": {"skill_out_of_sample": SKILL,
                         "importance": {"sev": 1.0},
                         "effect": {"sev": {
                             "shape": "monotone", "grid": list(GRID),
                             "response": [p + 0.25 * g
                                          for g in GRID]}}}})
    return {"columns": cols, "relationships": rels,
            "patients": {"count": 1200,
                         "visits": {"q": [0.0, 0.5, 1.0],
                                    "v": [3.0, 4.0, 5.0],
                                    "mean": 4.0}, "dials": {}}}


SHARES = [0.12, 0.22, 0.35, 0.50]


def arm(flag, seed, shares=None):
    shares = shares or SHARES
    rep = {}
    g = generate(blueprint(shares), n_patients=1200, seed=seed,
                 report=rep, solve_token_targets=flag)
    s = g["sets"].fillna("")
    out = []
    for i in range(len(shares)):
        x = s.str.contains("t%d" % i).astype(float)
        sd = float(x.std())
        out.append((float(x.mean()), 0.0 if sd == 0 else
                    float(np.corrcoef(
                        x, g["sev"].astype(float))[0, 1])))
    return out, rep


def main():
    # ---- the fixture must ROUTE, or nothing below is exercised ----
    off0, rep_off = arm(False, 3)
    on0, rep_on = arm(True, 3)
    informed = [e for e in (rep_on.get("sets_informed") or [])
                if e.get("column") == "sets"]
    check("the fixture actually ROUTES its token children - a set "
          "column with no routed children is BIT-IDENTICAL on both "
          "arms, and a fixture that does not reach the tier would "
          "pass every check below without exercising anything",
          bool(informed)
          and len(informed[0].get("children") or []) == len(SHARES))

    # ---- the tier is OFF by default -------------------------------
    d = inspect.signature(generate).parameters[
        "solve_token_targets"].default
    check("the tier is OFF by default - three FIXTURE seeds are not "
          "the standard a default meets here; the co-occurrence "
          "default was earned on three seeds of the REAL EXTRACT, "
          "and this file records a gain a real control arm later "
          "overturned", d is False)

    check("...and `_informative_sets` defaults it ON, so the ONE "
          "place that decides is `generate`'s signature and the CLI "
          "flag, rather than two halves each carrying a default",
          inspect.signature(_informative_sets).parameters[
              "solve_token_targets"].default is True)

    # ---- THE MECHANISM: the plain blend tracks the PREDICTED
    # ceiling. This is the load-bearing check in the suite: a
    # measured shortfall is a tuning observation, while a measured
    # shortfall that matches a bound computed independently is a
    # mechanism.
    # The bound applies to an UNCONSTRAINED placement, so it
    # predicts the FIRST-placed child. Every later child is
    # rank-matched inside the groups the earlier ones fixed, and that
    # starvation can only subtract - which is why the prediction and
    # the starvation are two checks rather than one average.
    share0, got0 = off0[0]
    pred0 = ceiling(share0) * TARGET
    check("WITHOUT the tier the FIRST-placed token lands at exactly "
          "ceiling(share) x target - the point-biserial bound, "
          "computed here independently of the engine: predicted "
          "{:+.4f} at share {:.3f}, measured {:+.4f}. Two routes to "
          "one number is what makes this a mechanism rather than an "
          "observed fade".format(pred0, share0, got0),
          abs(got0 - pred0) < 0.03)

    check("...and every LATER token lands at or below its own bound, "
          "because each is rank-matched inside the groups the "
          "earlier ones fixed and that can only subtract: {}"
          .format(" ".join("%+.3f/%.3f" % (g, ceiling(sh) * TARGET)
                           for sh, g in off0[1:])),
          all(g <= ceiling(sh) * TARGET + 0.03
              for sh, g in off0[1:]))

    check("...and that starvation is REAL on this fixture rather "
          "than assumed - the last-placed token reaches only {:.2f} "
          "of the bound its own share allows"
          .format(off0[-1][1] / (ceiling(off0[-1][0]) * TARGET)),
          off0[-1][1] < 0.6 * ceiling(off0[-1][0]) * TARGET)

    check("...and the ceiling is a real function of the share rather "
          "than a constant, so a single-share fixture could not tell "
          "this from a flat fade: predicted {:.3f} at the sparsest "
          "token against {:.3f} at the broadest"
          .format(ceiling(off0[0][0]), ceiling(off0[-1][0])),
          abs(ceiling(off0[0][0]) - ceiling(off0[-1][0])) > 0.05)

    # ---- the tier closes most of that gap, on every seed ----------
    rows = []
    for sd in (3, 4, 5):
        o, _ = arm(False, sd)
        n, _ = arm(True, sd)
        rows.append((
            float(np.mean([abs(g - TARGET) for _, g in o])),
            float(np.mean([abs(g - TARGET) for _, g in n])),
            sum(1 for _, g in o if abs(g - TARGET) <= 0.2),
            sum(1 for _, g in n if abs(g - TARGET) <= 0.2),
            sum(1 for _, g in o if g > 0),
            sum(1 for _, g in n if g > 0)))
    check("the tier lowers the mean absolute miss against the target "
          "on EVERY seed ({} -> {}) - this project reports the range "
          "and never a point"
          .format(" ".join("%.3f" % r[0] for r in rows),
                  " ".join("%.3f" % r[1] for r in rows)),
          all(r[1] < r[0] - 0.05 for r in rows))

    check("...and raises the count landing within the gate's own 0.2 "
          "band on every seed ({} -> {} of {})"
          .format(" ".join(str(r[2]) for r in rows),
                  " ".join(str(r[3]) for r in rows), len(SHARES)),
          all(r[3] > r[2] for r in rows))

    # ---- AND IT MUST NOT BUY THAT WITH AN INVERSION ---------------
    # The solve ALONE lost a sign on the fourth child: tightening the
    # earlier children fixes the groups and starves the last one.
    # Placing the LEAST-FREE token first is what removed it, and that
    # ordering is part of this tier rather than a separate tidy-up.
    check("...and loses NO sign the control arm keeps, on every seed "
          "({} -> {} of {}) - a pair counted close is worth less "
          "than a pair counted INVERTED, which this project calls an "
          "absolute stop, so a count that trades one inversion for "
          "one close pair is read the wrong way round"
          .format(" ".join(str(r[4]) for r in rows),
                  " ".join(str(r[5]) for r in rows), len(SHARES)),
          all(r[5] >= r[4] for r in rows))

    # ---- the run SAYS what it asked for and what it got -----------
    asked = (informed[0].get("asked") or {})
    check("the run REPORTS each child's asked-for correlation beside "
          "what it achieved in place, so a tier that silently stopped "
          "landing is visible from the run's own artifacts rather "
          "than from a fixture - a fix that cannot prove it executed "
          "has not been deployed, which cost this project two round "
          "trips to the data machine",
          len(asked) == len(SHARES)
          and all(isinstance(v, (list, tuple)) and len(v) == 3
                  for v in asked.values()))

    check("...and the control arm reports NO such record, because it "
          "asks for nothing - absent is not the same fact as asked "
          "and missed",
          not any((e.get("asked") or {})
                  for e in (rep_off.get("sets_informed") or [])))

    # ---- inert where its mechanism is absent ----------------------
    bp_none = blueprint(SHARES)
    bp_none["relationships"] = []
    g1 = generate(bp_none, n_patients=300, seed=7,
                  solve_token_targets=False)
    g2 = generate(bp_none, n_patients=300, seed=7,
                  solve_token_targets=True)
    check("a blueprint with NO routed children generates "
          "BYTE-IDENTICALLY on both arms, so the tier cannot act "
          "where its mechanism is absent", g1.equals(g2))

    # ---- and the ask can exceed what the share allows -------------
    # A token on a twentieth of rows cannot correlate above ~0.45
    # with anything. The tier must not pretend otherwise.
    tiny, _ = arm(True, 3, shares=[0.05, 0.40])
    sh, got = tiny[0]
    check("a token too sparse to carry the ask falls short and that "
          "shortfall is REAL rather than a tuning miss: share {:.3f} "
          "bounds any arrangement at {:.3f} against an ask of {:.3f}, "
          "and the achieved {:+.3f} does not exceed the bound"
          .format(sh, ceiling(sh), TARGET, got),
          got <= ceiling(sh) + 0.05)

    # ---- AND THROUGH THE REAL CLI, ON DISK --------------------
    #
    # Everything above calls `generate` directly. The CLI is the
    # path a run takes, and this project has twice found a fault
    # that only the reader could produce - so the flag is exercised
    # end to end. The fixture makes each token a NEARLY
    # DETERMINISTIC threshold on a continuous column, because that
    # is what makes discovery orient the INDICATOR as the child and
    # route it: a noisy threshold orients the other way and the tier
    # is never reached.
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        rng = np.random.default_rng(4)
        n = 3000
        sev = rng.normal(0, 1, n)
        load = rng.normal(0, 1, n)
        rows = []
        for i in range(n):
            tk = []
            for name, sh, drv in (("t_rare", 0.12, sev),
                                  ("t_mid", 0.24, sev),
                                  ("t_wide", 0.38, load),
                                  ("t_broad", 0.55, load)):
                if drv[i] + rng.normal(0, 0.18) > np.quantile(
                        drv, 1 - sh):
                    tk.append(name)
            rows.append(";".join(sorted(set(tk))) or "t_broad")
        src = t / "src.csv"
        pd.DataFrame({
            "person_id": ["P%04d" % (i // 3) for i in range(n)],
            "severity": np.round(sev, 4),
            "load": np.round(load, 4),
            "drugs": rows}).to_csv(src, index=False)

        res = {}
        for name, extra in (("off", []), ("on",
                                          ["--solve-token-targets"])):
            r = subprocess.run(
                [sys.executable,
                 str(ROOT / "scripts" / "run_discovery.py"),
                 "--src", str(src), "--out", str(t / name),
                 "--group-by", "person_id", "--generate",
                 "--seed", "5"] + extra,
                capture_output=True, text=True, cwd=str(ROOT))
            res[name] = r.returncode
        check("both CLI arms complete", res["off"] == 0
              and res["on"] == 0)

        fo = json.loads((t / "off" / "fidelity.json")
                        .read_text(encoding="utf-8"))
        fn = json.loads((t / "on" / "fidelity.json")
                        .read_text(encoding="utf-8"))

        # THE RUN MUST BE ABLE TO PROVE THE TIER EXECUTED. `generate`
        # has recorded this since the routing existed and the CLI
        # consumed exactly one field of its report - so reading the
        # key off fidelity.json returned None and read as "the
        # routing never fired". It had fired, in both arms.
        si_o = fo.get("sets_informed") or []
        si_n = fn.get("sets_informed") or []
        check("the CLI WRITES which set columns were informed, so a "
              "real run can prove the routing executed rather than "
              "leaving a reader to infer it from a field the CLI "
              "never wrote",
              bool(si_o) and bool(si_n)
              and len(si_n[0].get("children") or []) >= 4)

        check("...and the asked-for correlations are recorded on the "
              "arm that asks and ABSENT on the control, because "
              "asking for nothing is a different fact from asking "
              "and missing",
              bool(si_n[0].get("asked"))
              and not (si_o[0].get("asked") or {}))

        # THE FLAG MUST REACH GENERATION. A flag read but not used is
        # this project's own named failure mode, and it has cost two
        # round trips to the machine that holds the extract.
        go = pd.read_csv(t / "off" / "generated.csv", dtype=str,
                         keep_default_na=False)
        gn = pd.read_csv(t / "on" / "generated.csv", dtype=str,
                         keep_default_na=False)
        moved = int(sum(1 for a, b in zip(go["drugs"], gn["drugs"])
                        if a != b))
        check("the flag CHANGES the written file - {} of {} rows "
              "carry a different set".format(moved, len(go)),
              moved > 0)

        # AND THE CHANGE IS A PERMUTATION, which is what makes the
        # whole tier safe: the sizes, the token shares and the
        # combinations are exactly what the draw produced, so no
        # published marginal can drift by construction.
        check("...and the set column holds the IDENTICAL MULTISET on "
              "both arms, so the tier only rearranges which row gets "
              "which drawn set and no published share can move",
              sorted(go["drugs"]) == sorted(gn["drugs"]))

        # PROPAGATION IS NOT A LEAK, and which is which is read
        # from the blueprint's own graph rather than assumed. A
        # column that is a CHILD of one of this set's indicators is
        # drawn FROM it, so rearranging the sets moves that column
        # too - that is the relationship working. A column the graph
        # does not connect to the set has no such excuse. The first
        # version of this check asserted that every other column was
        # byte-identical and went red on `severity`, which the
        # blueprint declares a child of two token indicators.
        bpn = json.loads((t / "on" / "blueprint.json")
                         .read_text(encoding="utf-8"))
        downstream = set()
        for _ in range(4):          # follow the chain, not one hop
            for r in (bpn.get("relationships") or []):
                ps = r.get("parents") or []
                if any(pp == "drugs" or pp.startswith("drugs__")
                       or pp in downstream for pp in ps):
                    downstream.add(r.get("child"))
        moved_cols = [c for c in go.columns
                      if c != "drugs"
                      and not bool((go[c] == gn[c]).all())]
        unexplained = [c for c in moved_cols if c not in downstream]
        check("...and every column that moved is one the blueprint "
              "declares DOWNSTREAM of this set - moved {}, of which "
              "unexplained {}. A column drawn from a token indicator "
              "must move when the sets are rearranged; one the graph "
              "does not connect has no such excuse"
              .format(moved_cols or "none", unexplained or "none"),
              not unexplained)

        check("...and NO column's multiset changes anywhere in the "
              "file, downstream ones included, so nothing the tier "
              "touches or reaches can drift a published marginal",
              all(sorted(go[c]) == sorted(gn[c])
                  for c in go.columns))

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
