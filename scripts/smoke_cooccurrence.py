"""Which tokens travel together, and whether the draw carries it.

WHY THIS EXISTS. On the real extract about 110 of the 135 pairs the
gate relates are SET-TOKEN pairs - roughly four fifths of everything
graded. `_token_weights` solves weights so each token's MARGINAL
share measures back and says nothing about which tokens co-occur, so
within a row they were drawn independently. The draw now reweights
each pick by the published structure of what the row already holds,
including tokens drawn in EARLIER set columns, which is the shape
auto-regressive tabular generators use over discretised columns.

THIS SUITE RUNS THE REAL CLI, AND THAT IS THE LOAD-BEARING PART.
An earlier version built its blueprint with
`discover(df, targets=["x"])`, which EXCLUDES the relationships the
pipeline discovers between the set-size and token-indicator
scaffolding columns - and the engine already draws those through its
ordinary relationship machinery. On that restricted blueprint the
baseline read -0.003 and this feature looked like it took a planted
implication from absent to +0.419. Measured through the real path the
baseline is **+0.497** and the feature adds essentially nothing on a
fixture: the gain this feature has is on the REAL EXTRACT, where it
moved close 102/125 to 106/125, and a fixture cannot show it.

So what is asserted here is MECHANISM, not fidelity: the structure is
detected and published correctly, the k floor applies to the JOINT,
the flag genuinely changes the output, the marginals still land, and
it is inert where its mechanism is absent. The fidelity claim lives
where it was measured.
"""
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from synthkit import sets as S                        # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def fixture(n=4000, seed=3):
    """A drug that IMPLIES its route - the extract's own shape."""
    rng = np.random.default_rng(seed)
    drugs, routes = [], []
    for _ in range(n):
        d, r = [], []
        if rng.random() < 0.30:
            d.append("heparin"); r.append("Flush")
        if rng.random() < 0.25:
            d.append("albuterol"); r.append("Neb")
        if rng.random() < 0.40:
            d.append("acetaminophen")
            r.append("Oral" if rng.random() < 0.9 else "IV")
        if rng.random() < 0.20:
            r.append("Oral")
        # A PAIR WHOSE TOKENS ARE COMMON AND WHOSE JOINT IS RARE.
        # Each is held by hundreds of patients; the pair by a
        # handful, and that handful is who it would describe.
        u = rng.random()
        if u < 0.22:
            d.append("twinA")
        elif u < 0.44:
            r.append("twinB")
        elif u < 0.4406:
            d.append("twinA"); r.append("twinB")
        drugs.append(";".join(sorted(set(d))))
        routes.append(";".join(sorted(set(r))))
    return pd.DataFrame({
        "person_id": ["P%04d" % (i // 4) for i in range(n)],
        "active_drugs": drugs, "drug_routes": routes,
        "x": np.round(rng.normal(0, 1, n), 4)})


def phi(f, a, b):
    A = S.has_token(f["active_drugs"], ";", a).astype(float)
    B = S.has_token(f["drug_routes"], ";", b).astype(float)
    return float(np.corrcoef(A, B)[0, 1])


def share(f, col, t):
    return float(S.has_token(f[col], ";", t).astype(float).mean())


def run_cli(src, out, *flags):
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_discovery.py"),
         "--src", str(src), "--out", str(out),
         "--group-by", "person_id", "--generate", "--seed", "5"]
        + list(flags),
        capture_output=True, text=True, cwd=str(ROOT))
    return r


def main():
    df = fixture()
    gid = df["person_id"].to_numpy()

    # ---- the detector -----------------------------------------
    co = S.cooccurrence(df, ["active_drugs", "drug_routes"], gid,
                        k=10)
    check("the detector publishes token pairs ({} of them)".format(
        len(co)), len(co) > 0)

    hf = [c for c in co
          if c["a"] == "active_drugs__has__heparin"
          and c["b"] == "drug_routes__has__Flush"]
    check("...and it finds the planted implication by name, with "
          "its CONDITIONAL near the source's 1.0 - a lift is a "
          "nudge, the conditional is the thing a draw has to land",
          len(hf) == 1 and hf[0]["cond"] > 0.9
          and hf[0]["lift"] > 1.0)

    # THE JOINT IS WHAT CLEARS k. Each token has already passed the
    # vocabulary's floor; a PAIR held by three patients describes
    # those three people however common each token is alone.
    _ma = S.has_token(df["active_drugs"], ";", "twinA"
                      ).fillna(False).to_numpy(dtype=bool)
    _mb = S.has_token(df["drug_routes"], ";", "twinB"
                      ).fillna(False).to_numpy(dtype=bool)
    _pa, _pb = len(set(gid[_ma])), len(set(gid[_mb]))
    _pj = len(set(gid[_ma & _mb]))
    check("the fixture carries the thing the k floor is about: two "
          "tokens held by {} and {} patients whose JOINT is held "
          "by {}".format(_pa, _pb, _pj),
          _pa > 100 and _pb > 100 and 0 < _pj < 10)

    check("...and that rare JOINT is NOT published, however common "
          "each token is alone",
          not any({c["a"], c["b"]}
                  == {"active_drugs__has__twinA",
                      "drug_routes__has__twinB"} for c in co))

    check("...and every published pair clears the floor on the "
          "joint",
          all(c["patients"] >= 10 for c in co))

    check("...and repulsion is published as well as attraction - a "
          "lift below 1 is structure, and this project carried "
          "'negative token associations are not modeled' as a gap "
          "for months",
          any(c["lift"] < 1.0 for c in co))

    # THE SCREEN IS PHI, THE CORRELATION THE GRADER COMPUTES.
    # Lift and phi agree on strong pairs and on rare ones; they
    # part on a MODERATE conditional on a COMMON token, where lift
    # divides by a large base and stays near 1.
    rg = np.random.default_rng(31)
    nn = 20000
    ca, cb = [], []
    for _ in range(nn):
        a, b = [], []
        ha = rg.random() < 0.30
        if ha:
            a.append("tokA")
        if rg.random() < (0.62 if ha else 0.449):
            b.append("tokB")
        for i in range(4):
            if rg.random() < 0.35:
                a.append("fa%d" % i)
            if rg.random() < 0.35:
                b.append("fb%d" % i)
        ca.append(";".join(sorted(set(a))))
        cb.append(";".join(sorted(set(b))))
    mdf = pd.DataFrame({
        "person_id": ["P%05d" % (i // 2) for i in range(nn)],
        "ca": ca, "cb": cb})
    mgid = mdf["person_id"].to_numpy()
    mphi = float(np.corrcoef(
        S.has_token(mdf["ca"], ";", "tokA").astype(float),
        S.has_token(mdf["cb"], ";", "tokB").astype(float))[0, 1])
    mco = S.cooccurrence(mdf, ["ca", "cb"], mgid, k=10)
    mhit = [c for c in mco
            if "tokA" in c["a"] and "tokB" in c["b"]]
    check("the screen is PHI, and the fixture carries where that "
          "differs from a LIFT bar: measured phi {:+.4f} - over "
          "the gate's 0.1 relate floor - with a lift of {:.3f}"
          .format(mphi, mhit[0]["lift"] if mhit else float("nan")),
          abs(mphi) > 0.10 and mhit and mhit[0]["lift"] < 1.5)

    check("...and it IS published, because the screen is the "
          "correlation the grader measures",
          bool(mhit))

    # ---- through the REAL CLI ---------------------------------
    # A blueprint built with a restricted `targets=` EXCLUDES the
    # scaffolding relationships the engine already draws through,
    # and measuring against that baseline overstated this feature
    # by an order of magnitude. The CLI is the path a run takes.
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        src = t / "src.csv"
        df.to_csv(src, index=False)
        r_off = run_cli(src, t / "off",
                        "--no-token-cooccurrence")
        r_on = run_cli(src, t / "on")
        check("both CLI arms complete", r_off.returncode == 0
              and r_on.returncode == 0)

        # THE DEFAULT IS THE CLAIM. The plain invocation above is
        # the `on` arm, so if somebody flips the default back the
        # two arms become identical and every comparison below
        # silently stops comparing anything - two arms that agree
        # are one arm, and this suite would read green.
        import inspect
        from synthkit.generate import generate as _gen
        _d = inspect.signature(_gen).parameters[
            "use_cooccurrence"].default
        check("the conditional draw is ON BY DEFAULT, which is the "
              "claim three paired seeds of the real extract earned: "
              "close +4 / +7 / +5 pairs, direction +2 on every seed, "
              "zero own-column inversions throughout - a range that "
              "does not cross zero", _d is True)

        bp = json.loads((t / "on" / "blueprint.json")
                        .read_text(encoding="utf-8"))
        check("the blueprint PUBLISHES the structure, so the "
              "contract carries it rather than the engine keeping "
              "it privately",
              len(bp.get("token_cooccurrence") or []) > 0)

        g_off = pd.read_csv(t / "off" / "generated.csv", dtype=str,
                            keep_default_na=False)
        g_on = pd.read_csv(t / "on" / "generated.csv", dtype=str,
                           keep_default_na=False)

        # THE FLAG MUST REACH GENERATION. A flag that is read and
        # not used is this project's own named failure mode.
        check("the flag CHANGES the output - a flag read but not "
              "used is worse than no flag",
              not g_off.equals(g_on))

        # AND THE MARGINALS MUST STILL LAND, or the conditional is
        # paid for out of the published shares.
        worst = max(abs(share(df, c, t_) - share(g_on, c, t_))
                    for c, t_ in (("active_drugs", "heparin"),
                                  ("active_drugs", "acetaminophen"),
                                  ("drug_routes", "Flush"),
                                  ("drug_routes", "Oral")))
        check("...and every token's MARGINAL share still lands "
              "(worst miss {:.3f}) - the conditional is solved by "
              "the same draw-measure-adjust loop as the marginal, "
              "and the SOLVE and the DRAW call one pick function "
              "so they cannot disagree".format(worst),
              worst < 0.06)

        # WHAT IS **NOT** CLAIMED HERE, and the number says why.
        # Through the real path the baseline already carries most
        # of the implication, because the pipeline discovers
        # relationships between the scaffolding columns and draws
        # them. The gain this feature has is on the real extract.
        b_off, b_on = phi(g_off, "heparin", "Flush"), \
            phi(g_on, "heparin", "Flush")
        check("the BASELINE already carries the implication "
              "through the real path ({:+.3f} of a source {:+.3f}) "
              "- so no fidelity gain is claimed on a fixture here; "
              "the measured gain is on the real extract, close "
              "102/125 to 106/125".format(
                  b_off, phi(df, "heparin", "Flush")),
              b_off > 0.3)

        check("...and the conditional draw does not DAMAGE it "
              "({:+.3f} with the flag on)".format(b_on),
              b_on > b_off - 0.10)

    # ---- inert where its mechanism is absent -------------------
    rng2 = np.random.default_rng(11)
    n2 = 2000
    flat = pd.DataFrame({
        "person_id": ["P%03d" % (i // 4) for i in range(n2)],
        "a": np.round(rng2.normal(0, 1, n2), 4),
        "b": np.round(rng2.normal(0, 1, n2), 4)})
    check("a frame with no set column publishes NO structure, so "
          "the change cannot act where its mechanism is absent",
          S.cooccurrence(flat, [], flat["person_id"].to_numpy(),
                         k=10) == [])

    from synthkit import blueprint as B                # noqa: E402
    from synthkit.discover import discover             # noqa: E402
    from synthkit.generate import generate             # noqa: E402
    fcat = discover(flat, group_by="person_id", targets=["b"])
    fbp = B.build(flat, fcat, group_by="person_id")
    g1 = generate(fbp, n_patients=300, seed=7)
    fbp2 = copy.deepcopy(fbp)
    fbp2.pop("token_cooccurrence", None)
    g2 = generate(fbp2, n_patients=300, seed=7)
    # AND THE DEFAULT PATH IS THE ONE EXERCISED, so the identity
    # above is a statement about the shipped behaviour rather than
    # about a branch nobody takes.
    check("...and a blueprint written BEFORE this field existed "
          "generates BYTE-IDENTICALLY - an old artifact must still "
          "load, and absent is not a different fact from empty",
          g1.equals(g2))

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
