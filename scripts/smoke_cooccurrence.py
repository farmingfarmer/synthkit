"""Which tokens travel together, and whether generation carries it.

WHY THIS EXISTS. On the real extract about 110 of the 135 pairs the
gate relates are SET-TOKEN pairs - roughly four fifths of everything
graded - and the rules engine's draw modelled none of it.
`_token_weights` solved weights so each token's MARGINAL share
measured back and there was nothing in it about which tokens
co-occur, so tokens were drawn independently and "a drug implies its
route" survived by chance.

The approach is the one auto-regressive tabular generators use over
discretised columns: approximate the joint as a chain of
CONDITIONALS rather than a product of marginals. Each pick reweights
the remaining candidates by the published structure of what the row
already holds, including tokens drawn in EARLIER set columns - which
is where `drug -> route` lives.

WHAT IS LOAD-BEARING: the JOINT clears the patient floor (a pair
held by three patients describes those three people), the marginals
still land (or the fix trades one criterion for another), and the
whole thing is INERT where nothing is published, so an older
blueprint generates unchanged.
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from synthkit import blueprint as B                   # noqa: E402
from synthkit import sets as S                        # noqa: E402
from synthkit.discover import discover                # noqa: E402
from synthkit.generate import generate                # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def fixture(n=6000, seed=3):
    """A drug that IMPLIES its route, which is the extract's own
    dominant shape and the one no fixture here carried."""
    rng = np.random.default_rng(seed)
    pid = ["P%04d" % (i // 6) for i in range(n)]
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
        # Each of these is held by hundreds of patients, so a floor
        # applied to the two tokens SEPARATELY waves the pair
        # through; the pair itself is held by a handful, and that
        # handful is who it would describe. Without this shape the
        # joint screen and a per-token screen publish the same list
        # and no check can tell them apart.
        u = rng.random()
        if u < 0.22:
            d.append("twinA")
        elif u < 0.44:
            r.append("twinB")
        elif u < 0.4406:
            d.append("twinA"); r.append("twinB")
        drugs.append(";".join(sorted(set(d))))
        routes.append(";".join(sorted(set(r))))
    return pd.DataFrame({"person_id": pid, "active_drugs": drugs,
                         "drug_routes": routes,
                         "x": np.round(rng.normal(0, 1, n), 4)})


def phi(f, a, b):
    A = S.has_token(f["active_drugs"], ";", a).astype(float)
    Bv = S.has_token(f["drug_routes"], ";", b).astype(float)
    return float(np.corrcoef(A, Bv)[0, 1])


def share(f, col, t):
    return float(S.has_token(f[col], ";", t).astype(float).mean())


def main():
    df = fixture()
    gid = df["person_id"].to_numpy()

    # ---- the detector -----------------------------------------
    co = S.cooccurrence(df, ["active_drugs", "drug_routes"], gid,
                        k=10)
    check("the detector publishes token pairs at all ({} of "
          "them)".format(len(co)), len(co) > 0)

    hf = [c for c in co
          if c["a"] == "active_drugs__has__heparin"
          and c["b"] == "drug_routes__has__Flush"]
    check("...and it finds the planted implication by name, with "
          "its LIFT above 1 and its CONDITIONAL near the source's "
          "1.0 - the lift is a nudge, the conditional is the thing "
          "a draw has to land",
          len(hf) == 1 and hf[0]["lift"] > 1.5
          and hf[0]["cond"] > 0.9)

    # THE JOINT IS WHAT CLEARS k. Each token has already passed the
    # vocabulary's floor; a PAIR held by three patients describes
    # those three people however common each token is alone.
    check("...and EVERY published pair clears the patient floor on "
          "the JOINT, not on the two tokens separately",
          all(c["patients"] >= 10 for c in co))

    check("...and a k above every joint publishes NOTHING, so the "
          "floor is what gates it rather than the shape of the "
          "data",
          S.cooccurrence(df, ["active_drugs", "drug_routes"], gid,
                         k=100000) == [])

    # BOTH DIRECTIONS. A token that is LESS likely in another's
    # presence is structure too, and "negative token associations
    # are not modeled" has been a named gap here for months.
    check("...and repulsion is published as well as attraction - a "
          "lift below 1 is structure, and this project has carried "
          "'negative token associations are not modeled' as a gap",
          any(c["lift"] < 1.0 for c in co))

    # THE JOINT SCREEN, WITH A SHAPE THAT CAN SEPARATE IT.
    #
    # A first version of this asserted only that every published
    # pair clears k on the joint - and a mutation screening the two
    # tokens SEPARATELY broke nothing, because on this fixture
    # every pair cleared both ways. The fixture now plants a pair
    # whose tokens are each held by hundreds of patients and whose
    # JOINT is held by a handful.
    _ja = "active_drugs__has__twinA"
    _jb = "drug_routes__has__twinB"
    _ma = S.has_token(df["active_drugs"], ";", "twinA"
                      ).fillna(False).to_numpy(dtype=bool)
    _mb = S.has_token(df["drug_routes"], ";", "twinB"
                      ).fillna(False).to_numpy(dtype=bool)
    _pa = len(set(gid[_ma]))
    _pb = len(set(gid[_mb]))
    _both = _ma & _mb
    _pj = len(set(gid[_both]))
    check("the fixture carries the thing THIS check is about: two "
          "tokens held by {} and {} patients whose JOINT is held "
          "by {} - a per-token floor waves that pair through"
          .format(_pa, _pb, _pj),
          _pa > 100 and _pb > 100 and 0 < _pj < 10)

    check("...and the rare JOINT is NOT published, however common "
          "each token is alone - the pair is what would describe "
          "those few patients",
          not any(c["a"] == _ja and c["b"] == _jb for c in co)
          and not any(c["a"] == _jb and c["b"] == _ja for c in co))

    # ---- does generation carry it? ----------------------------
    cat = discover(df, group_by="person_id", targets=["x"])
    bp = B.build(df, cat, group_by="person_id")
    check("the blueprint PUBLISHES the structure, so the contract "
          "carries it rather than the engine keeping it privately",
          len(bp.get("token_cooccurrence") or []) > 0)

    g = generate(bp, n_patients=1000, seed=5,
                 use_cooccurrence=True)
    bp0 = copy.deepcopy(bp)
    bp0["token_cooccurrence"] = []
    g0 = generate(bp0, n_patients=1000, seed=5,
                  use_cooccurrence=True)

    for a, b in (("heparin", "Flush"), ("albuterol", "Neb")):
        s_, n_, y_ = phi(df, a, b), phi(g0, a, b), phi(g, a, b)
        check("`{} <-> {}` is ABSENT without the published "
              "structure ({:+.3f} against a source {:+.3f}) and "
              "CARRIED with it ({:+.3f}) - the control arm is what "
              "makes the gain a measurement".format(
                  a, b, n_, s_, y_),
              abs(n_) < 0.08 and y_ > 0.25)

    # AND THE MARGINALS MUST STILL LAND, or this trades one
    # criterion for another - which is what two reverted fidelity
    # tiers in this project did.
    worst = 0.0
    for col, t in (("active_drugs", "heparin"),
                   ("active_drugs", "acetaminophen"),
                   ("drug_routes", "Flush"), ("drug_routes", "Oral")):
        worst = max(worst, abs(share(df, col, t) - share(g, col, t)))
    check("...and every token's MARGINAL share still lands (worst "
          "miss {:.3f}) - the conditional is solved by the same "
          "draw-measure-adjust loop as the marginal, and the SOLVE "
          "and the DRAW call one pick function so they cannot "
          "disagree".format(worst),
          worst < 0.05)

    # CONTENTION: TEN DRUGS COMPETING FOR ONE ROUTE'S BUDGET.
    #
    # A token's marginal share is pinned, so placing it in one
    # partner's rows takes it out of another's. This is the shape
    # that made the real-extract arm a trade - four token pairs
    # flipped out of close and they FADED rather than overshot.
    _rg2 = np.random.default_rng(13)
    _n2 = 6000
    _DR = ["d%02d" % i for i in range(10)]
    _OT = ["r%02d" % i for i in range(6)]
    _dg, _rt = [], []
    for _ in range(_n2):
        _d, _r = [], []
        for _g in _DR:
            if _rg2.random() < 0.10:
                _d.append(_g)
                if _rg2.random() < 0.9:
                    _r.append("Oral")
        for _o in _OT:
            if _rg2.random() < 0.25:
                _r.append(_o)
        _dg.append(";".join(sorted(set(_d))))
        _rt.append(";".join(sorted(set(_r))))
    _cdf = pd.DataFrame({
        "person_id": ["P%04d" % (i // 6) for i in range(_n2)],
        "active_drugs": _dg, "drug_routes": _rt,
        "x": np.round(_rg2.normal(0, 1, _n2), 4)})
    _cgid = _cdf["person_id"].to_numpy()
    _oral = float(S.has_token(_cdf["drug_routes"], ";", "Oral")
                  .astype(float).mean())
    _cco = S.cooccurrence(_cdf, ["active_drugs", "drug_routes"],
                          _cgid, k=10)
    _into = [c for c in _cco
             if c["b"] == "drug_routes__has__Oral"]

    check("the fixture carries the thing this screen is about: a "
          "route on {:.0%} of rows that ten drugs imply, so its "
          "LIFT is small while its pair correlation is not"
          .format(_oral),
          _oral > 0.5 and len(_DR) == 10)

    check("...and the structure IS published on it ({} "
          "conditionals into that route)".format(len(_into)),
          len(_into) >= 5)

    check("...and every published pair carries the PHI the screen "
          "used, so a reader can see why it was kept",
          all(abs(float(c.get("phi") or 0)) >= 0.10
              for c in _cco))

    _ccat = discover(_cdf, group_by="person_id", targets=["x"])
    _cbp = B.build(_cdf, _ccat, group_by="person_id")
    _cg = generate(_cbp, n_patients=1000, seed=5,
                   use_cooccurrence=True)
    _cbp0 = copy.deepcopy(_cbp)
    _cbp0["token_cooccurrence"] = []
    _cg0 = generate(_cbp0, n_patients=1000, seed=5,
                    use_cooccurrence=True)

    def _mphi(f):
        return float(np.mean([phi(f, d, "Oral") for d in _DR]))
    _ms, _m0, _mc = _mphi(_cdf), _mphi(_cg0), _mphi(_cg)
    check("...and under genuine CONTENTION - ten drugs competing "
          "for one route's fixed budget - the association goes "
          "from {:+.3f} to {:+.3f} against a source {:+.3f}, so "
          "the allocation carries most of it rather than "
          "collapsing".format(_m0, _mc, _ms),
          abs(_m0) < 0.05 and _mc > 0.5 * _ms)

    _os = float(S.has_token(_cg["drug_routes"], ";", "Oral")
                .astype(float).mean())
    check("...and the contended token's own MARGINAL still lands "
          "({:.3f} against a source {:.3f}) - satisfying ten "
          "conditionals must not be paid for out of the published "
          "share".format(_os, _oral),
          abs(_os - _oral) < 0.05)

    # THE SCREEN IS PHI, NOT LIFT, AND HERE IS WHERE THEY PART.
    #
    # A lift threshold and the gate's relate floor disagree about
    # what counts as structure. They agree on strong pairs and on
    # rare ones - a first attempt to separate them used both and
    # the mutation found NOTHING, because on those shapes the two
    # screens publish the same list. Where they part is a MODERATE
    # conditional on a COMMON token: lift divides by a large base
    # so it stays near 1, while phi - the binary-pair correlation
    # the gate would compute - is comfortably over the floor.
    _rg3 = np.random.default_rng(31)
    _n3 = 20000
    _ca, _cb = [], []
    for _ in range(_n3):
        _a, _b = [], []
        _ha = _rg3.random() < 0.30
        if _ha:
            _a.append("tokA")
        if _rg3.random() < (0.62 if _ha else 0.449):
            _b.append("tokB")
        for _i in range(4):
            if _rg3.random() < 0.35:
                _a.append("fa%d" % _i)
            if _rg3.random() < 0.35:
                _b.append("fb%d" % _i)
        _ca.append(";".join(sorted(set(_a))))
        _cb.append(";".join(sorted(set(_b))))
    _mdf = pd.DataFrame({
        "person_id": ["P%05d" % (i // 2) for i in range(_n3)],
        "ca": _ca, "cb": _cb})
    _mgid = _mdf["person_id"].to_numpy()
    _ma = S.has_token(_mdf["ca"], ";", "tokA").astype(float)
    _mb = S.has_token(_mdf["cb"], ";", "tokB").astype(float)
    _mphi_v = float(np.corrcoef(_ma, _mb)[0, 1])
    _mco = S.cooccurrence(_mdf, ["ca", "cb"], _mgid, k=10)
    _mhit = [c for c in _mco
             if "tokA" in c["a"] and "tokB" in c["b"]]

    check("the fixture carries the shape the two screens disagree "
          "on: a MODERATE conditional on a COMMON token, measured "
          "phi {:+.4f} - over the gate's 0.1 relate floor - with a "
          "lift of only {:.3f}".format(
              _mphi_v, _mhit[0]["lift"] if _mhit else float("nan")),
          abs(_mphi_v) > 0.10
          and _mhit and _mhit[0]["lift"] < 1.5)

    check("...and it IS published, because the screen is the "
          "correlation the grader measures rather than a lift "
          "threshold that would drop it",
          bool(_mhit))

    # ---- inert where its mechanism is absent -------------------
    rng = np.random.default_rng(11)
    nn = 2000
    flat = pd.DataFrame({
        "person_id": ["P%03d" % (i // 4) for i in range(nn)],
        "a": np.round(rng.normal(0, 1, nn), 4),
        "b": np.round(rng.normal(0, 1, nn), 4)})
    fcat = discover(flat, group_by="person_id", targets=["b"])
    fbp = B.build(flat, fcat, group_by="person_id")
    check("a frame with no set column publishes NO structure, so "
          "the change cannot act where its mechanism is absent",
          (fbp.get("token_cooccurrence") or []) == [])

    g1 = generate(fbp, n_patients=300, seed=7)
    fbp2 = copy.deepcopy(fbp)
    fbp2.pop("token_cooccurrence", None)
    g2 = generate(fbp2, n_patients=300, seed=7)
    check("...and a blueprint written BEFORE this field existed "
          "generates BYTE-IDENTICALLY - an old artifact must still "
          "load, and absent is not a different fact from empty "
          "here",
          g1.equals(g2))

    # OFF BY DEFAULT, AND THAT DEFAULT IS LOAD-BEARING.
    #
    # The control arm on the real extract called this a TRADE:
    # direction kept 118/125 -> 120/125 and close 102/125 -> 99/125,
    # with all four flipped pairs being token pairs that FADED
    # rather than overshot. Trading one criterion for another is an
    # operator decision, and this project has twice reverted a tier
    # that shipped on a fixture gain a control arm later overturned.
    g_def = generate(bp, n_patients=1000, seed=5)
    g_off = generate(bp0, n_patients=1000, seed=5)
    check("the conditional draw is OFF BY DEFAULT - a run that "
          "does not ask for it is BYTE-IDENTICAL to one whose "
          "blueprint carries no structure at all, so the measured "
          "trade cannot land on anybody by accident",
          g_def.equals(g_off))

    check("...and asking for it CHANGES the output, so the flag is "
          "not read-but-unused - which this project has recorded "
          "as its own failure mode more than once",
          not g_def.equals(g))

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
