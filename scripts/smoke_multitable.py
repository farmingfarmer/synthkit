"""Smoke: several tables, keys from the values, integrity kept.

WHY THIS EXISTS. The fitted path has always been single-table,
so an operator with patients, visits and labs had to join them
by hand first - and a hand join is where row counts and
missingness quietly change before this tool ever sees them.
M3's gate is three multi-table fixtures, keys auto-detected,
referential integrity preserved.

THE NEGATIVE CONTROLS ARE THE POINT. A key detector that finds
a key in everything also finds one in nothing, so the suite
plants columns that MUST NOT be keys beside the ones that must:
a continuous measurement (unique on every row, and the first
version of this module made one a primary key and would have
linked tables on it), a two-value flag, and a table unrelated
to the others.

AND THE HEADERS LIE ON PURPOSE. One fixture names its key
`subj` and puts lab values in a column called `patient_id`,
because this project's rule for PHI is that values decide and
headers only assist, and a key detector has no business
following a different rule.
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

from synthkit import multitable as MT                 # noqa: E402
from synthkit.multitable import (                     # noqa: E402
    assign_keys, check_integrity, cross_couplings, detect,
    integrity, render)

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:3d}  {}".format(PASS, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def star(seed=3):
    """One parent, two children, with planted orphans.

    The orphans are deliberate: a real extract has visits whose
    patient was filtered out by somebody's cohort query, and a
    generator that emits 100% integrity where the source had
    97.8% has invented a cleanliness the data never had.
    """
    rng = np.random.default_rng(seed)
    n = 200
    pat = pd.DataFrame({
        "mrn": ["P{:05d}".format(i) for i in range(n)],
        "sex": rng.choice(["F", "M"], n),
        "yob": rng.integers(1940, 2005, n)})
    vn = 900
    vis = pd.DataFrame({
        "visit_id": ["V{:06d}".format(i) for i in range(vn)],
        "mrn": (list(rng.choice(pat["mrn"], vn - 20))
                + ["P99999"] * 20),
        "los": rng.integers(1, 14, vn)})
    rx = pd.DataFrame({
        "rx_id": ["R{:06d}".format(i) for i in range(400)],
        "mrn": rng.choice(pat["mrn"], 400),
        "dose": np.round(rng.normal(10, 3, 400), 2)})
    return {"patients": pat, "visits": vis, "meds": rx}


def chain(seed=5):
    """patients -> visits -> labs, each a child of the last."""
    rng = np.random.default_rng(seed)
    n = 150
    pat = pd.DataFrame({
        "mrn": ["Q{:05d}".format(i) for i in range(n)],
        "site": rng.choice(["a", "b", "c"], n)})
    vis = pd.DataFrame({
        "visit_id": ["W{:06d}".format(i) for i in range(600)],
        "mrn": rng.choice(pat["mrn"], 600)})
    lab = pd.DataFrame({
        "lab_id": ["L{:07d}".format(i) for i in range(2000)],
        "visit_id": rng.choice(vis["visit_id"], 2000),
        # A MEASUREMENT, unique on nearly every row - the
        # column the first version of this module made a key.
        "value": np.round(rng.normal(5, 1, 2000), 4)})
    return {"patients": pat, "visits": vis, "labs": lab}


def lying_headers(seed=7):
    """The key is `subj`; `patient_id` holds lab values.

    Integer keys too, because a patient number is usually an
    integer and a detector that refuses those rejects the most
    common key there is.
    """
    rng = np.random.default_rng(seed)
    n = 180
    pat = pd.DataFrame({
        "subj": list(range(90000, 90000 + n)),
        "patient_id": np.round(rng.normal(5, 1, n), 3),
        "flag": rng.choice(["y", "n"], n)})
    enc = pd.DataFrame({
        "enc": ["E{:06d}".format(i) for i in range(700)],
        "subj": rng.choice(pat["subj"], 700),
        "admitting": rng.choice(pat["subj"], 700)})
    return {"people": pat, "encounters": enc}


def main():
    # ---- FIXTURE ONE: A STAR
    fr = star()
    found = detect(fr)
    keys = {n: t.primary_key for n, t in found["tables"].items()}
    check("a star schema's three keys are found from the "
          "values: {}".format(keys),
          keys == {"patients": "mrn", "visits": "visit_id",
                   "meds": "rx_id"})
    pairs = {(lk.child, lk.parent) for lk in found["links"]}
    check("...and both children are linked to the parent, by "
          "value overlap rather than by column name: {}"
          .format(sorted(pairs)),
          ("visits", "patients") in pairs
          and ("meds", "patients") in pairs)
    src_int = integrity(fr, found["links"])
    vis_row = [r for r in src_int if r["child"] == "visits"][0]
    check("...and the 20 planted ORPHANS are measured in the "
          "SOURCE - {} of {} visits have a parent ({:.1%}). A "
          "generator reporting 100% without this line is "
          "taking credit for somebody else's cleaning"
          .format(vis_row["with_parent"], vis_row["rows"],
                  vis_row["share"]),
          vis_row["orphans"] == 20)

    gen = assign_keys(fr, found, seed=1)
    gen_int = check_integrity(gen, found)
    check("the generated tables keep referential integrity BY "
          "CONSTRUCTION - every child key is drawn from keys "
          "that already exist, so there is no step where a "
          "child can point at a parent that is not there",
          all(r["share"] >= 0.97 for r in gen_int))
    gvis = [r for r in gen_int if r["child"] == "visits"][0]
    check("...and the orphan SHARE is reproduced rather than "
          "cleaned away: source {:.1%}, generated {:.1%}"
          .format(vis_row["share"], gvis["share"]),
          abs(gvis["share"] - vis_row["share"]) < 0.02)
    check("...and NOT ONE real key is republished - an MRN is "
          "a direct identifier, and a generated file carrying "
          "one has republished it however synthetic everything "
          "around it is",
          not (set(gen["patients"]["mrn"].astype(str))
               & set(fr["patients"]["mrn"].astype(str))))
    check("...and the orphans point at keys that belong to "
          "NOBODY, so an orphan is an orphan rather than a "
          "real identifier that happens not to match",
          any(str(v).startswith("ORPHAN")
              for v in gen["visits"]["mrn"]))

    # ---- FIXTURE TWO: A CHAIN, AND A MEASUREMENT THAT IS NOT A KEY
    fr2 = chain()
    f2 = detect(fr2)
    k2 = {n: t.primary_key for n, t in f2["tables"].items()}
    check("a chain's keys are found at every level: {}"
          .format(k2),
          k2 == {"patients": "mrn", "visits": "visit_id",
                 "labs": "lab_id"})
    check("...and a CONTINUOUS MEASUREMENT is not a key, even "
          "though it is unique on every row and present on "
          "every row - the first version of this module made a "
          "lab value a primary key, and that key would then "
          "have been used to LINK TABLES",
          "value" not in f2["tables"]["labs"].pk_candidates)
    p2 = {(lk.child, lk.parent) for lk in f2["links"]}
    check("...and the chain is linked end to end: {}"
          .format(sorted(p2)),
          ("visits", "patients") in p2
          and ("labs", "visits") in p2)
    g2 = assign_keys(fr2, f2, seed=2)
    check("...and the generated chain holds integrity at every "
          "level",
          all(r["share"] >= 0.999
              for r in check_integrity(g2, f2)))

    # ---- FIXTURE THREE: HEADERS THAT LIE
    fr3 = lying_headers()
    f3 = detect(fr3)
    check("the key is found as `subj` even though another "
          "column is NAMED patient_id - values decide and "
          "headers only assist, which is the rule `scrub` "
          "already uses for PHI",
          f3["tables"]["people"].primary_key == "subj")
    check("...and the column NAMED patient_id, which holds lab "
          "values, is not a key candidate at all",
          "patient_id"
          not in f3["tables"]["people"].pk_candidates)
    check("...and an INTEGER key works, because a patient "
          "number usually is one and a detector that refuses "
          "those rejects the most common key there is",
          f3["tables"]["people"].primary_key == "subj"
          and str(fr3["people"]["subj"].iloc[0]).isdigit())
    check("...and TWO columns pointing at the same parent is "
          "reported as something to CHECK - it can be real "
          "(admitting and discharging clinician) and it is "
          "also what a mis-detection looks like",
          any("mis-detection" in w for w in f3["warnings"]))

    # ---- NEGATIVE CONTROLS
    rng = np.random.default_rng(11)
    unrelated = pd.DataFrame({
        "code": ["X{:04d}".format(i) for i in range(120)],
        "w": rng.normal(0, 1, 120)})
    f4 = detect({"patients": fr["patients"],
                 "codes": unrelated})
    check("two UNRELATED tables produce NO links - a detector "
          "that finds a key in everything also finds one in "
          "nothing",
          not f4["links"])
    check("...and the report SAYS so rather than printing an "
          "empty section, naming the likely cause",
          "NO LINKS FOUND" in render(f4, []))
    twoval = pd.DataFrame({"flag": rng.choice(["y", "n"], 300),
                           "n": rng.normal(0, 1, 300)})
    f5 = detect({"patients": fr["patients"], "flags": twoval})
    check("a two-value column is not a key - it matches any "
          "other two-value column by accident, which is the "
          "commensurability rule the constraint finder already "
          "applies",
          f5["tables"]["flags"].primary_key is None)

    # ---- AND IT REACHES THE CLI
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for name, f in fr.items():
            f.to_csv(d / (name + ".csv"), index=False)
        out = d / "schema.json"
        r = subprocess.run(
            [sys.executable, "-m", "synthkit.cli", "schema",
             str(d / "patients.csv"), str(d / "visits.csv"),
             str(d / "meds.csv"), "--json-out", str(out)],
            capture_output=True, text=True, cwd=str(ROOT),
            timeout=900)
        check("`synthkit schema` reads the tables and reports "
              "the schema - a capability wired to nothing is "
              "this repo's named silent-failure class",
              r.returncode == 0
              and "SCHEMA FOUND FROM THE VALUES" in r.stdout
              and "REFERENTIAL INTEGRITY IN THE SOURCE"
              in r.stdout)
        got = json.loads(out.read_text(encoding="utf-8"))
        check("...and the JSON carries the keys, the links "
              "with their match SHARE, and the integrity - a "
              "link holding on 94% of rows must read as 94% "
              "rather than as a yes",
              got["tables"]["patients"]["primary_key"] == "mrn"
              and any(0.0 < lk["match"] < 1.0
                      for lk in got["links"])
              and got["integrity"])
        r2 = subprocess.run(
            [sys.executable, "-m", "synthkit.cli", "schema",
             str(d / "patients.csv")],
            capture_output=True, text=True, cwd=str(ROOT),
            timeout=900)
        check("...and ONE table is refused with the reason and "
              "the command that does fit - with one table "
              "there is nothing to link",
              r2.returncode != 0
              and "at least two tables" in (r2.stdout + r2.stderr)
              and "synthkit types" in (r2.stdout + r2.stderr))

    # ---- WHAT MULTI-TABLE DOES NOT YET DO, AS A NUMBER
    #
    # Detection is done: keys from the values, links by overlap,
    # integrity reproduced. But each table still goes through
    # the single-table pipeline ALONE, so a child column that
    # depends on its PARENT's column has no path to learn it.
    # That was stated as a limitation and never measured, and a
    # limitation without a number is one nobody can tell they
    # have fixed.
    #
    # Measured here: a lab value planted at 2.0x its visit's
    # length of stay. This is a CHARACTERIZATION test - it pins
    # the gap as it stands so the net stays green for the team
    # while the gap is open, and the day joint fitting lands it
    # goes RED and says so. Arming it is raising the bound.
    import numpy as _np
    import pandas as _pd
    from synthkit import blueprint as _Bx
    from synthkit.discover import discover as _dx
    from synthkit.generate import generate as _gx

    _rg = _np.random.default_rng(7)
    _NP, _NV = 200, 4
    _vid = ["V{:05d}".format(i) for i in range(_NP * _NV)]
    _los = _np.round(_rg.lognormal(1.0, 0.6, _NP * _NV), 2)
    _vis = _pd.DataFrame({
        "visit_id": _vid,
        "person_id": _np.repeat(
            ["P{:04d}".format(i) for i in range(_NP)], _NV),
        "length_of_stay": _los,
        "age": _np.round(_rg.normal(60, 12, _NP * _NV), 0)})
    _rows = []
    for _i, _v in enumerate(_vid):
        for _ in range(int(_rg.integers(1, 4))):
            _rows.append((_v, round(float(
                2.0 * _los[_i] + _rg.normal(0, 1.2)), 3)))
    _labs = _pd.DataFrame(_rows,
                          columns=["visit_id", "lab_value"])

    def _xcorr(lab_df, vis_df):
        _m = lab_df.merge(
            vis_df[["visit_id", "length_of_stay"]],
            on="visit_id", how="inner")
        _a = _pd.to_numeric(_m["lab_value"], errors="coerce")
        _b = _pd.to_numeric(_m["length_of_stay"],
                            errors="coerce")
        _k = _a.notna() & _b.notna()
        if int(_k.sum()) < 30:
            return None
        return float(_a[_k].corr(_b[_k], method="spearman"))

    _src_x = _xcorr(_labs, _vis)
    check("the fixture really carries a CROSS-TABLE "
          "relationship - a lab value against its visit's "
          "length of stay, source {:+.3f} - because a fixture "
          "that does not cannot show the gap"
          .format(_src_x or 0.0),
          _src_x is not None and _src_x > 0.8)

    _gen = {}
    for _nm, _fr, _gb in (("visits", _vis, "person_id"),
                          ("labs", _labs, "visit_id")):
        _bpx = _Bx.build(_fr, _dx(_fr, group_by=_gb, seed=1),
                         group_by=_gb)
        _gen[_nm] = _gx(_bpx,
                        n_patients=_fr[_gb].nunique(), seed=3)
    _gv, _gl = _gen["visits"], _gen["labs"]
    _keys = sorted(set(str(x) for x in _gv.get("visit_id", [])))
    check("...and the parent's KEY survives generation, so the "
          "join can be made at all - without it the "
          "relationship could not even be asked about",
          bool(_keys))
    _gl = _gl.copy()
    _gl["visit_id"] = _np.random.default_rng(11).choice(
        _keys, size=len(_gl))
    _gen_x = _xcorr(_gl, _gv)
    check("fitting each table alone DOES destroy it - source "
          "{:+.3f} against generated {:+.3f} with no coupling, "
          "which is why the assignment cannot be left uniform"
          .format(_src_x or 0.0, _gen_x or 0.0),
          _gen_x is not None and abs(_gen_x) < 0.25)

    # ---- MT-1: THE COUPLING IS MEASURED AND ENFORCED
    #
    # Each table is still fitted alone - that part is
    # unchanged. What changed is WHICH PARENT a child row
    # points at. The child's values and the parent's values
    # are already drawn and carry their marginals; the
    # assignment is the only free variable, and it is enough
    # to set the correlation across the link.
    #
    # It is a REARRANGEMENT, the instrument the repair and the
    # dials already use: no marginal can move and referential
    # integrity is untouched, because the pool assigned from is
    # the parent keys that exist.
    _found = detect({"visits": _vis, "labs": _labs})
    _cps = cross_couplings({"visits": _vis, "labs": _labs},
                              _found)
    check("the cross-table coupling is FOUND in the source - "
          "numeric pairs over the join, rank correlation, "
          "screened for rows and for the gate's own floor so "
          "noise is not chased across a key",
          any(c["child_column"] == "lab_value"
              and c["parent_column"] == "length_of_stay"
              and abs(c["rho"]) > 0.8 for c in _cps))

    # ---- A COLLIDING COLUMN NAME MUST READ THE PARENT'S
    #
    # With pandas' default merge suffixes only the colliding
    # name gets suffixed and the bare one refers to the CHILD,
    # so a parent column whose name also exists in the child
    # was read from the child - and a within-child correlation
    # came back labelled cross-table. `provider_id` collides in
    # real OMOP, which is where this was found; this fixture
    # had no colliding non-key column and could not reach it.
    #
    # The planted collision is unambiguous: `site` is CONSTANT
    # in the parent and random in the child, so reading the
    # wrong side gives a different answer every time.
    _vis2 = _vis.copy()
    _labs2 = _labs.copy()
    _vis2["site"] = 1.0
    _labs2["site"] = np.random.default_rng(3).normal(
        0, 1, len(_labs2))
    _f2 = detect({"visits": _vis2, "labs": _labs2})
    _c2 = cross_couplings({"visits": _vis2, "labs": _labs2},
                          _f2)
    check("a parent column whose NAME collides with a child "
          "column is read from the PARENT - the parent's `site` "
          "is constant, so any coupling reported for it would "
          "be the child's column wearing the parent's name",
          not any(c["parent_column"] == "site" for c in _c2))
    check("...while the genuine coupling still comes through "
          "alongside the collision, so the fix did not simply "
          "stop looking",
          any(c["child_column"] == "lab_value"
              and c["parent_column"] == "length_of_stay"
              for c in _c2))

    # ---- AN IDENTIFIER IS NOT A QUANTITY
    #
    # The first real read came back led by
    # `measurement_id ~ provider_id` and
    # `value_as_concept_id ~ care_site_id`. Enforcing one of
    # those would attach a child to a parent by ID SIMILARITY,
    # which is worse than not enforcing at all. The fixture
    # carried only genuine quantities, so nothing here could
    # show it.
    _vis3 = _vis.copy()
    _labs3 = _labs.copy()
    # THE PLANTED `_id` MUST ACTUALLY BE OFFERABLE, or the
    # name test is not load-bearing and its mutation finds
    # nothing - which is what happened first time, because a
    # modular counter correlates with nothing. This one is a
    # rounded copy of the length of stay, so it carries the
    # SAME strong signal as the genuine coupling and only its
    # NAME keeps it out.
    _vis3["provider_id"] = _vis3["length_of_stay"].round(0)
    _labs3["measurement_id"] = np.arange(len(_labs3))
    _f3 = detect({"visits": _vis3, "labs": _labs3})
    _c3 = cross_couplings({"visits": _vis3, "labs": _labs3},
                          _f3)
    check("an `_id` column is never offered as a coupling, on "
          "either side - a rank correlation between two keys "
          "means nothing, and enforcing it would attach a "
          "child to a parent by ID similarity",
          not any(c["child_column"].endswith("_id")
                  or c["parent_column"].endswith("_id")
                  for c in _c3))
    check("...and a near-unique numeric column is refused "
          "whatever it is called, because a renamed counter is "
          "still a counter",
          not MT._is_quantity(
              pd.Series(np.arange(500), name="serial")))
    check("...while an ordinary measured amount passes that "
          "same test, so the screen is narrow and not a "
          "blanket",
          MT._is_quantity(_labs3["lab_value"].rename(
              "lab_value")))

    _before = _gen["labs"]["lab_value"].copy()
    _par_before = _gen["visits"]["length_of_stay"].copy()
    _out = assign_keys(_gen, _found, seed=5,
                          couplings=_cps)
    _rec = _xcorr(_out["labs"], _out["visits"])
    check("and ENFORCING it recovers the relationship: "
          "{:+.3f} against a source {:+.3f}, where the uniform "
          "assignment gave {:+.3f}"
          .format(_rec or 0.0, _src_x or 0.0, _gen_x or 0.0),
          _rec is not None and abs(_rec - _src_x) < 0.15)
    check("...and NO VALUE MOVED - the child's column and the "
          "parent's are byte-identical, because only the "
          "assignment was permuted",
          _out["labs"]["lab_value"].equals(_before)
          and _out["visits"]["length_of_stay"].equals(
              _par_before))
    check("...and referential integrity is untouched: every "
          "child still points at a parent that exists, because "
          "the pool assigned from IS the parents that exist",
          _out["labs"]["visit_id"].isin(
              set(_out["visits"]["visit_id"])).all())
    _uncoupled = assign_keys(_gen, _found, seed=5,
                                couplings=None)
    check("...and the coupling is what did it - the same seed "
          "with no couplings passed still reads {:+.3f}, so "
          "the recovery is not the seed"
          .format(_xcorr(_uncoupled["labs"],
                         _uncoupled["visits"]) or 0.0),
          abs(_xcorr(_uncoupled["labs"],
                     _uncoupled["visits"]) or 0.0) < 0.25)

    # ------------------------------------------------ MT-2
    # ONE FOREIGN KEY IS ONE CHOICE PER ROW, so the
    # correlations a link can carry have RANK ONE. Two child
    # columns riding the SAME parent are rank one and both
    # land; two riding DIFFERENT parents are not, and the
    # honest move is to refuse rather than land one and lose
    # the other - which is measured below as the thing the
    # single-coupling version actually did.
    def _two(shared, seed=7):
        rng = np.random.default_rng(seed)
        npat, nv = 250, 4
        vid = ["V{:05d}".format(i) for i in range(npat * nv)]
        pid = np.repeat(["P{:04d}".format(i) for i in range(npat)],
                        nv)
        los = np.round(rng.lognormal(1.0, 0.6, npat * nv), 2)
        sev = np.round(rng.normal(5, 1.5, npat * nv), 2)
        vis = pd.DataFrame({"visit_id": vid, "person_id": pid,
                            "length_of_stay": los,
                            "severity": sev})
        rows = []
        for i, v in enumerate(vid):
            for _ in range(int(rng.integers(1, 4))):
                drv = los[i] if shared else sev[i]
                rows.append((
                    v,
                    round(float(2.0 * los[i] + rng.normal(0, 1.2)), 3),
                    round(float(1.5 * drv + rng.normal(0, 1.0)), 3)))
        lab = pd.DataFrame(rows, columns=["visit_id", "lab_value",
                                          "lab_b"])
        return vis, lab

    def _xc2(lab, vis, c, pcol):
        m = lab.merge(vis[["visit_id", pcol]], on="visit_id",
                      how="inner")
        a = pd.to_numeric(m[c], errors="coerce")
        b = pd.to_numeric(m[pcol], errors="coerce")
        k = a.notna() & b.notna()
        if k.sum() < 30:
            return None
        return float(a[k].corr(b[k], method="spearman"))

    def _fit2(vis, lab):
        frames = {"visits": vis, "labs": lab}
        found = detect(frames)
        cps = cross_couplings(frames, found)
        gen = {}
        for nm, fr, gb in (("visits", vis, "person_id"),
                           ("labs", lab, "visit_id")):
            bp = _Bx.build(fr, _dx(fr, group_by=gb, seed=1),
                           group_by=gb)
            gen[nm] = _gx(bp, n_patients=fr[gb].nunique(),
                          seed=3)
        return found, cps, gen

    _v1, _l1 = _two(True)
    _found1, _cps1, _gen1 = _fit2(_v1, _l1)
    # A BLOCK ABOUT TWO COUPLINGS NEEDS TWO, or every check
    # under it passes on a shape it was not written for.
    check("the rank-one fixture genuinely offers TWO couplings "
          "on one link, so the joint solve has something to be "
          "joint about ({} found)".format(len(_cps1)),
          len(_cps1) >= 2)
    _rep1 = []
    _o1 = assign_keys(_gen1, _found1, seed=5, couplings=_cps1,
                      report_into=_rep1)
    check("...and both children ride the SAME parent, so the "
          "target structure has rank one and `carried` says so "
          "at {:.2f}".format(_rep1[0]["carried"]),
          _rep1[0]["carried"] >= MT.JOINT_MIN_CARRIED)
    _a1 = _xc2(_o1["labs"], _o1["visits"], "lab_value",
               "length_of_stay")
    _b1 = _xc2(_o1["labs"], _o1["visits"], "lab_b",
               "length_of_stay")
    _sa = _xc2(_l1, _v1, "lab_value", "length_of_stay")
    _sb = _xc2(_l1, _v1, "lab_b", "length_of_stay")
    check("...and BOTH land: lab_value {:+.3f} against a source "
          "{:+.3f}, lab_b {:+.3f} against {:+.3f} - one "
          "assignment carrying two couplings, which is the "
          "whole of MT-2".format(_a1, _sa, _b1, _sb),
          abs(_a1 - _sa) < 0.15 and abs(_b1 - _sb) < 0.15)

    # THE PAIRED CONTROL ARM. Same fixture, same seed, only the
    # coupling set differs - which is the comparison shape two
    # reverted fidelity tiers lacked.
    _best1 = max(_cps1, key=lambda c: abs(c["rho"]))
    _o1b = assign_keys(_gen1, _found1, seed=5,
                       couplings=[_best1])
    _a1b = _xc2(_o1b["labs"], _o1b["visits"], "lab_value",
                "length_of_stay")
    _b1b = _xc2(_o1b["labs"], _o1b["visits"], "lab_b",
                "length_of_stay")
    _err_joint = abs(_a1 - _sa) + abs(_b1 - _sb)
    _err_one = abs(_a1b - _sa) + abs(_b1b - _sb)
    check("...and the joint solve BEATS the strongest-only arm "
          "it replaces, paired at one seed: total error {:.3f} "
          "against {:.3f}, because enforcing one coupling "
          "leaves the second at {:+.3f} where its source is "
          "{:+.3f}".format(_err_joint, _err_one, _b1b, _sb),
          _err_joint < _err_one / 2.0)
    check("...and no per-table number can degrade, because the "
          "assignment is a PERMUTATION of keys: every child "
          "column holds the identical multiset it was "
          "generated with",
          all(sorted(_o1["labs"][c].astype(str).tolist())
              == sorted(_gen1["labs"][c].astype(str).tolist())
              for c in ("lab_value", "lab_b")))

    _v2, _l2 = _two(False)
    _found2, _cps2, _gen2 = _fit2(_v2, _l2)
    _rep2 = []
    _o2 = assign_keys(_gen2, _found2, seed=5, couplings=_cps2,
                      report_into=_rep2)
    _sa2 = _xc2(_l2, _v2, "lab_value", "length_of_stay")
    _sb2 = _xc2(_l2, _v2, "lab_b", "severity")
    check("two couplings on DIFFERENT parents are not jointly "
          "reachable at all - `carried` reads {:.2f} against a "
          "bar of {:.2f}, and the solve REFUSES the joint path "
          "rather than spending its one degree of freedom and "
          "calling it done"
          .format(_rep2[0]["carried"], MT.JOINT_MIN_CARRIED),
          _rep2[0]["carried"] < MT.JOINT_MIN_CARRIED
          and _rep2[0].get("joint") is False)
    _b2 = _xc2(_o2["labs"], _o2["visits"], "lab_b", "severity")
    check("...and it enforces the STRONGEST, which lands: {} at "
          "{:+.3f} against a source {:+.3f}"
          .format(_rep2[0].get("enforced", "(none)"), _b2, _sb2),
          abs(_b2 - _sb2) < 0.15)
    check("...and it NAMES what cannot come with it, rather "
          "than reporting a count a reader cannot act on: {}"
          .format(_rep2[0].get("not_reachable", "(none)")),
          "lab_value ~ length_of_stay"
          in _rep2[0].get("not_reachable", []))
    check("...and it says WHY in the record, naming the one "
          "foreign key as the reason rather than leaving a "
          "refusal to be read as a bug",
          "rank" in _rep2[0].get("why", "")
          and "one foreign key"
          in _rep2[0].get("why", ""))
    # A MAGIC KEY IN THE RETURNED DICT would hand a list to
    # anything iterating those values as tables, and fail
    # somewhere else entirely - `check_integrity` does exactly
    # that. The report has its own channel.
    check("...and the report travels by its own parameter, so "
          "every value in the returned mapping is still a "
          "table",
          all(isinstance(v, pd.DataFrame) for v in _o2.values()))

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
