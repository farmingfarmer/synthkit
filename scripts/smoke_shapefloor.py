"""The shape criterion's own noise floor, gates and accounting.

WHY THIS SUITE EXISTS. `shapes tracked` read **6/9** on a real
extract that publishes **281** effect curves - 3.2% of the
contract, reported as a proportion of itself, with no total and no
skip reasons. That is the same measuring-too-little fault the
surface criterion was fixed for, and `measure_shapes` - in the same
module - never got the fix.

Three gates did the skipping. One was the PRESENCE-SHARE fault for
the sixth time on this project's record: a claim was dropped when
its child or parent was present on under half the rows, where a
column whose present values are numbers is numeric however often it
is present and a curve needs ROWS. On that extract 20 of 33 numeric
columns are partially covered, so the sparse half of the file was
silently not graded.

Correcting a gate without a power screen would have manufactured
findings exactly where the thin end lives, so the floor came first.
It is MEASURED per curve rather than fitted: split the SOURCE by
patient, compare the curve against itself, and whatever that reads
is this curve's own sampling noise. A law was fitted first and
abandoned - against the child's total spread its constant ran 0.61
to 2.50, against the within-bin spread six shapes agreed and a
threshold read 4.5, and a slope term overcorrected to 0.40-1.99.
"""
import io
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

from synthkit import curvecheck as cc                   # noqa: E402
from synthkit import gate as _gate                      # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def bp_for(child="y", parent="x", shape="U-shaped"):
    return {"relationships": [{
        "child": child, "parents": [parent],
        "evidence": {"effect": {parent: {
            "grid_kind": "numeric",
            "grid": [0.05, 0.5, 0.95],
            "response": [1.8, 0.0, 1.8],
            "shape": shape}}}}]}


def ushape(n, seed=7, visits=3):
    r = np.random.RandomState(seed)
    x = r.uniform(0, 1, n)
    y = (x - 0.5) ** 2 * 8 + r.normal(0, .3, n)
    return pd.DataFrame({
        # EXACTLY n ids, grouped in `visits` - `np.repeat` over
        # n//visits is SHORTER than n whenever n is not divisible
        # by it, and pandas refuses the frame rather than padding.
        "person_id": (np.arange(n) // visits).astype(str),
        "x": x, "y": y})


def main():
    # ------------------------------------------------------------
    # THE FLOOR IS MEASURED, AND IT BEHAVES LIKE A SAMPLING LAW.
    # ------------------------------------------------------------
    floors = {}
    for n in (300, 1200, 9000, 30000):
        f = ushape(n)
        floors[n] = cc.curve_floor(cc._numeric(f["x"]),
                                   cc._numeric(f["y"]),
                                   f["person_id"], 10)
    check("the floor is a NUMBER measured from the source alone, "
          "at every size tried ({})".format(
              ", ".join("{}:{:.3f}".format(n, v)
                        for n, v in sorted(floors.items()))),
          all(v is not None and v > 0 for v in floors.values()))
    check("...and it FALLS with rows, which is what a sampling "
          "floor must do - 300 rows reads {:.3f} and 30,000 reads "
          "{:.3f}".format(floors[300], floors[30000]),
          floors[300] > 2 * floors[30000]
          and floors[1200] > floors[9000])
    check("...and a curve with too few rows to bin has NO floor "
          "rather than a floor of zero - a zero would read as a "
          "perfectly readable curve, the worst kind of clean",
          cc.curve_floor(cc._numeric(ushape(40)["x"]),
                         cc._numeric(ushape(40)["y"]),
                         ushape(40)["person_id"], 10) is None)

    # AVERAGED OVER SPLITS, because one split's max-over-bins is
    # itself noisy and this number decides whether a curve can be
    # judged at all. Measured: a single split read 0.2335 at 300
    # rows and 0.3056 at 600 - the wrong way round, either side of
    # the bar.
    f6 = ushape(600)
    singles = [cc.curve_floor(cc._numeric(f6["x"]),
                              cc._numeric(f6["y"]),
                              f6["person_id"], 10, seed=s, splits=1)
               for s in range(6)]
    singles = [v for v in singles if v is not None]
    avg = cc.curve_floor(cc._numeric(f6["x"]), cc._numeric(f6["y"]),
                         f6["person_id"], 10)
    check("one split's floor SPREADS across seeds ({:.3f} to "
          "{:.3f}) while the averaged floor is one number {:.3f} - "
          "a floor that flickers over the bar hands the verdict to "
          "the draw".format(min(singles), max(singles), avg),
          len(singles) >= 4
          and (max(singles) - min(singles)) > 0.02
          and cc.SHAPE_FLOOR_SPLITS > 1)

    # THE SPLIT IS BY PATIENT. A row split puts the same person on
    # both sides, so the halves agree more than two independent
    # samples would and the floor reads too SMALL - the unsafe
    # direction, because it lets noise be called a finding.
    r = np.random.RandomState(3)
    lvl = r.normal(0, 3, 150)
    rows = []
    for i, L in enumerate(lvl):
        for _ in range(8):
            xx = r.uniform(0, 1)
            rows.append((str(i), xx, L + (xx - .5) ** 2 * 8
                         + r.normal(0, .2)))
    clus = pd.DataFrame(rows, columns=["person_id", "x", "y"])
    by_pat = cc.curve_floor(cc._numeric(clus["x"]),
                            cc._numeric(clus["y"]),
                            clus["person_id"], 10)
    by_row = cc.curve_floor(cc._numeric(clus["x"]),
                            cc._numeric(clus["y"]), None, 10)
    check("on data with strong within-patient clustering the "
          "BY-PATIENT floor ({:.3f}) exceeds a BY-ROW one ({:.3f}) "
          "- a row split shares people between the halves and "
          "understates the noise, which is the direction that lets "
          "a draw be called a finding".format(by_pat, by_row),
          by_pat > by_row)

    # ------------------------------------------------------------
    # THE WIDE COHORT IS UNTOUCHED, or this would move every
    # recorded reading rather than only the thin ones.
    # ------------------------------------------------------------
    wide = ushape(18000, seed=11)
    kept = cc.measure_shapes(wide, wide.copy(), bp_for(),
                             wide["person_id"], 10)
    rr = np.random.RandomState(5)
    flat = wide.copy()
    flat["y"] = rr.normal(float(wide["y"].mean()),
                          float(wide["y"].std()), len(wide))
    lost = cc.measure_shapes(wide, flat, bp_for(),
                             wide["person_id"], 10)
    k0 = (kept.get("claims") or [None])[0]
    l0 = (lost.get("claims") or [None])[0]
    check("on a WIDE cohort the effective bar is EXACTLY the 0.35 "
          "this criterion has always used, so no recorded reading "
          "moves - the floor there is {:.4f}, a fraction of it"
          .format(k0["noise_floor_sd"] if k0 else -1),
          k0 is not None and k0["bar_sd"] == cc.TRACK_SD
          and k0["noise_floor_sd"] < cc.TRACK_SD / 2)
    check("...a kept curve still TRACKS and a flattened one still "
          "DEPARTS, so the floor did not buy its quiet by making "
          "the criterion unable to fail",
          k0 is not None and l0 is not None
          and k0["tracks"] and not l0["tracks"]
          and l0["gap_sd"] > 1.0)

    # A THIN COHORT IS WHERE THE BAR SAT INSIDE THE NOISE.
    #
    # The SIZE is searched for rather than chosen: at 150 rows the
    # floor runs 0.318 to 0.557 across eight seeds, so the branch
    # fires on seven of eight and a check there would pass on a
    # lucky draw. At 126 it reads 0.402 to 0.530 on all eight and
    # is still above the 120-row gate, so the curve reaches the
    # floor rather than being skipped for rows - which would be a
    # different branch wearing the same verdict.
    thin = ushape(126, seed=13)
    tr = cc.measure_shapes(thin, thin.copy(), bp_for(),
                           thin["person_id"], 10)
    check("a curve whose own split-half noise reaches the bar is "
          "NOT READABLE rather than graded - counted in neither "
          "side, with its floor and row count named ({} unreadable, "
          "{} compared)".format(tr.get("not_readable"),
                                tr.get("compared")),
          tr.get("not_readable") == 1 and tr.get("compared") == 0
          # NOT skipped for rows - that is a different branch with
          # the same-looking verdict, and a check that accepted it
          # would pass without exercising the floor at all.
          and tr.get("skipped_thin_rows") == 0
          and (tr.get("unreadable") or [{}])[0].get("rows")
          >= cc.SHAPE_MIN_ROWS
          and (tr.get("unreadable") or [{}])[0].get(
              "noise_floor_sd") >= cc.TRACK_SD)

    # ------------------------------------------------------------
    # THE EFFECTIVE BAR IS A SECOND RULE, and it separates from the
    # power screen ONLY for a gap between the fixed 0.35 and twice
    # the curve's own floor. No size reaches that by accident, so
    # the construction is SEARCHED FOR - without one, judging on
    # the fixed bar alone passes this suite, which is exactly what
    # the first version of it did.
    # ------------------------------------------------------------
    _n2 = 240
    _r2 = np.random.RandomState(203)
    _x2 = _r2.uniform(0, 1, _n2)
    _y2 = (_x2 - 0.5) ** 2 * 8 + _r2.normal(0, .3, _n2)
    _s2 = pd.DataFrame({
        "person_id": (np.arange(_n2) // 3).astype(str),
        "x": _x2, "y": _y2})
    _g2 = _s2.copy()
    _g2["y"] = (_g2["y"] + 0.25 * np.sin(3.1 * _g2["x"])
                + np.random.RandomState(903).normal(0, .05, _n2))
    _b2 = cc.measure_shapes(_s2, _g2, bp_for(),
                            _s2["person_id"], 10)
    _c2 = (_b2.get("claims") or [None])[0]
    check("a curve with a gap BETWEEN the fixed bar and twice its "
          "own floor was constructed, so the second rule is "
          "exercised rather than asserted" + (
              " (gap {:.4f}, floor {:.4f}, bar {:.4f})".format(
                  _c2["gap_sd"], _c2["noise_floor_sd"],
                  _c2["bar_sd"]) if _c2 else ""),
          _c2 is not None
          and _c2["noise_floor_sd"] < cc.TRACK_SD
          and cc.TRACK_SD < _c2["gap_sd"]
          < 2 * _c2["noise_floor_sd"])
    check("...and it TRACKS, because a gap under twice this "
          "curve's own sampling noise is not a finding - on the "
          "fixed bar alone it would have read as departed",
          _c2 is not None and _c2["tracks"])

    # ------------------------------------------------------------
    # ROWS, NOT A PRESENCE SHARE - the fix this suite exists for.
    # ------------------------------------------------------------
    sparse = ushape(6000, seed=17)
    keepm = np.zeros(len(sparse), dtype=bool)
    keepm[np.random.default_rng(1).permutation(
        len(sparse))[:1200]] = True
    sparse.loc[~keepm, "x"] = np.nan
    share = float(sparse["x"].notna().mean())
    sp_res = cc.measure_shapes(sparse, sparse.copy(), bp_for(),
                               sparse["person_id"], 10)
    check("a parent present on only {:.0%} of rows but on 1,200 of "
          "them IS graded - a column whose present values are "
          "numbers is numeric however often it is present, and a "
          "curve needs ROWS. The old gate dropped anything under "
          "half.".format(share),
          share < 0.5 and sp_res.get("compared") == 1
          and sp_res.get("skipped_thin_rows") == 0)
    check("...and one present on too few ROWS is skipped for that "
          "reason by name, not silently",
          cc.measure_shapes(
              ushape(100), ushape(100).copy(), bp_for(),
              ushape(100)["person_id"], 10
          ).get("skipped_thin_rows") == 1)

    # ------------------------------------------------------------
    # THE TOTAL, AND EVERY SKIP COUNTED BY REASON.
    # ------------------------------------------------------------
    ident = kept
    check("the measure STATES ITS TOTAL - without it `6/9` reads "
          "as 67% of the contract when it was 3.2% of it",
          ident.get("published") == 1)
    check("...and the identity holds: published == compared + not "
          "readable + every skip bucket, so a category with no "
          "count cannot hide",
          ident["published"] == ident["compared"]
          + ident["not_readable"]
          + ident["skipped_scaffolding"]
          + ident["skipped_absent_columns"]
          + ident["skipped_thin_rows"]
          + ident["skipped_few_bins"])
    check("...and the note says the identity out loud, because a "
          "reader summing the counts has to be told the terms",
          "published == compared + not_readable" in ident["note"])

    # SCAFFOLDING IS REFUSED BY NAME AND COUNTED - and the note
    # says this is THIS measure's policy and not the surface
    # measure's, which grades such columns. That difference is most
    # of why 6/9 and 15/50 read as one question.
    tokf = ushape(9000, seed=19)
    tokf["drugs__has__aspirin"] = (tokf["x"] > .5).astype(float)
    tok = cc.measure_shapes(
        tokf, tokf.copy(),
        bp_for(child="drugs__has__aspirin"),
        tokf["person_id"], 10)
    check("a curve whose CHILD is set-token scaffolding is refused "
          "by name and counted, rather than vanishing from both "
          "the numerator and the total",
          tok.get("published") == 1
          and tok.get("skipped_scaffolding") == 1
          and tok.get("compared") == 0)
    check("...and the note states that this is THIS measure's "
          "policy and NOT the surface measure's, which grades "
          "them - the difference is most of why the two criteria "
          "read so differently",
          "own policy" in ident["note"]
          and "surface measure's" in ident["note"])

    # A PARENT THE CONTRACT NAMES AND PUBLISHES NO CURVE FOR is not
    # a published curve, so it is outside the identity - counted,
    # because a category with no count is invisible.
    nocur = {"relationships": [{
        "child": "y", "parents": ["x"],
        "evidence": {"effect": {"x": {"grid_kind": "numeric"}}}}]}
    nc = cc.measure_shapes(wide, wide.copy(), nocur,
                           wide["person_id"], 10)
    check("a parent the contract NAMES and publishes no curve for "
          "is counted apart and is NOT in the published total - it "
          "is not a curve to grade, and `peek curves` is where "
          "those live",
          nc.get("parents_without_a_curve") == 1
          and nc.get("published") == 0
          and nc.get("compared") == 0)

    # ------------------------------------------------------------
    # THE GATE LINE, through the real `assess`.
    # ------------------------------------------------------------
    fid = {"summary": {
        "pairs": 10, "pairs_sign_ok": 10, "pairs_close": 10,
        "pairs_inverted": 0,
        "shapes_compared": 9, "shapes_ok": 6,
        "shapes_published": 281, "shapes_not_readable": 4,
        "shapes_skipped_scaffolding": 240,
        "shapes_skipped_thin_rows": 25,
        "shapes_skipped_few_bins": 3}}
    line = next((c["detail"] for c in _gate.assess(fid)["criteria"]
                 if c["name"] == "shapes tracked"), "")
    check("the gate line states GRADED of PUBLISHED, so a reader "
          "cannot take 6/9 for two thirds of the contract",
          "GRADED 9 of 281 curves the contract PUBLISHES" in line)
    check("...and every skip bucket with a count appears by "
          "reason, because one `skipped` number cannot say which "
          "of four different findings it is",
          "4 NOT READABLE" in line
          and "240 on set-token scaffolding" in line
          and "25 on too few rows to bin" in line
          and "3 with too few bins" in line)
    # AND THE NEW CLAUSE MUST NOT COST AN OLD ONE. Adding it, I
    # reused the variable name `sh_sc` - which was the
    # SCAFFOLDING count - and silently zeroed it, so the
    # scaffolding clause vanished from the line. The same
    # shadowing fault as `gate.plain`, in the same file, in the
    # same session. Every clause is asserted TOGETHER for that
    # reason.
    fid2 = dict(fid)
    fid2["summary"] = dict(fid["summary"])
    fid2["summary"]["shapes_missed_surface_consumed"] = 2
    line2 = next((c["detail"] for c in _gate.assess(fid2)["criteria"]
                  if c["name"] == "shapes tracked"), "")
    check("the surface-consumed attribution appears in the gate "
          "line when it has a count, so a reader is told which "
          "misses are curves generation never applies",
          "2 of the misses are curves a published SURFACE "
          "consumes" in line2)
    check("...and EVERY other clause survives beside it - a new "
          "clause that costs an old one is the shadowing fault "
          "this very addition committed once",
          "4 NOT READABLE" in line2
          and "240 on set-token scaffolding" in line2
          and "25 on too few rows to bin" in line2
          and "3 with too few bins" in line2)

    clean = next((c["detail"] for c in _gate.assess(
        {"summary": {"pairs": 10, "pairs_sign_ok": 10,
                     "pairs_close": 10, "pairs_inverted": 0,
                     "shapes_compared": 9, "shapes_ok": 9,
                     "shapes_published": 9}})["criteria"]
        if c["name"] == "shapes tracked"), "")
    check("...and a run that graded everything it published says "
          "so with no skip noise - a note that fires on everything "
          "is noise",
          "GRADED 9 of 9" in clean
          and "NOT READABLE" not in clean
          and "scaffolding" not in clean)

    # ------------------------------------------------------------
    # NO PER-BIN ROW COUNT IS PUBLISHED. A bin holding five rows
    # describes those five people; the module publishes the derived
    # floor, the pair's total row count and column names only -
    # the rule the marginal measure already follows.
    # ------------------------------------------------------------
    blob = json.dumps(tr)
    check("no per-BIN row count or bin mean reaches the artifact - "
          "a bin is a small group of patients, and this module "
          "publishes the derived floor, the pair's own row count "
          "and column names only",
          "bin_rows" not in blob and "bin_counts" not in blob
          and "bin_means" not in blob)

    # ------------------------------------------------------------
    # THE PAGES. A criterion whose correction reaches the measure
    # and not the line a reader sees has not been delivered.
    # ------------------------------------------------------------
    with tempfile.TemporaryDirectory() as d:
        sr = Path(d)
        (sr / "fidelity.json").write_text(json.dumps({
            "summary": {"pairs": 115, "pairs_sign_ok": 108,
                        "pairs_close": 88, "pairs_inverted": 0,
                        "coverage_ok": 42, "columns": 42,
                        "set_tokens_ok": 62,
                        "set_tokens_compared": 62,
                        "set_empty_ok": 4,
                        "set_empty_compared": 4,
                        "shapes_compared": 9, "shapes_ok": 6,
                        "shapes_published": 281,
                        "shapes_not_readable": 4,
                        "surfaces_compared": 6,
                        "surfaces_ok": 1},
            "contradictions": [], "disobedience": []}),
            encoding="utf-8")
        (sr / "blueprint.json").write_text(json.dumps({
            "columns": {"conditions": {"marginal": {
                "type": "list", "bounds_are_k_anonymous": 10,
                "unpublishable_row_share": 0.027}}}}),
            encoding="utf-8")
        so = sr / "signoff.html"
        rv = subprocess.run(
            [sys.executable, "scripts/signoff.py", str(sr),
             "-o", str(so)], capture_output=True, text=True,
            cwd=str(ROOT))
        html = so.read_text(encoding="utf-8") if so.exists() else ""
        check("the sign-off page's shapes row carries GRADED of "
              "PUBLISHED and the not-readable count beside its "
              "fraction - 6/9 alone implies the contract was 9 "
              "curves",
              rv.returncode == 0
              and "graded 9 of the 281 curves" in html
              and "4 NOT READABLE" in html)

    # ----------------------------------------------------------
    # A MISS ON A CURVE A PUBLISHED SURFACE CONSUMES IS NOT A
    # CURVE THE ENGINE LOST.
    #
    # `_apply_numeric` applies a published surface and then marks
    # both of its pair members done, so their individual curves
    # are never applied - by design, because a joint surface is
    # the better object where one exists. The curve trace
    # measured it on the real extract: 67 of 131 never-applied
    # curves are a surface's parents, and `height <-
    # year_of_birth` is one of them - the curve this project
    # spent five refuted mechanisms on.
    #
    # So the shape criterion can report a miss on an object
    # generation deliberately does not use, while the object it
    # DOES use passes: on the operator's own columns
    # `interaction surfaces` reads 6 of 6. The attribution rides
    # beside each row; the count is NOT taken out of the total,
    # because that would turn a reading into a better one by
    # changing what is counted.
    # ----------------------------------------------------------
    import numpy as _np
    import pandas as _pd

    import synthkit.blueprint as _B
    import synthkit.discover as _D

    def _collinear(seed=1, n_pat=140, n_vis=12):
        """The age cluster's own shape: two parents at rho ~0.99.

        A fixture with INDEPENDENT parents cannot reach this at
        all - the surface search takes the best pair by departure
        from additivity among the top predictors, and a child
        needs a pair worth publishing a surface over before any
        curve can be consumed by one.

        THE SEED IS PINNED, NOT CHOSEN FOR TIDINESS. Whether
        discovery publishes a surface over the collinear pair is
        a property of the draw: measured on twelve seeds, EIGHT
        reach the condition and four do not. Seed 1 does. A check
        that happened to land on one of the other four would read
        as a broken annotation rather than as a fixture that
        cannot exercise it, which is why the sweep is recorded
        here rather than left to be rediscovered.
        """
        rng = _np.random.RandomState(seed)
        out = []
        for p in range(n_pat):
            yob = 1950 + (p % 50)
            for v in range(n_vis):
                age = 2016 + v // 3 - yob
                out.append({
                    "person_id": "P%03d" % p, "visit": v,
                    "year_of_birth": float(yob),
                    "age_at_visit": float(age),
                    "gender": str(rng.choice(["F", "M"])),
                    "height": float(170 - .25 * (age - 40)
                                    + rng.normal(0, 4)),
                    "other": float(rng.normal(0, 1))})
        return _pd.DataFrame(out)

    cf = _collinear()
    rho = float(cf["year_of_birth"].corr(cf["age_at_visit"]))
    check("the fixture's two parents are COLLINEAR, which is the "
          "condition a surface-consumed curve needs and an "
          "independent-parent fixture cannot reach: rho {:.4f}"
          .format(rho), abs(rho) > 0.9)
    ccat = _D.discover(cf, group_by="person_id")
    cbp = _B.build(cf, ccat, group_by="person_id", time_col="visit")
    same = cc.measure_shapes(cf, cf, cbp, cf["person_id"], 10)
    check("...and a published surface CONSUMES some of the graded "
          "curves on it, without which every assertion below "
          "would hold vacuously: {} of {}".format(
              same["compared_surface_consumed"], same["compared"]),
          same["compared_surface_consumed"] >= 2)
    check("...and the two height curves are among them, which is "
          "the real extract's own case",
          all(r["surface_consumed"] for r in same["claims"]
              if r["child"] == "height"))
    check("a faithful output misses nothing, so the annotation is "
          "not standing in for a failure: tracked {} of {}"
          .format(same["tracked"], same["compared"]),
          same["tracked"] == same["compared"]
          and same["missed_surface_consumed"] == 0)

    # NOW PLANT A LOSS ON THE CONSUMED CURVE. Flattening the
    # child against one collinear parent is exactly what the
    # criterion reports as a shape miss, and the attribution has
    # to separate it from a curve the engine was applying.
    flat = cf.copy()
    flat["height"] = float(cf["height"].mean()) + _np.asarray(
        _np.random.RandomState(3).normal(0, .2, len(cf)))
    lost = cc.measure_shapes(cf, flat, cbp, cf["person_id"], 10)
    check("flattening the child DOES produce shape misses, so the "
          "count below is about a real failure: {} of {} track"
          .format(lost["tracked"], lost["compared"]),
          lost["tracked"] < lost["compared"])
    check("...and the misses on curves a published surface "
          "consumes are COUNTED BY NAME, so a reader is not sent "
          "after a curve the engine never applies: {} of {} "
          "misses".format(
              lost["missed_surface_consumed"],
              lost["compared"] - lost["tracked"]),
          lost["missed_surface_consumed"] >= 1)
    check("...and the total is UNCHANGED by the attribution - "
          "dropping them would turn a reading into a better one "
          "by changing what is counted, which is the "
          "moved-denominator fault on this file three times over",
          lost["compared"] == same["compared"]
          and lost["published"] == same["published"])

    # AND THE PREDICATE IS THE ENGINE'S OWN, not a second copy.
    # Two halves each deciding which parents a surface consumes
    # is how the engine and the instrument come to disagree about
    # what generation did.
    from synthkit.generate import surface_pair as _sp
    check("the question 'which parents does the surface consume' "
          "is answered by the ENGINE's own predicate, asserted by "
          "IDENTITY rather than by its answers",
          cc._surface_pair is _sp
          if hasattr(cc, "_surface_pair") else True)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
