"""The surface criterion's own noise floor, and the ceiling tool.

WHY THIS SUITE EXISTS. `interaction surfaces` read 14 of 53 on the
real extract and the record named the suspect as a pair member
trimmed to order the graph. Measured on run_tail_ctl_current, that
suspect is REFUTED: ZERO trimmed edges, ZERO lost lag parents, ZERO
surfaces whose pair member is missing from its parents, and all 75
published surfaces carry a grid with both members among the parents -
so every one passes `_apply_numeric`'s gate and is applied at full
strength.

Which moved the question to the MEASURE, and the measure had a bar
inside its own noise: the gap is the WORST of nine conditional-mean
cells and a cell was kept at FIVE rows. Measured by splitting a
cohort in half BY PATIENT - nothing generated, so every gap is
sampling noise - a faithful output reads 0.38 to 0.51 sd at 360 rows
per half against a 0.35 bar.

This suite pins the law, the screen, and the tool that reads it.
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
import surface_ceiling as SC                            # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


BP = {"columns": {},
      "relationships": [{"child": "y", "parents": ["a", "b"],
                         "evidence": {"interaction": {
                             "pair": ["a", "b"],
                             "grid_a": [0.0, 1.0],
                             "grid_b": [0.0, 1.0],
                             "response": [[0.0, 1.0],
                                          [1.0, 0.0]]}}}]}


def frame(n, seed=7, noise=0.15, strength=2.0):
    r = np.random.RandomState(seed)
    a = r.uniform(0, 1, n)
    b = r.uniform(0, 1, n)
    y = (((a > .5) ^ (b > .5)).astype(float) * strength
         + r.normal(0, noise, n))
    return pd.DataFrame({
        "person_id": np.repeat(np.arange(n // 3), 3).astype(str),
        "a": a, "b": b, "y": y})


def main():
    # ------------------------------------------------------------
    # THE LAW, ASSERTED AGAINST THE NOISE IT DESCRIBES.
    #
    # `surface_floor` is a PURE function on two cell occupancies,
    # so the constant can be held to the thing it was fitted to
    # rather than to whatever a fixture happens to produce. A
    # constant that drifts away from the measured noise would make
    # every screen decision rest on a number nobody re-ran.
    # ------------------------------------------------------------
    obs = []
    for s in range(12):
        f = frame(1800, seed=300 + s)
        A = f.iloc[:900].reset_index(drop=True)
        B = f.iloc[900:].reset_index(drop=True)
        res = cc.measure_surfaces(A, B, BP, A["person_id"], 10)
        for r in (res.get("surfaces") or []):
            obs.append((r["gap_sd"], r["cells_compared"]))
        for r in (res.get("unreadable") or []):
            # An unreadable surface still carries the floor, and the
            # floor is what is being checked here.
            obs.append((None, r["cells_compared"]))
    gaps = [g for g, _ in obs if g is not None]
    check("the probe produced readings at all, so the law below is "
          "checked against measured noise rather than an empty "
          "list ({} draws)".format(len(obs)), len(obs) >= 10)
    pred = SC and cc.surface_floor(900 // 9, 900 // 9)
    check("the PREDICTED floor at 100 rows per cell per side is "
          "{:.4f} sd, and the OBSERVED mean worst gap on a "
          "FAITHFUL output at that occupancy is {:.4f} - the "
          "constant describes the noise it was fitted to, within a "
          "factor of two in either direction"
          .format(pred, float(np.mean(gaps)) if gaps else -1.0),
          bool(gaps) and 0.5 <= pred / float(np.mean(gaps)) <= 2.0)
    check("the floor FALLS with occupancy, which is the whole "
          "shape of a sampling law - 100 rows reads {:.4f} and "
          "2000 reads {:.4f}".format(
              cc.surface_floor(100, 100), cc.surface_floor(2000,
                                                           2000)),
          cc.surface_floor(100, 100)
          > 3 * cc.surface_floor(2000, 2000))
    check("...and it is SYMMETRIC in the two sides and dominated "
          "by the THINNER one, because that is what "
          "1/n_src + 1/n_gen says - 50 against 5000 reads close "
          "to 50 against 50",
          abs(cc.surface_floor(50, 5000)
              - cc.surface_floor(50, 50)) < 0.5
          * cc.surface_floor(50, 50)
          and cc.surface_floor(50, 5000)
          == cc.surface_floor(5000, 50))
    check("an EMPTY cell has no floor rather than a floor of zero "
          "- a zero would read as a perfectly readable surface, "
          "which is the worst kind of clean",
          cc.surface_floor(0, 100) == float("inf")
          and cc.surface_floor(100, 0) == float("inf"))

    # ------------------------------------------------------------
    # THE SCREEN. A wide cohort must be UNTOUCHED, or this would
    # move every recorded reading rather than only the thin ones.
    # ------------------------------------------------------------
    wide = frame(18000, seed=11)
    kept = cc.measure_surfaces(wide, wide.copy(), BP,
                               wide["person_id"], 10)
    r = np.random.RandomState(5)
    flat = wide.copy()
    flat["y"] = r.normal(float(wide["y"].mean()),
                         float(wide["y"].std()), len(wide))
    lost = cc.measure_surfaces(wide, flat, BP,
                               wide["person_id"], 10)
    k0 = (kept.get("surfaces") or [None])[0]
    l0 = (lost.get("surfaces") or [None])[0]
    check("on a WIDE cohort the effective bar is EXACTLY the 0.35 "
          "the criterion has always used, so no recorded reading "
          "moves - the floor there is {:.4f}, a fraction of it"
          .format(k0["noise_floor_sd"] if k0 else -1.0),
          k0 is not None and k0["bar_sd"] == cc.TRACK_SD
          and k0["noise_floor_sd"] < cc.TRACK_SD / 2.0)
    check("...a kept surface still TRACKS and a surface "
          "generation lost still DEPARTS, so the screen did not "
          "buy its quiet by making the criterion unable to fail",
          k0 is not None and l0 is not None
          and k0["tracks"] and not l0["tracks"]
          and l0["gap_sd"] > 0.5)
    check("...and nothing on a wide cohort is unreadable",
          kept.get("not_readable") == 0
          and lost.get("not_readable") == 0)

    # A THIN COHORT IS WHERE THE BAR SAT INSIDE THE NOISE. Both
    # sides are REAL data here - one half of a frame against the
    # other - so a FINDING would be manufactured from the draw.
    thin = frame(720, seed=13)
    A = thin.iloc[:360].reset_index(drop=True)
    B = thin.iloc[360:].reset_index(drop=True)
    tres = cc.measure_surfaces(A, B, BP, A["person_id"], 10)
    check("a FAITHFUL output on a thin cohort is NOT READABLE "
          "rather than a finding - counted in neither tracked nor "
          "compared, with its floor named ({} unreadable, {} "
          "compared)".format(tres.get("not_readable"),
                             tres.get("compared")),
          tres.get("not_readable") == 1
          and tres.get("compared") == 0
          and (tres.get("unreadable") or [{}])[0].get(
              "noise_floor_sd", 0) >= cc.TRACK_SD)
    check("...and the identity still holds with the third term, so "
          "nothing moved from one bucket to another unnoticed",
          tres["published"] == tres["compared"]
          + tres["not_readable"]
          + tres["skipped_token_columns"]
          + tres["skipped_absent_columns"]
          + tres["skipped_thin_data"])
    check("the note SAYS a surface can be not readable and states "
          "the identity, because a reader summing the counts has "
          "to be told the term exists",
          "NOT READABLE" in tres["note"]
          and "not_readable" in tres["note"])
    check("...and NO per-cell ROW COUNT is published - a 3x3 cell "
          "is a small group of patients, and this module publishes "
          "distances and column names only",
          all("rows" not in k for u in (tres.get("unreadable") or [])
              for k in u)
          and all("rows" not in k
                  for row in (tres.get("surfaces") or [])
                  for k in row))

    # ------------------------------------------------------------
    # THE EFFECTIVE BAR, which is a SECOND rule and separates from
    # the screen only for a reading BETWEEN 0.35 and twice the
    # floor. No occupancy reaches that by accident, so it is
    # SEARCHED FOR - the same way the marginal measure's own
    # effective bar had to be.
    # ------------------------------------------------------------
    found = None
    for s in range(60):
        f = frame(2160, seed=700 + s)
        A2 = f.iloc[:1080].reset_index(drop=True)
        B2 = f.iloc[1080:].reset_index(drop=True)
        res2 = cc.measure_surfaces(A2, B2, BP, A2["person_id"], 10)
        for row in (res2.get("surfaces") or []):
            fl = row["noise_floor_sd"]
            if (cc.TRACK_SD / 2.0 < fl < cc.TRACK_SD
                    and cc.TRACK_SD < row["gap_sd"] < 2.0 * fl):
                found = row
                break
        if found:
            break
    check("a reading BETWEEN the fixed bar and twice the "
          "surface's own floor was found, so the second rule is "
          "exercised rather than asserted - without one, judging "
          "on the fixed bar alone passes this suite" + (
              " (gap {:.4f}, floor {:.4f}, bar {:.4f})".format(
                  found["gap_sd"], found["noise_floor_sd"],
                  found["bar_sd"]) if found else ""),
          found is not None)
    if found:
        check("...and that surface TRACKS, because a gap under "
              "twice its own sampling noise is not a finding - on "
              "the fixed bar alone it would have read as departed",
              found["tracks"] and found["gap_sd"] > cc.TRACK_SD)

    # ------------------------------------------------------------
    # THE PER-CELL RULE, ON PLANTED CELL OCCUPANCIES.
    #
    # No frame can exercise this exactly: `grid_means` keeps a cell
    # at five rows and drops it below, so the thin-but-kept band the
    # rule turns on - 5 to about 86 rows per side - cannot be
    # dialled through a quantile-binned fixture on demand. Six
    # constructions were tried and every one gave either nine fat
    # cells or seven, the corners vanishing rather than thinning.
    # So the decision is a PURE FUNCTION and is asserted directly,
    # which is both exact and what this project does with every
    # other load-bearing rule.
    # ------------------------------------------------------------
    FAT, THIN = 800, 20
    check("a well-populated surface that MATCHES tracks, and the "
          "bar it was judged against is exactly 0.35",
          cc.judge_cells({(0, 0): (1.0, FAT), (1, 1): (2.0, FAT)},
                         {(0, 0): (1.0, FAT), (1, 1): (2.0, FAT)},
                         1.0)[0] is True
          and cc.judge_cells(
              {(0, 0): (1.0, FAT)}, {(0, 0): (1.0, FAT)},
              1.0)[3] == cc.TRACK_SD)
    check("...and one that genuinely departs in a well-populated "
          "cell is DEPARTED",
          cc.judge_cells({(0, 0): (1.0, FAT)},
                         {(0, 0): (2.0, FAT)}, 1.0)[0] is False)
    check("a surface EVERY cell of which is too thin to judge is "
          "NOT READABLE - tracks is None rather than True or "
          "False, because neither would be a measurement",
          cc.judge_cells({(0, 0): (1.0, THIN), (1, 1): (1.0, THIN)},
                         {(0, 0): (9.0, THIN), (1, 1): (1.0, THIN)},
                         1.0)[0] is None)
    # THE CASE THAT SEPARATES THE TWO RULES, and the whole reason
    # for the refactor: a THIN cell with a huge noise gap beside a
    # FAT cell with a real departure. Judging the surface by its
    # largest gap and that cell's own power reads NOT READABLE and
    # loses the finding; judging each cell against its own noise
    # reads DEPARTED.
    _mix_s = {(0, 0): (1.0, THIN), (2, 2): (1.0, FAT)}
    _mix_g = {(0, 0): (9.0, THIN), (2, 2): (2.0, FAT)}
    _mix = cc.judge_cells(_mix_s, _mix_g, 1.0)
    check("a THIN cell with a 8.0 sd noise gap beside a FAT cell "
          "with a real 1.0 sd departure reads DEPARTED, not NOT "
          "READABLE - the old rule took the largest gap, found it "
          "in the unjudgeable cell, and lost the finding",
          _mix[0] is False and _mix[4] == 1)
    check("...and the gap it REPORTS is the readable cell's 1.0, "
          "not the thin cell's 8.0 - a gap taken from a cell "
          "nobody can judge is a number about the draw, and this "
          "figure is read off the artifact and quoted ({})"
          .format(_mix[1]),
          abs(_mix[1] - 1.0) < 1e-9)
    check("...and the thin cell cannot RESCUE a departure either: "
          "a fat cell that matches beside a thin cell with a huge "
          "gap TRACKS, because the thin cell is not evidence in "
          "either direction",
          cc.judge_cells({(0, 0): (1.0, THIN), (2, 2): (1.0, FAT)},
                         {(0, 0): (9.0, THIN), (2, 2): (1.0, FAT)},
                         1.0)[0] is True)
    check("a surface with NO overlapping cell has no power rather "
          "than a clean verdict",
          cc.judge_cells({(0, 0): (1.0, FAT)},
                         {(1, 1): (1.0, FAT)}, 1.0)[0] is None)
    check("the gap is in units of the child's own SPREAD, so the "
          "same absolute difference on a wider column is a smaller "
          "gap - a bar in raw units would mean a different thing "
          "per column",
          cc.judge_cells({(0, 0): (0.0, FAT)},
                         {(0, 0): (1.0, FAT)}, 1.0)[1]
          > cc.judge_cells({(0, 0): (0.0, FAT)},
                           {(0, 0): (1.0, FAT)}, 10.0)[1])

    # ------------------------------------------------------------
    # THE VERDICT IS A PURE FUNCTION, asserted on PLANTED numbers.
    # A fixture that cannot reach one branch leaves it untested
    # inside a tool whose entire output is the branch it chose.
    # ------------------------------------------------------------
    check("the ceiling's verdict is MEASURE-LIMITED when a "
          "faithful output misses any surface",
          SC.verdict(12, 53) == "measure-limited"
          and SC.verdict(52, 53) == "measure-limited")
    check("...SOUND when a faithful output tracks every one, "
          "which is the answer that points the work at generation",
          SC.verdict(53, 53) == "sound"
          and SC.verdict(1, 1) == "sound")
    check("...and NO-SURFACES is its own answer rather than being "
          "folded into either - an absent measurement is not a "
          "clean one",
          SC.verdict(0, 0) == "no-surfaces"
          and SC.verdict(0, None) == "no-surfaces")

    # THE SPLIT IS BY PATIENT, which is the one way this tool
    # could lie in the flattering direction: splitting ROWS puts
    # the same person on both sides, so the agreement measured is
    # the same records twice.
    f = frame(900, seed=3)
    rng = np.random.default_rng(1)
    A3, B3 = SC.halves(f, "person_id", rng)
    sa = set(A3["person_id"].astype(str))
    sb = set(B3["person_id"].astype(str))
    check("the halves share NO patient, so a patient's own rows "
          "never appear on both sides",
          not (sa & sb) and len(sa) > 10 and len(sb) > 10)
    check("...and every row is on exactly one side, so the split "
          "is a partition rather than a sample",
          len(A3) + len(B3) == len(f))

    # ------------------------------------------------------------
    # THE TOOL, END TO END THROUGH ITS OWN CLI, on a CSV on disk -
    # the reader is part of the suspect path and an in-memory
    # fixture cannot reach it.
    # ------------------------------------------------------------
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        frame(3000, seed=21).to_csv(p / "thin.csv", index=False)
        (p / "bp.json").write_text(json.dumps(BP), encoding="utf-8")
        rv = subprocess.run(
            [sys.executable, "scripts/surface_ceiling.py",
             "--src", str(p / "thin.csv"),
             "--blueprint", str(p / "bp.json"),
             "--splits", "2"],
            capture_output=True, text=True, cwd=str(ROOT))
        check("the ceiling tool runs end to end on a CSV on disk "
              "and reports a verdict (exit {})".format(rv.returncode),
              rv.returncode in (0, 3)
              and "FROM SAMPLING NOISE" in rv.stdout)
        check("...and it says NOTHING IS GENERATED, which is the "
              "whole basis of the reading - a reader who misses "
              "that would take it for a fidelity measurement",
              "nothing is generated here" in rv.stdout
              and "REAL data" in rv.stdout)
        check("...and every figure crosses in WORDS beside its "
              "digits, because the screen channel this project "
              "reads these on turns a zero into an at-sign",
              "zero" in rv.stdout or "one" in rv.stdout
              or "two" in rv.stdout)
        check("...and it REFUSES without a blueprint rather than "
              "measuring a ceiling over no surfaces and reporting "
              "it as a number",
              subprocess.run(
                  [sys.executable, "scripts/surface_ceiling.py",
                   "--src", str(p / "thin.csv")],
                  capture_output=True, text=True,
                  cwd=str(ROOT)).returncode == 2)
        check("...and it REFUSES a group column that is not there, "
              "because the split has to be by PATIENT",
              subprocess.run(
                  [sys.executable, "scripts/surface_ceiling.py",
                   "--src", str(p / "thin.csv"),
                   "--blueprint", str(p / "bp.json"),
                   "--group-by", "nope"],
                  capture_output=True, text=True,
                  cwd=str(ROOT)).returncode == 2)
        rvh = subprocess.run(
            [sys.executable, "scripts/surface_ceiling.py",
             "--help"], capture_output=True, text=True,
            cwd=str(ROOT))
        check("...and --help exits 0 rather than running a "
              "measurement nobody asked for, which this project "
              "has caught three scripts doing",
              rvh.returncode == 0 and "usage:" in rvh.stdout)

    # ------------------------------------------------------------
    # THE OWN/SCAFFOLDING SPLIT. On the real extract 35 of 35
    # FAILING surfaces had a derived token indicator as their
    # CHILD, while 6 of 6 surfaces whose child is a column the
    # operator receives tracked - so one percentage over both reads
    # as a fidelity gap on their data when it is search machinery
    # the engine never drew.
    # ------------------------------------------------------------
    _tok = "drugs__has__aspirin"
    _mix = frame(18000, seed=31)
    _mix[_tok] = (_mix["a"] > 0.5).astype(float)
    _bp2 = {"relationships": [
        {"child": "y", "parents": ["a", "b"],
         "evidence": {"interaction": {
             "pair": ["a", "b"], "grid_a": [0.0, 1.0],
             "grid_b": [0.0, 1.0],
             "response": [[0.0, 1.0], [1.0, 0.0]]}}},
        {"child": _tok, "parents": ["a", "b"],
         "evidence": {"interaction": {
             "pair": ["a", "b"], "grid_a": [0.0, 1.0],
             "grid_b": [0.0, 1.0],
             "response": [[0.0, 1.0], [1.0, 0.0]]}}}]}
    _sp = cc.measure_surfaces(_mix, _mix.copy(), _bp2,
                              _mix["person_id"], 10)
    check("the measure counts the surfaces on the operator's OWN "
          "columns apart from those whose CHILD is a derived token "
          "indicator, and the two sum to the whole ({} own + {} "
          "scaffold = {} compared)".format(
              _sp.get("compared_own"),
              _sp.get("compared_scaffold"), _sp.get("compared")),
          _sp.get("compared_own") == 1
          and _sp.get("compared_scaffold") == 1
          and (_sp["compared_own"] + _sp["compared_scaffold"]
               == _sp["compared"]))
    check("...and every surface carries WHOSE column it is about, "
          "so a reader of the artifact can make the same split "
          "rather than inferring it from a name",
          all("child_is_scaffolding" in r and "own_columns" in r
              for r in (_sp.get("surfaces") or []))
          and sum(1 for r in _sp["surfaces"]
                  if r["child_is_scaffolding"]) == 1)
    check("...and the own-column TRACKED count is counted over the "
          "own-column surfaces alone, not the whole set - a "
          "numerator from one population over a denominator from "
          "another is the fault this project records three times",
          _sp.get("tracked_own") <= _sp.get("compared_own"))

    # AND THE GATE LINE SAYS IT, through the real `assess`. A
    # measure that records the split while the line a reader
    # actually sees states one percentage has not delivered it -
    # the value-wired-to-nothing fault this project records more
    # than any other.
    from synthkit import gate as _gate
    _fid = {"summary": {
        "pairs": 10, "pairs_sign_ok": 10, "pairs_close": 10,
        "pairs_inverted": 0,
        "surfaces_compared": 50, "surfaces_ok": 15,
        "surfaces_own_compared": 6, "surfaces_own_ok": 6,
        "surfaces_scaffold": 44, "surfaces_not_readable": 3}}
    _v = _gate.assess(_fid)
    _line = next((c["detail"] for c in _v["criteria"]
                  if c["name"] == "interaction surfaces"), "")
    check("the gate's surfaces line states the own/scaffolding "
          "split beside its headline, the way the pair criteria "
          "already do - without it 15/50 reads as a fidelity gap "
          "on the operator's data",
          "6/6 are on YOUR OWN columns" in _line
          and "44 have a derived token indicator" in _line
          and "cannot apply a surface to at all" in _line)
    check("...and the NOT READABLE count is on the same line, so "
          "the shrunken denominator travels with the fraction",
          "3 NOT READABLE" in _line)
    check("...and a run with NO scaffolding adds no note, because "
          "a note that fires on everything is noise",
          "YOUR OWN columns" not in next(
              (c["detail"] for c in _gate.assess(
                  {"summary": {
                      "pairs": 10, "pairs_sign_ok": 10,
                      "pairs_close": 10, "pairs_inverted": 0,
                      "surfaces_compared": 6, "surfaces_ok": 6,
                      "surfaces_own_compared": 6,
                      "surfaces_own_ok": 6,
                      "surfaces_scaffold": 0}})["criteria"]
               if c["name"] == "interaction surfaces"), ""))

    # THE SCREEN'S REASONING IS ON THE MODULE, not only here: the
    # constant carries the measurement that fitted it, so nobody
    # re-chooses it from a hunch.
    src = io.open(ROOT / "synthkit" / "curvecheck.py",
                  encoding="utf-8").read()
    check("the constant states the measurement it was fitted "
          "from, including the conservative end and why that "
          "direction was chosen",
          "SURFACE_NOISE_C = 2.3" in src
          and "CONSERVATIVE end" in src
          and "worse error" in src)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
