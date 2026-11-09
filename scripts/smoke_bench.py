"""Smoke: the benchmark can see the fitted path, and the fixture can
contain the failure the real extract has.

WHY THIS EXISTS. The hard benchmark already existed and nothing
pointed it at the new path. `make_tidy_fixture` plants a U-shape with
zero linear correlation, an XOR with no main effect, a Simpson's
reversal, a three-way and a lagged pair; `score_discovery` scores
recall BY KIND. But `score_discovery` read condnet models and
confirmation reports only, so discover / blueprint / generate had
never been scored against planted truth at all.

Pointed at it, the answer was not the expected one:

    FOUND     11-12/13 planted kinds, 0 noise edges
    SURVIVED  10/10 of the kinds with an honest survival test

including the XOR the fixture itself predicted would be missed. So the
search is not the weak part.

THE FOUND COUNT IS A RANGE, and this docstring said 12/13 for months
because that was the seed it was first run on. Over seeds 0-4 it reads
11, 12, 11, 11, 11 - the only moving kind is `heterogeneous`, which
sits at discovery's inclusion threshold: missed at source 0.234, found
at 0.247. A run reading 11 is not a regression against the 12, which
is what the sprint board carried as an unattributed debt until the
sweep was run.

WHAT THE FIXTURE COULD NOT CONTAIN was the shape of the real graph.
Its planted structure is disjoint pairs and triples - a FOREST - so
nothing is ever dropped to break a cycle. The real extract drops 16 of
26 relationships to make its graph sampleable, and a relationship that
is found and then dropped is indistinguishable in the output from one
never found.

`--tangled N` adds a ring where every column is both a parent and a
child. Measured: 27 of 42 relationships dropped, 64%, against the real
extract's 62% - and `heterogeneous`, which was found and survived on
the forest, disappeared entirely.
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
from synthkit import blueprint as BB                   # noqa: E402
from synthkit.blueprint import resolve                 # noqa: E402
from synthkit.discover import discover as DISC         # noqa: E402
from synthkit.generate import _order                   # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def rel(child, parents, skill):
    return {"child": child, "parents": parents,
            "dials": {"strength": None},
            "evidence": {"skill_out_of_sample": skill,
                         "importance": dict((p, 1.0) for p in parents),
                         "effect": {}}}


def bp_from(rels, cols):
    return {"columns": dict(
        (c, {"kind": "numeric", "level": "visit", "coverage": 1.0,
             "marginal": {"type": "quantiles", "mean": 0.0,
                          "q": [0.0, 0.5, 1.0], "v": [-1.0, 0.0, 1.0]},
             "dials": {}}) for c in cols),
        "relationships": rels, "patients": {"count": 10, "dials": {}}}


def main():
    # ---- a FOREST drops nothing; a RING must drop something -------
    cols = ["a", "b", "c", "d"]
    forest = bp_from([rel("b", ["a"], 0.8), rel("d", ["c"], 0.7)],
                     cols)
    _o, _p, dropped_f, _r, _d, _c, _s = _order(resolve(forest))
    check("a forest-shaped graph drops nothing - which is why a "
          "fixture built from disjoint pairs scores 10/10 survival "
          "and says nothing about a real graph",
          len([x for x in dropped_f if not x.get("harmless")]) == 0)

    ring = bp_from([rel("b", ["a"], 0.9), rel("c", ["b"], 0.85),
                    rel("d", ["c"], 0.8), rel("a", ["d"], 0.75)],
                   cols)
    _o, _p, dropped_r, _r, _d, _c, _s = _order(resolve(ring))
    check("a RING cannot be ordered without cutting an edge, so the "
          "sampler drops one and says so - this is the mechanism "
          "that loses 16 of 26 relationships on the real extract",
          len(dropped_r) >= 1)

    # ---- the fixture can now build one ---------------------------
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "fx"
        r = subprocess.run(
            [sys.executable,
             str(ROOT / "scripts" / "make_tidy_fixture.py"),
             "-o", str(out), "--patients", "45",
             "--tangled", "6", "--skew-planted", "2.8"],
            capture_output=True, text=True)
        check("the fixture builds with a tangled ring and skew",
              r.returncode == 0)
        truth = json.loads((out / "ground_truth.json").read_text(
            encoding="utf-8"))
        df = pd.read_csv(out / "tidy_visits_labeled.csv")

        ring_rows = [x for x in truth["relationships"]
                     if x["kind"] == "tangled-ring"]
        check("every ring member is recorded as planted truth, so a "
              "dropped one is a MEASURED loss rather than a surprise",
              len(ring_rows) == 6)
        check("...and each is both a parent and a child, which is "
              "what makes the graph cyclic rather than deep",
              all(len(x["parents"]) == 2 for x in ring_rows))

        T = [c for c in df.columns if c.startswith("tangle_")]
        nb = [abs(float(df[T[i]].corr(df[T[(i + 1) % len(T)]])))
              for i in range(len(T))]
        check("neighbors in the ring really do predict each other "
              "(mean |r| {:.2f}), or the cycle is nominal".format(
                  float(np.mean(nb))), float(np.mean(nb)) > 0.5)

        check("the ring carries the requested skew ({:+.1f}), so the "
              "cycle is not also the easy shape".format(
                  float(df["tangle_00"].skew())),
              float(df["tangle_00"].skew()) > 1.5)

        # THE DAMAGE THAT SKEW DID, asserted so it cannot come back.
        # Applied to the planted columns it took the U-shape's linear
        # correlation from ~0 to +0.27 - the ground truth says that
        # relationship has NONE, so the fixture stopped being true of
        # itself.
        pear = abs(float(df["planted_ushape_x"].corr(
            df["planted_ushape_y"])))
        check("skew does NOT touch the planted columns - it took the "
              "U-shape's linear correlation from ~0 to +0.27 when it "
              "did, and 'zero linear correlation' is the property "
              "that relationship exists to test (now {:+.3f})".format(
                  pear), pear < 0.1)

    # ---- the scorer can read the fitted path ---------------------
    with tempfile.TemporaryDirectory() as td:
        cat = Path(td) / "catalogue.json"
        cat.write_text(json.dumps({"claims": [
            {"child": "y", "predictors": [{"column": "x"}]}],
            "unexplained": [], "skipped": []}), encoding="utf-8")
        tr = Path(td) / "truth.json"
        tr.write_text(json.dumps({
            "relationships": [{"child": "y", "parents": ["x"],
                               "kind": "linear"}],
            "noise_columns": []}), encoding="utf-8")
        r = subprocess.run(
            [sys.executable,
             str(ROOT / "scripts" / "score_discovery.py"),
             "--truth", str(tr), "--model", str(cat)],
            capture_output=True, text=True)
        out = r.stdout or ""
        check("score_discovery reads a catalogue.json - it read "
              "condnet models and confirmation reports only, so the "
              "fitted path had never been scored against planted "
              "truth at all", "catalogue" in out)
        check("...and scores it, rather than reading zero edges and "
              "reporting a clean sheet",
              "recovered 1" in out or "recall 100%" in out)

    # ---- THE BENCH'S FOUND COUNT IS A RANGE, NOT A POINT
    #
    # The sprint board carried "the bench reads 11/13 against a
    # recorded 12/13 - still unattributed" as a debt. A seed
    # sweep attributed it: 11, 12, 11, 11, 11 over seeds 0-4,
    # the only moving kind `heterogeneous`, whose measured
    # source strength ran 0.182-0.247 and which was found only
    # at 0.247. A miss at 0.234 and a find at 0.247 are the same
    # effect either side of discovery's inclusion cut, so a run
    # reading 11 is not a regression.
    #
    # Running the bench takes over ten minutes, so it cannot be
    # exercised here. What CAN be is the note's own arithmetic,
    # which is why it is a function returning lines rather than
    # print statements - a content check cannot see a dead code
    # path, and this one computes its range from the recorded
    # sweep.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bench_new_path import SEED_SWEEP, seed_range_note
    _counts = [c for _s, c, _v, _f in SEED_SWEEP]
    _note = "\n".join(seed_range_note())
    check("the recorded seed sweep has more than one seed, "
          "because a range derived from one point is a point",
          len(SEED_SWEEP) >= 5)
    check("...and the counts genuinely VARY across it ({}), so "
          "there is a range to report rather than a constant "
          "dressed as one".format(sorted(set(_counts))),
          len(set(_counts)) > 1)
    check("...and the note states that range computed from the "
          "sweep - {}-{} of 13 - rather than a number somebody "
          "typed".format(min(_counts), max(_counts)),
          "{}-{} of 13".format(min(_counts), max(_counts))
          in _note)
    check("...and it NAMES the kind that moves and says a low "
          "run is NOT a regression, which is the whole point of "
          "reporting a range",
          "heterogeneous" in _note
          and "NOT a regression" in _note)
    check("...and it names the one kind that misses on EVERY "
          "seed with the reason, so a permanent miss is not read "
          "as the same thing as a seed-fragile one",
          "lagged-cross-column" in _note
          and "--lags" in _note)
    _het_found = [v for _s, _c, v, f in SEED_SWEEP if f]
    _het_missed = [v for _s, _c, v, f in SEED_SWEEP if not f]
    check("and the sweep shows the THRESHOLD rather than a weak "
          "effect: the kind was missed at {:.3f} and found at "
          "{:.3f}, which is the same strength either side of "
          "discovery's inclusion cut"
          .format(max(_het_missed), min(_het_found)),
          _het_found and _het_missed
          and min(_het_found) - max(_het_missed) < 0.05)

    # ---- `--lags` IS A TRADE, AND THE NOTE MUST SAY BOTH HALVES
    #
    # The bench scored `lagged-cross-column` as a permanent miss
    # for months, and the reason was not the engine: the bench
    # never built the lag features `synthkit fit --lags` builds,
    # so the one kind that needs them could not be found by
    # construction. That is a capability present in the pipeline
    # and wired to nothing in the instrument that grades it.
    #
    # With them it is found on EVERY seed - and the output then
    # carries a SAME-VISIT correlation of ~0.97 on a pair the
    # source relates at 0.00, where without them that pair
    # correctly reads 0.00. A note that reported only the extra
    # point would be advertising a regression.
    import bench_new_path as BN
    _TEXT = io.open(str(ROOT / "scripts" / "bench_new_path.py"),
                    encoding="utf-8").read()
    _lag = dict(BN.LAG_SWEEP)
    _base = {sd: c for sd, c, _v, _f in BN.SEED_SWEEP}
    check("the recorded `--lags` sweep covers the SAME seeds as "
          "the base sweep ({} against {}), because a count "
          "measured on other seeds is not a comparison"
          .format(sorted(_lag), sorted(_base)),
          sorted(_lag) == sorted(_base))
    check("...and it adds EXACTLY ONE kind on every seed ({}), "
          "which is the lagged one and not a general improvement"
          .format(["{}->{}".format(_base[k], _lag[k])
                   for k in sorted(_lag)]),
          all(_lag[k] - _base[k] == 1 for k in _lag))
    _cost = BN.LAG_COST
    check("the recorded COST is an invented same-visit pattern - "
          "source {:+.3f}..{:+.3f} against generated "
          "{:.3f}..{:.3f} - and not a rounding difference"
          .format(min(f for _s, f, _g in _cost),
                  max(f for _s, f, _g in _cost),
                  min(g for _s, _f, g in _cost),
                  max(g for _s, _f, g in _cost)),
          all(abs(f) < 0.1 and g >= 0.3 for _s, f, g in _cost))
    _note = "\n".join(BN.seed_range_note())
    check("...and the note states BOTH halves in its own words, "
          "computed from those tables rather than typed, so the "
          "extra point cannot be reported without its price",
          "--lags" in _note and "not a free point" in _note.lower()
          and "same-visit" in _note.lower()
          and "invented" in _note.lower())
    # ---- THE BLUEPRINT'S OWN FIELD NAMES, READ LIVE ------------
    #
    # The lag diagnostic printed `blueprint planted_lag_y <- []`
    # for every record, on a blueprint whose relationships plainly
    # work - because it read `predictors[].column`, which is the
    # CATALOGUE's shape, where a blueprint record carries
    # `parents` as a list of NAMES. The ninth wrong-key fault
    # here, and it sent me hunting an empty-predictor bug that
    # does not exist. Asserted against a LIVE build so a rename
    # breaks this suite rather than somebody's diagnostic.
    _rng = np.random.default_rng(0)
    _n, _per = 200, 6
    _pid = np.repeat(["P{:04d}".format(i) for i in range(_n)], _per)
    _N = len(_pid)
    _a = _rng.normal(50, 10, _N)
    _b = 0.9 * (_a - 50) + 50 + 4.4 * _rng.normal(0, 1, _N)
    _bp = BB.build(
        pd.DataFrame({"person_id": _pid, "a": _a,
                      "b": _b}).astype(str),
        DISC(pd.DataFrame({"person_id": _pid, "a": _a,
                           "b": _b}).astype(str),
             group_by="person_id", seed=1),
        group_by="person_id")
    _recs = _bp.get("relationships") or []
    check("a blueprint relationship record carries `parents` as a "
          "list of NAMES ({} record(s), keys {}) - and NOT "
          "`predictors`, which is the catalogue's shape"
          .format(len(_recs),
                  sorted(_recs[0]) if _recs else []),
          bool(_recs)
          and all(isinstance(r.get("parents"), list) for r in _recs)
          and all(all(isinstance(x, str) for x in r["parents"])
                  for r in _recs)
          and not any("predictors" in r for r in _recs))
    check("...and at least one of them is NON-EMPTY, so a "
          "diagnostic reading the wrong key cannot print an empty "
          "list and be believed",
          any(r.get("parents") for r in _recs))
    check("...and the diagnostic reads that key rather than the "
          "catalogue's",
          'r.get("parents")' in _TEXT
          and '[p_["column"]\n' not in _TEXT)

    check("the lagged survival test RECOMPUTES the lag from each "
          "table's own ordering rather than reading the emitted "
          "__prev column, which generation draws rather than "
          "carries - so a file whose lag column contradicts its "
          "own previous visit cannot pass",
          "def lag_corr" in _TEXT
          and "groupby(group, sort=False)[parent].shift(1)" in _TEXT)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
