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

    FOUND     12/13 planted kinds, 0 noise edges
    SURVIVED  10/10 of the kinds with an honest survival test

including the XOR the fixture itself predicted would be missed. So the
search is not the weak part.

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
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit.blueprint import resolve                 # noqa: E402
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
    _o, _p, dropped_f, _r, _d, _c = _order(resolve(forest))
    check("a forest-shaped graph drops nothing - which is why a "
          "fixture built from disjoint pairs scores 10/10 survival "
          "and says nothing about a real graph",
          len([x for x in dropped_f if not x.get("harmless")]) == 0)

    ring = bp_from([rel("b", ["a"], 0.9), rel("c", ["b"], 0.85),
                    rel("d", ["c"], 0.8), rel("a", ["d"], 0.75)],
                   cols)
    _o, _p, dropped_r, _r, _d, _c = _order(resolve(ring))
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
        check("neighbours in the ring really do predict each other "
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

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
