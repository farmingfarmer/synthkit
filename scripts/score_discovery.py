"""Score what a search found against what was actually planted.

    python scripts/score_discovery.py --truth ground_truth.json
                                      --model condnet_model.json
                                      [--report]

Until now every claim about discovery here has been PRECISION-only:
"58 relationships found, 16 reproduced". Nobody could say whether 16
was most of what was there or a tenth of it, because the truth was
unknown. Against a planted fixture both halves are computable, and
recall is the half that says whether an automatic pattern finder
works at all.

THREE THINGS IT REPORTS SEPARATELY, because they are different
failures:

  recall BY KIND   a search that finds every linear relationship and
                   no interaction is not 50% good at its job; it has
                   one capability and lacks another, and an average
                   hides that
  noise found      a noise column is planted to be unrelated to
                   everything. An edge touching one is unambiguously
                   wrong
  unplanted        an edge among real columns that was not planted.
                   NOT counted as a false positive: the fixture gives
                   its columns per-patient level and shared anchors,
                   so incidental structure genuinely exists. Reported,
                   not judged

Matching ignores DIRECTION. Dependence is symmetric and the arrow
comes from the column ordering, so a search that finds `x <- y` has
found the same relationship as `y <- x`. Scoring it otherwise once
made me report three planted relationships as missed when all three
had been found.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:                       # console-safe on Windows terminals
    from _console import console_safe
    console_safe()
except Exception:
    pass


def edges_from(path):
    """Accepts a condnet model, a confirm_patterns report, or a plain
    list of {child, parents}."""
    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(blob, list):
        return blob, "list"
    rep = blob.get("report") or {}
    conf = rep.get("confirmation") or {}
    if conf.get("confirmed_edges"):
        return conf["confirmed_edges"], "confirmed"
    if rep.get("edges"):
        return rep["edges"], "model"
    for arm in blob.get("arms") or []:
        if arm.get("edges"):
            return [e for e in arm["edges"] if e.get("reproduced")], \
                "confirm_report"
    return [], "none"


def group_of(e):
    return frozenset([e.get("child")] + list(e.get("parents") or []))


def main():
    ap = argparse.ArgumentParser(
        description="Precision and recall against planted truth.")
    ap.add_argument("--truth", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    tp = Path(a.truth)
    mp = Path(a.model)
    for p in (tp, mp):
        if not p.exists():
            sys.exit("not found: {}".format(p))
    truth = json.loads(tp.read_text(encoding="utf-8"))
    planted = truth.get("relationships") or []
    noise = set(truth.get("noise_columns") or [])
    if not planted:
        sys.exit("{} lists no planted relationships - nothing to "
                 "score against".format(tp))
    edges, source = edges_from(mp)

    found_groups = [group_of(e) for e in edges]
    results = []
    for rel in planted:
        want = frozenset([rel["child"]] + list(rel["parents"]))
        hit = full = None
        for g in found_groups:
            if want <= g:
                full = g
                break
            if len(want & g) >= 2 and hit is None:
                hit = g            # some of it, not all
        results.append({
            "child": rel["child"], "parents": rel["parents"],
            "kind": rel.get("kind", "?"),
            "found": full is not None,
            "partial": full is None and hit is not None,
            "expected_miss": "cannot see this" in (rel.get("note")
                                                   or ""),
        })

    planted_groups = [frozenset([r["child"]] + list(r["parents"]))
                      for r in planted]
    noise_edges, unplanted = [], []
    for e, g in zip(edges, found_groups):
        if g & noise:
            noise_edges.append(e)
        elif not any(len(pg & g) >= 2 for pg in planted_groups):
            unplanted.append(e)

    recovered = sum(1 for r in results if r["found"])
    expected = [r for r in results if not r["expected_miss"]]
    rec_exp = sum(1 for r in expected if r["found"])
    by_kind = {}
    for r in results:
        k = by_kind.setdefault(r["kind"], {"n": 0, "found": 0})
        k["n"] += 1
        k["found"] += 1 if r["found"] else 0

    out = {
        "source": source,
        "edges_reported": len(edges),
        "planted": len(planted),
        "recovered": recovered,
        "recall": round(recovered / float(len(planted)), 4),
        "recall_excluding_expected_misses": (
            round(rec_exp / float(len(expected)), 4)
            if expected else None),
        "noise_edges": len(noise_edges),
        "unplanted_edges": len(unplanted),
        "by_kind": by_kind,
        "detail": results,
    }
    print(render(out, a.width, a.report))
    return out


def render(o, width, full):
    L = []
    L.append("DISCOVERY SCORE (edges read as: {})".format(o["source"]))
    L.append("edges reported {}".format(o["edges_reported"]))
    L.append("planted {}".format(o["planted"]))
    L.append("recovered {}".format(o["recovered"]))
    L.append("recall {:.0%}".format(o["recall"]))
    if o["recall_excluding_expected_misses"] is not None:
        L.append("recall excluding known blind spots {:.0%}".format(
            o["recall_excluding_expected_misses"]))
    L.append("noise edges {}  (any is unambiguously wrong)".format(
        o["noise_edges"]))
    L.append("unplanted edges {}  (reported, not judged)".format(
        o["unplanted_edges"]))
    L.append("--- recall by kind ---")
    for k in sorted(o["by_kind"]):
        v = o["by_kind"][k]
        L.append("{} {}/{}".format(k, v["found"], v["n"]))
    L.append("--- each planted relationship ---")
    for r in o["detail"]:
        mark = ("FOUND" if r["found"] else
                "partial" if r["partial"] else "MISSED")
        if not r["found"] and r["expected_miss"]:
            mark = "MISSED (expected)"
        L.append("{} : {} <- {}".format(
            mark, r["child"], ", ".join(r["parents"])))
    return "\n".join(L)


if __name__ == "__main__":
    main()
