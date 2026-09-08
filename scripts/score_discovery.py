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
    # THE NEW PATH'S CATALOGUE, which this could not read - so the
    # discover / blueprint / generate path had never been scored
    # against the planted truth at all. The fixture has carried a
    # U-shape, an XOR with no main effect, a Simpson's reversal, a
    # lagged cross-column pair and a three-way since it was written,
    # and nobody could say which of them the new search finds.
    if blob.get("claims") is not None:
        return ([{"child": c["child"],
                  "parents": [p_["column"]
                              for p_ in (c.get("predictors") or [])]}
                 for c in blob["claims"]], "catalogue")
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


LAG_SUFFIXES = ("__prev", "__delta")
PRODUCT_SEP = "__x__"


def expand(c):
    """A parent may be an engineered feature standing for one or two
    source columns. `a__x__b` is how a PURE interaction is found - the
    only way, since neither factor has a marginal signal for a
    pairwise search to latch onto - so it must count as both."""
    if c and PRODUCT_SEP in c:
        left, right = c.split(PRODUCT_SEP, 1)
        return [base_name(left), base_name(right)]
    return [base_name(c)]


def base_name(c):
    """An engineered lag feature stands for its source column.

    A lagged relationship can ONLY be found through the feature: the
    search sees `y <- x__prev`, and the planted truth says `y <- x`.
    Scored literally, a correct finding reads as a miss - which is
    exactly what happened, and is the same class of error as demanding
    a particular edge direction."""
    for suf in LAG_SUFFIXES:
        if c and c.endswith(suf):
            return c[:-len(suf)]
    return c


def group_of(e):
    """PARENTS may be lag features; the CHILD may not.

    `y <- x__prev` is how a lagged relationship is found, and is the
    whole point of engineering the feature. But `y__prev <- x__prev`
    is a relationship among previous-visit COPIES: the model has
    learned nothing about generating y at this visit, so counting it
    would credit a capability that is not there.

    Normalizing both ends scored the XOR as recovered - a
    relationship established as structurally invisible - via
    `xor_b__prev <- xor_a__prev, xor_y__prev`. That reversal of a
    settled result is what exposed the error."""
    child = e.get("child")
    cols = [child]
    for c in (e.get("parents") or []):
        cols.extend(expand(c))
    # An engineered feature as the CHILD is a different relationship -
    # `y__prev <- x__prev` is about the previous visit, and
    # `a__x__b <- a, b` is the product's own arithmetic. Neither is
    # evidence the model can generate y here, so the child is never
    # normalized.
    return frozenset(cols)


def used_lag(e):
    return any(str(c).endswith(LAG_SUFFIXES)
               for c in [e.get("child")] + list(e.get("parents") or []))


def used_product(e):
    return any(PRODUCT_SEP in str(c)
               for c in (e.get("parents") or []))


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
        via_lag = via_prod = False
        for e, g in zip(edges, found_groups):
            if want <= g:
                full = g
                via_lag = used_lag(e)
                via_prod = used_product(e)
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
            "via_lag_feature": via_lag,
            "via_product_feature": via_prod,
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
        L.append("{} : {} <- {}{}".format(
            mark, r["child"], ", ".join(r["parents"]),
            ("   [via product feature]"
             if r.get("via_product_feature")
             else "   [via lag feature]" if r.get("via_lag_feature")
             else "")))
    return "\n".join(L)


if __name__ == "__main__":
    main()
