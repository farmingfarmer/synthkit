"""Smoke: manufacture the features a pairwise search cannot see,
without manufacturing all of them.

The centring is the load-bearing part. An exclusive-or is visible in
(a - median_a)*(b - median_b) because that product is negative exactly
when one factor is high and the other low; an UNCENTRED product cannot
express it. The suite proves the centred form separates the cases and
the uncentred one does not, rather than taking it on faith.
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit.condnet import CondNet                  # noqa: E402
from synthkit.engineered import (                     # noqa: E402
    add_product_features, unexplained_columns, base_columns,
    PRODUCT_SEP)

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def main():
    rnd = random.Random(4)
    rows = []
    for p in range(200):
        for _v in range(8):
            a = rnd.uniform(0, 1)
            b = rnd.uniform(0, 1)
            x = rnd.uniform(0, 1)
            rows.append({
                "person_id": "P{:04d}".format(p),
                # a pure XOR: neither factor predicts y alone
                "a": round(a, 4), "b": round(b, 4),
                "y": round((1.0 if (a > 0.5) != (b > 0.5) else 0.0)
                           + rnd.gauss(0, 0.05), 4),
                # an ordinary relationship, so something IS explained
                "x": round(x, 4),
                # noisy enough to be an EDGE rather than a
                # deterministic derivation - at rho 0.98 condnet
                # files it as arithmetic and it never becomes an edge
                "z": round(2.0 * x + rnd.gauss(0, 0.55), 4),
                "flat": "1",
                "sparse": (round(rnd.gauss(0, 1), 3)
                           if rnd.random() < 0.02 else ""),
            })

    net = CondNet(k=10).learn(rows, group_by="person_id",
                              multilevel=True)
    un = set(unexplained_columns(net, rows))
    check("the XOR factors come back UNEXPLAINED, which is what makes "
          "them findable by pairing - neither has a marginal signal "
          "for the search to latch onto",
          {"a", "b"} <= un)
    check("a column already in a relationship is not re-paired - "
          "the budget belongs where nothing was explained",
          "x" not in un and "z" not in un)
    check("a constant column is not paired", "flat" not in un)
    check("a column too sparse to pair is not either",
          "sparse" not in un)

    out, rep = add_product_features(rows, sorted(un), budget=400)
    check("products are made and counted against what was possible, "
          "so partial coverage is visible rather than implied",
          rep["pairs_tried"] > 0
          and rep["pairs_possible"] >= rep["pairs_tried"])
    name = None
    for c in out[0]:
        if PRODUCT_SEP in c and set(base_columns(c) or []) == {"a", "b"}:
            name = c
    check("the pair that matters is among them", name is not None)

    # the centring is what makes XOR visible
    def corr(xs, ys):
        m1, m2 = sum(xs) / len(xs), sum(ys) / len(ys)
        nu = sum((p - m1) * (q - m2) for p, q in zip(xs, ys))
        de = (sum((p - m1) ** 2 for p in xs)
              * sum((q - m2) ** 2 for q in ys)) ** 0.5
        return nu / de if de else 0.0
    ys = [float(r["y"]) for r in out]
    centred = [float(r[name]) for r in out]
    uncentred = [float(r["a"]) * float(r["b"]) for r in out]
    check("neither factor alone predicts the outcome - the definition "
          "of a pure interaction, and why a pairwise search is blind "
          "to it",
          abs(corr([float(r["a"]) for r in out], ys)) < 0.12
          and abs(corr([float(r["b"]) for r in out], ys)) < 0.12)
    check("the CENTRED product does predict it",
          abs(corr(centred, ys)) > 0.6)
    check("...and an UNCENTRED product does NOT, which is why the "
          "centring is not a detail",
          abs(corr(uncentred, ys)) < abs(corr(centred, ys)) / 2.0)

    # budget
    _o2, r2 = add_product_features(rows, sorted(un), budget=2)
    check("a budget is honoured and the truncation is REPORTED, not "
          "silent - a partial search that reads as complete is worse "
          "than one that says so",
          r2["pairs_tried"] == 2 and r2["truncated"] is True
          and "BUDGET REACHED" in r2["note"])
    check("the note states what the rule cannot cover",
          "still missed" in r2["note"])

    # nothing to pair
    _o3, r3 = add_product_features(rows, [], budget=10)
    check("no unexplained columns means no products, said plainly",
          r3["added"] == 0 and "no unexplained" in r3["note"])

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
