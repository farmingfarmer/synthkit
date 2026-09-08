"""Smoke: manufacture the features a pairwise search cannot see,
without manufacturing all of them.

The centering is the load-bearing part. An exclusive-or is visible in
(a - median_a)*(b - median_b) because that product is negative exactly
when one factor is high and the other low; an UNCENTERED product cannot
express it. The suite proves the centered form separates the cases and
the uncentered one does not, rather than taking it on faith.
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit.condnet import CondNet                  # noqa: E402
from synthkit.engineered import (                     # noqa: E402
    add_product_features, unexplained_columns, base_columns,
    _num, PRODUCT_SEP)

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
                # more unexplained columns, so a budget below the
                # possible-pair count is a real truncation and the
                # sampling can be tested at all
                "u1": round(rnd.gauss(0, 1), 4),
                "u2": round(rnd.gauss(0, 1), 4),
                "u3": round(rnd.gauss(0, 1), 4),
                "u4": round(rnd.gauss(0, 1), 4),
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

    # the centering is what makes XOR visible
    def corr(xs, ys):
        m1, m2 = sum(xs) / len(xs), sum(ys) / len(ys)
        nu = sum((p - m1) * (q - m2) for p, q in zip(xs, ys))
        de = (sum((p - m1) ** 2 for p in xs)
              * sum((q - m2) ** 2 for q in ys)) ** 0.5
        return nu / de if de else 0.0
    ys = [float(r["y"]) for r in out]
    centered = [float(r[name]) for r in out]
    uncentered = [float(r["a"]) * float(r["b"]) for r in out]
    check("neither factor alone predicts the outcome - the definition "
          "of a pure interaction, and why a pairwise search is blind "
          "to it",
          abs(corr([float(r["a"]) for r in out], ys)) < 0.12
          and abs(corr([float(r["b"]) for r in out], ys)) < 0.12)
    check("the CENTERED product does predict it",
          abs(corr(centered, ys)) > 0.6)
    check("...and an UNCENTERED product does NOT, which is why the "
          "centering is not a detail",
          abs(corr(uncentered, ys)) < abs(corr(centered, ys)) / 2.0)

    # budget
    _o2, r2 = add_product_features(rows, sorted(un), budget=2)
    check("a budget is honored and the truncation is REPORTED, not "
          "silent - a partial search that reads as complete is worse "
          "than one that says so",
          r2["pairs_tried"] == 2 and r2["truncated"] is True
          and "PARTIAL" in r2["note"])
    check("the note states what the rule cannot cover",
          "still missed" in r2["note"])
    # Taking the first N of a sorted pair list makes recoverability
    # depend on column naming: measured, the interaction was found at
    # budget 400 and missed at 150 because its pair fell past the cut.
    check("the fixture has enough unexplained columns for a budget "
          "to actually truncate, or the sampling test is vacuous",
          len(un) * (len(un) - 1) // 2 > 6)
    o4, r4 = add_product_features(rows, sorted(un), budget=3, seed=1)
    o5, r5 = add_product_features(rows, sorted(un), budget=3, seed=2)
    n4 = {c for c in o4[0] if PRODUCT_SEP in c}
    n5 = {c for c in o5[0] if PRODUCT_SEP in c}
    check("a truncated budget SAMPLES the pairs rather than taking "
          "them alphabetically - which interaction is findable must "
          "not depend on column naming", n4 != n5)
    o6, _r6 = add_product_features(rows, sorted(un), budget=3, seed=1)
    check("...and the sample is reproducible from its seed",
          {c for c in o6[0] if PRODUCT_SEP in c} == n4)
    check("the note names the seed it sampled with",
          "seed 1" in r4["note"])
    check("partial coverage is reported as the CEILING on interaction "
          "recall, since a pair outside the sample cannot be found "
          "however good the search is",
          r4["coverage"] < 1.0 and "IS the ceiling" in r4["note"])
    check("complete coverage is stated too, so a full search is "
          "distinguishable from a lucky partial one",
          rep["coverage"] == 1.0 and "COMPLETE" in rep["note"])

    # nothing to pair
    _o3, r3 = add_product_features(rows, [], budget=10)
    check("no unexplained columns means no products, said plainly",
          r3["added"] == 0 and "no unexplained" in r3["note"])

    # ---- engineered features are PARENT-ONLY -------------------
    # Measured: adding products produced 114 spurious edges, of which
    # 87 had a product as the CHILD, 19 were a product against its own
    # two factors, and 8 were a product explaining a column it
    # CONTAINS. All three are arithmetic rather than findings, and all
    # three are structural - so they cost nothing to remove. With both
    # rules the noise edges went 82 to 0 while the XOR stayed found.
    from synthkit.engineered import feature_sources
    fs = feature_sources(out)
    check("every engineered feature knows the columns it came from",
          fs.get(name) and set(fs[name]) == {"a", "b"})
    net2 = CondNet(k=10).learn(out, group_by="person_id",
                               multilevel=True, feature_sources=fs)
    kids = {e["child"] for e in net2.report["edges"]}
    check("no engineered feature is ever a CHILD - it exists to "
          "explain other columns, not to be explained or generated",
          not any(PRODUCT_SEP in c for c in kids))
    circular = []
    for e in net2.report["edges"]:
        for par in e["parents"]:
            if e["child"] in fs.get(par, ()):
                circular.append((e["child"], par))
    check("no feature explains a column it CONTAINS - `noise_02 <- "
          "height__x__noise_02` is the product carrying noise_02 "
          "inside it, not a finding", not circular)
    check("...and the interaction is still recovered, so the rules "
          "cost precision nothing and recall nothing",
          any(e["child"] == "y"
              and any(PRODUCT_SEP in q for q in e["parents"])
              for e in net2.report["edges"]))

    # An XOR is SYMMETRIC - any two of {a, b, y} determine the third -
    # so the search legitimately finds y <- a__x__b, b <- a__x__y and
    # a <- b__x__y. All true, and together a cycle: nothing can be
    # drawn first. Measured, the generic cycle-repair stranded all six
    # columns and the XOR vanished from the generated data while the
    # model went on reporting it.
    back = [(e["child"], p) for e in net2.report["edges"]
            for p in e["parents"] if PRODUCT_SEP in p
            and not all(net2.order.index(s) < net2.order.index(e["child"])
                        for s in fs.get(p, ()))]
    check("a product only explains a column whose BOTH factors are "
          "drawn before it, so the direction is decided rather than "
          "left to form a cycle", not back)

    # ---- the GENERATED data must carry them --------------------
    # Parent-only settles learning and says nothing about generation,
    # where the default is actively wrong: a feature drawn from its own
    # marginal is a previous value belonging to no patient and a
    # product of nothing. The model would REPORT the interaction and
    # the data would not contain it - failure that reads as success.
    gen = net2.sample_patients(400, seed=7)
    check("engineered features are NOT emitted - they are scaffolding "
          "for the search, not columns of the user's data",
          gen and not any(PRODUCT_SEP in c for c in gen[0]))
    lo = [float(r["y"]) for r in gen
          if (float(r["a"]) > 0.5) == (float(r["b"]) > 0.5)]
    hi = [float(r["y"]) for r in gen
          if (float(r["a"]) > 0.5) != (float(r["b"]) > 0.5)]
    check("the XOR survives INTO the generated data: y is higher when "
          "exactly one factor is high, which is the relationship the "
          "product was manufactured to find",
          lo and hi and (sum(hi) / len(hi)) - (sum(lo) / len(lo)) > 0.3)

    # and the same for a lag feature
    from synthkit.temporal import add_lag_features, LAG_SUFFIX
    lrows = []
    lr = random.Random(11)
    for p in range(220):
        prev = None
        for v in range(8):
            x = round(lr.uniform(0, 1), 4)
            lrows.append({"person_id": "P{:04d}".format(p),
                          "t": v + 1, "x": x,
                          "y": round((3.0 * prev if prev is not None
                                      else lr.uniform(0, 3))
                                     + lr.gauss(0, 0.08), 4)})
            prev = x
    lag_rows, _lrep = add_lag_features(lrows, "person_id", "t",
                                       "sequence")
    from synthkit.engineered import feature_sources as fsrc
    lnet = CondNet(k=10).learn(lag_rows, group_by="person_id",
                               multilevel=True,
                               feature_sources=fsrc(lag_rows))
    check("the lagged relationship is found, or the generation test "
          "below has nothing to preserve",
          any(e["child"] == "y"
              and any(str(q).endswith(LAG_SUFFIX) for q in e["parents"])
              for e in lnet.report["edges"]))
    g2 = lnet.sample_patients(300, seed=7)
    by = {}
    for r in g2:
        by.setdefault(r["person_id"], []).append(r)
    pairs = []
    for g in by.values():
        for i in range(1, len(g)):
            xp, yc = _num(g[i - 1].get("x")), _num(g[i].get("y"))
            if xp is not None and yc is not None:
                pairs.append((xp, yc))
    check("the LAG survives into generated data too: y at a visit "
          "tracks x at the PREVIOUS visit, so the feature was rebuilt "
          "from this patient's own history rather than redrawn",
          len(pairs) > 500
          and corr([p for p, _ in pairs], [q for _, q in pairs]) > 0.45)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
