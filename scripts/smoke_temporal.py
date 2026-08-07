"""Smoke: find the time axis whatever shape it arrived in, and lag
only what is worth lagging.

An automated system cannot be told which column orders its visits, so
the detection is tested against a date, against a bare sequence
number, and against a table with neither - where the honest answer is
to say file order was assumed rather than to pretend.
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit.temporal import (                    # noqa: E402
    detect_time_column, add_lag_features, LAG_SUFFIX)

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def build(kind, n_pat=120, n_vis=8, seed=3):
    rnd = random.Random(seed)
    rows = []
    for p in range(n_pat):
        trait = rnd.choice(["A", "B"])
        prev = None
        for v in range(n_vis):
            r = {"person_id": "P{:04d}".format(p),
                 # a fixed trait: its previous value IS its value
                 "sex": trait,
                 # varies within a patient
                 "lab": round(rnd.gauss(50, 8), 3),
                 # only 3% coverage - not worth a lag feature
                 "rare": (round(rnd.gauss(0, 1), 3)
                          if rnd.random() < 0.03 else ""),
                 # y follows the PREVIOUS x
                 "x": round(rnd.uniform(0, 1), 4)}
            r["y"] = round((2.0 * prev if prev is not None
                            else rnd.uniform(0, 2))
                           + rnd.gauss(0, 0.1), 4)
            prev = r["x"]
            if kind == "date":
                r["when"] = "2021-{:02d}-{:02d}".format(
                    (v % 12) + 1, (v % 27) + 1)
            elif kind == "sequence":
                r["visit_no"] = v + 1
            rows.append(r)
    return rows


def main():
    # ---- detection ------------------------------------------------
    col, kind, ev = detect_time_column(build("date"), "person_id")
    check("a DATE column is found and named as the time axis",
          col == "when" and kind == "date-like" and "parse" in ev)
    col, kind, ev = detect_time_column(build("sequence"), "person_id")
    check("a bare SEQUENCE number is found when there is no date - "
          "an automated system cannot be told which column it is",
          col == "visit_no" and kind == "sequence")
    col, kind, ev = detect_time_column(build("none"), "person_id")
    check("with neither, it says FILE ORDER was assumed rather than "
          "inventing an axis",
          col is None and kind == "file-order"
          and "assumption" in ev)
    check("...and says so in words a reader can act on",
          "rather than a measurement" in ev)

    # ---- what earns a lag -----------------------------------------
    rows = build("date")
    col, kind, _ = detect_time_column(rows, "person_id")
    out, rep = add_lag_features(rows, "person_id", col, kind)
    lagged = set(rep["lagged_columns"])
    check("a column that VARIES within a patient earns a lag",
          "x" in lagged and "lab" in lagged)
    check("a FIXED trait does not - its previous value is its value, "
          "and the search would rediscover that as a finding",
          "sex" not in lagged)
    check("a column too sparse to carry one does not either",
          "rare" not in lagged)
    check("the time column is not lagged against itself",
          col not in lagged)
    check("the report names the axis it used, so the choice is "
          "visible rather than implicit",
          rep["time_column"] == "when"
          and rep["time_kind"] == "date-like")

    # ---- the feature is correct -----------------------------------
    by = {}
    for r in out:
        by.setdefault(r["person_id"], []).append(r)
    one = by["P0000"]
    check("the FIRST visit has no previous value, and says so with a "
          "blank rather than a fabricated number",
          one[0]["x" + LAG_SUFFIX] == "")
    check("each later visit carries the previous visit's value "
          "exactly",
          all(abs(float(one[i]["x" + LAG_SUFFIX])
                  - float(one[i - 1]["x"])) < 1e-9
              for i in range(1, len(one))))

    # last OBSERVED, not last row
    gap = [{"person_id": "P1", "t": 1, "v": 5.0},
           {"person_id": "P1", "t": 2, "v": ""},
           {"person_id": "P1", "t": 3, "v": ""},
           {"person_id": "P1", "t": 4, "v": 9.0}]
    g2, _ = add_lag_features(gap * 30, "person_id", "t", "sequence",
                             columns=["v"])
    fourth = [r for r in g2 if str(r["v"]) == "9.0"]
    check("the lag carries the last OBSERVED value across a gap, not "
          "the last row's blank - at 11% coverage the previous row is "
          "almost always empty and the feature would be useless",
          fourth and str(fourth[0]["v" + LAG_SUFFIX]) == "5.0")

    # ---- the relationship becomes findable ------------------------
    def corr(xs, ys):
        m1, m2 = sum(xs) / len(xs), sum(ys) / len(ys)
        nu = sum((a - m1) * (b - m2) for a, b in zip(xs, ys))
        de = (sum((a - m1) ** 2 for a in xs)
              * sum((b - m2) ** 2 for b in ys)) ** 0.5
        return nu / de if de else 0.0
    pairs = [(float(r["x" + LAG_SUFFIX]), float(r["y"]))
             for r in out if str(r["x" + LAG_SUFFIX]).strip()]
    same = [(float(r["x"]), float(r["y"])) for r in out]
    check("the engineered feature makes an invisible relationship "
          "VISIBLE: y correlates with x__prev while showing nothing "
          "against x at the same visit",
          corr([a for a, _ in pairs], [b for _, b in pairs]) > 0.9
          and abs(corr([a for a, _ in same],
                       [b for _, b in same])) < 0.15)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
