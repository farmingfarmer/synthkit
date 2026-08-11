"""Smoke: a model built only from structure that reproduced.

What this fixture CAN show: the mechanism is sound - discovery on one
set of patients, confirmation on another, parameters refitted on all
of them, evidence attached to every relationship, and planted truth
kept while planted noise is not.

What it CANNOT show, stated plainly: the real extract found 58
relationships of which 16 reproduced, and no fixture here reproduces
that 72% drop - the search finds no spurious edges on data this clean.
The VALUE of confirmation is measured on the real extract; only its
correctness is measured here.
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit.condnet import CondNet              # noqa: E402
from synthkit.confirmed import (                  # noqa: E402
    learn_confirmed, split_by_patient, test_edge)

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def build(n_pat=500, n_vis=8, n_noise=14, seed=4):
    rnd = random.Random(seed)
    rows = []
    for p in range(n_pat):
        for _v in range(n_vis):
            xl = rnd.uniform(0, 1)
            xn = rnd.uniform(0, 1)
            r = {"person_id": "P{:05d}".format(p),
                 "x_lin": round(xl, 4),
                 "y_lin": round(2.0 * xl + rnd.gauss(0, 0.15), 4),
                 "x_non": round(xn, 4),
                 "y_non": round(4.0 * (xn - 0.5) ** 2
                                + rnd.gauss(0, 0.05), 4)}
            for i in range(n_noise):
                r["noise_{:02d}".format(i)] = round(rnd.gauss(0, 1), 4)
            rows.append(r)
    return rows


def main():
    rows = build()

    # ---- the split ------------------------------------------------
    tr, te, n_tr, n_te = split_by_patient(rows, "person_id")
    ids_tr = set(r["person_id"] for r in tr)
    ids_te = set(r["person_id"] for r in te)
    check("the split shares NO patient between halves - a row-wise "
          "split would put the same person on both sides and confirm "
          "nearly anything", not (ids_tr & ids_te))
    check("every patient lands on exactly one side",
          len(ids_tr) + len(ids_te) == 500
          and n_tr == len(ids_tr) and n_te == len(ids_te))
    tr2, _te2, _a, _b = split_by_patient(rows, "person_id")
    check("the split is deterministic for a given seed",
          [r["person_id"] for r in tr2] == [r["person_id"] for r in tr])

    net = learn_confirmed(rows, "person_id", k=10)
    c = net.report["confirmation"]

    # ---- the mechanism --------------------------------------------
    check("parameters are refitted on ALL patients, not just the half "
          "the search used - discovery can afford half the data, "
          "estimation should not",
          net.report["persons"] == 500
          and c["train_patients"] + c["holdout_patients"] == 500)
    check("confirmed is a SUBSET of discovered",
          c["confirmed"] <= c["discovered"]
          and c["confirmed"] + c["not_reproduced"] == c["discovered"])
    check("the holdout is reported as having no correction applied - "
          "each edge is one pre-specified hypothesis",
          "does not search" in c["note"]
          and c["holdout_alpha"] > 0)

    # ---- the referee ----------------------------------------------
    kept = set()
    for e in c["confirmed_edges"]:
        kept |= {e["child"]} | set(e["parents"])
    check("the planted LINEAR relationship survives confirmation",
          {"x_lin", "y_lin"} <= kept)
    check("the planted U-SHAPE survives too - it has no linear "
          "correlation to find", {"x_non", "y_non"} <= kept)
    check("NO noise column reaches the confirmed model",
          not any(n.startswith("noise") for n in kept))
    modelled = set()
    for e in net.report.get("edges", []):
        modelled |= {e["child"]} | set(e["parents"])
    check("...and none reaches the refitted model either",
          not any(n.startswith("noise") for n in modelled))

    # ---- the evidence ---------------------------------------------
    ok = True
    for e in c["confirmed_edges"]:
        h, f = e["held_out"], e["found_on"]
        if not (h["g"] >= h["crit"] and h["patients"] > 0
                and f["patients"] > 0 and h["df"] >= 1):
            ok = False
    check("every confirmed relationship carries its strength on BOTH "
          "halves and the bar it had to clear", ok)
    check("holdout evidence is measured on the HOLDOUT's patients",
          all(e["held_out"]["patients"] <= c["holdout_patients"]
              for e in c["confirmed_edges"]))
    check("relationships that did not reproduce are kept on record, "
          "not silently discarded",
          isinstance(c["dropped_edges"], list))

    # ---- a fabricated edge must NOT confirm -----------------------
    scout = CondNet(k=10).learn(tr, group_by="person_id",
                                multilevel=True)
    fake = test_edge(scout, te, "y_lin", ["noise_00"], "person_id",
                     0.01)
    real = test_edge(scout, te, "y_lin", ["x_lin"], "person_id", 0.01)
    check("a relationship between an outcome and pure noise does NOT "
          "reproduce out of sample",
          fake is not None and not fake["reproduced"])
    check("...while the real one does, by a wide margin",
          real is not None and real["reproduced"]
          and real["g"] > 3 * real["crit"])

    # ---- guards ---------------------------------------------------
    try:
        learn_confirmed(build(n_pat=15), "person_id", k=10)
        check("too few patients to split is refused rather than "
              "confirmed on a handful", False)
    except ValueError as e:
        check("too few patients to split is refused rather than "
              "confirmed on a handful",
              "enough patients" in str(e))

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
