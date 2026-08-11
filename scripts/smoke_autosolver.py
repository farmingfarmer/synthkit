"""SYNTH_C1 smoke: the autosolver proven — encoder sniffs types
through mess, the stdlib baseline actually LEARNS planted signal
(closing most of the gap to the ceiling), autoclean fixes exactly
what heuristics honestly can, and the showdown prints the sentence
the phase exists for.

Run from the repo root:

    python scripts/smoke_autosolver.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synthkit.autosolver import (
    FeatureEncoder,
    LogisticBaseline,
    autoclean,
    autosolver,
    run_showdown,
)
from synthkit.campaign import compile_campaign, run_campaign
from synthkit.examples import reference_table
from synthkit.tableeval import evaluate_cleaning, \
    evaluate_prediction
from synthkit.tableplan import plan_table
from synthkit.tablespec import TableSpec

PASS = 0


def check(label, cond):
    global PASS
    if not cond:
        print("FAIL: {}".format(label))
        sys.exit(1)
    PASS += 1
    print("ok  {:02d}  {}".format(PASS, label))


def predict_table(rows=250, seed=11) -> TableSpec:
    spec = reference_table(rows=rows, master_seed=seed)
    spec.outcomes = [{
        "name": "readmitted",
        "kind": "logistic",
        "intercept": -3.2,
        "coefficients": {"los_days": 0.18, "age": 0.015,
                         "department=oncology": 0.9,
                         "active": -0.3},
    }]
    return spec


def main():
    # ---------- encoder ----------
    rows = [
        {"amt": "$1,234", "when": "03/15/2026", "flag": "YES",
         "dept": "cardiology", "_meta": "x"},
        {"amt": "980", "when": "2026-04-01", "flag": "no",
         "dept": "ONCOLOGY"},
        {"amt": "N/A", "when": "April 2, 2026", "flag": "y",
         "dept": "cardiology"},
        {"amt": "2 . 5", "when": "2026.05.09", "flag": "0",
         "dept": "emergency"},
    ]
    enc = FeatureEncoder().fit(rows)
    kinds = {c.name: c.kind for c in enc.columns}
    check("encoder sniffs numeric, date, bool, categorical "
          "through mess; ignores _columns",
          kinds == {"amt": "numeric", "when": "date",
                    "flag": "bool", "dept": "categorical"})
    v1 = enc.encode(rows[0])
    v3 = enc.encode(rows[2])
    check("encoding is fixed-width; missing numerics impute to "
          "the train mean (standardized zero)",
          len(v1) == len(v3)
          and abs(v3[0]) < 1e-9)
    check("case-insensitive category matching",
          enc.encode({"dept": "Cardiology"})
          == enc.encode({"dept": "cardiology"}))

    # ---------- the baseline learns ----------
    spec = predict_table(rows=300, seed=11)
    train_spec = predict_table(rows=300, seed=11)
    train_spec.master_seed += 1000
    test_bp = plan_table(spec)
    train_bp = plan_table(train_spec)
    outcome = "readmitted"
    train_rows = [{k: v for k, v in r.items() if k != outcome}
                  for r in train_bp.dirty_rows[:300]]
    train_labels = [1 if train_bp.clean_rows[i][outcome] == "True"
                    else 0 for i in range(300)]
    test_rows = [{k: v for k, v in r.items() if k != outcome}
                 for r in test_bp.dirty_rows[:300]]
    solve = autosolver()
    scores = solve(train_rows, train_labels, test_rows)
    rep = evaluate_prediction(test_bp, outcome, scores,
                              "baseline")
    check("the stdlib baseline learns the planted signal "
          "(auroc > 0.7, most of the ceiling closed)",
          rep.auroc > 0.7 and rep.auroc_gap < 0.12)
    scores2 = autosolver()(train_rows, train_labels, test_rows)
    check("training is deterministic",
          scores == scores2)
    single = [float(r["los_days"]) / 45.0 for r in test_rows]
    single_rep = evaluate_prediction(test_bp, outcome, single,
                                     "los-only")
    check("the multi-feature baseline beats the single-feature "
          "ranker",
          rep.auroc > single_rep.auroc)

    # ---------- autoclean ----------
    bp = plan_table(reference_table(rows=200))
    rep_c = evaluate_cleaning(bp, autoclean(bp.dirty_rows),
                              "autoclean")
    check("autoclean fixes whitespace completely",
          rep_c.ops["space"].fix_rate == 1.0)
    check("autoclean unifies date formats",
          rep_c.ops["format"].fix_rate > 0.6)
    check("autoclean restores casing to column convention",
          rep_c.ops["case"].fix_rate > 0.6)
    check("autoclean never guesses: missing and wrong untouched, "
          "near-zero overcorrection",
          rep_c.ops["missing"].fix_rate == 0.0
          and rep_c.ops.get("wrong") is not None
          and rep_c.ops["wrong"].fix_rate < 0.05
          and rep_c.overcorrection_rate < 0.03)
    check("autoclean flags a meaningful share of exact "
          "duplicates",
          rep_c.dup_flag_rate > 0.3)

    # ---------- autoclean up a campaign ladder ----------
    camp = compile_campaign("clean", reference_table(rows=120),
                            bars={"fix_rate": 0.35,
                                  "overcorrection_max": 0.05,
                                  "detect_rate": 0.5})
    result = run_campaign(camp, autoclean, "autoclean")
    check("autoclean clears normalization tiers, falls at "
          "wrong-value detection",
          result.tier_results[0].passed
          and not result.tier_results[2].passed
          and "wrong.detect_rate"
          in result.tier_results[2].failed_conditions)

    # ---------- the showdown ----------
    camp = compile_campaign("predict", predict_table(rows=200),
                            bars={"auroc": 0.62,
                                  "gap_max": 0.35},
                            outcome="readmitted")

    def vendor(train_rows, train_labels, test_rows):
        # age-only, with a crude imputation for missing cells
        def age_of(r):
            try:
                return float(r["age"])
            except ValueError:
                return 45.0
        return [age_of(r) / 105.0 for r in test_rows]

    sd = run_showdown(camp, vendor, "acme-predictor")
    text = sd.format_text()
    check("showdown walks every tier with ceiling, baseline, "
          "and vendor per line",
          len(sd.tiers) == 3
          and all("ceiling" in line
                  for line in text.splitlines()[1:]))
    check("the baseline outranks the age-only vendor on every "
          "tier",
          all(t.baseline_auroc > t.vendor_auroc
              for t in sd.tiers))
    check("the showdown names the verdict",
          "loses to a stdlib baseline" in text)
    sd2 = run_showdown(camp, vendor, "acme-predictor")
    check("showdowns are deterministic",
          sd2.format_text() == text)
    try:
        run_showdown(compile_campaign(
            "clean", reference_table(rows=40)), vendor)
        check("showdowns refuse non-predict campaigns", False)
    except ValueError:
        check("showdowns refuse non-predict campaigns", True)

    print("\nAll {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
