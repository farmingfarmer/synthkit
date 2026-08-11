"""SYNTH_H1 smoke: hybrid tables — a note column whose planted
contents drive the outcome. Validation teaching, deterministic
note assembly, the excludes trap guard, presence-in-the-logit,
the text-blind encoder, the mining hybrid solver beating the
tabular baseline, selectable showdown baselines, and the demo
spec's calibration contract.

Run from the repo root:

    python scripts/smoke_hybrid.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synthkit.autosolver import (FeatureEncoder, TextMiner,
                                 autosolver, autosolver_hybrid,
                                 run_showdown)
from synthkit.campaign import compile_campaign, run_campaign
from synthkit.mlmetrics import auroc
from synthkit.tableplan import plan_table
from synthkit.tablespec import TableSpec, TableSpecError

PASS = 0


def check(label, cond):
    global PASS
    if not cond:
        print("FAIL: {}".format(label))
        sys.exit(1)
    PASS += 1
    print("ok  {:02d}  {}".format(PASS, label))


def hybrid_spec(rows=400, seed=5) -> TableSpec:
    spec = {
     "title": "hybrid fixture", "rows": rows,
     "master_seed": seed,
     "columns": [
      {"name": "pid", "ctype": "str_id",
       "distribution": {"kind": "sequence", "prefix": "P",
                        "start": 1}},
      {"name": "age", "ctype": "int",
       "distribution": {"kind": "normal", "mean": 70,
                        "std": 10, "min": 40, "max": 95}},
      {"name": "note", "ctype": "note", "note": {
       "elements": [
        {"id": "nonadherence", "density": 0.25, "weight": 1.6,
         "phrasings": [
          "patient reports missing several doses",
          "poor adherence to medication regimen"]},
        {"id": "no_support", "density": 0.3, "weight": 1.2,
         "phrasings": [
          "lives alone with no home support",
          "no family available to assist at home"]}],
       "distractors": [
        {"id": "denies", "density": 0.5,
         "excludes": "nonadherence",
         "phrasings": ["denies missing any doses",
                       "adherent to all medications"]}],
       "fillers": ["Vital signs stable at discharge",
                   "Follow-up appointment scheduled",
                   "Diet counseling provided"]}}],
     "outcomes": [{"name": "readmit", "kind": "logistic",
       "intercept": -3.4,
       "coefficients": {"age": 0.02,
                        "note.nonadherence": 1.6,
                        "note.no_support": 1.2},
       "target_prevalence": [0.05, 0.25]}]}
    return TableSpec.from_json(json.dumps(spec))


def main():
    # ---------- validation teaches ----------
    bad = hybrid_spec()
    bad.columns[2].note["elements"] = []
    try:
        bad.validate()
        check("note columns without elements are refused with "
              "teaching", False)
    except TableSpecError as e:
        check("note columns without elements are refused with "
              "teaching", "note.elements" in str(e))
    bad = hybrid_spec()
    bad.outcomes[0]["coefficients"]["note.ghost"] = 1.0
    try:
        bad.validate()
        check("dotted coefficients naming unplanted elements "
              "are refused, listing the real ones", False)
    except TableSpecError as e:
        check("dotted coefficients naming unplanted elements "
              "are refused, listing the real ones",
              "ghost" in str(e) and "nonadherence" in str(e))
    bad = hybrid_spec()
    bad.columns[2].mess.typo_rate = 0.1
    try:
        bad.validate()
        check("cell mess on note columns is refused", False)
    except TableSpecError as e:
        check("cell mess on note columns is refused",
              "mess IS the text" in str(e))

    # ---------- planning ----------
    spec = hybrid_spec()
    bp = plan_table(spec)
    bp2 = plan_table(spec)
    check("note assembly is deterministic",
          [r["note"] for r in bp.dirty_rows]
          == [r["note"] for r in bp2.dirty_rows])
    truth = bp.note_truth["note"]
    check("presence truth is ledgered per row per element",
          len(truth) == 400
          and set(truth[0]) == {"nonadherence", "no_support"})
    leak = [i for i, p in enumerate(truth)
            if p["nonadherence"]
            and ("denies missing" in bp.clean_rows[i]["note"]
                 or "adherent to all" in
                 bp.clean_rows[i]["note"])]
    check("`excludes` keeps the denial trap out of truly "
          "nonadherent patients' notes", not leak)
    probs = bp.true_probs["readmit"]
    w = sum(probs[i] for i, p in enumerate(truth)
            if p["nonadherence"])
    w /= max(sum(1 for p in truth if p["nonadherence"]), 1)
    wo = sum(probs[i] for i, p in enumerate(truth)
             if not p["nonadherence"])
    wo /= max(sum(1 for p in truth
                  if not p["nonadherence"]), 1)
    check("planted note elements move the outcome logit",
          w > wo + 0.1)
    prev = sum(1 for r in bp.clean_rows
               if r["readmit"] == "True") / 400
    check("the outcome stays a minority class",
          0.03 < prev < 0.3)

    # ---------- solvers ----------
    enc = FeatureEncoder().fit(bp.dirty_rows)
    check("the tabular encoder is text-blind by design "
          "(note column skipped)",
          all(c.name != "note" for c in enc.columns))
    train = hybrid_spec(seed=1005)
    tr, te = plan_table(train), plan_table(spec)
    y = lambda b: [r["readmit"] == "True" for r in b.clean_rows]
    strip = lambda b: [{k: v for k, v in r.items()
                        if k != "readmit"}
                       for r in b.dirty_rows[:400]]
    tab = auroc(autosolver()(strip(tr), y(tr)[:400],
                             strip(te)), y(te)[:400])
    hyb = auroc(autosolver_hybrid()(strip(tr), y(tr)[:400],
                                    strip(te)), y(te)[:400])
    ceil = auroc(te.true_probs["readmit"], y(te))
    check("the mining hybrid decisively beats the text-blind "
          "tabular baseline (the demo's spine)",
          hyb > tab + 0.05 and ceil >= hyb)
    miner = TextMiner().fit(strip(tr), y(tr)[:400])
    check("the miner learned real note terms from labels "
          "alone", len(miner.terms) >= 10
          and any("doses" in t or "adherence" in t
                  or "support" in t for t in miner.terms))

    # ---------- campaign + showdown ----------
    camp = compile_campaign("predict", spec,
                            bars={"auroc": 0.6,
                                  "gap_max": 0.35},
                            outcome="readmit")
    res = run_campaign(camp, autosolver_hybrid(),
                       "hybrid-baseline")
    check("hybrid specs walk the predict ladder",
          res.highest_passed >= 2)
    sd = run_showdown(camp, autosolver(),
                      vendor_name="VendorCo",
                      baseline=autosolver_hybrid(),
                      baseline_name="synthkit hybrid")
    text = sd.format_text()
    check("showdowns take a selectable baseline and name it",
          "synthkit hybrid" in text and "VendorCo" in text)
    check("with the hybrid as the floor, the text-blind "
          "vendor loses on signal-bearing tiers",
          any(t.vendor_auroc < t.baseline_auroc - 0.03
              for t in sd.tiers))

    # ---------- the demo spec's contract ----------
    demo = TableSpec.from_json(
        Path("readmit_demo.json").read_text(encoding="utf-8"))
    demo.validate()
    check("the committed demo spec validates with its note "
          "column, mixtures, rules, and declared imbalance",
          demo.rows == 1000
          and demo.outcomes[0]["target_prevalence"]
          == [0.05, 0.12]
          and any(c.ctype == "note" for c in demo.columns))

    print("\nAll {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
