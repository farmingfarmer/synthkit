"""SYNTH_B1 smoke: ML metrics exact on known cases, outcome
generation with planted signal and true probabilities, ceiling
AUROC, split-blinded prediction evaluation, and campaign compilation
plus runs for all three goals.

Run from the repo root:

    python scripts/smoke_campaign.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synthkit.campaign import (
    Campaign,
    CampaignError,
    CampaignIntegrityError,
    compile_campaign,
    load_campaign,
    run_campaign,
    write_campaign,
)
from synthkit.examples import (
    reference_spec,
    reference_table,
    regex_extract,
)
from synthkit.harness import MetricError
from synthkit.mlmetrics import (
    MetricInputError,
    at_threshold,
    auroc,
    brier,
    ceiling_auroc,
)
from synthkit.tableeval import (
    evaluate_cleaning,
    evaluate_prediction,
    resolve_prediction_metric,
)
from synthkit.tableplan import load_table, plan_table, write_table
from synthkit.tablespec import TableSpec, TableSpecError

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
    # ---------- metrics on known cases ----------
    check("perfect ranking scores AUROC 1.0",
          auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0)
    check("inverted ranking scores AUROC 0.0",
          auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0)
    check("all-tied scores land at 0.5",
          auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5)
    try:
        auroc([0.1, 0.2], [1, 1])
        check("single-class AUROC refused", False)
    except MetricInputError:
        check("single-class AUROC refused", True)
    check("perfect probabilities score Brier 0",
          brier([0.0, 1.0], [0, 1]) == 0.0)
    th = at_threshold([0.9, 0.8, 0.2, 0.4], [1, 1, 0, 1])
    check("threshold metrics computed",
          th["accuracy"] == 0.75 and th["precision"] == 1.0)

    # ---------- outcome validation ----------
    bad = predict_table()
    bad.outcomes.append({
        "name": "age", "kind": "mystery",
        "coefficients": {"ghost": 1.0,
                         "patient_name": 2.0,
                         "age=old": 1.0}})
    try:
        bad.validate()
        check("broken outcomes rejected", False)
    except TableSpecError as e:
        msg = str(e)
        check("outcome validation reports every problem",
              "collides with a column" in msg
              and "kind must be `logistic` or `linear`" in msg
              and "requires `intercept`" in msg
              and "names no column" in msg
              and "needs a numeric or bool column" in msg
              and "needs a category column" in msg)

    # ---------- generation with planted signal ----------
    spec = predict_table()
    bp = plan_table(spec)
    bp2 = plan_table(predict_table())
    check("outcome generation is deterministic",
          bp.clean_rows == bp2.clean_rows
          and bp.true_probs == bp2.true_probs)
    labels = [r["readmitted"] == "True" for r in bp.clean_rows]
    rate = sum(labels) / len(labels)
    check("outcome column generated with sensible prevalence",
          "readmitted" in bp.columns
          and len(bp.true_probs["readmitted"]) == 250
          and 0.05 < rate < 0.6)
    long_rate = sum(
        1 for r in bp.clean_rows
        if int(r["los_days"]) >= 8 and r["readmitted"] == "True"
    ) / max(sum(1 for r in bp.clean_rows
                if int(r["los_days"]) >= 8), 1)
    short_rate = sum(
        1 for r in bp.clean_rows
        if int(r["los_days"]) <= 4 and r["readmitted"] == "True"
    ) / max(sum(1 for r in bp.clean_rows
                if int(r["los_days"]) <= 4), 1)
    check("planted signal shows: long stays readmit more",
          long_rate > short_rate + 0.1)
    check("outcome labels never messy, dirty rows carry them",
          all(row["readmitted"] in ("True", "False")
              for row in bp.dirty_rows))

    ceiling = ceiling_auroc(bp.true_probs["readmitted"],
                            [1 if x else 0 for x in labels])
    check("ceiling AUROC is strong on strong signal",
          ceiling > 0.7)

    # ---------- persistence ----------
    import shutil
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="synthkit_b_"))
    run = write_table(tmp / "p1", spec, bp)
    _spec_l, bp_l = load_table(run)
    check("true probabilities and outcome column round-trip",
          bp_l.true_probs["readmitted"]
          == [round(x, 6) for x in bp.true_probs["readmitted"]]
          and "readmitted" in bp_l.columns)
    shutil.rmtree(tmp)

    # ---------- prediction evaluation ----------
    int_labels = [1 if x else 0 for x in labels]
    oracle = evaluate_prediction(
        bp, "readmitted", bp.true_probs["readmitted"], "oracle")
    check("the oracle meets the ceiling exactly",
          abs(oracle.auroc - oracle.ceiling_auroc) < 1e-9
          and oracle.auroc_gap < 1e-9)
    flat = evaluate_prediction(
        bp, "readmitted", [0.5] * 250, "coinflip")
    check("an uninformative solver shows the full gap",
          flat.auroc == 0.5
          and flat.auroc_gap > 0.2
          and "ceiling" in flat.format_text())
    try:
        evaluate_prediction(bp, "readmitted", [0.5] * 10)
        check("score-length contract enforced", False)
    except ValueError:
        check("score-length contract enforced", True)
    try:
        evaluate_prediction(bp, "ghost", [0.5] * 250)
        check("unknown outcome refused", False)
    except ValueError:
        check("unknown outcome refused", True)
    check("prediction metrics resolve; unknown paths raise",
          resolve_prediction_metric(flat, "predict.auroc") == 0.5)
    try:
        resolve_prediction_metric(flat, "predict.magic")
        check("unknown prediction metric raises", False)
    except MetricError:
        check("unknown prediction metric raises", True)

    # ---------- campaign: clean ----------
    camp = compile_campaign("clean", reference_table(rows=120),
                            bars={"fix_rate": 0.95,
                                  "detect_rate": 0.5})
    t1 = TableSpec.from_json(camp.tiers[0].spec_json)
    t3 = TableSpec.from_json(camp.tiers[2].spec_json)
    ref = reference_table(rows=120)
    check("clean tiers escalate: half mess, no lies -> heavier "
          "mess with lies",
          all(c.mess.wrong_rate == 0.0 for c in t1.columns)
          and any(c.mess.wrong_rate > 0 for c in t3.columns)
          and t1.columns[2].mess.missing_rate
          < ref.columns[2].mess.missing_rate
          and t3.columns[2].mess.missing_rate
          > ref.columns[2].mess.missing_rate)
    check("only the adversarial tier demands wrong-value "
          "detection",
          not any("wrong.detect_rate" in c
                  for c in camp.tiers[0].conditions)
          and any("wrong.detect_rate" in c
                  for c in camp.tiers[2].conditions))

    def oracle_cleaner_no_detect(rows):
        # Perfect restoration, flags duplicates, but never
        # suspects wrong values — a top-tier normalizer.
        spec_now = oracle_cleaner_no_detect.bp
        out = []
        for i, _r in enumerate(rows):
            src = spec_now.duplicate_of.get(i, i)
            row = dict(spec_now.clean_rows[src])
            if i in spec_now.duplicate_of:
                row["_duplicate"] = "true"
            out.append(row)
        return out

    def clean_solver(rows):
        from synthkit.tableplan import plan_table as _pt
        # re-derive the tier blueprint the runner planned: the
        # solver here is an oracle for test purposes only.
        return oracle_cleaner_no_detect(rows)

    # Wire the oracle per tier by monkey-patching plan lookup:
    # simpler — run each tier manually mirroring run_campaign.
    from synthkit.tableeval import resolve_table_metric
    from synthkit.harness import Condition
    tier_pass = []
    for tier in camp.tiers:
        tspec = TableSpec.from_json(tier.spec_json)
        tbp = plan_table(tspec)
        oracle_cleaner_no_detect.bp = tbp
        rep = evaluate_cleaning(
            tbp, oracle_cleaner_no_detect(tbp.dirty_rows),
            "oracle-no-detect")
        ok = all(Condition.parse(c).holds(
            resolve_table_metric(rep, Condition.parse(c).metric))
            for c in tier.conditions)
        tier_pass.append(ok)
    check("oracle normalizer clears tiers 1-2, falls at "
          "adversarial detection",
          tier_pass == [True, True, False])

    # ---------- campaign: predict (split-blinded) ----------
    camp = compile_campaign("predict", predict_table(rows=200),
                            bars={"auroc": 0.62, "gap_max": 0.35},
                            outcome="readmitted")

    def los_solver(train_rows, train_labels, test_rows):
        # Learns nothing; scores by a single sensible feature.
        return [float(r["los_days"]) / 45.0 for r in test_rows]

    result = run_campaign(camp, los_solver, "los-ranker")
    ceilings = [tr.measured["predict.ceiling_auroc"]
                for tr in result.tier_results]
    check("signal scaling moves the ceiling monotonically",
          ceilings[0] > ceilings[1] > ceilings[2])
    check("split-blinded predict campaign runs and reports the "
          "ladder",
          result.tier_results[0].passed
          and result.highest_passed >= 1
          and "ceiling" in result.format_text())
    result2 = run_campaign(camp, los_solver, "los-ranker")
    check("campaign runs are deterministic",
          result.format_text() == result2.format_text())

    # ---------- campaign: extract (compile) ----------
    camp = compile_campaign("extract", reference_spec(size=16),
                            bars={"recall": 0.8})
    import json as _json
    gentle = _json.loads(camp.tiers[0].spec_json)
    adver = _json.loads(camp.tiers[2].spec_json)

    def difficulty_counts(d):
        out = {"easy": 0, "medium": 0, "hard": 0}
        for uf in d["unstructured_fields"]:
            for el in uf["target_elements"]:
                out[el["difficulty"]] += 1
        return out

    g_c, a_c = difficulty_counts(gentle), difficulty_counts(adver)
    g_dis = gentle["unstructured_fields"][0]["distractors"][0]
    a_dis = adver["unstructured_fields"][0]["distractors"][0]
    check("extract tiers shift difficulty and distractor density",
          g_c["hard"] < a_c["hard"]
          and g_c["easy"] > a_c["easy"]
          and a_dis["density"] > g_dis["density"]
          and not any("fp_rate" in c
                      for c in camp.tiers[0].conditions)
          and any("fp_rate" in c
                  for c in camp.tiers[1].conditions))
    ext_result = run_campaign(camp, regex_extract, "regex-naive")
    check("extract campaign runs stub-rendered and the naive "
          "extractor hits the trap wall",
          ext_result.highest_passed <= 1
          and "CAMPAIGN" in ext_result.format_text())

    # ---------- persistence ----------
    import shutil as _sh
    import tempfile as _tf
    tmp2 = Path(_tf.mkdtemp(prefix="synthkit_camp_"))
    run_dir = write_campaign(tmp2 / "c1", camp, ext_result)
    camp_l = load_campaign(run_dir)
    check("campaigns round-trip with tiers, specs, conditions",
          camp_l.goal == camp.goal
          and len(camp_l.tiers) == 3
          and camp_l.tiers[2].conditions
          == camp.tiers[2].conditions
          and camp_l.tiers[0].spec_json is not None)
    import json as _j
    saved = _j.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    check("campaign results persist with the full ladder",
          saved["highest_passed"] == ext_result.highest_passed
          and len(saved["tiers"]) == 3)
    # ---------- append-only trials ----------
    from synthkit.campaign import format_trials, read_trials
    write_campaign(run_dir, camp, ext_result,
                   result_name="second-arm")
    trials = read_trials(run_dir)
    check("trial history is append-only, never clobbered",
          len(trials) == 2
          and trials[0]["solver"] == "unnamed"
          and trials[1]["solver"] == "second-arm")
    table = format_trials(run_dir)
    check("the trials table compares arms per tier",
          "2 arm(s)" in table
          and "second-arm" in table
          and "cleared" in table)

    victim = run_dir / "campaign.json"
    victim.write_text(victim.read_text(encoding="utf-8") + " ", encoding="utf-8")
    try:
        load_campaign(run_dir)
        check("tampered campaign refuses to load", False)
    except CampaignIntegrityError as e:
        check("tampered campaign refuses to load",
              "campaign.json" in str(e))

    # ---------- CLI end-to-end ----------
    import contextlib
    import io as _io
    from synthkit.cli import main as cli
    spec_path = tmp2 / "base_table.json"
    spec_path.write_text(reference_table(rows=60).to_json(),
                         encoding="utf-8")

    def run_cli(argv):
        out = _io.StringIO()
        with contextlib.redirect_stdout(out):
            try:
                rc = cli(argv)
            except SystemExit as e:
                rc = int(e.code or 0)
        return rc, out.getvalue()

    rc, out = run_cli(["campaign-compile", "--goal", "clean",
                       "--spec", str(spec_path),
                       "--bars", "fix_rate=0.9,detect_rate=0.5",
                       "-o", str(tmp2 / "c2")])
    check("campaign-compile CLI writes the ladder",
          rc == 0 and "3 tier(s)" in out
          and (tmp2 / "c2" / "manifest.json").is_file())
    rc, out = run_cli(["campaign-run", str(tmp2 / "c2"),
                       "--solver",
                       "synthkit.examples:strip_cleaner"])
    check("campaign-run CLI walks the ladder, persists the "
          "result, exits nonzero on failure",
          rc == 1 and "CAMPAIGN [clean]" in out
          and (tmp2 / "c2" / "result.json").is_file())
    _sh.rmtree(tmp2)

    # ---------- campaign errors ----------
    try:
        compile_campaign("levitate", reference_table())
        check("unknown goal refused", False)
    except CampaignError:
        check("unknown goal refused", True)
    try:
        compile_campaign("predict", reference_table(),
                         outcome="ghost")
        check("predict without a generated outcome refused",
              False)
    except CampaignError:
        check("predict without a generated outcome refused", True)

    print("\nAll {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
