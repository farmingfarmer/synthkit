"""SYNTH_R1 smoke: regression outcomes and relational tables
proven — linear truth with known noise sets an R^2 ceiling the
baseline nearly reaches, the regress ladder walks blinded, and
linked tables plant seeded foreign keys with ledgered orphans a
perfect flagger recovers exactly.

Run from the repo root:

    python scripts/smoke_regress_relational.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synthkit.autosolver import autosolver_regress
from synthkit.campaign import CampaignError, compile_campaign, \
    run_campaign
from synthkit.examples import reference_table
from synthkit.lint import lint_table
from synthkit.relational import (
    Link,
    RelationalBlueprint,
    RelationalIntegrityError,
    RelationalSpec,
    RelationalSpecError,
    evaluate_links,
    load_relational,
    plan_relational,
    write_relational,
)
from synthkit.tableeval import evaluate_regression
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


def cost_spec(rows=300, sigma=800.0, target=None) -> TableSpec:
    spec = reference_table(rows=rows, master_seed=17)
    oc = {"name": "episode_cost", "kind": "linear",
          "intercept": 500.0,
          "coefficients": {"los_days": 2100.0, "age": 12.0,
                           "department=oncology": 3000.0},
          "noise_sigma": sigma}
    if target:
        oc["target_range"] = target
    spec.outcomes = [oc]
    return spec


def main():
    # ================= regression =================
    try:
        bad = cost_spec()
        del bad.outcomes[0]["noise_sigma"]
        bad.validate()
        check("linear outcomes demand noise_sigma with the "
              "ceiling rationale", False)
    except TableSpecError as e:
        check("linear outcomes demand noise_sigma with the "
              "ceiling rationale",
              "irreducible error" in str(e))

    spec = cost_spec()
    bp = plan_table(spec)
    y = [float(r["episode_cost"]) for r in bp.clean_rows]
    los = [float(r["los_days"]) for r in bp.clean_rows]
    check("linear outcome is a continuous column tracking its "
          "sources",
          len(set(y)) > 250
          and sum(1 for a, b in zip(los, y)
                  if (a - sum(los) / len(los))
                  * (b - sum(y) / len(y)) > 0) > 200)
    bp2 = plan_table(cost_spec())
    check("linear planning is deterministic",
          [r["episode_cost"] for r in bp.clean_rows]
          == [r["episode_cost"] for r in bp2.clean_rows])

    oracle = evaluate_regression(bp, "episode_cost",
                                 bp.true_probs["episode_cost"],
                                 "oracle")
    check("the oracle (noiseless signal) scores R^2 at the "
          "ceiling",
          abs(oracle.r2 - oracle.ceiling_r2) < 0.005
          and oracle.r2_gap <= 0.005)
    tight = plan_table(cost_spec(sigma=100.0))
    loose = plan_table(cost_spec(sigma=25000.0))
    c_tight = evaluate_regression(
        tight, "episode_cost", tight.true_probs["episode_cost"],
        "o").ceiling_r2
    c_loose = evaluate_regression(
        loose, "episode_cost", loose.true_probs["episode_cost"],
        "o").ceiling_r2
    check("less noise, higher ceiling: sigma is the dial",
          c_tight > 0.98 and c_loose < 0.5
          and c_tight > c_loose)

    camp = compile_campaign("regress", cost_spec(rows=250),
                            bars={"r2": 0.5, "gap_max": 0.25},
                            outcome="episode_cost")
    check("regress campaigns compile three tiers",
          len(camp.tiers) == 3
          and "regression of `episode_cost`" in camp.title)
    result = run_campaign(camp, autosolver_regress(),
                          "baseline")
    text = result.format_text()
    check("the linear baseline clears the blinded ladder with "
          "ceilings reported",
          result.highest_passed == 3
          and "ceiling" in text and "R^2" in text)
    try:
        compile_campaign("regress", reference_table(rows=40),
                         outcome="nope")
        check("regress refuses non-linear outcomes", False)
    except CampaignError as e:
        check("regress refuses non-linear outcomes",
              "kind-linear" in str(e))

    report = lint_table(cost_spec(target=[20000, 30000]))
    check("lint L1-range warns when the realized mean misses "
          "the declared range",
          any(f.code == "L1-range" and f.level == "WARN"
              for f in report.findings))
    report = lint_table(cost_spec(target=[5000, 20000]))
    check("lint L1-range passes an honest range as INFO",
          any(f.code == "L1-range" and f.level == "INFO"
              for f in report.findings))

    # ================= relational =================
    def rel_spec(orphan_rate=0.08) -> RelationalSpec:
        patients = reference_table(rows=60, master_seed=1)
        encounters = reference_table(rows=200, master_seed=2)
        # strip the id column collision: encounters keep their
        # own patient-independent columns; the link injects the
        # fk. Rename encounter ids to avoid confusion.
        for c in encounters.columns:
            if c.name == "patient_id":
                c.name = "encounter_id"
        for rule in encounters.rules:
            pass
        return RelationalSpec(
            title="patients + encounters",
            master_seed=99,
            tables={"patients": patients,
                    "encounters": encounters},
            links=[Link(child="encounters", parent="patients",
                        parent_key="patient_id",
                        fk_column="patient_ref",
                        orphan_rate=orphan_rate)])

    try:
        bad = rel_spec()
        bad.links[0].parent_key = "age"
        bad.validate()
        check("parent keys must be str_id sequences", False)
    except RelationalSpecError as e:
        check("parent keys must be str_id sequences",
              "uniqueness is the whole point" in str(e))
    try:
        bad = rel_spec()
        bad.links[0].fk_column = "age"
        bad.validate()
        check("fk columns must not collide with child columns",
              False)
    except RelationalSpecError as e:
        check("fk columns must not collide with child columns",
              "the link injects it" in str(e))

    rbp = plan_relational(rel_spec())
    patients = rbp.blueprints["patients"]
    encounters = rbp.blueprints["encounters"]
    parent_keys = {r["patient_id"] for r in patients.clean_rows}
    check("every CLEAN fk references a real parent — truth "
          "keeps its integrity",
          all(r["patient_ref"] in parent_keys
              for r in encounters.clean_rows))
    orphans = rbp.link_ledger
    dirty_fks = [r["patient_ref"]
                 for r in encounters.dirty_rows]
    check("orphans are planted in dirty rows at roughly the "
          "declared rate, every one ledgered",
          10 <= len(orphans) <= 25
          and all(dirty_fks[m.row] == m.orphan_key
                  and m.orphan_key not in parent_keys
                  for m in orphans))
    rbp2 = plan_relational(rel_spec())
    check("relational planning is deterministic",
          [r["patient_ref"] for r in
           rbp2.blueprints["encounters"].dirty_rows]
          == dirty_fks)

    link = rel_spec().links[0]
    perfect = [i in {m.row for m in orphans}
               for i in range(len(dirty_fks))]
    rep = evaluate_links(rbp, link, perfect, "oracle-flagger")
    check("a perfect flagger scores recall 1.0 precision 1.0",
          rep.recall == 1.0 and rep.precision == 1.0)
    naive = [fk not in parent_keys for fk in dirty_fks]
    rep_naive = evaluate_links(rbp, link, naive, "set-checker")
    check("a set-membership checker also aces it — the honest "
          "baseline for orphan detection",
          rep_naive.recall == 1.0
          and rep_naive.precision == 1.0)
    mute = [False] * len(dirty_fks)
    rep_mute = evaluate_links(rbp, link, mute, "mute")
    check("a mute flagger scores zero recall",
          rep_mute.recall == 0.0)

    tmp = Path(tempfile.mkdtemp(prefix="synthkit_rel_"))
    try:
        write_relational(tmp / "r1", rel_spec(), rbp)
        loaded = load_relational(tmp / "r1")
        check("relational artifacts round-trip under a layered "
              "manifest",
              len(loaded["orphans"]) == len(orphans)
              and (tmp / "r1" / "encounters"
                   / "dirty.csv").is_file())
        victim = tmp / "r1" / "links.json"
        victim.write_text(victim.read_text() + " ",
                          encoding="utf-8")
        try:
            load_relational(tmp / "r1")
            check("tampering with the link ledger is caught",
                  False)
        except RelationalIntegrityError:
            check("tampering with the link ledger is caught",
                  True)
    finally:
        shutil.rmtree(tmp)

    print("\nAll {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
