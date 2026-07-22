"""Builds notebooks/synthkit_demo.ipynb — the guided tour for
SageMaker or any Jupyter kernel.

Strategy: reuse the standalone notebook's inlined-module cells
(zero imports beyond stdlib, no network, no pip), then append a
narrated demo: the encounter dataset from an English paragraph's
calibrated spec, semantic lint, a predict ladder with intervals,
the showdown sentence, a regression ceiling, and relational join
mess with the precision trap. Ends with the vendor-study summary.

The generated notebook is headless-executed by the caller to
prove it runs top to bottom.

Run from the repo root:

    python scripts/build_demo_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STANDALONE = ROOT / "notebooks" / "synthkit_standalone.ipynb"
OUT = ROOT / "notebooks" / "synthkit_demo.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {},
            "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


ENCOUNTER_SPEC = {
    "title": "Inpatient encounters (readmission study)",
    "rows": 400,
    "master_seed": 12345,
    "duplicate_rate": 0.03,
    "columns": [
        {"name": "patient_id", "ctype": "str_id",
         "distribution": {"kind": "sequence", "prefix": "P",
                          "start": 20000}},
        {"name": "patient_name", "ctype": "person_name",
         "mess": {"case_rate": 0.05, "space_rate": 0.02}},
        {"name": "age", "ctype": "int",
         "distribution": {"kind": "normal", "mean": 62,
                          "std": 15, "min": 18, "max": 100},
         "mess": {"missing_rate": 0.12}},
        {"name": "admitting_department", "ctype": "category",
         "distribution": {"kind": "categorical",
                          "choices": ["internal medicine",
                                      "cardiology", "oncology",
                                      "emergency"],
                          "weights": [0.5, 0.2, 0.2, 0.1]},
         "mess": {"typo_rate": 0.05}},
        {"name": "length_of_stay", "ctype": "int",
         "distribution": {"kind": "mixture",
                          "components": [
                              {"kind": "uniform", "min": 1,
                               "max": 4},
                              {"kind": "uniform", "min": 10,
                               "max": 18}],
                          "weights": [0.75, 0.25]},
         "mess": {"outlier_rate": 0.01, "outlier_factor": 2}},
        {"name": "total_charges", "ctype": "float",
         "mess": {"outlier_rate": 0.02,
                  "outlier_factor": 40.0}},
        {"name": "admission_date", "ctype": "date",
         "distribution": {"kind": "date_range",
                          "start": "2026-01-01",
                          "end": "2026-06-30"},
         "mess": {"format_rate": 0.05}},
        {"name": "discharge_date", "ctype": "date",
         "mess": {"wrong_rate": 0.02}},
    ],
    "rules": [
        {"kind": "date_after", "earlier": "admission_date",
         "later": "discharge_date",
         "days_from": "length_of_stay"},
        {"kind": "derived", "target": "total_charges",
         "source": "length_of_stay", "factor": 2100,
         "noise_sigma": 0.15},
    ],
    "outcomes": [
        {"name": "readmitted", "kind": "logistic",
         "intercept": -3.8,
         "coefficients": {"length_of_stay": 0.1, "age": 0.02,
                          "admitting_department=oncology": 0.8},
         "target_prevalence": [0.10, 0.25]},
    ],
}

DEMO_CELLS = [
    md("# synthkit — the guided tour\n\n"
       "Everything above this cell is the LIBRARY, inlined so "
       "this notebook needs nothing but a Python kernel: no "
       "pip, no network, no files. Everything below is the "
       "demo: an English paragraph's dataset, born with an "
       "answer key, measured to its ceiling.\n\n"
       "*Provenance: this spec was compiled from plain English "
       "by a local LLM through synthkit's assisted-never-"
       "trusted loop, then human-calibrated at the review "
       "gate. The compiler converged over four benchmark runs "
       "(6 problems -> 2 -> 2 -> 0).*"),
    code("ENCOUNTER = " + json.dumps(ENCOUNTER_SPEC, indent=2)),
    md("## 1. The dataset that knows its own truth\n\n"
       "400 encounters: bimodal stays, charges derived at "
       "\\$2100/day, discharge dates COUPLED to length of stay, "
       "a readmission outcome with declared prevalence — and "
       "seven kinds of deliberate mess, every corruption "
       "ledgered."),
    code("import json as _json\n"
         "spec = TableSpec.from_json(_json.dumps(ENCOUNTER))\n"
         "spec.validate()\n"
         "bp = plan_table(spec)\n"
         "print('rows:', len(bp.clean_rows), '  dirty:',\n"
         "      len(bp.dirty_rows), ' (duplicates included)')\n"
         "ops = {}\n"
         "for m in bp.ledger:\n"
         "    ops[m.op] = ops.get(m.op, 0) + 1\n"
         "print('mess by op:', dict(sorted(ops.items())))\n"
         "prev = sum(1 for r in bp.clean_rows\n"
         "           if r['readmitted'] == 'True') / 400\n"
         "print('readmission prevalence: {:.1%}'.format(prev))\n"
         "print()\n"
         "cols = bp.columns\n"
         "print(' | '.join(c[:12] for c in cols))\n"
         "for row in bp.dirty_rows[:5]:\n"
         "    print(' | '.join(str(row[c])[:12] for c in cols))"),
    md("## 2. Semantic lint — does the spec mean what was "
       "meant?\n\nValidation checks lawfulness; lint plans a "
       "probe and checks INTENT: realized prevalence against "
       "the declared target, derived magnitudes, rule "
       "coupling. (In live use, a miscalibrated intercept that "
       "realized 67% against a 15-20% ask was caught exactly "
       "here, before any rendering.)"),
    code("print(lint_table(spec).format_text())"),
    md("## 3. The ladder — a goal becomes a measured claim\n\n"
       "Goal `predict` compiles to three tiers (amplified / "
       "as-specified / attenuated signal). The baseline trains "
       "on a BLINDED shifted-seed table. Every evidence line "
       "carries a confidence interval; verdicts say DECISIVE "
       "or INCONCLUSIVE — and inconclusive ones prescribe the "
       "sample size that would resolve them, which synthkit "
       "can generate on request."),
    code("camp = compile_campaign('predict', spec,\n"
         "                        bars={'auroc': 0.6,\n"
         "                              'gap_max': 0.3},\n"
         "                        outcome='readmitted')\n"
         "result = run_campaign(camp, autosolver(),\n"
         "                      'synthkit-baseline')\n"
         "print(result.format_text())"),
    md("## 4. The showdown sentence\n\nThe number a vendor "
       "meeting actually needs: ceiling / baseline / vendor, "
       "per tier. Here the 'vendor' is a naive age-only "
       "scorer."),
    code("def age_vendor(train_rows, train_labels, test_rows):\n"
         "    def age(r):\n"
         "        try:\n"
         "            return float(r['age'])\n"
         "        except ValueError:\n"
         "            return 62.0\n"
         "    return [age(r) / 100.0 for r in test_rows]\n"
         "\n"
         "print(run_showdown(camp, age_vendor,\n"
         "                   'age-only-vendor').format_text())"),
    md("## 5. Regression — the ceiling for continuous "
       "targets\n\nA linear outcome's declared noise IS the "
       "irreducible error, so the R^2 ceiling is exact by "
       "construction. 'This cost data supports at most R^2=X' "
       "is a sentence no real dataset can say."),
    code("rspec = TableSpec.from_json(_json.dumps(ENCOUNTER))\n"
         "rspec.outcomes = [{\n"
         "    'name': 'episode_cost', 'kind': 'linear',\n"
         "    'intercept': 500.0,\n"
         "    'coefficients': {'length_of_stay': 2100.0,\n"
         "                     'age': 12.0,\n"
         "                     'admitting_department=oncology':\n"
         "                     3000.0},\n"
         "    'noise_sigma': 4000.0}]\n"
         "rcamp = compile_campaign('regress', rspec,\n"
         "                         bars={'r2': 0.5,\n"
         "                               'gap_max': 0.25},\n"
         "                         outcome='episode_cost')\n"
         "rres = run_campaign(rcamp, autosolver_regress(),\n"
         "                    'synthkit-baseline')\n"
         "print(rres.format_text())"),
    md("## 6. Relational — join mess with an answer key\n\n"
       "Patients + encounters linked by seeded foreign keys. "
       "Join mess has tiers: random orphans (easy), and FORMAT "
       "DRIFT — valid references with mangled formatting, the "
       "precision trap that makes naive set-membership "
       "checkers false-flag good data."),
    code("patients = TableSpec.from_json(_json.dumps(\n"
         "    ENCOUNTER))\n"
         "patients.title = 'patients'\n"
         "patients.rows = 60\n"
         "patients.outcomes = []\n"
         "child = TableSpec.from_json(_json.dumps(ENCOUNTER))\n"
         "child.title = 'encounters'\n"
         "child.rows = 200\n"
         "child.outcomes = []\n"
         "for c in child.columns:\n"
         "    if c.name == 'patient_id':\n"
         "        c.name = 'encounter_id'\n"
         "rel = RelationalSpec(\n"
         "    title='patients+encounters', master_seed=99,\n"
         "    tables={'patients': patients,\n"
         "            'encounters': child},\n"
         "    links=[Link(child='encounters',\n"
         "                parent='patients',\n"
         "                parent_key='patient_id',\n"
         "                fk_column='patient_ref',\n"
         "                orphan_rate=0.06,\n"
         "                drift_rate=0.10)])\n"
         "rbp = plan_relational(rel)\n"
         "keys = {r['patient_id'] for r in\n"
         "        rbp.blueprints['patients'].clean_rows}\n"
         "dirty = [r['patient_ref'] for r in\n"
         "         rbp.blueprints['encounters'].dirty_rows]\n"
         "naive = [fk not in keys for fk in dirty]\n"
         "print('set-checker        ',\n"
         "      evaluate_links(rbp, rel.links[0], naive,\n"
         "                     'set-checker').format_text())\n"
         "fixed = [fk.strip().upper() not in\n"
         "         {k.upper() for k in keys} for fk in dirty]\n"
         "print('normalizing-checker',\n"
         "      evaluate_links(rbp, rel.links[0], fixed,\n"
         "                     'normalizing')\n"
         "      .format_text())"),
    md("## 7. What this instrument has already done\n\n"
       "In one day of live use, synthkit conducted a complete "
       "vendor study on a local LLM as a clinical extractor:\n\n"
       "- **Characterized** a dangerous failure: 26% of "
       "discontinued medications extracted as current "
       "(decisive, tier-stable, density-independent)\n"
       "- **Intervened** with a committed prompt treatment: "
       "26% -> ~4%, recall held at a decisive 1.000\n"
       "- **Replicated** across models: the gap is endemic to "
       "small models; the treatment transfers\n"
       "- **Hardened itself**: temperature-0 evaluation after "
       "sampling variance flipped a bar-edge verdict; "
       "intervals and prescriptions after n=19 proved too "
       "small to resolve a 0.10 bar\n"
       "- **Confirmed** at the sample size the instrument "
       "itself prescribed: 750 calls, 0 malformed, residual "
       "3.7% [1.6-8.4]\n\n"
       "Every number regenerates from a spec and a seed. "
       "That is the product: truth you planted, claims you "
       "can measure, and an instrument that knows what it "
       "knows."),
]


def main():
    nb = json.loads(STANDALONE.read_text(encoding="utf-8"))
    nb["cells"] = nb["cells"] + DEMO_CELLS
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("demo notebook -> {} ({} cells)".format(
        OUT, len(nb["cells"])))


if __name__ == "__main__":
    main()
