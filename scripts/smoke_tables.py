"""SYNTH_A1 smoke: the tabular half proven — total validation,
determinism, column-keyed seed independence, ledger exactness (the
dirty table differs from the clean table exactly where the ledger
says), realized mess rates, duplicates, integrity-checked IO, and
cleaning evaluation against three characterized cleaners (perfect,
whitespace-only, and a destructive one that overcorrects).

Run from the repo root:

    python scripts/smoke_tables.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synthkit.harness import Condition, MetricError
from synthkit.tableeval import (
    TableExperiment,
    evaluate_cleaning,
    resolve_table_metric,
    run_table_experiment,
)
from synthkit.tableplan import (
    TableIntegrityError,
    load_table,
    plan_table,
    write_table,
)
from synthkit.tablespec import (
    ColumnMess,
    ColumnSpec,
    TableSpec,
    TableSpecError,
)

PASS = 0


def check(label, cond):
    global PASS
    if not cond:
        print("FAIL: {}".format(label))
        sys.exit(1)
    PASS += 1
    print("ok  {:02d}  {}".format(PASS, label))


def reference_table(rows=200, seed=7) -> TableSpec:
    """Schema-authoring example; must stay equal to the library
    copy in synthkit.examples (parity-checked below)."""
    return TableSpec(
        title="Encounter billing extract",
        rows=rows,
        master_seed=seed,
        duplicate_rate=0.1,
        columns=[
            ColumnSpec("patient_id", "str_id",
                       {"kind": "sequence", "prefix": "PT-",
                        "start": 5000}),
            ColumnSpec("patient_name", "person_name", {},
                       ColumnMess(case_rate=0.3, space_rate=0.2)),
            ColumnSpec("age", "int",
                       {"kind": "normal", "mean": 58, "std": 18,
                        "min": 0, "max": 105},
                       ColumnMess(missing_rate=0.15)),
            ColumnSpec("department", "category",
                       {"kind": "categorical",
                        "choices": ["cardiology", "oncology",
                                    "orthopedics", "emergency"],
                        "weights": [5, 2, 2, 1]},
                       ColumnMess(typo_rate=0.2)),
            ColumnSpec("los_days", "int",
                       {"kind": "mixture",
                        "components": [
                            {"kind": "uniform", "min": 1,
                             "max": 4},
                            {"kind": "normal", "mean": 18,
                             "std": 5, "min": 8, "max": 45},
                        ],
                        "weights": [0.7, 0.3]}),
            ColumnSpec("total_cost", "float",
                       {"kind": "lognormal", "mu": 7.5,
                        "sigma": 0.8, "min": 50},
                       ColumnMess(outlier_rate=0.05,
                                  outlier_factor=100.0,
                                  wrong_rate=0.1)),
            ColumnSpec("visit_date", "date",
                       {"kind": "date_range",
                        "start": "2026-01-01",
                        "end": "2026-06-30"},
                       ColumnMess(format_rate=0.4,
                                  wrong_rate=0.08)),
            ColumnSpec("discharge_date", "date",
                       {"kind": "date_range",
                        "start": "2026-01-01",
                        "end": "2026-06-30"}),
            ColumnSpec("active", "bool",
                       {"kind": "bernoulli", "p": 0.7},
                       ColumnMess(format_rate=0.3)),
        ],
        rules=[
            {"kind": "date_after", "earlier": "visit_date",
             "later": "discharge_date", "min_days": 1,
             "max_days": 45},
            {"kind": "derived", "target": "total_cost",
             "source": "los_days", "factor": 1150.0,
             "noise_sigma": 0.2},
        ],
    )


def main():
    # ---------- validation ----------
    bad = TableSpec(title="", rows=0, columns=[
        ColumnSpec("x", "int", {"kind": "categorical"}),
        ColumnSpec("x", "wat", {"kind": "uniform"}),
        ColumnSpec("d", "category",
                   {"kind": "categorical",
                    "choices": ["a", "b"], "weights": [1]}),
        ColumnSpec("m", "float",
                   {"kind": "mixture",
                    "components": [{"kind": "categorical"}],
                    "weights": [1, 2]}),
        ColumnSpec("charges", "float", {"kind": "derived"}),
        ColumnSpec("flag", "bool", {"kind": "bernoulli",
                                    "p": 0.2}),
    ], rules=[
        {"kind": "date_after", "earlier": "ghost",
         "later": "ghost"},
        {"kind": "teleport"},
        {"kind": "derived", "target": "d", "source": "m",
         "factor": 2.0},
        {"kind": "derived", "target": "flag", "source": "m",
         "factor": 1.0},
    ])
    try:
        bad.validate()
        check("broken spec rejected", False)
    except TableSpecError as e:
        msg = str(e)
        check("validation reports EVERY problem at once",
              "title" in msg and "rows" in msg
              and "duplicate name" in msg
              and "invalid for type" in msg
              and "unknown type" in msg
              and "weights length" in msg
              and "must name a column" in msg
              and "unknown kind `teleport`" in msg
              and "top-level `duplicate_rate` field" in msg
              and "`derived` requires numeric columns" in msg
              and "is a RULE kind, not a distribution" in msg
              and "`outcomes` array (kind logistic)" in msg
              and "indicator" in msg
              and "component #1" in msg
              and "requires param `choices`" in msg
              and "mixture weights must match" in msg)

    spec = reference_table()
    spec.validate()
    check("reference table spec validates", True)
    from synthkit.examples import reference_table as lib_table
    check("library reference table matches the inline schema "
          "example",
          lib_table(rows=200, master_seed=7).to_json()
          == spec.to_json())
    spec2 = TableSpec.from_json(spec.to_json())
    check("table spec JSON round-trips exactly",
          spec2.to_json() == spec.to_json())

    # ---------- planning determinism ----------
    bp = plan_table(spec)
    bp2 = plan_table(reference_table())
    check("planning is deterministic",
          bp.clean_rows == bp2.clean_rows
          and bp.dirty_rows == bp2.dirty_rows
          and len(bp.ledger) == len(bp2.ledger))
    check("duplicates appended and mapped",
          len(bp.dirty_rows) == 200 + 20
          and len(bp.duplicate_of) == 20
          and all(bp.dirty_rows[i] == bp.dirty_rows[src]
                  for i, src in bp.duplicate_of.items()))

    # Column-keyed seeds: adding a column never changes others.
    wider = reference_table()
    wider.columns.append(ColumnSpec(
        "readmitted", "bool", {"kind": "bernoulli", "p": 0.2}))
    bp_w = plan_table(wider)
    check("adding a column leaves existing columns untouched",
          all(
              {k: row[k] for k in bp.columns}
              == {k: bp_w.clean_rows[i][k] for k in bp.columns}
              for i, row in enumerate(bp.clean_rows)
          ))

    # ---------- ledger exactness ----------
    mess = bp.ledger_index()
    exact = True
    for r in range(200):
        for c in bp.columns:
            differs = bp.dirty_rows[r][c] != bp.clean_rows[r][c]
            in_ledger = (r, c) in mess
            if differs != in_ledger:
                exact = False
    check("dirty differs from clean EXACTLY at ledger cells",
          exact and len(mess) > 100)

    # ---------- realized rates ----------
    n_missing = sum(1 for m in bp.ledger if m.op == "missing")
    n_format = sum(1 for m in bp.ledger if m.op == "format")
    check("mess rates realized near their specs (n=200)",
          20 <= n_missing <= 42
          and abs(n_format / (200 * 0.7) - 1.0) < 0.5)
    dept = [row["department"] for row in bp.clean_rows]
    check("weighted categorical skews as weighted",
          dept.count("cardiology") > dept.count("emergency") * 2)

    # ---------- rules ----------
    from datetime import date as _date
    ordered = all(
        _date.fromisoformat(r["discharge_date"])
        > _date.fromisoformat(r["visit_date"])
        for r in bp.clean_rows)
    check("date_after rule holds in every clean row", ordered)
    import math
    ratios = [float(r["total_cost"]) / (int(r["los_days"]) * 1150.0)
              for r in bp.clean_rows]
    check("derived rule correlates cost with stay (noisy but "
          "bounded)",
          all(0.4 < x < 2.6 for x in ratios)
          and 0.85 < sum(ratios) / len(ratios) < 1.2)

    # ---------- mixture ----------
    los = sorted(int(r["los_days"]) for r in bp.clean_rows)
    short = sum(1 for v in los if v <= 5)
    check("mixture produces the bimodal split",
          0.55 < short / len(los) < 0.85
          and max(los) > 10)

    # ---------- wrong values: plausible lies ----------
    wrongs = [m for m in bp.ledger if m.op == "wrong"]
    check("wrong values planted and format-valid",
          len(wrongs) > 15
          and all(m.dirty != m.clean for m in wrongs)
          and all(
              _date.fromisoformat(m.dirty) is not None
              for m in wrongs if m.column == "visit_date"))

    # ---------- IO round trip + integrity ----------
    tmp = Path(tempfile.mkdtemp(prefix="synthkit_tbl_"))
    run = write_table(tmp / "t1", spec, bp)
    spec_l, bp_l = load_table(run)
    check("table round-trips: spec, rows, ledger, dup map",
          spec_l.to_json() == spec.to_json()
          and bp_l.dirty_rows == bp.dirty_rows
          and bp_l.clean_rows == bp.clean_rows
          and len(bp_l.ledger) == len(bp.ledger)
          and bp_l.duplicate_of == bp.duplicate_of)
    victim = run / "clean.csv"
    victim.write_text(victim.read_text() + "x", encoding="utf-8")
    try:
        load_table(run)
        check("tampered table refuses to load", False)
    except TableIntegrityError as e:
        check("tampered table refuses to load",
              "clean.csv" in str(e))

    # ---------- cleaners ----------
    def perfect(rows):
        wrong_by_row = {}
        for m in bp.ledger:
            if m.op == "wrong":
                wrong_by_row.setdefault(m.row, []).append(m.column)
        out = []
        for i, _row in enumerate(rows):
            src = bp.duplicate_of.get(i, i)
            row = dict(bp.clean_rows[src])
            if i in bp.duplicate_of:
                row["_duplicate"] = "true"
            if i in wrong_by_row:
                row["_suspect"] = ",".join(wrong_by_row[i])
            out.append(row)
        return out

    rep = evaluate_cleaning(bp, perfect(bp.dirty_rows), "perfect")
    check("perfect cleaner: total fix rate, zero overcorrection, "
          "all duplicates flagged",
          rep.fix_rate == 1.0
          and rep.overcorrection_rate == 0.0
          and rep.dup_flag_rate == 1.0
          and rep.dup_false_flags == 0)
    check("perfect cleaner detects every planted wrong value",
          rep.wrong_detect_rate == 1.0
          and rep.suspect_false == 0)

    def whitespace_only(rows):
        return [{k: v.strip() for k, v in row.items()}
                for row in rows]

    rep = evaluate_cleaning(bp, whitespace_only(bp.dirty_rows),
                            "strip-only")
    check("whitespace-only cleaner fixes exactly the space op",
          rep.ops["space"].fix_rate == 1.0
          and rep.ops["missing"].fix_rate == 0.0
          and rep.ops["format"].fix_rate == 0.0
          and 0.0 < rep.fix_rate < 0.5
          and rep.overcorrection_rate == 0.0)
    check("naive cleaner detects zero wrong values — the honest "
          "gap",
          rep.wrong_detect_rate == 0.0
          and rep.ops.get("wrong") is not None
          and rep.ops["wrong"].fix_rate == 0.0)
    check("wrong metrics resolve for the harness",
          resolve_table_metric(rep, "wrong.detect_rate") == 0.0
          and resolve_table_metric(rep, "wrong.total") > 15)

    def destructive(rows):
        return [{k: v.strip().upper() for k, v in row.items()}
                for row in rows]

    rep_d = evaluate_cleaning(bp, destructive(bp.dirty_rows),
                              "uppercaser")
    check("destructive cleaner caught overcorrecting clean cells",
          rep_d.overcorrection_rate > 0.2)
    text = rep_d.format_text()
    check("cleaning report carries the sliced money lines",
          "FIX RATE BY MESS TYPE" in text
          and "CELL ACCURACY BY COLUMN" in text
          and "overcorrection" in text)

    # ---------- metrics + experiment ----------
    check("table metrics resolve by dotted path",
          resolve_table_metric(rep_d,
                               "overall.overcorrection_rate") > 0.2
          and 0.0 <= resolve_table_metric(
              rep_d, "ops.typo.fix_rate") <= 1.0)
    try:
        resolve_table_metric(rep_d, "ops.nonexistent.fix_rate")
        check("unknown table metrics raise MetricError", False)
    except MetricError:
        check("unknown table metrics raise MetricError", True)

    exp = TableExperiment(
        name="strip-only against the cleaning bar",
        spec=reference_table(),
        cleaner=whitespace_only,
        cleaner_name="strip-only",
        conditions=[
            Condition.parse("overall.fix_rate >= 0.9"),
            Condition.parse("overall.overcorrection_rate <= 0.01"),
            Condition.parse("ops.space.fix_rate >= 0.95"),
        ],
    )
    result = run_table_experiment(exp)
    check("table experiment fails for exactly the right reasons",
          not result.passed
          and result.failed_conditions == ["overall.fix_rate"]
          and "FAILED" in result.finding()
          and "HOLDS" in result.finding())

    # ---------- table compiler: assisted, never trusted ----------
    from synthkit.compiler import compile_table_spec

    class CannedBackend:
        def __init__(self, payloads):
            self.payloads = list(payloads)
            self.prompts = []

        def complete(self, prompt, *, system="", max_tokens=3000,
                     temperature=0.3):
            self.prompts.append(prompt)
            return self.payloads.pop(0)

    good = reference_table(rows=50).to_json()
    broken = reference_table(rows=0).to_json()
    backend = CannedBackend([broken, good])
    result = compile_table_spec("fifty billing rows", backend)
    check("table compiler repairs via a problems-fed second round",
          result.ok
          and result.spec.rows == 50
          and "rows must be >= 1" in backend.prompts[1])
    backend = CannedBackend([broken, broken])
    result = compile_table_spec("fifty billing rows", backend)
    check("unrepaired table compile hands the human a draft plus "
          "problems",
          not result.ok
          and result.spec is not None
          and "rows must be >= 1" in result.problems)

    shutil.rmtree(tmp)
    print("\nAll {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
