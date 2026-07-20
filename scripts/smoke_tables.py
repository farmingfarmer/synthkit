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
            ColumnSpec("total_cost", "float",
                       {"kind": "lognormal", "mu": 7.5,
                        "sigma": 0.8, "min": 50},
                       ColumnMess(outlier_rate=0.05,
                                  outlier_factor=100.0)),
            ColumnSpec("visit_date", "date",
                       {"kind": "date_range",
                        "start": "2026-01-01",
                        "end": "2026-06-30"},
                       ColumnMess(format_rate=0.4)),
            ColumnSpec("active", "bool",
                       {"kind": "bernoulli", "p": 0.7},
                       ColumnMess(format_rate=0.3)),
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
              and "weights length" in msg)

    spec = reference_table()
    spec.validate()
    check("reference table spec validates", True)
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
        out = []
        for i, _row in enumerate(rows):
            src = bp.duplicate_of.get(i, i)
            row = dict(bp.clean_rows[src])
            if i in bp.duplicate_of:
                row["_duplicate"] = "true"
            out.append(row)
        return out

    rep = evaluate_cleaning(bp, perfect(bp.dirty_rows), "perfect")
    check("perfect cleaner: total fix rate, zero overcorrection, "
          "all duplicates flagged",
          rep.fix_rate == 1.0
          and rep.overcorrection_rate == 0.0
          and rep.dup_flag_rate == 1.0
          and rep.dup_false_flags == 0)

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

    def destructive(rows):
        return [{k: v.strip().upper() for k, v in row.items()}
                for row in rows]

    rep_d = evaluate_cleaning(bp, destructive(bp.dirty_rows),
                              "uppercaser")
    check("destructive cleaner caught overcorrecting clean cells",
          rep_d.overcorrection_rate > 0.3)
    text = rep_d.format_text()
    check("cleaning report carries the sliced money lines",
          "FIX RATE BY MESS TYPE" in text
          and "CELL ACCURACY BY COLUMN" in text
          and "overcorrection" in text)

    # ---------- metrics + experiment ----------
    check("table metrics resolve by dotted path",
          resolve_table_metric(rep_d,
                               "overall.overcorrection_rate") > 0.3
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

    shutil.rmtree(tmp)
    print("\nAll {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
