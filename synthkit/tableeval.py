"""SYNTH_A1: cleaning evaluation — exact, cell-level, sliced.

Contract: the cleaner receives the DIRTY rows and returns the same
number of rows in the same order (duplicate handling is flag-based:
a cleaner may add a "_duplicate" column with truthy values on rows
it identifies as duplicates). Every cell is then judged against the
clean truth:

    messy cell   -> FIXED (matches clean) or MISSED
    clean cell   -> PRESERVED or OVERCORRECTED (the silent killer:
                    a "cleaner" that mangles good data)
    dup rows     -> flagged or missed (via _duplicate column)

Slices: fix_rate by mess op (missing/typo/format/outlier/case/
space) and per column — the report that says WHICH kinds of mess a
vendor's cleaner actually handles. Metric paths resolve for the
harness: overall.fix_rate, overall.overcorrection_rate,
ops.typo.fix_rate, columns.age.cell_accuracy,
duplicates.flag_rate.

Python 3.8 compatible. Stdlib only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .harness import Condition, MetricError
from .tableplan import TableBlueprint


def _norm(s: str) -> str:
    return str(s).strip()


@dataclass
class OpScore:
    op: str
    total: int = 0
    fixed: int = 0

    @property
    def fix_rate(self) -> float:
        return self.fixed / self.total if self.total else 0.0


@dataclass
class ColumnScore:
    column: str
    cells: int = 0
    correct: int = 0

    @property
    def cell_accuracy(self) -> float:
        return self.correct / self.cells if self.cells else 0.0


@dataclass
class CleaningReport:
    cleaner_name: str
    rows: int
    mess_cells: int
    fixed: int
    missed: int
    clean_cells: int
    overcorrected: int
    ops: Dict[str, OpScore]
    columns: Dict[str, ColumnScore]
    dup_total: int
    dup_flagged: int
    dup_false_flags: int
    wrong_total: int = 0
    wrong_detected: int = 0
    suspect_false: int = 0

    @property
    def fix_rate(self) -> float:
        return self.fixed / self.mess_cells if self.mess_cells \
            else 0.0

    @property
    def overcorrection_rate(self) -> float:
        return self.overcorrected / self.clean_cells \
            if self.clean_cells else 0.0

    @property
    def wrong_detect_rate(self) -> float:
        return self.wrong_detected / self.wrong_total \
            if self.wrong_total else 0.0

    @property
    def dup_flag_rate(self) -> float:
        return self.dup_flagged / self.dup_total if self.dup_total \
            else 0.0

    def format_text(self) -> str:
        lines = [
            "CLEANING EVALUATION: {} on {} row(s), {} messy "
            "cell(s)".format(self.cleaner_name, self.rows,
                             self.mess_cells),
            "  fix rate: {:.1%}   ({} fixed / {} missed)".format(
                self.fix_rate, self.fixed, self.missed),
            "  overcorrection: {:.2%} of {} clean cell(s) "
            "damaged".format(self.overcorrection_rate,
                             self.clean_cells),
        ]
        if self.wrong_total:
            lines.append(
                "  wrong-value detection: {:.1%} ({}/{}, {} false "
                "suspicion(s))".format(
                    self.wrong_detect_rate, self.wrong_detected,
                    self.wrong_total, self.suspect_false))
        if self.dup_total:
            lines.append(
                "  duplicates flagged: {:.1%} ({}/{}, {} false "
                "flag(s))".format(self.dup_flag_rate,
                                  self.dup_flagged, self.dup_total,
                                  self.dup_false_flags))
        lines.append("")
        lines.append("FIX RATE BY MESS TYPE:")
        for op, sc in sorted(self.ops.items()):
            lines.append("  {:<10} {:>6.1%}  ({}/{})".format(
                op, sc.fix_rate, sc.fixed, sc.total))
        lines.append("")
        lines.append("CELL ACCURACY BY COLUMN:")
        for name, sc in sorted(self.columns.items()):
            lines.append("  {:<20} {:>6.1%}".format(
                name, sc.cell_accuracy))
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps({
            "cleaner": self.cleaner_name,
            "fix_rate": round(self.fix_rate, 4),
            "overcorrection_rate": round(
                self.overcorrection_rate, 4),
            "dup_flag_rate": round(self.dup_flag_rate, 4),
            "ops": {k: {"fix_rate": round(v.fix_rate, 4),
                        "total": v.total}
                    for k, v in self.ops.items()},
            "columns": {k: round(v.cell_accuracy, 4)
                        for k, v in self.columns.items()},
        }, indent=2)


def evaluate_cleaning(bp: TableBlueprint,
                      cleaned_rows: List[Dict[str, str]],
                      cleaner_name: str = "cleaner",
                      ) -> CleaningReport:
    if len(cleaned_rows) != len(bp.dirty_rows):
        raise ValueError(
            "cleaner must return {} row(s) in order, got {} — "
            "flag duplicates via a `_duplicate` column instead of "
            "dropping rows".format(len(bp.dirty_rows),
                                   len(cleaned_rows)))
    mess = bp.ledger_index()
    n_originals = len(bp.clean_rows)
    ops: Dict[str, OpScore] = {}
    cols: Dict[str, ColumnScore] = {
        c: ColumnScore(c) for c in bp.columns}
    fixed = missed = clean_cells = overcorrected = 0

    for r in range(n_originals):
        for c in bp.columns:
            truth = _norm(bp.clean_rows[r][c])
            out = _norm(cleaned_rows[r].get(c, ""))
            entry = mess.get((r, c))
            col_sc = cols[c]
            col_sc.cells += 1
            if entry is not None:
                sc = ops.setdefault(entry.op, OpScore(entry.op))
                sc.total += 1
                if out == truth:
                    sc.fixed += 1
                    fixed += 1
                    col_sc.correct += 1
                else:
                    missed += 1
            else:
                clean_cells += 1
                if out == truth:
                    col_sc.correct += 1
                else:
                    overcorrected += 1

    dup_total = len(bp.duplicate_of)
    dup_flagged = dup_false = 0
    # Wrong-value detection: the cleaner may emit a `_suspect`
    # column per row listing comma-separated column names it
    # believes carry wrong values. Fixing a plausible lie is
    # usually impossible; FLAGGING it is the measurable skill.
    wrong_cells = {(m.row, m.column) for m in bp.ledger
                   if m.op == "wrong"}
    wrong_detected = suspect_false = 0
    for idx, row in enumerate(cleaned_rows):
        flagged = str(row.get("_duplicate", "")).strip().lower() \
            in ("1", "true", "yes", "y")
        if idx in bp.duplicate_of:
            if flagged:
                dup_flagged += 1
        elif flagged:
            dup_false += 1
        suspects = [s.strip() for s in
                    str(row.get("_suspect", "")).split(",")
                    if s.strip()]
        for col in suspects:
            if (idx, col) in wrong_cells:
                wrong_detected += 1
            elif idx < n_originals:
                suspect_false += 1

    return CleaningReport(
        cleaner_name=cleaner_name,
        rows=len(bp.dirty_rows),
        mess_cells=len(mess),
        fixed=fixed,
        missed=missed,
        clean_cells=clean_cells,
        overcorrected=overcorrected,
        ops=ops,
        columns=cols,
        dup_total=dup_total,
        dup_flagged=dup_flagged,
        dup_false_flags=dup_false,
        wrong_total=len(wrong_cells),
        wrong_detected=wrong_detected,
        suspect_false=suspect_false,
    )


# ===================================================================
# Harness integration
# ===================================================================

def resolve_table_metric(report: CleaningReport,
                         path: str) -> float:
    parts = path.split(".")
    try:
        if parts[0] == "overall":
            return {"fix_rate": report.fix_rate,
                    "overcorrection_rate":
                        report.overcorrection_rate}[parts[1]]
        if parts[0] == "ops":
            return {"fix_rate": report.ops[parts[1]].fix_rate,
                    "total": float(report.ops[parts[1]].total)
                    }[parts[2]]
        if parts[0] == "columns":
            return report.columns[parts[1]].cell_accuracy
        if parts[0] == "wrong":
            return {"detect_rate": report.wrong_detect_rate,
                    "false_suspects":
                        float(report.suspect_false),
                    "total": float(report.wrong_total)}[parts[1]]
        if parts[0] == "duplicates":
            return {"flag_rate": report.dup_flag_rate,
                    "false_flags":
                        float(report.dup_false_flags)}[parts[1]]
    except KeyError:
        raise MetricError(
            "metric not present in this report: {}".format(path))
    raise MetricError("unknown metric family: {}".format(path))


@dataclass
class TableExperiment:
    name: str
    spec: "TableSpec"                     # noqa: F821
    cleaner: object                       # callable(rows)->rows
    cleaner_name: str
    conditions: List[Condition] = field(default_factory=list)


@dataclass
class TableExperimentResult:
    name: str
    passed: bool
    measured: Dict[str, float]
    failed_conditions: List[str]
    report: CleaningReport
    measured_descriptions: List[str] = field(default_factory=list)

    def finding(self) -> str:
        lines = ["Table experiment '{}' on {} row(s): {}.".format(
            self.name, self.report.rows,
            "PASSED" if self.passed else "FAILED")]
        lines += ["  " + d for d in self.measured_descriptions]
        return "\n".join(lines)


def run_table_experiment(exp: TableExperiment
                         ) -> TableExperimentResult:
    from .tableplan import plan_table
    bp = plan_table(exp.spec)
    cleaned = exp.cleaner([dict(r) for r in bp.dirty_rows])
    report = evaluate_cleaning(bp, cleaned, exp.cleaner_name)
    measured: Dict[str, float] = {}
    descriptions: List[str] = []
    failed: List[str] = []
    for cond in exp.conditions:
        value = resolve_table_metric(report, cond.metric)
        measured[cond.metric] = round(value, 4)
        descriptions.append(cond.describe(value))
        if not cond.holds(value):
            failed.append(cond.metric)
    result = TableExperimentResult(
        name=exp.name, passed=not failed, measured=measured,
        failed_conditions=failed, report=report)
    result.measured_descriptions = descriptions
    return result


# ===================================================================
# Prediction evaluation — scored against the ceiling.
# ===================================================================

@dataclass
class PredictionReport:
    solver_name: str
    outcome: str
    n: int
    n_pos: int
    auroc: float
    ceiling_auroc: float
    brier_score: float
    accuracy: float
    f1: float

    @property
    def auroc_gap(self) -> float:
        return self.ceiling_auroc - self.auroc

    def format_text(self) -> str:
        return (
            "PREDICTION EVALUATION: {} on `{}` ({} row(s), {} "
            "positive)\n"
            "  AUROC:   {:.3f}   (ceiling {:.3f} — gap {:.3f})\n"
            "  Brier:   {:.4f}\n"
            "  acc@0.5: {:.1%}   F1: {:.3f}".format(
                self.solver_name, self.outcome, self.n,
                self.n_pos, self.auroc, self.ceiling_auroc,
                self.auroc_gap, self.brier_score, self.accuracy,
                self.f1))


def evaluate_prediction(bp: TableBlueprint, outcome: str,
                        scores: List[float],
                        solver_name: str = "solver",
                        ) -> PredictionReport:
    from .mlmetrics import (at_threshold, auroc, brier,
                            ceiling_auroc)
    if outcome not in bp.true_probs:
        raise ValueError(
            "outcome `{}` not generated in this table (have: {})"
            .format(outcome, sorted(bp.true_probs)))
    n = len(bp.clean_rows)
    if len(scores) != n:
        raise ValueError(
            "need one score per ORIGINAL row ({}) in order, "
            "got {}".format(n, len(scores)))
    labels = [1 if bp.clean_rows[r][outcome] == "True" else 0
              for r in range(n)]
    thresh = at_threshold(scores, labels)
    return PredictionReport(
        solver_name=solver_name,
        outcome=outcome,
        n=n,
        n_pos=sum(labels),
        auroc=auroc(scores, labels),
        ceiling_auroc=ceiling_auroc(
            bp.true_probs[outcome], labels),
        brier_score=brier(
            [min(max(s, 0.0), 1.0) for s in scores], labels),
        accuracy=thresh["accuracy"],
        f1=thresh["f1"],
    )


def resolve_prediction_metric(report: PredictionReport,
                              path: str) -> float:
    try:
        return {
            "predict.auroc": report.auroc,
            "predict.ceiling_auroc": report.ceiling_auroc,
            "predict.auroc_gap": report.auroc_gap,
            "predict.brier": report.brier_score,
            "predict.accuracy": report.accuracy,
            "predict.f1": report.f1,
        }[path]
    except KeyError:
        raise MetricError(
            "unknown prediction metric: {}".format(path))
