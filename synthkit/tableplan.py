"""SYNTH_A1: table planning — clean truth first, mess second,
ledger always.

plan_table(spec) generates the CLEAN table (per-cell seeds keyed by
(master_seed, row_index, column_NAME) — adding a column never
changes any other column's values, growing rows never reshuffles
existing ones), then applies each column's mess policy with
independent per-cell-per-op seeds, recording every corruption:

    CellMess(row, column, op, clean, dirty)

The dirty table is what a cleaner sees; the clean table plus the
ledger is the exact answer key. Duplicate rows are appended and
ledgered with the source row index.

write_table/load_table: manifest with the verbatim spec and sha256
of every artifact (dirty.csv, clean.csv, ledger.json) — tampering
refuses to load, same law as the document corpus.

Python 3.8 compatible. Stdlib only.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .tablespec import ColumnSpec, TableSpec

_FIRST = ["Alex", "Sam", "Jordan", "Morgan", "Riley", "Casey",
          "Devon", "Harper", "Rowan", "Quinn", "Avery", "Jules"]
_LAST = ["Reyes", "Kim", "Okafor", "Marsh", "Ito", "Alvarez",
         "Novak", "Singh", "Bauer", "Fontaine", "Walsh", "Osei"]

_DATE_FORMATS = ["%m/%d/%Y", "%d-%b-%Y", "%B %d, %Y", "%Y.%m.%d"]


def _cell_rng(master: int, row: int, column: str,
              op: str = "value") -> random.Random:
    key = "{}:{}:{}:{}".format(master, row, column, op)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


# ===================================================================
# Clean generation
# ===================================================================

def _gen_clean(col: ColumnSpec, row: int, master: int) -> Any:
    rng = _cell_rng(master, row, col.name)
    p = col.distribution
    kind = col.dist_kind()
    if col.ctype == "person_name":
        return "{} {}".format(rng.choice(_FIRST), rng.choice(_LAST))
    if kind == "sequence":
        return "{}{}".format(p["prefix"], int(p["start"]) + row)
    if kind == "uniform":
        lo, hi = float(p["min"]), float(p["max"])
        val = rng.uniform(lo, hi)
        return int(round(val)) if col.ctype == "int" else round(
            val, 4)
    if kind == "normal":
        val = rng.gauss(float(p["mean"]), float(p["std"]))
        val = min(max(val, float(p.get("min", -math.inf))),
                  float(p.get("max", math.inf)))
        return int(round(val)) if col.ctype == "int" else round(
            val, 4)
    if kind == "lognormal":
        val = rng.lognormvariate(float(p["mu"]), float(p["sigma"]))
        val = min(max(val, float(p.get("min", 0))),
                  float(p.get("max", math.inf)))
        return int(round(val)) if col.ctype == "int" else round(
            val, 4)
    if kind == "beta":
        val = rng.betavariate(float(p["alpha"]), float(p["beta"]))
        return round(val * float(p.get("scale", 1.0)), 4)
    if kind == "categorical":
        choices = list(p["choices"])
        weights = p.get("weights")
        if weights:
            return rng.choices(choices, weights=weights, k=1)[0]
        return rng.choice(choices)
    if kind == "date_range":
        start = date.fromisoformat(p["start"])
        end = date.fromisoformat(p["end"])
        span = max((end - start).days, 0)
        return (start + timedelta(days=rng.randint(0, span)))
    if kind == "bernoulli":
        return rng.random() < float(p["p"])
    if kind == "mixture":
        pick = _cell_rng(master, row, col.name, "mixture_pick")
        comps = p["components"]
        weights = p["weights"]
        comp = pick.choices(list(range(len(comps))),
                            weights=weights, k=1)[0]
        sub = ColumnSpec(name=col.name, ctype=col.ctype,
                         distribution=comps[comp])
        return _gen_clean(sub, row, master)
    raise ValueError("unhandled distribution: {}".format(kind))


def _feature(key: str, vals: Dict[str, Any]) -> float:
    if "=" in key:
        col, want = key.split("=", 1)
        return 1.0 if str(vals[col]) == want else 0.0
    v = vals[key]
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    return float(v)


def _apply_rules(spec: TableSpec, row: int,
                 vals: Dict[str, Any]) -> None:
    """Cross-column rules, in declared order, overwriting the
    target's base value. Seeds key on the TARGET column name, so
    the column-independence law survives: adding an unrelated
    column changes nothing, and a rule target depends only on its
    sources plus its own seed."""
    for i, rule in enumerate(spec.rules):
        kind = rule["kind"]
        if kind == "date_after":
            earlier = vals[rule["earlier"]]
            rng = _cell_rng(spec.master_seed, row,
                            rule["later"], "rule{}".format(i))
            delta = rng.randint(int(rule["min_days"]),
                                int(rule["max_days"]))
            vals[rule["later"]] = earlier + timedelta(days=delta)
        elif kind == "derived":
            source = vals[rule["source"]]
            base = float(source) * float(rule["factor"])
            sigma = float(rule.get("noise_sigma", 0.0))
            if sigma > 0:
                rng = _cell_rng(spec.master_seed, row,
                                rule["target"],
                                "rule{}".format(i))
                base *= rng.lognormvariate(0.0, sigma)
            target_col = next(c for c in spec.columns
                              if c.name == rule["target"])
            vals[rule["target"]] = (int(round(base))
                                    if target_col.ctype == "int"
                                    else round(base, 4))


def clean_str(value: Any) -> str:
    """Canonical string form of a clean value — the form a perfect
    cleaner should output. Dates ISO, bools True/False, floats
    without trailing zeros beyond 4 places."""
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return ("{:.4f}".format(value)).rstrip("0").rstrip(".")
    return str(value)


# ===================================================================
# Mess application
# ===================================================================

def _typo(text: str, rng: random.Random) -> str:
    if len(text) < 2:
        return text + text[-1:] if text else text
    i = rng.randrange(len(text) - 1)
    op = rng.choice(("swap", "drop", "double"))
    if op == "swap":
        return text[:i] + text[i + 1] + text[i] + text[i + 2:]
    if op == "drop":
        return text[:i] + text[i + 1:]
    return text[:i] + text[i] + text[i:]


def _format_drift(col: ColumnSpec, value: Any,
                  rng: random.Random) -> str:
    if isinstance(value, date):
        return value.strftime(rng.choice(_DATE_FORMATS))
    if isinstance(value, bool):
        return rng.choice(("yes", "no")) if not value else \
            rng.choice(("yes", "Y", "TRUE", "1"))
    if isinstance(value, (int, float)):
        s = clean_str(value)
        style = rng.choice(("thousands", "currency", "spaces"))
        if style == "thousands" and isinstance(value, int) \
                and abs(value) >= 1000:
            return "{:,}".format(value)
        if style == "currency":
            return "$" + s
        return s.replace(".", " . ") if "." in s else s + " "
    return clean_str(value)


def _case_drift(text: str, rng: random.Random) -> str:
    return rng.choice((text.upper(), text.lower(), text.title()))


def _wrong_value(col: ColumnSpec, value: Any,
                 rng: random.Random) -> str:
    """A FORMAT-VALID lie: stays parseable in the column's normal
    form so no normalizer can fix it — only detection helps."""
    if isinstance(value, date):
        shift = rng.choice((-1, 1)) * rng.randint(7, 90)
        return (value + timedelta(days=shift)).isoformat()
    if isinstance(value, bool):
        return clean_str(not value)
    if isinstance(value, int):
        s = str(abs(value))
        if len(s) >= 2:
            i = rng.randrange(len(s) - 1)
            s = s[:i] + s[i + 1] + s[i] + s[i + 2:]
            out = int(s) * (1 if value >= 0 else -1)
            if out != value:
                return str(out)
        return str(value + rng.choice((-1, 1))
                   * max(1, abs(value) // 3))
    if isinstance(value, float):
        return clean_str(round(
            value * rng.uniform(1.25, 2.5)
            * rng.choice((1, 1, -1 if value < 0 else 1)), 4))
    if col.dist_kind() == "categorical":
        choices = [c for c in col.distribution["choices"]
                   if c != value]
        if choices:
            return str(rng.choice(choices))
    if col.ctype == "person_name":
        for _ in range(4):
            cand = "{} {}".format(rng.choice(_FIRST),
                                  rng.choice(_LAST))
            if cand != value:
                return cand
    return clean_str(value)


@dataclass
class CellMess:
    row: int
    column: str
    op: str          # missing|typo|format|outlier|case|space|duplicate
    clean: str
    dirty: str


@dataclass
class TableBlueprint:
    spec_title: str
    master_seed: int
    columns: List[str]
    clean_rows: List[Dict[str, str]]
    dirty_rows: List[Dict[str, str]]
    ledger: List[CellMess]
    duplicate_of: Dict[int, int] = field(default_factory=dict)
    # outcome name -> per-ORIGINAL-row true probabilities (the
    # generating model's P(y=1|x); ceiling metrics live on these)
    true_probs: Dict[str, List[float]] = field(
        default_factory=dict)
    # dirty row index -> source clean row index (for appended dups)

    def ledger_index(self) -> Dict[Tuple[int, str], CellMess]:
        return {(m.row, m.column): m for m in self.ledger
                if m.op != "duplicate"}


def plan_table(spec: TableSpec) -> TableBlueprint:
    spec.validate()
    columns = [c.name for c in spec.columns]
    clean_rows: List[Dict[str, str]] = []
    clean_vals: List[Dict[str, Any]] = []
    for r in range(spec.rows):
        vals = {c.name: _gen_clean(c, r, spec.master_seed)
                for c in spec.columns}
        _apply_rules(spec, r, vals)
        clean_vals.append(vals)
        clean_rows.append({k: clean_str(v)
                           for k, v in vals.items()})

    # Outcomes: labels from CLEAN values (truth reflects reality;
    # mess on features is what makes prediction hard).
    true_probs: Dict[str, List[float]] = {}
    for oc in spec.outcomes:
        name = oc["name"]
        probs: List[float] = []
        for r in range(spec.rows):
            z = float(oc["intercept"])
            for key, w in oc["coefficients"].items():
                z += float(w) * _feature(key, clean_vals[r])
            prob = 1.0 / (1.0 + math.exp(-z))
            probs.append(prob)
            label = _cell_rng(spec.master_seed, r, name,
                              "outcome").random() < prob
            clean_vals[r][name] = label
            clean_rows[r][name] = clean_str(label)
        true_probs[name] = probs
    outcome_names = [oc["name"] for oc in spec.outcomes]
    columns = columns + outcome_names

    dirty_rows: List[Dict[str, str]] = []
    ledger: List[CellMess] = []
    for r in range(spec.rows):
        row_out: Dict[str, str] = {
            name: clean_rows[r][name] for name in outcome_names}
        for col in spec.columns:
            cval = clean_vals[r][col.name]
            cstr = clean_rows[r][col.name]
            dirty = cstr
            op_applied = None
            m = col.mess
            # Ops are mutually exclusive per cell, priority order;
            # each op draws its own seeded RNG so toggling one rate
            # never shifts another's draws. `wrong` is first: a
            # plausible lie is the deepest corruption and must not
            # be masked by surface mess.
            if m.wrong_rate > 0 and _cell_rng(
                    spec.master_seed, r, col.name,
                    "wrong").random() < m.wrong_rate:
                cand = _wrong_value(col, cval, _cell_rng(
                    spec.master_seed, r, col.name, "wrong_pick"))
                if cand != cstr:
                    dirty = cand
                    op_applied = "wrong"
            elif m.missing_rate > 0 and _cell_rng(
                    spec.master_seed, r, col.name,
                    "missing").random() < m.missing_rate:
                rng = _cell_rng(spec.master_seed, r, col.name,
                                "missing_tok")
                dirty = rng.choice(m.missing_tokens)
                op_applied = "missing"
            elif m.outlier_rate > 0 and _cell_rng(
                    spec.master_seed, r, col.name,
                    "outlier").random() < m.outlier_rate:
                scaled = (cval * m.outlier_factor
                          if isinstance(cval, (int, float))
                          and not isinstance(cval, bool) else cval)
                dirty = clean_str(
                    int(scaled) if isinstance(cval, int)
                    and not isinstance(cval, bool)
                    else round(float(scaled), 4))
                op_applied = "outlier"
            elif m.format_rate > 0 and _cell_rng(
                    spec.master_seed, r, col.name,
                    "format").random() < m.format_rate:
                dirty = _format_drift(
                    col, cval, _cell_rng(spec.master_seed, r,
                                         col.name, "format_pick"))
                op_applied = "format"
            elif m.typo_rate > 0 and _cell_rng(
                    spec.master_seed, r, col.name,
                    "typo").random() < m.typo_rate:
                dirty = _typo(cstr, _cell_rng(
                    spec.master_seed, r, col.name, "typo_pick"))
                op_applied = "typo"
            elif m.case_rate > 0 and _cell_rng(
                    spec.master_seed, r, col.name,
                    "case").random() < m.case_rate:
                dirty = _case_drift(cstr, _cell_rng(
                    spec.master_seed, r, col.name, "case_pick"))
                op_applied = "case"
            elif m.space_rate > 0 and _cell_rng(
                    spec.master_seed, r, col.name,
                    "space").random() < m.space_rate:
                rng = _cell_rng(spec.master_seed, r, col.name,
                                "space_pick")
                dirty = " " * rng.randint(1, 3) + cstr + \
                    " " * rng.randint(0, 2)
                op_applied = "space"
            if op_applied and dirty != cstr or op_applied == \
                    "missing":
                ledger.append(CellMess(r, col.name, op_applied,
                                       cstr, dirty))
                row_out[col.name] = dirty
            else:
                row_out[col.name] = cstr
        dirty_rows.append(row_out)

    duplicate_of: Dict[int, int] = {}
    if spec.duplicate_rate > 0:
        n_dups = int(round(spec.rows * spec.duplicate_rate))
        rng = _cell_rng(spec.master_seed, -1, "__dups__")
        for k in range(n_dups):
            src = rng.randrange(spec.rows)
            idx = len(dirty_rows)
            dirty_rows.append(dict(dirty_rows[src]))
            duplicate_of[idx] = src
            ledger.append(CellMess(idx, "*", "duplicate",
                                   str(src), str(idx)))

    return TableBlueprint(
        spec_title=spec.title,
        master_seed=spec.master_seed,
        columns=columns,
        clean_rows=clean_rows,
        dirty_rows=dirty_rows,
        ledger=ledger,
        duplicate_of=duplicate_of,
        true_probs=true_probs,
    )


# ===================================================================
# I/O — the auditable table artifact
# ===================================================================

class TableIntegrityError(RuntimeError):
    pass


def _csv_text(columns: List[str],
              rows: List[Dict[str, str]]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns,
                       lineterminator="\n")
    w.writeheader()
    for row in rows:
        w.writerow(row)
    return buf.getvalue()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_table(run_dir: Path, spec: TableSpec,
                bp: TableBlueprint) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "dirty.csv": _csv_text(bp.columns, bp.dirty_rows),
        "clean.csv": _csv_text(bp.columns, bp.clean_rows),
        "ledger.json": json.dumps(
            {"ledger": [asdict(m) for m in bp.ledger],
             "duplicate_of": {str(k): v for k, v
                              in bp.duplicate_of.items()},
             "true_probs": {k: [round(x, 6) for x in v]
                            for k, v in bp.true_probs.items()}},
            indent=2),
    }
    hashes = {}
    for name, text in artifacts.items():
        (run_dir / name).write_text(text, encoding="utf-8")
        hashes[name] = _sha(text)
    manifest = {
        "synthkit_table_manifest": 1,
        "spec": json.loads(spec.to_json()),
        "rows": spec.rows,
        "dirty_rows": len(bp.dirty_rows),
        "mess_cells": len([m for m in bp.ledger
                           if m.op != "duplicate"]),
        "hashes": hashes,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return run_dir


def load_table(run_dir: Path
               ) -> Tuple[TableSpec, TableBlueprint]:
    run_dir = Path(run_dir)
    manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8"))
    bad = []
    texts = {}
    for name, expected in sorted(manifest["hashes"].items()):
        text = (run_dir / name).read_text(encoding="utf-8")
        if _sha(text) != expected:
            bad.append(name)
        texts[name] = text
    if bad:
        raise TableIntegrityError(
            "table integrity failed for: {}".format(
                ", ".join(bad)))
    spec = TableSpec.from_json(json.dumps(manifest["spec"]))

    def rows_of(text: str) -> List[Dict[str, str]]:
        return list(csv.DictReader(io.StringIO(text)))

    raw = json.loads(texts["ledger.json"])
    bp = TableBlueprint(
        spec_title=spec.title,
        master_seed=spec.master_seed,
        columns=[c.name for c in spec.columns],
        clean_rows=rows_of(texts["clean.csv"]),
        dirty_rows=rows_of(texts["dirty.csv"]),
        ledger=[CellMess(**m) for m in raw["ledger"]],
        duplicate_of={int(k): v for k, v
                      in raw["duplicate_of"].items()},
        true_probs=raw.get("true_probs", {}),
    )
    bp.columns = (bp.columns
                  + [oc["name"] for oc in spec.outcomes])
    return spec, bp
