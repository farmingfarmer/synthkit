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
    if not kind:
        return None    # rule-produced; the rules pass fills it
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
        w = p.get("weights")
        if w and span > 0:
            # Real activity is never uniform across a date range —
            # it clusters. `weights` divides the span into equal
            # buckets and draws a bucket by weight, then a day
            # inside it, so profiled temporal shape survives.
            pick = _cell_rng(master, row, col.name, "date_bucket")
            b = pick.choices(list(range(len(w))), weights=w,
                             k=1)[0]
            lo = int(span * b / len(w))
            hi = int(span * (b + 1) / len(w)) - 1
            hi = max(lo, min(hi, span))
            return start + timedelta(days=rng.randint(lo, hi))
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
    if "." in key.split("=", 1)[0]:
        # note-element presence: 'notecol.element_id'
        ncol, eid = key.split(".", 1)
        return float(vals.get(
            "_note::{}::{}".format(ncol, eid), 0.0))
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
            days_from = rule.get("days_from")
            if days_from is not None:
                # The gap IS another column's value — dates and
                # durations stay consistent, the way a real table
                # would be. (A live compile wanted exactly this
                # and the schema could not say it.)
                delta = max(int(round(
                    float(vals[days_from]))), 0)
            else:
                rng = _cell_rng(spec.master_seed, row,
                                rule["later"],
                                "rule{}".format(i))
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
    # note column -> per-row {element_id: planted?} — the text
    # ground truth (what the note ACTUALLY asserts, distractors
    # excluded), for scoring text-mining solvers honestly.
    note_truth: Dict[str, List[Dict[str, bool]]] = field(
        default_factory=dict)
    true_probs: Dict[str, List[float]] = field(
        default_factory=dict)
    # dirty row index -> source clean row index (for appended dups)

    def ledger_index(self) -> Dict[Tuple[int, str], CellMess]:
        return {(m.row, m.column): m for m in self.ledger
                if m.op != "duplicate"}


def _build_note(col, r: int, seed: int):
    """Assemble one patient's note deterministically: seeded
    presence draws per element/distractor, seeded phrasing
    picks, fillers, seeded sentence order. Returns (text,
    presence) where presence maps element_id -> bool — the
    ground truth that feeds the outcome logit."""
    note = col.note or {}
    rng = _cell_rng(seed, r, col.name, "note")
    sentences = []
    presence = {}
    for el in note.get("elements", []):
        hit = rng.random() < float(el.get("density", 0.0))
        presence[el["id"]] = hit
        if hit:
            phr = el["phrasings"][
                rng.randrange(len(el["phrasings"]))]
            sentences.append(phr)
    for dis in note.get("distractors", []):
        # `excludes`: only plant this trap when the named
        # element is ABSENT — "denies missing doses" belongs in
        # the notes of patients who did NOT miss doses; that is
        # what makes naive keyword matching dangerous.
        excl = dis.get("excludes", "")
        if excl and presence.get(excl, False):
            continue
        if rng.random() < float(dis.get("density", 0.0)):
            phr = dis["phrasings"][
                rng.randrange(len(dis["phrasings"]))]
            sentences.append(phr)
    fillers = list(note.get("fillers", []))
    if fillers:
        n_fill = 1 + rng.randrange(min(3, len(fillers)))
        picks = list(range(len(fillers)))
        for _ in range(n_fill):
            k = picks.pop(rng.randrange(len(picks)))
            sentences.append(fillers[k])
            if not picks:
                break
    order = list(range(len(sentences)))
    shuffled = []
    while order:
        shuffled.append(sentences[order.pop(
            rng.randrange(len(order)))])
    text = " ".join(s.rstrip(".") + "." for s in shuffled)
    return text, presence



# =====================================================================
# CORRELATION IMPOSITION (Iman-Conover)
# ---------------------------------------------------------------------
# Columns are drawn independently from their declared marginals, then
# REORDERED so their rank pattern matches a reference sample carrying
# the target correlation. Because reordering only permutes the values
# already drawn, every marginal survives EXACTLY as specified — the
# fitted distribution is untouched while the joint structure appears.
# This is what lets a profiled spec reproduce "how the fields move
# together" without disturbing "what each field looks like".
# =====================================================================
def _cholesky(m):
    n = len(m)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                d = m[i][i] - s
                # nearest-PSD nudge: profiled matrices from real data
                # are not always positive definite
                L[i][j] = math.sqrt(d) if d > 1e-12 else 1e-6
            else:
                L[i][j] = (m[i][j] - s) / L[j][j] if L[j][j] else 0.0
    return L


def _rank_order(xs):
    """Positions that would sort xs, i.e. rank of each element."""
    idx = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0] * len(xs)
    for rank, i in enumerate(idx):
        out[i] = rank
    return out


def _apply_correlations(spec, clean_vals, clean_rows):
    pairs = getattr(spec, "correlations", None) or []
    if not pairs:
        return
    names = []
    for pr in pairs:
        for k in ("a", "b"):
            if pr[k] not in names:
                names.append(pr[k])
    n = len(names)
    idx = {nm: i for i, nm in enumerate(names)}
    target = [[1.0 if i == j else 0.0 for j in range(n)]
              for i in range(n)]
    for pr in pairs:
        i, j = idx[pr["a"]], idx[pr["b"]]
        # Pearson correlation of the normal scores that yields the
        # requested SPEARMAN correlation (Pearson-Spearman bridge)
        rho = max(-0.999, min(0.999, float(pr["spearman"])))
        target[i][j] = target[j][i] = 2.0 * math.sin(
            math.pi * rho / 6.0)
    L = _cholesky(target)
    rows = len(clean_vals)
    rng = _cell_rng(spec.master_seed, 0, "__correlation__", "ref")
    ref = []
    for _ in range(rows):
        z = [rng.gauss(0.0, 1.0) for _ in range(n)]
        ref.append([sum(L[i][k] * z[k] for k in range(i + 1))
                    for i in range(n)])
    for nm in names:
        col = idx[nm]
        vals = [clean_vals[r].get(nm) for r in range(rows)]
        if any(not isinstance(v, (int, float)) for v in vals):
            continue                    # numeric columns only
        want = _rank_order([ref[r][col] for r in range(rows)])
        ordered = sorted(vals)
        for r in range(rows):
            clean_vals[r][nm] = ordered[want[r]]
            clean_rows[r][nm] = clean_str(ordered[want[r]])


def plan_table(spec: TableSpec) -> TableBlueprint:
    spec.validate()
    columns = [c.name for c in spec.columns]
    clean_rows: List[Dict[str, str]] = []
    clean_vals: List[Dict[str, Any]] = []
    note_truth: Dict[str, List[Dict[str, bool]]] = {
        c.name: [] for c in spec.columns if c.ctype == "note"}
    for r in range(spec.rows):
        vals = {}
        for c in spec.columns:
            if c.ctype == "note":
                text, presence = _build_note(
                    c, r, spec.master_seed)
                vals[c.name] = text
                note_truth[c.name].append(presence)
                for eid, hit in presence.items():
                    vals["_note::{}::{}".format(
                        c.name, eid)] = 1.0 if hit else 0.0
            else:
                vals[c.name] = _gen_clean(
                    c, r, spec.master_seed)
        _apply_rules(spec, r, vals)
        clean_vals.append(vals)
        clean_rows.append({k: clean_str(v)
                           for k, v in vals.items()
                           if not k.startswith("_")})

    # Joint structure BEFORE outcomes, so planted coefficients act
    # on the correlated values a model will actually see.
    _apply_correlations(spec, clean_vals, clean_rows)

    # Outcomes: labels from CLEAN values (truth reflects reality;
    # mess on features is what makes prediction hard).
    true_probs: Dict[str, List[float]] = {}
    for oc in spec.outcomes:
        name = oc["name"]
        kind = oc.get("kind", "logistic")
        probs: List[float] = []
        for r in range(spec.rows):
            z = float(oc["intercept"])
            for key, w in oc["coefficients"].items():
                z += float(w) * _feature(key, clean_vals[r])
            rng = _cell_rng(spec.master_seed, r, name,
                            "outcome")
            if kind == "linear":
                # The noiseless signal is the TRUTH; additive
                # gaussian noise of known sigma is the
                # irreducible error that sets the R^2 ceiling.
                probs.append(z)
                value = z + rng.gauss(0.0, float(
                    oc.get("noise_sigma", 0.0)))
                clean_vals[r][name] = round(value, 4)
                clean_rows[r][name] = clean_str(
                    round(value, 4))
            else:
                prob = 1.0 / (1.0 + math.exp(-z))
                probs.append(prob)
                label = rng.random() < prob
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
            if col.ctype == "note":
                row_out[col.name] = dirty
                continue
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
        note_truth=note_truth,
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
