"""SYNTH_R1: relational tables — multi-table truth with join
mess.

Real hospital data is never one table: encounters reference
patients, labs reference encounters, and the mess that matters
most lives in the JOINS — orphaned foreign keys pointing at
parents that do not exist. No flat-table product can test a
vendor's referential-integrity claims; a generator that plants
orphans and keeps the answer key can.

Design:
  - A RelationalSpec holds named TableSpecs plus links. Each link
    declares child/parent tables, the parent's key column (must be
    a `sequence` str_id — guaranteed unique), and the fk column to
    INJECT into the child (it does not appear in the child's own
    spec; the link owns it).
  - Planning is layered determinism: each table plans under its
    own derived seed; fk values are drawn from the parent's real
    keys by seeded RNG; then `orphan_rate` replaces a seeded
    fraction of DIRTY fk cells with format-valid keys that do not
    exist — every injection ledgered in a LinkLedger. Clean rows
    keep true references always.
  - evaluate_links scores an orphan-flagger (a vendor claiming to
    detect referential breaks) with precision/recall against the
    ledger.

Python 3.8 compatible. Stdlib only.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .tableplan import TableBlueprint, plan_table
from .tablespec import TableSpec


class RelationalSpecError(ValueError):
    pass


class RelationalIntegrityError(ValueError):
    pass


@dataclass
class Link:
    """orphan_style:
      'random' — fake keys far from any real one (a set-membership
                 check catches them; the easy tier)
      'near'   — fake keys one digit-transposition from a REAL key
                 (still absent from the key set, but they defeat
                 fuzzy matchers that 'repair' near-misses)
    drift_rate — the PRECISION trap: valid references whose
      formatting is mangled (case flips, stray whitespace); a
      naive set-membership checker false-flags every one."""
    child: str
    parent: str
    parent_key: str
    fk_column: str
    orphan_rate: float = 0.0
    orphan_style: str = "random"
    drift_rate: float = 0.0


@dataclass
class RelationalSpec:
    title: str
    master_seed: int
    tables: Dict[str, TableSpec]
    links: List[Link] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps({
            "title": self.title,
            "master_seed": self.master_seed,
            "tables": {name: json.loads(t.to_json())
                       for name, t in self.tables.items()},
            "links": [{"child": l.child, "parent": l.parent,
                       "parent_key": l.parent_key,
                       "fk_column": l.fk_column,
                       "orphan_rate": l.orphan_rate,
                       "orphan_style": l.orphan_style,
                       "drift_rate": l.drift_rate}
                      for l in self.links],
        }, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "RelationalSpec":
        d = json.loads(raw)
        return cls(
            title=d.get("title", ""),
            master_seed=int(d.get("master_seed", 0)),
            tables={name: TableSpec.from_json(json.dumps(t))
                    for name, t in d.get("tables", {}).items()},
            links=[Link(child=l["child"], parent=l["parent"],
                        parent_key=l["parent_key"],
                        fk_column=l["fk_column"],
                        orphan_rate=float(
                            l.get("orphan_rate", 0.0)),
                        orphan_style=l.get("orphan_style",
                                           "random"),
                        drift_rate=float(
                            l.get("drift_rate", 0.0)))
                   for l in d.get("links", [])],
        )

    def validate(self) -> None:
        problems: List[str] = []
        if not self.tables:
            problems.append("a relational spec needs tables")
        for name, table in self.tables.items():
            try:
                table.validate()
            except Exception as e:
                problems.append(
                    "table `{}`: {}".format(name, e))
        for i, link in enumerate(self.links):
            tag = "link #{}".format(i + 1)
            if link.child not in self.tables:
                problems.append("{}: unknown child table `{}`"
                                .format(tag, link.child))
            if link.parent not in self.tables:
                problems.append("{}: unknown parent table `{}`"
                                .format(tag, link.parent))
                continue
            parent = self.tables[link.parent]
            key_cols = [c for c in parent.columns
                        if c.name == link.parent_key]
            if not key_cols:
                problems.append(
                    "{}: parent `{}` has no column `{}`".format(
                        tag, link.parent, link.parent_key))
            elif key_cols[0].ctype != "str_id" or \
                    key_cols[0].dist_kind() != "sequence":
                problems.append(
                    "{}: parent key `{}` must be a str_id "
                    "sequence column — uniqueness is the whole "
                    "point of a key".format(
                        tag, link.parent_key))
            if link.child in self.tables and any(
                    c.name == link.fk_column
                    for c in self.tables[link.child].columns):
                problems.append(
                    "{}: fk column `{}` must NOT appear in the "
                    "child spec — the link injects it".format(
                        tag, link.fk_column))
            if not (0.0 <= link.orphan_rate <= 0.5):
                problems.append(
                    "{}: orphan_rate must be in [0, 0.5]"
                    .format(tag))
            if link.orphan_style not in ("random", "near"):
                problems.append(
                    "{}: orphan_style must be `random` or "
                    "`near`".format(tag))
            if not (0.0 <= link.drift_rate <= 0.5):
                problems.append(
                    "{}: drift_rate must be in [0, 0.5]"
                    .format(tag))
        if problems:
            raise RelationalSpecError(
                "{} problem(s):\n".format(len(problems))
                + "\n".join("  - " + p for p in problems))


@dataclass
class OrphanMess:
    child: str
    row: int
    fk_column: str
    true_key: str
    orphan_key: str


@dataclass
class DriftMess:
    child: str
    row: int
    fk_column: str
    true_key: str
    drifted: str


@dataclass
class RelationalBlueprint:
    blueprints: Dict[str, TableBlueprint]
    link_ledger: List[OrphanMess] = field(default_factory=list)
    drift_ledger: List[DriftMess] = field(default_factory=list)


def _table_seed(master_seed: int, name: str) -> int:
    digest = hashlib.sha256(
        "{}:{}".format(master_seed, name).encode()).hexdigest()
    return int(digest[:12], 16)


def plan_relational(spec: RelationalSpec) -> RelationalBlueprint:
    spec.validate()
    blueprints: Dict[str, TableBlueprint] = {}
    for name, table in spec.tables.items():
        derived = TableSpec.from_json(table.to_json())
        derived.master_seed = _table_seed(spec.master_seed, name)
        blueprints[name] = plan_table(derived)
    ledger: List[OrphanMess] = []
    drift: List[DriftMess] = []
    for link in spec.links:
        parent_bp = blueprints[link.parent]
        child_bp = blueprints[link.child]
        parent_keys = [r[link.parent_key]
                       for r in parent_bp.clean_rows]
        prefix = "".join(
            ch for ch in parent_keys[0]
            if not ch.isdigit()) if parent_keys else "X"
        width = len(parent_keys[0]) - len(prefix) \
            if parent_keys else 5
        rng = random.Random(_table_seed(
            spec.master_seed,
            "link:{}:{}".format(link.child, link.fk_column)))
        n_clean = len(child_bp.clean_rows)
        assignments = [rng.choice(parent_keys)
                       for _ in range(n_clean)]
        for i, row in enumerate(child_bp.clean_rows):
            row[link.fk_column] = assignments[i]
        # Dirty rows include duplicates; map each dirty row back
        # to its clean origin via the trailing append order: the
        # first n_clean dirty rows correspond 1:1, appended
        # duplicates copy their source row's assignment.
        for i, row in enumerate(child_bp.dirty_rows):
            src_idx = i if i < n_clean else \
                _dup_source(child_bp, i, n_clean)
            row[link.fk_column] = assignments[src_idx]
        # Join mess on DIRTY only, seeded per row; orphan and
        # drift are mutually exclusive per cell (orphan first —
        # a broken reference is the deeper corruption).
        existing = set(parent_keys)
        for i in range(len(child_bp.dirty_rows)):
            src_idx = i if i < n_clean else \
                _dup_source(child_bp, i, n_clean)
            true_key = assignments[src_idx]
            r = random.Random(_table_seed(
                spec.master_seed,
                "orphan:{}:{}:{}".format(
                    link.child, link.fk_column, i)))
            if r.random() < link.orphan_rate:
                fake = _make_orphan(link.orphan_style, true_key,
                                    prefix, width, existing, r)
                if fake is None:
                    continue
                ledger.append(OrphanMess(
                    child=link.child, row=i,
                    fk_column=link.fk_column,
                    true_key=true_key, orphan_key=fake))
                child_bp.dirty_rows[i][link.fk_column] = fake
                continue
            rd = random.Random(_table_seed(
                spec.master_seed,
                "drift:{}:{}:{}".format(
                    link.child, link.fk_column, i)))
            if rd.random() < link.drift_rate:
                mangled = _drift_key(true_key, rd)
                if mangled == true_key:
                    continue
                drift.append(DriftMess(
                    child=link.child, row=i,
                    fk_column=link.fk_column,
                    true_key=true_key, drifted=mangled))
                child_bp.dirty_rows[i][link.fk_column] = mangled
        if link.fk_column not in child_bp.columns:
            child_bp.columns.append(link.fk_column)
    return RelationalBlueprint(blueprints=blueprints,
                               link_ledger=ledger,
                               drift_ledger=drift)


def _make_orphan(style: str, true_key: str, prefix: str,
                 width: int, existing, r) -> Optional[str]:
    if style == "near":
        digits = list(true_key[len(prefix):])
        for _attempt in range(20):
            if len(digits) < 2:
                break
            j = r.randint(0, len(digits) - 2)
            swapped = list(digits)
            swapped[j], swapped[j + 1] = \
                swapped[j + 1], swapped[j]
            cand = prefix + "".join(swapped)
            if cand not in existing and cand != true_key:
                return cand
        return None
    for _attempt in range(20):
        cand = "{}{:0{}d}".format(
            prefix, r.randint(10 ** width,
                              2 * 10 ** width - 1), width)
        if cand not in existing:
            return cand
    return None


def _drift_key(key: str, r) -> str:
    choice = r.randint(0, 2)
    if choice == 0:
        return key.lower()
    if choice == 1:
        return " " + key
    return key + "  "


def _dup_source(bp: TableBlueprint, dirty_idx: int,
                n_clean: int) -> int:
    """Appended duplicates carry their source row index when the
    planner recorded one; fall back to a stable wrap."""
    dup_map = getattr(bp, "duplicate_of", None)
    if isinstance(dup_map, dict) and dirty_idx in dup_map:
        return dup_map[dirty_idx]
    return dirty_idx % max(n_clean, 1)


@dataclass
class LinkReport:
    link: str
    orphans_planted: int
    orphans_flagged: int
    false_flags: int
    n_rows: int
    drift_planted: int = 0
    drift_false_flagged: int = 0

    @property
    def recall(self) -> float:
        return (self.orphans_flagged / self.orphans_planted
                if self.orphans_planted else 0.0)

    @property
    def precision(self) -> float:
        total = self.orphans_flagged + self.false_flags
        return self.orphans_flagged / total if total else 0.0

    def format_text(self) -> str:
        text = ("LINK {}: {} orphan(s) planted in {} rows — "
                "recall {:.3f}, precision {:.3f}".format(
                    self.link, self.orphans_planted,
                    self.n_rows, self.recall, self.precision))
        if self.drift_planted:
            text += ("; {} drifted-but-VALID ref(s), {} "
                     "false-flagged".format(
                         self.drift_planted,
                         self.drift_false_flagged))
        return text


def evaluate_links(rbp: RelationalBlueprint, link: Link,
                   flags: List[bool],
                   flagger_name: str = "flagger") -> LinkReport:
    """Scores a vendor's orphan flags (one bool per dirty child
    row) against the ledger."""
    child_bp = rbp.blueprints[link.child]
    n = len(child_bp.dirty_rows)
    if len(flags) != n:
        raise RelationalIntegrityError(
            "flags must cover every dirty child row: got {} "
            "for {}".format(len(flags), n))
    planted = {m.row for m in rbp.link_ledger
               if m.child == link.child
               and m.fk_column == link.fk_column}
    drifted = {m.row for m in rbp.drift_ledger
               if m.child == link.child
               and m.fk_column == link.fk_column}
    flagged = {i for i, f in enumerate(flags) if f}
    return LinkReport(
        link="{}.{} -> {}.{}".format(
            link.child, link.fk_column, link.parent,
            link.parent_key),
        orphans_planted=len(planted),
        orphans_flagged=len(planted & flagged),
        false_flags=len(flagged - planted),
        n_rows=n,
        drift_planted=len(drifted),
        drift_false_flagged=len(drifted & flagged))


def write_relational(run_dir, spec: RelationalSpec,
                     rbp: RelationalBlueprint) -> Path:
    from .tableplan import write_table
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    hashes: Dict[str, str] = {}
    for name, table in spec.tables.items():
        derived = TableSpec.from_json(table.to_json())
        derived.master_seed = _table_seed(spec.master_seed, name)
        write_table(run_dir / name, derived,
                    rbp.blueprints[name])
        sub = (run_dir / name / "manifest.json").read_text(
            encoding="utf-8")
        hashes["{}/manifest.json".format(name)] = \
            hashlib.sha256(sub.encode()).hexdigest()
    links_payload = json.dumps({
        "spec": json.loads(spec.to_json()),
        "orphans": [{"child": m.child, "row": m.row,
                     "fk_column": m.fk_column,
                     "true_key": m.true_key,
                     "orphan_key": m.orphan_key}
                    for m in rbp.link_ledger],
        "drift": [{"child": m.child, "row": m.row,
                   "fk_column": m.fk_column,
                   "true_key": m.true_key,
                   "drifted": m.drifted}
                  for m in rbp.drift_ledger],
    }, indent=2)
    (run_dir / "links.json").write_text(links_payload,
                                        encoding="utf-8")
    hashes["links.json"] = hashlib.sha256(
        links_payload.encode()).hexdigest()
    (run_dir / "manifest.json").write_text(json.dumps({
        "synthkit_relational_manifest": 1,
        "hashes": hashes,
    }, indent=2), encoding="utf-8")
    return run_dir


def load_relational(run_dir) -> dict:
    run_dir = Path(run_dir)
    manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8"))
    for rel, expected in manifest["hashes"].items():
        actual = hashlib.sha256(
            (run_dir / rel).read_text(
                encoding="utf-8").encode()).hexdigest()
        if actual != expected:
            raise RelationalIntegrityError(
                "`{}` does not match its manifest hash — the "
                "artifact has been altered".format(rel))
    return json.loads(
        (run_dir / "links.json").read_text(encoding="utf-8"))
