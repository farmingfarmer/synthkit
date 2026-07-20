"""SYNTH_A1: TableSpec — arbitrary tabular datasets, mess included.

The tabular half of synthkit: the user declares columns (type +
distribution) and MESS POLICIES (missingness, typos, format drift,
outliers, case drift, whitespace, duplicate rows). The planner
generates the CLEAN truth first, then applies mess deterministically
and records every corruption in a ledger — so "can vendor X clean
this?" becomes an exact, cell-level, sliced measurement instead of
an opinion.

Column types: int, float, category, str_id, person_name, date, bool
Distributions (per type where sensible):
    uniform      {min,max}
    normal       {mean,std,min?,max?}
    lognormal    {mu,sigma,min?,max?}
    beta         {alpha,beta,scale?}          (floats in [0,scale])
    categorical  {choices,[weights]}          (weights optional)
    date_range   {start,end}                  (ISO dates)
    sequence     {prefix,start}               (ids)
    bernoulli    {p}                          (bools)

Mess (per column, all rates in [0,1], all OFF by default):
    missing_rate    cell replaced by a missing token
    typo_rate       strings: one seeded char swap/drop/double
    format_rate     dates -> mixed formats; numbers -> separators
    outlier_rate    numerics scaled by outlier_factor
    case_rate       strings: upper/lower/title drift
    space_rate      leading/trailing whitespace
Table-level:
    duplicate_rate  fraction of rows duplicated (appended, marked
                    in the ledger)

Validation is total (every problem reported at once); JSON
round-trips exactly. Python 3.8 compatible. Stdlib only.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

MISSING_TOKENS = ["", "NULL", "N/A", "?"]

COLUMN_TYPES = ("int", "float", "category", "str_id",
                "person_name", "date", "bool")
DIST_KINDS = ("uniform", "normal", "lognormal", "beta",
              "categorical", "date_range", "sequence", "bernoulli",
              "mixture")

RULE_KINDS = ("date_after", "derived")

_TYPE_DISTS = {
    "int": {"uniform", "normal", "lognormal", "sequence",
            "mixture"},
    "float": {"uniform", "normal", "lognormal", "beta",
              "mixture"},
    "category": {"categorical"},
    "str_id": {"sequence"},
    "person_name": {"categorical"},   # ignored; names synthesized
    "date": {"date_range"},
    "bool": {"bernoulli"},
}


class TableSpecError(ValueError):
    pass


@dataclass
class ColumnMess:
    missing_rate: float = 0.0
    missing_tokens: List[str] = field(
        default_factory=lambda: list(MISSING_TOKENS))
    typo_rate: float = 0.0
    format_rate: float = 0.0
    outlier_rate: float = 0.0
    outlier_factor: float = 10.0
    case_rate: float = 0.0
    space_rate: float = 0.0
    # The plausible-lie tier: the cell stays FORMAT-VALID but
    # carries a wrong value (date shifted, digits transposed,
    # category swapped, bool flipped). Normalizers cannot fix
    # these; real data-quality tools must DETECT them.
    wrong_rate: float = 0.0

    def any_active(self) -> bool:
        return any(r > 0 for r in (
            self.missing_rate, self.typo_rate, self.format_rate,
            self.outlier_rate, self.case_rate, self.space_rate,
            self.wrong_rate))


@dataclass
class ColumnSpec:
    name: str
    ctype: str
    distribution: Dict[str, Any] = field(default_factory=dict)
    mess: ColumnMess = field(default_factory=ColumnMess)

    def dist_kind(self) -> str:
        return str(self.distribution.get("kind", ""))


@dataclass
class TableSpec:
    title: str
    columns: List[ColumnSpec]
    rows: int = 100
    master_seed: int = 42
    duplicate_rate: float = 0.0
    # Cross-column rules, applied in order over generated columns:
    #  {"kind":"date_after","earlier":"admit","later":"discharge",
    #   "min_days":1,"max_days":30}
    #  {"kind":"derived","target":"total_cost","source":"los_days",
    #   "factor":1200,"noise_sigma":0.15}  (multiplicative noise)
    rules: List[Dict[str, Any]] = field(default_factory=list)

    # ---------------- validation (total) ----------------
    def validate(self) -> None:
        problems: List[str] = []
        if not self.title.strip():
            problems.append("table title is required")
        if self.rows < 1:
            problems.append("rows must be >= 1")
        if not (0.0 <= self.duplicate_rate <= 0.5):
            problems.append("duplicate_rate must be in [0, 0.5]")
        if not self.columns:
            problems.append("at least one column is required")
        seen = set()
        for col in self.columns:
            tag = "column `{}`".format(col.name or "?")
            if not col.name.strip():
                problems.append("a column is missing a name")
            elif col.name in seen:
                problems.append("{}: duplicate name".format(tag))
            seen.add(col.name)
            if col.ctype not in COLUMN_TYPES:
                problems.append("{}: unknown type `{}`".format(
                    tag, col.ctype))
                continue
            kind = col.dist_kind()
            if col.ctype == "person_name":
                pass  # synthesized; distribution optional
            elif kind not in DIST_KINDS:
                problems.append("{}: unknown distribution `{}`"
                                .format(tag, kind))
            elif kind not in _TYPE_DISTS[col.ctype]:
                problems.append(
                    "{}: distribution `{}` invalid for type `{}`"
                    .format(tag, kind, col.ctype))
            else:
                problems.extend(self._check_params(tag, col))
            m = col.mess
            for rname in ("missing_rate", "typo_rate",
                          "format_rate", "outlier_rate",
                          "case_rate", "space_rate"):
                if not (0.0 <= getattr(m, rname) <= 1.0):
                    problems.append("{}: {} must be in [0, 1]"
                                    .format(tag, rname))
            if m.outlier_rate > 0 and col.ctype not in (
                    "int", "float"):
                problems.append("{}: outlier_rate needs a numeric "
                                "column".format(tag))
            if m.wrong_rate > 0 and col.ctype == "str_id":
                problems.append("{}: wrong_rate is not supported "
                                "on str_id columns".format(tag))
        names = {c.name for c in self.columns}
        for i, rule in enumerate(self.rules):
            rtag = "rule #{}".format(i + 1)
            kind = rule.get("kind")
            if kind not in RULE_KINDS:
                problems.append("{}: unknown kind `{}`".format(
                    rtag, kind))
                continue
            if kind == "date_after":
                for key in ("earlier", "later"):
                    if rule.get(key) not in names:
                        problems.append(
                            "{}: `{}` must name a column".format(
                                rtag, key))
                if rule.get("earlier") == rule.get("later"):
                    problems.append("{}: earlier and later must "
                                    "differ".format(rtag))
                for key in ("min_days", "max_days"):
                    if key not in rule:
                        problems.append("{}: requires `{}`".format(
                            rtag, key))
            if kind == "derived":
                for key in ("target", "source", "factor"):
                    if key == "factor":
                        if key not in rule:
                            problems.append("{}: requires "
                                            "`factor`".format(rtag))
                    elif rule.get(key) not in names:
                        problems.append(
                            "{}: `{}` must name a column".format(
                                rtag, key))
                if rule.get("target") == rule.get("source"):
                    problems.append("{}: target and source must "
                                    "differ".format(rtag))
        if problems:
            raise TableSpecError("\n".join(problems))

    @staticmethod
    def _check_params(tag: str, col: ColumnSpec) -> List[str]:
        p = col.distribution
        kind = col.dist_kind()
        out: List[str] = []

        def need(*keys):
            for k in keys:
                if k not in p:
                    out.append("{}: `{}` requires param `{}`"
                               .format(tag, kind, k))
        if kind == "uniform":
            need("min", "max")
        elif kind == "normal":
            need("mean", "std")
        elif kind == "lognormal":
            need("mu", "sigma")
        elif kind == "beta":
            need("alpha", "beta")
        elif kind == "categorical":
            need("choices")
            choices = p.get("choices") or []
            weights = p.get("weights")
            if weights is not None:
                if len(weights) != len(choices):
                    out.append("{}: weights length must match "
                               "choices".format(tag))
                elif any(w < 0 for w in weights) or \
                        sum(weights) <= 0:
                    out.append("{}: weights must be non-negative "
                               "with a positive sum".format(tag))
        elif kind == "date_range":
            need("start", "end")
        elif kind == "sequence":
            need("prefix", "start")
        elif kind == "bernoulli":
            need("p")
            if "p" in p and not (0.0 <= p["p"] <= 1.0):
                out.append("{}: p must be in [0, 1]".format(tag))
        elif kind == "mixture":
            comps = p.get("components") or []
            weights = p.get("weights") or []
            if not comps:
                out.append("{}: mixture requires components"
                           .format(tag))
            if len(weights) != len(comps):
                out.append("{}: mixture weights must match "
                           "components".format(tag))
            for j, comp in enumerate(comps):
                ck = comp.get("kind")
                if ck not in DIST_KINDS or ck == "mixture":
                    out.append("{}: component #{} has invalid "
                               "kind `{}`".format(tag, j + 1, ck))
                else:
                    sub = ColumnSpec(name=col.name,
                                     ctype=col.ctype,
                                     distribution=comp)
                    out.extend(TableSpec._check_params(
                        "{} component #{}".format(tag, j + 1),
                        sub))
        return out

    # ---------------- JSON ----------------
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2,
                          ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "TableSpec":
        d = json.loads(raw)
        return cls(
            title=d["title"],
            rows=d.get("rows", 100),
            master_seed=d.get("master_seed", 42),
            duplicate_rate=d.get("duplicate_rate", 0.0),
            rules=d.get("rules", []),
            columns=[
                ColumnSpec(
                    name=c["name"],
                    ctype=c["ctype"],
                    distribution=c.get("distribution", {}),
                    mess=ColumnMess(**c.get("mess", {})),
                )
                for c in d.get("columns", [])
            ],
        )
