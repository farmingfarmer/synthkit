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
    # Generated labels with PLANTED SIGNAL: a logistic model over
    # this table's own columns. Every row gets a true probability,
    # which makes ceiling metrics computable.
    #  {"name":"readmitted","kind":"logistic","intercept":-2.0,
    #   "coefficients":{"age":0.03,"los_days":0.1,
    #                   "department=oncology":0.8,"active":-0.4}}
    # Coefficient keys: numeric column, bool column (0/1), or
    # "column=value" indicator for categoricals.
    outcomes: List[Dict[str, Any]] = field(default_factory=list)

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
        rule_targets = set()
        for rule in self.rules:
            if rule.get("kind") == "date_after":
                rule_targets.add(rule.get("later"))
            elif rule.get("kind") == "derived":
                rule_targets.add(rule.get("target"))
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
            elif not col.distribution:
                # Rule-produced columns need no distribution —
                # requiring a throwaway one made a live compile
                # invent placeholders.
                if col.name not in rule_targets:
                    problems.append(
                        "{}: needs a distribution (or a rule "
                        "that produces it)".format(tag))
            elif kind in DIST_KINDS and kind not in                     _TYPE_DISTS.get(col.ctype, set())                     and col.name in rule_targets:
                problems.append(
                    "{}: distribution `{}` invalid for type "
                    "`{}` — this column is produced by a rule, "
                    "so you may omit its distribution entirely"
                    .format(tag, kind, col.ctype))
            elif kind not in DIST_KINDS:
                if kind in RULE_KINDS:
                    problems.append(
                        "{}: `{}` is a RULE kind, not a "
                        "distribution — give this column a "
                        "normal distribution for its type and "
                        "add a top-level rule referencing it"
                        .format(tag, kind))
                else:
                    problems.append(
                        "{}: unknown distribution `{}` — valid "
                        "kinds: {}".format(
                            tag, kind, ", ".join(DIST_KINDS)))
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
        ctypes = {c.name: c.ctype for c in self.columns}
        names = set(ctypes)
        for i, rule in enumerate(self.rules):
            rtag = "rule #{}".format(i + 1)
            kind = rule.get("kind")
            if kind not in RULE_KINDS:
                problems.append(
                    "{}: unknown kind `{}` — valid rule kinds are "
                    "date_after and derived; row duplication is "
                    "the top-level `duplicate_rate` field, not a "
                    "rule; per-column mess lives in each column's "
                    "`mess` object".format(rtag, kind))
                continue
            if kind == "date_after":
                for key in ("earlier", "later"):
                    col = rule.get(key)
                    if col not in names:
                        problems.append(
                            "{}: `{}` must name a column".format(
                                rtag, key))
                    elif ctypes[col] != "date":
                        problems.append(
                            "{}: `{}` column `{}` must have type "
                            "date (it is {})".format(
                                rtag, key, col, ctypes[col]))
                if rule.get("earlier") == rule.get("later"):
                    problems.append("{}: earlier and later must "
                                    "differ".format(rtag))
                days_from = rule.get("days_from")
                if days_from is not None:
                    if days_from not in names:
                        problems.append(
                            "{}: `days_from` must name a column"
                            .format(rtag))
                    elif ctypes[days_from] not in ("int",
                                                   "float"):
                        problems.append(
                            "{}: `days_from` column `{}` must be "
                            "numeric".format(rtag, days_from))
                else:
                    for key in ("min_days", "max_days"):
                        if key not in rule:
                            problems.append(
                                "{}: requires `{}` (or "
                                "`days_from` naming a numeric "
                                "column to drive the gap)".format(
                                    rtag, key))
            if kind == "derived":
                # A validated spec MUST plan: derived multiplies,
                # so both ends must be numeric — a live compile
                # authored derived over date columns, passed the
                # old checks, and crashed at plan time. The message
                # teaches the repair round which rule to use
                # instead.
                for key in ("target", "source"):
                    col = rule.get(key)
                    if col not in names:
                        problems.append(
                            "{}: `{}` must name a column".format(
                                rtag, key))
                    elif ctypes[col] not in ("int", "float"):
                        hint = ("for date ordering use kind "
                                "`date_after` — with `days_from` "
                                "naming a numeric column if the "
                                "gap should track that column"
                                if ctypes[col] == "date" else
                                "a generated label belongs in "
                                "the top-level `outcomes` array "
                                "(kind logistic), not in rules"
                                if ctypes[col] == "bool" else
                                "use a `column=value` indicator "
                                "coefficient in `outcomes` for "
                                "category influence")
                        problems.append(
                            "{}: `derived` requires numeric "
                            "columns, but `{}` column `{}` has "
                            "type {} — {}".format(
                                rtag, key, col, ctypes[col],
                                hint))
                if "factor" not in rule:
                    problems.append("{}: requires `factor`"
                                    .format(rtag))
                sigma = rule.get("noise_sigma", 0.0)
                if not (0.0 <= float(sigma) <= 3.0):
                    problems.append(
                        "{}: noise_sigma={} is not a noise "
                        "level — it is the sigma of "
                        "MULTIPLICATIVE lognormal noise, where "
                        "0.15 means roughly ±15% scatter and "
                        "anything above ~1 is extreme. Scale "
                        "belongs in `factor`; keep noise_sigma "
                        "in [0, 3]. (A live compile put a "
                        "dollar amount here and produced 10^150-"
                        "dollar hospital stays.)".format(
                            rtag, sigma))
                if rule.get("target") == rule.get("source"):
                    problems.append("{}: target and source must "
                                    "differ".format(rtag))
        for i, oc in enumerate(self.outcomes):
            otag = "outcome #{}".format(i + 1)
            name = oc.get("name", "")
            if not str(name).strip():
                problems.append("{}: requires a name".format(otag))
            elif name in names:
                problems.append("{}: name `{}` collides with a "
                                "column".format(otag, name))
            if oc.get("kind") != "logistic":
                problems.append("{}: kind must be `logistic`"
                                .format(otag))
            if "intercept" not in oc:
                problems.append("{}: requires `intercept`"
                                .format(otag))
            target = oc.get("target_prevalence")
            if target is not None:
                bad = (not isinstance(target, (list, tuple))
                       or len(target) != 2)
                if not bad:
                    lo, hi = target
                    bad = not (0.0 < float(lo) <= float(hi)
                               < 1.0)
                if bad:
                    problems.append(
                        "{}: target_prevalence must be "
                        "[lo, hi] with 0 < lo <= hi < 1"
                        .format(otag))
            coeffs = oc.get("coefficients") or {}
            if not coeffs:
                problems.append("{}: requires non-empty "
                                "`coefficients`".format(otag))
            for key in coeffs:
                base = key.split("=", 1)[0]
                if base not in names:
                    problems.append(
                        "{}: coefficient `{}` names no column"
                        .format(otag, key))
                    continue
                ctype = ctypes[base]
                if "=" in key and ctype != "category":
                    problems.append(
                        "{}: indicator `{}` needs a category "
                        "column".format(otag, key))
                if "=" not in key and ctype not in (
                        "int", "float", "bool"):
                    problems.append(
                        "{}: coefficient `{}` needs a numeric or "
                        "bool column (use `{}=value` for "
                        "categories)".format(otag, key, base))
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
            outcomes=d.get("outcomes", []),
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
