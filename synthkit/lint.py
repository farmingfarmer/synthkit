"""SYNTH_L1: semantic lint — the checks a validator cannot make.

Validation answers "is this spec lawful?"; lint answers "will this
spec produce what you MEANT?" by planning a deterministic probe
and inspecting the realized data. Every check here was a human
catch first, at station 03, squinting at a preview:

  L1 outcome prevalence      (a live -1.4 intercept yielded 71%
                              against a requested 15-20%)
  L2 derived-value sanity    (a dollar-amount noise_sigma made
                              10^150-dollar hospital stays)
  L3 rule-coupling truth     (an edit dropped a date rule and
                              gaps silently decoupled from stays)
  L4 fossil distributions    (rule targets carrying ignored
                              distributions mislead readers)
  L5 ignored name pools      (person_name synthesizes; provided
                              choices are silently unused)
  L6 description coverage    (the English asked for outliers;
                              the spec forgot them)

Findings are WARN (probably not what you meant) or INFO (worth
knowing); ERRORs belong to validation. `target_prevalence` on an
outcome makes L1 enforceable: "around 15-20 percent" in English
becomes [0.15, 0.20] in the spec becomes a measured check.

Python 3.8 compatible. Stdlib only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .tablespec import TableSpec

_MESS_KEYWORDS = [
    ("duplicate", "duplicate_rate",
     lambda spec: spec.duplicate_rate > 0),
    ("missing", "missing_rate",
     lambda spec: any(c.mess.missing_rate > 0
                      for c in spec.columns)),
    ("outlier", "outlier_rate",
     lambda spec: any(c.mess.outlier_rate > 0
                      for c in spec.columns)),
    ("typo", "typo_rate",
     lambda spec: any(c.mess.typo_rate > 0
                      for c in spec.columns)),
    ("wrong", "wrong_rate",
     lambda spec: any(c.mess.wrong_rate > 0
                      for c in spec.columns)),
    ("casing", "case_rate",
     lambda spec: any(c.mess.case_rate > 0
                      for c in spec.columns)),
    ("whitespace", "space_rate",
     lambda spec: any(c.mess.space_rate > 0
                      for c in spec.columns)),
]


@dataclass
class LintFinding:
    level: str          # WARN | INFO
    code: str
    message: str


@dataclass
class LintReport:
    findings: List[LintFinding] = field(default_factory=list)
    probe_rows: int = 0

    @property
    def ok(self) -> bool:
        return not any(f.level == "WARN" for f in self.findings)

    def format_text(self) -> str:
        if not self.findings:
            return ("SEMANTIC LINT: clean ({} probe rows) — the "
                    "spec appears to mean what it says."
                    .format(self.probe_rows))
        lines = ["SEMANTIC LINT ({} probe rows): {} finding(s)"
                 .format(self.probe_rows, len(self.findings))]
        for f in self.findings:
            lines.append("  {:<4} [{}] {}".format(
                f.level, f.code, f.message))
        return "\n".join(lines)


def lint_table(spec: TableSpec, description: str = "",
               probe_rows: int = 300) -> LintReport:
    """Plan a deterministic probe and inspect what the spec
    actually produces. Never raises for content reasons; the spec
    must already validate."""
    from .tableplan import plan_table
    spec.validate()
    probe = TableSpec.from_json(spec.to_json())
    probe.rows = min(spec.rows, probe_rows)
    bp = plan_table(probe)
    n = len(bp.clean_rows)
    report = LintReport(probe_rows=n)
    add = report.findings.append

    rule_targets = {r.get("later") for r in spec.rules
                    if r.get("kind") == "date_after"}
    rule_targets |= {r.get("target") for r in spec.rules
                     if r.get("kind") == "derived"}

    # ---- L1: outcome prevalence ----
    for oc in spec.outcomes:
        name = oc["name"]
        if oc.get("kind") == "linear":
            vals = [float(r[name]) for r in bp.clean_rows]
            mean = sum(vals) / n
            var = sum((v - mean) ** 2 for v in vals) / n
            tr_ = oc.get("target_range")
            if var < 1e-12:
                add(LintFinding(
                    "WARN", "L1-degenerate",
                    "outcome `{}` has zero variance — no signal "
                    "to regress".format(name)))
            elif tr_ and not (float(tr_[0]) <= mean
                              <= float(tr_[1])):
                add(LintFinding(
                    "WARN", "L1-range",
                    "outcome `{}` realizes mean {:.2f} against "
                    "the declared target range [{}, {}] — "
                    "adjust the intercept".format(
                        name, mean, tr_[0], tr_[1])))
            else:
                add(LintFinding(
                    "INFO", "L1-range",
                    "outcome `{}` realizes mean {:.2f}, sd "
                    "{:.2f}".format(name, mean, var ** 0.5)))
            continue
        rate = sum(1 for r in bp.clean_rows
                   if r[name] == "True") / n
        target = oc.get("target_prevalence")
        if target:
            lo, hi = float(target[0]), float(target[1])
            if not (lo <= rate <= hi):
                add(LintFinding(
                    "WARN", "L1-prevalence",
                    "outcome `{}` realizes {:.1%} against the "
                    "declared target [{:.0%}, {:.0%}] — adjust "
                    "the intercept (more negative = rarer)"
                    .format(name, rate, lo, hi)))
            else:
                add(LintFinding(
                    "INFO", "L1-prevalence",
                    "outcome `{}` realizes {:.1%}, inside its "
                    "declared target".format(name, rate)))
        elif rate < 0.01 or rate > 0.9:
            add(LintFinding(
                "WARN", "L1-degenerate",
                "outcome `{}` realizes {:.1%} — nearly single-"
                "class; prediction campaigns will be vacuous"
                .format(name, rate)))
        else:
            add(LintFinding(
                "INFO", "L1-prevalence",
                "outcome `{}` realizes {:.1%} (no declared "
                "target — add `target_prevalence: [lo, hi]` to "
                "enforce)".format(name, rate)))

    # ---- L2: derived-value sanity ----
    for rule in spec.rules:
        if rule.get("kind") != "derived":
            continue
        target = rule["target"]
        vals = []
        for r in bp.clean_rows:
            try:
                vals.append(float(r[target]))
            except ValueError:
                pass
        if not vals:
            continue
        top = max(abs(v) for v in vals)
        if not all(math.isfinite(v) for v in vals) or top > 1e12:
            add(LintFinding(
                "WARN", "L2-derived",
                "derived column `{}` reaches magnitude {:.2e} — "
                "check `factor` and `noise_sigma` (scale belongs "
                "in factor)".format(target, top)))
        elif all(v == 0 for v in vals):
            add(LintFinding(
                "WARN", "L2-derived",
                "derived column `{}` is all zeros — factor or "
                "source likely wrong".format(target)))

    # ---- L3: rule-coupling truth ----
    from datetime import date as _date
    for rule in spec.rules:
        if rule.get("kind") != "date_after" \
                or not rule.get("days_from"):
            continue
        later, earlier = rule["later"], rule["earlier"]
        src = rule["days_from"]
        bad = 0
        for r in bp.clean_rows:
            gap = (_date.fromisoformat(r[later])
                   - _date.fromisoformat(r[earlier])).days
            if gap != max(int(round(float(r[src]))), 0):
                bad += 1
        if bad:
            add(LintFinding(
                "WARN", "L3-coupling",
                "date gap `{}`-`{}` disagrees with `{}` in "
                "{}/{} probe rows — rule ordering or an "
                "overriding rule is interfering".format(
                    later, earlier, src, bad, n)))

    # ---- L4: fossil distributions ----
    for col in spec.columns:
        if col.name in rule_targets and col.distribution:
            add(LintFinding(
                "INFO", "L4-fossil",
                "column `{}` carries a distribution that its "
                "rule overwrites — omit it to say what you mean"
                .format(col.name)))

    # ---- L5: ignored name pools ----
    for col in spec.columns:
        if col.ctype == "person_name" and col.distribution:
            add(LintFinding(
                "INFO", "L5-names",
                "column `{}`: person_name synthesizes from a "
                "built-in pool; the provided distribution is "
                "ignored".format(col.name)))

    # ---- L6: description coverage ----
    if description:
        text = description.lower()
        for keyword, knob, present in _MESS_KEYWORDS:
            if keyword in text and not present(spec):
                add(LintFinding(
                    "WARN", "L6-coverage",
                    "the description mentions `{}` but no "
                    "column sets {} — a mess clause was dropped"
                    .format(keyword, knob)))
    return report


# ===================================================================
# Corpus lint — the document side of "did you mean this?"
# ===================================================================

def lint_corpus(spec, description: str = "",
                probe_docs: int = 30) -> LintReport:
    """Plans a deterministic probe corpus and inspects what the
    document spec actually plants."""
    from .planner import plan_corpus
    from .spec import DataSpec
    assert isinstance(spec, DataSpec)
    spec.validate()
    probe = DataSpec.from_json(spec.to_json())
    probe.corpus.size = min(spec.corpus.size, probe_docs)
    blueprints = plan_corpus(probe)
    n = len(blueprints)
    report = LintReport(probe_rows=n)
    add = report.findings.append

    planted: dict = {}
    distracted: dict = {}
    styles: dict = {}
    for bp in blueprints:
        for note in bp.notes:
            for el in note.elements:
                planted[el.element_id] = planted.get(
                    el.element_id, 0) + 1
            for d in note.distractors:
                distracted[d.distractor_id] = distracted.get(
                    d.distractor_id, 0) + 1
            for axis, val in note.style.items():
                styles.setdefault(axis, set()).add(val)

    element_ids = set()
    distractor_ids = set()
    for uf in spec.unstructured_fields:
        for el in uf.target_elements:
            element_ids.add(el.element_id)
            if planted.get(el.element_id, 0) == 0:
                add(LintFinding(
                    "WARN", "D1-never-planted",
                    "element `{}` was never planted across {} "
                    "probe documents — its recall will divide "
                    "by zero conceptually; check probability "
                    "and difficulty".format(el.element_id, n)))
        for dis in uf.distractors:
            distractor_ids.add(dis.distractor_id)
            if dis.density > 0 and distracted.get(
                    dis.distractor_id, 0) == 0:
                add(LintFinding(
                    "WARN", "D2-toothless",
                    "distractor `{}` has density {} but landed "
                    "in zero probe documents — the trap is "
                    "unloaded".format(dis.distractor_id,
                                      dis.density)))

    overlap = element_ids & distractor_ids
    if overlap:
        add(LintFinding(
            "WARN", "D3-collision",
            "ids used as BOTH element and distractor: {} — "
            "scoring cannot tell reward from trap".format(
                ", ".join(sorted(overlap)))))

    for axis, vals in sorted(styles.items()):
        if len(vals) == 1:
            add(LintFinding(
                "INFO", "D4-flat-style",
                "style axis `{}` drew a single value across "
                "the probe — the corpus will read uniformly"
                .format(axis)))

    if description:
        text = description.lower()
        if ("trap" in text or "distractor" in text) \
                and not distractor_ids:
            add(LintFinding(
                "WARN", "D5-coverage",
                "the description asks for traps/distractors "
                "but the spec defines none"))
    return report
