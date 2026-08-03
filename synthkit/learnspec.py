"""Turn a learned model into things a person can read and edit.

The conditional network knows the joint distribution, but a report
of `conditions::chf <- active_drugs::furosemide` is a sentence for
an engineer, not a clinician. And the transcriber needs to know
which columns are conditions, which are measurements, and which
carry list content — knowledge the profile already holds.

So this module does three jobs, all of them translation:

  narrate    what the model found, in clinical English, separated
             from what it filed as the pipeline's own arithmetic
  dials      the relationships a person can strengthen, weaken or
             switch off, with plain labels
  facts      the note-column plan derived from the learned
             columns, so free text can be generated from real
             structure instead of hand-authored guesses

Nothing here learns anything new. It reads what was learned and
makes it legible.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .transcribe import CORRUPTIONS, FactSpec


# Real notes spell a diagnosis out at least once and abbreviate it
# thereafter. Coded column values ("chf", "htn") read as database
# fields rather than clinical prose, so the label is expanded to
# the full term and the `abbreviation` corruption shortens it back
# where a clinician would.
TERM_EXPANSIONS = {
    "chf": "congestive heart failure",
    "htn": "hypertension",
    "ckd": "chronic kidney disease",
    "dm": "diabetes mellitus",
    "copd": "chronic obstructive pulmonary disease",
    "cad": "coronary artery disease",
    "afib": "atrial fibrillation",
    "mi": "myocardial infarction",
    "cva": "cerebrovascular accident",
    "gerd": "gastroesophageal reflux disease",
    "ckd3": "chronic kidney disease stage 3",
    "esrd": "end stage renal disease",
    "sob": "shortness of breath",
    "uti": "urinary tract infection",
    "dvt": "deep vein thrombosis",
    "pe": "pulmonary embolism",
    "osa": "obstructive sleep apnea",
    "ptsd": "post-traumatic stress disorder",
}


def _pretty(col: str) -> str:
    """A column name as a clinician would say it."""
    if "::" in col:
        base, item = col.split("::", 1)
        return TERM_EXPANSIONS.get(item.strip().lower(),
                                   item)
    plain = col.replace("_", " ")
    return TERM_EXPANSIONS.get(plain.strip().lower(), plain)


def _is_indicator(col: str) -> bool:
    return "::" in col


# -------------------------------------------------------------------
def narrate(net) -> Dict[str, Any]:
    """What the model found, in plain English.

    Findings and bookkeeping are reported separately and labelled
    as such. An instrument that presents its own arithmetic as a
    discovery is harder to trust than one that says which is which.
    """
    rep = net.report
    findings = []
    for e in rep.get("edges", []):
        child = _pretty(e["child"])
        parents = [_pretty(p) for p in e["parents"]]
        if len(parents) == 1:
            sentence = ("{} moves with {}".format(child, parents[0]))
        else:
            sentence = ("{} depends on {} together".format(
                child, " and ".join(parents)))
        findings.append({
            "sentence": sentence,
            "child": e["child"], "parents": e["parents"],
            "kind": ("interaction" if len(parents) > 1
                     else "relationship")})

    book = []
    for d in rep.get("derived_columns", []):
        book.append({
            "sentence": "{} is computed from {}".format(
                _pretty(d["column"]), _pretty(d["determined_by"])),
            "column": d["column"]})
    for b, a in (rep.get("derived_families") or {}).items():
        book.append({
            "sentence": "every {} follows from the {}".format(
                b.replace("_", " "), a.replace("_", " ")),
            "column": b})

    n_rows = rep.get("rows", 0)
    n_people = rep.get("persons", n_rows)
    return {
        "headline": (
            "Learned from {} records covering {} patients. "
            "{} real relationship{} found; {} column{} filed as "
            "arithmetic the data pipeline itself created.".format(
                n_rows, n_people, len(findings),
                "" if len(findings) == 1 else "s",
                len(book), "" if len(book) == 1 else "s")),
        "findings": findings,
        "bookkeeping": book,
        "resolution": rep.get("bins"),
        "effective_n": rep.get("effective_n", n_people),
        "caveat": (
            "Relationships are only as trustworthy as the number "
            "of patients behind them. At {} patients this model "
            "can support simple relationships; nonlinear ones need "
            "roughly 800 and interactions roughly 3,200.".format(
                rep.get("effective_n", n_people))),
    }


def dials(net) -> List[Dict[str, Any]]:
    """Every discovered relationship, as something to turn."""
    out = []
    for e in net.report.get("edges", []):
        out.append({
            "column": e["child"],
            "label": "{} given {}".format(
                _pretty(e["child"]),
                ", ".join(_pretty(p) for p in e["parents"])),
            "parents": e["parents"],
            "factor": 1.0,
            "guide": "0 removes this relationship entirely, 1 "
                     "leaves it as found, 2 makes it twice as "
                     "strong on the log-odds scale",
        })
    return out


def apply_dials(net, settings: Dict[str, float]):
    """Turn them. Returns the same net for chaining."""
    for col, factor in (settings or {}).items():
        if col in net.cpt:
            net.amplify(col, float(factor))
    return net


# -------------------------------------------------------------------
def facts_from_columns(rows: List[Dict[str, Any]],
                      max_facts: int = 12,
                      text_only: Optional[List[str]] = None,
                      skip: Optional[List[str]] = None
                      ) -> List[FactSpec]:
    """Derive a note plan from data alone, with no model.

    The pipeline and the bench were each sniffing columns their
    own way, which is two implementations of one idea and a
    guarantee they will drift. This is the single one; the
    model-based version adds what the model knows on top.
    """
    hide = set(text_only or [])
    avoid = set(skip or []) | {"person_id", "visit_id",
                               "visit_number"}
    facts: List[FactSpec] = []
    sample = rows[:400]
    for col in (rows[0] if rows else {}):
        if col in avoid or col.startswith("_"):
            continue
        vals = {str(r.get(col, "")).strip() for r in sample}
        vals.discard("")
        if not vals:
            continue
        place = "text_only" if col in hide else "both_agree"
        if vals <= {"0", "1", "True", "False", "true", "false"} \
                and len(vals) == 2:
            kind = ("med" if ("drug" in col or "med" in col)
                    else "condition")
            facts.append(FactSpec(
                col, _pretty(col), kind,
                section=("plan" if kind == "med"
                         else "assessment"),
                placement=place,
                corruptions=["abbreviation", "negation_simple",
                             "negation_scope_trap", "hedge",
                             "temporal_history"]))
        elif all(_isnum(v) for v in list(vals)[:40]) \
                and len(vals) > 8:
            facts.append(FactSpec(
                col, _pretty(col), "measurement",
                section="vitals", placement=place,
                corruptions=["transcription_error",
                             "omitted_units", "copy_forward",
                             "abbreviation"]))
        if len(facts) >= max_facts:
            break
    return facts


def _isnum(v) -> bool:
    try:
        float(str(v).replace(",", ""))
        return True
    except ValueError:
        return False


def facts_from_model(net, max_facts: int = 12,
                     note_column: str = "clinical_note",
                     text_only: Optional[List[str]] = None
                     ) -> List[FactSpec]:
    """Derive a note plan from the columns the model actually has.

    Binary columns and list indicators become conditions or
    medications; numerics become measurements. Each fact is given
    the corruptions that can sensibly apply to it — a number can be
    transposed or lose its units, a finding can be negated or
    hedged — so the generated prose is messy in ways that match
    what the field really is.
    """
    # Facts named in `text_only` are REMOVED from the structured
    # record and written only into the prose. Without this the
    # reading-versus-blind comparison is meaningless: if a cause
    # of the outcome is still sitting in a column, a model that
    # cannot read scores just as well by looking it up, and the
    # value of reading measures as zero.
    hide = set(text_only or [])
    facts: List[FactSpec] = []
    for col in net.order:
        if col in getattr(net, "derived", {}):
            continue
        b = net.binnings.get(col)
        if b is None:
            continue
        if _is_indicator(col):
            base = col.split("::", 1)[0]
            kind = ("med" if "drug" in base or "med" in base
                    else "condition")
            facts.append(FactSpec(
                col, _pretty(col), kind,
                section=("plan" if kind == "med" else "assessment"),
                placement=("text_only" if col in hide
                           else "both_agree"),
                corruptions=["abbreviation", "negation_simple",
                             "negation_scope_trap", "hedge",
                             "temporal_history"]))
        elif b.kind == "numeric":
            facts.append(FactSpec(
                col, _pretty(col), "measurement",
                section="vitals", placement=("text_only" if col in hide
                           else "both_agree"),
                corruptions=["transcription_error",
                             "omitted_units", "copy_forward",
                             "abbreviation"]))
        elif b.kind == "discrete" and len(b.levels) == 2:
            facts.append(FactSpec(
                col, _pretty(col), "condition",
                placement=("text_only" if col in hide
                           else "both_agree"),
                corruptions=["negation_simple",
                             "negation_scope_trap", "hedge"]))
        if len(facts) >= max_facts:
            break
    return facts


def columns_to_hide(net, text_only: List[str]) -> List[str]:
    """Everything that would leak a hidden fact back into a column.

    Hiding the indicator `active_drugs::furosemide` is not enough:
    the medication list is rebuilt in every generated row, and it
    still spells the drug out. A model that cannot read prose
    would simply look it up there, and the whole reading-versus-
    blind comparison would measure nothing. So the parent list
    goes too.
    """
    hide = set(text_only or [])
    for col in list(hide):
        if "::" in col:
            hide.add(col.split("::", 1)[0])
    return sorted(hide)


def blank_columns(rows: List[Dict[str, Any]],
                  cols: List[str]) -> List[Dict[str, Any]]:
    for r in rows:
        for c in cols:
            if c in r:
                r[c] = ""
    return rows


def default_rates() -> Dict[str, float]:
    r = {c: 0.30 for c in CORRUPTIONS}
    r["abbreviation"] = 0.5
    return r


def plan_summary(facts: List[FactSpec]) -> Dict[str, Any]:
    """What the note plan will produce, before producing it."""
    by_kind: Dict[str, int] = {}
    for f in facts:
        by_kind[f.kind] = by_kind.get(f.kind, 0) + 1
    return {
        "facts": len(facts),
        "by_kind": by_kind,
        "sentence": (
            "{} facts will be written into each note: {}. Each can "
            "appear plainly, in shorthand, denied, hedged, or as a "
            "past finding — and the ledger records which, so an "
            "extractor is graded on what the note actually says."
            .format(len(facts),
                    ", ".join("{} {}{}".format(v, k,
                                               "" if v == 1 else "s")
                              for k, v in sorted(by_kind.items())))),
    }


# -------------------------------------------------------------------
# Planting truth on learned data, and grading against it
# -------------------------------------------------------------------
def outcome_candidates(net) -> List[Dict[str, Any]]:
    """Columns a person could plausibly build an outcome from.

    Derived columns are excluded: an outcome computed from the
    pipeline's own arithmetic would be predictable for reasons
    that have nothing to do with the patient.
    """
    out = []
    for col in net.order:
        if col in getattr(net, "derived", {}):
            continue
        b = net.binnings.get(col)
        if b is None:
            continue
        if b.kind == "numeric":
            kind = "measurement"
        elif "::" in col:
            kind = "finding"
        elif b.kind == "discrete" and len(b.levels) <= 4:
            kind = "flag"
        else:
            continue
        out.append({"column": col, "label": _pretty(col),
                    "kind": kind, "weight": 0.0})
    return out


def _standardize(rows, col):
    xs = []
    for r in rows:
        try:
            xs.append(float(str(r.get(col, "")).strip()))
        except (TypeError, ValueError):
            xs.append(None)
    got = [x for x in xs if x is not None]
    if not got:
        return [0.0] * len(rows)
    m = sum(got) / len(got)
    var = sum((x - m) ** 2 for x in got) / max(len(got) - 1, 1)
    sd = var ** 0.5 or 1.0
    return [(0.0 if x is None else (x - m) / sd) for x in xs]


def plant_outcome(rows: List[Dict[str, Any]],
                  weights: Dict[str, float],
                  name: str = "outcome",
                  prevalence: Optional[float] = None,
                  intercept: float = 0.0,
                  seed: int = 20260803) -> Dict[str, Any]:
    """Compute an outcome from coefficients the user chose.

    This is what turns generated data into an EXAM. The weights
    are a human's causal claims — nothing infers them — so the
    probability behind every row is known exactly, and therefore
    so is the best score any model could possibly achieve on it.

    The intercept is only a base rate, so when a prevalence is
    declared it is solved for rather than guessed.
    """
    import math
    import random as _rnd
    cols = [c for c, w in (weights or {}).items() if w]
    if not cols:
        return {"error": "Give at least one column a weight — "
                         "those weights are the planted truth."}
    z = {c: _standardize(rows, c) for c in cols}
    base = [sum(weights[c] * z[c][i] for c in cols)
            for i in range(len(rows))]

    def realize(b0):
        return [1.0 / (1.0 + math.exp(-max(-30.0,
                                           min(30.0, b0 + v))))
                for v in base]

    solved = intercept
    if prevalence:
        lo, hi = -25.0, 25.0
        for _ in range(40):
            mid = (lo + hi) / 2.0
            p = realize(mid)
            if sum(p) / len(p) < prevalence:
                lo = mid
            else:
                hi = mid
        solved = round((lo + hi) / 2.0, 4)
    probs = realize(solved)
    r = _rnd.Random(seed)
    labels = [1 if r.random() < p else 0 for p in probs]
    for row, lab in zip(rows, labels):
        row[name] = lab
    return {"rows": rows, "probs": probs, "label": name,
            "intercept": solved,
            "realized_prevalence": round(
                sum(labels) / len(labels), 4),
            "weights": {c: weights[c] for c in cols},
            "note": "the coefficients are stated by a person, not "
                    "inferred; that is what makes the answer key "
                    "exact and the ceiling knowable"}


def showdown(rows: List[Dict[str, Any]], probs: List[float],
             label: str, note_column: str = "clinical_note",
             vendor: str = "") -> Dict[str, Any]:
    """Ceiling, a model that reads, and a model that cannot.

    The ceiling is the score of the planted probabilities
    themselves — the best any model could achieve. Everything else
    is measured against it rather than against imagination.
    """
    from .autosolver import autosolver, autosolver_hybrid
    from .mlmetrics import auroc, auroc_interval
    y = [1 if str(r.get(label)) in ("1", "True", "true") else 0
         for r in rows]
    if sum(y) in (0, len(y)):
        return {"error": "The outcome is all one value, so nothing "
                         "can be scored. Adjust the weights or the "
                         "prevalence."}
    ceiling = auroc(probs, y)
    cut = int(len(rows) * 0.7)
    ytr = [v == 1 for v in y[:cut]]
    yte = y[cut:]
    npos, nneg = sum(yte), len(yte) - sum(yte)
    if npos == 0 or nneg == 0:
        return {"error": "Too few positive cases to grade. Raise "
                         "the prevalence or generate more rows."}

    def feats(rs, blind):
        drop = {label} | ({note_column} if blind else set())
        return [{k: v for k, v in r.items() if k not in drop}
                for r in rs]

    has_notes = note_column in rows[0]
    reading = auroc(autosolver_hybrid()(
        feats(rows[:cut], False), ytr, feats(rows[cut:], False)),
        yte)
    blind = auroc(autosolver()(
        feats(rows[:cut], True), ytr, feats(rows[cut:], True)),
        yte) if has_notes else None
    lo_r, hi_r = auroc_interval(reading, npos, nneg)
    res = {"ceiling": round(ceiling, 4),
           "reading": round(reading, 4),
           "reading_ci": [round(lo_r, 4), round(hi_r, 4)],
           "blind": round(blind, 4) if blind is not None else None,
           "value_of_reading": (round(reading - blind, 4)
                                if blind is not None else None),
           "test_rows": len(yte), "test_positives": npos,
           "has_notes": has_notes}
    if vendor:
        try:
            from .gui import _solver
            fn = _solver(vendor)
            v = auroc(fn(feats(rows[:cut], True), ytr,
                         feats(rows[cut:], True)), yte)
            res["vendor"] = round(v, 4)
            res["vendor_name"] = vendor
            res["vendor_beats_ours"] = v > reading
        except Exception as e:
            res["vendor_error"] = str(e)[:200]
    return res
