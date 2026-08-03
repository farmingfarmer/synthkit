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
def facts_from_model(net, max_facts: int = 12,
                     note_column: str = "clinical_note"
                     ) -> List[FactSpec]:
    """Derive a note plan from the columns the model actually has.

    Binary columns and list indicators become conditions or
    medications; numerics become measurements. Each fact is given
    the corruptions that can sensibly apply to it — a number can be
    transposed or lose its units, a finding can be negated or
    hedged — so the generated prose is messy in ways that match
    what the field really is.
    """
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
                placement="both_agree",
                corruptions=["abbreviation", "negation_simple",
                             "negation_scope_trap", "hedge",
                             "temporal_history"]))
        elif b.kind == "numeric":
            facts.append(FactSpec(
                col, _pretty(col), "measurement",
                section="vitals", placement="both_agree",
                corruptions=["transcription_error",
                             "omitted_units", "copy_forward",
                             "abbreviation"]))
        elif b.kind == "discrete" and len(b.levels) == 2:
            facts.append(FactSpec(
                col, _pretty(col), "condition",
                placement="both_agree",
                corruptions=["negation_simple",
                             "negation_scope_trap", "hedge"]))
        if len(facts) >= max_facts:
            break
    return facts


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
