"""Render structured facts into messy clinical prose — with a
ledger of what the text actually asserts.

Why this exists. A benchmark whose facts appear identically in a
column and in a note measures nothing: a model can ignore the text
and score perfectly. Real charts are not like that. Some facts live
only in structured fields, some only in prose, and some appear in
both AND DISAGREE — which is the hard part of chart abstraction and
the part worth grading.

So each fact declares WHERE it lives, and the prose is corrupted by
a taxonomy rather than by undifferentiated noise. Each corruption
targets a distinct failure mode, and every one is ledgered with the
truth it obscures, so a vendor's report can read "recall 0.91
overall, but 0.34 on negated mentions and 0.12 on copy-forward"
instead of a single unhelpful grade.

Nothing here learns language from real notes. Facts are AUTHORED
and rendered, so no phrasing can be traced to a real patient — the
memorization risk that makes clinical text dangerous never arises.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------
# the corruption taxonomy
# ---------------------------------------------------------------
CORRUPTIONS = {
    "abbreviation":
        "written in shorthand a model may not expand",
    "negation_simple":
        "asserted as absent — the fact is NOT true of this patient",
    "negation_compound":
        "two findings negated by one operator ('no X or Y'), so "
        "the scope reaches further than a naive reader expects",
    "hedge":
        "asserted with uncertainty — not a confirmed finding",
    "temporal_history":
        "a PAST finding, not a current one",
    "copy_forward":
        "a stale value carried over from an earlier visit; the "
        "structured field holds the current one",
    "transcription_error":
        "digits transposed relative to the structured field",
    "omitted_units":
        "a number with no unit, so scale must be inferred",
    "negation_scope_trap":
        "a negation sits in the same clause but does NOT reach "
        "the finding ('no improvement in X' — X is present). Any "
        "extractor that looks for a negation word nearby gets "
        "this exactly backwards, which is why clause-level rules "
        "score well until they meet real prose",
}

PLACEMENTS = ("structured_only", "text_only", "both_agree",
              "both_disagree")

ABBREV = {
    "congestive heart failure": "CHF",
    "shortness of breath": "SOB",
    "chronic kidney disease": "CKD",
    "atrial fibrillation": "AFib",
    "diabetes mellitus": "DM",
    "hypertension": "HTN",
    "chest pain": "CP",
    "blood pressure": "BP",
    "heart rate": "HR",
    "coronary artery disease": "CAD",
    "chronic obstructive pulmonary disease": "COPD",
    "peripheral edema": "LE edema",
}


@dataclass
class FactSpec:
    """One clinical fact and where it is allowed to appear."""
    column: str
    label: str                       # human phrasing
    kind: str = "condition"          # condition | measurement | med
    unit: str = ""
    placement: str = "both_agree"
    corruptions: List[str] = field(default_factory=list)
    section: str = "assessment"


@dataclass
class TranscribeSpec:
    facts: List[FactSpec]
    rates: Dict[str, float] = field(default_factory=dict)
    filler_rate: float = 0.5
    seed: int = 20260731


FILLERS = [
    "Patient seen and examined at bedside.",
    "Vitals reviewed; no acute distress.",
    "Discussed plan with patient and family.",
    "Nursing notes reviewed.",
    "Will continue current management.",
    "Follow-up arranged with primary care.",
    "Labs pending at time of dictation.",
]
SECTIONS = ["hpi", "vitals", "assessment", "plan"]
SECTION_TITLES = {"hpi": "HPI", "vitals": "Vitals",
                  "assessment": "Assessment", "plan": "Plan"}


def _rng(seed: int, *parts) -> random.Random:
    key = "{}:{}".format(seed, ":".join(str(p) for p in parts))
    return random.Random(
        int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))


def _transpose(v: str) -> str:
    """A plausible keying error: swap two adjacent digits."""
    digits = [i for i, ch in enumerate(v) if ch.isdigit()]
    if len(digits) < 2:
        return v
    for a, b in zip(digits, digits[1:]):
        if b == a + 1 and v[a] != v[b]:
            return v[:a] + v[b] + v[a] + v[b + 1:]
    return v


def _present(fact: FactSpec, value: Any) -> bool:
    s = str(value).strip().lower()
    if fact.kind == "condition":
        return s in ("1", "true", "yes", "y", "present")
    return s not in ("", "0", "none", "nan", "false", "no")


# ---------------------------------------------------------------
def _phrase(fact, value, corrs, r) -> Tuple[str, Dict[str, Any]]:
    """Render one fact, applying its corruptions, and report what
    the resulting text actually ASSERTS."""
    label = fact.label
    # Only corruptions that actually reach the page belong in the
    # ledger. Several are mutually exclusive in rendering (a
    # hedged mention cannot also be a historical one), and
    # recording an intent that was never applied would grade an
    # extractor against text that does not exist.
    applied = [c for c in corrs
               if c in ("abbreviation", "omitted_units",
                        "transcription_error", "copy_forward")]
    for exclusive in ("negation_scope_trap", "negation_simple",
                      "negation_compound", "hedge",
                      "temporal_history"):
        if exclusive in corrs:
            applied.append(exclusive)
            break
    truth = {"fact": fact.column, "label": label,
             "corruptions": sorted(applied),
             "asserts_present": True, "asserted_value": None,
             "is_current": True, "is_certain": True}

    if "abbreviation" in corrs and label.lower() in ABBREV:
        label = ABBREV[label.lower()]

    if fact.kind == "measurement":
        # Clinicians do not write four decimal places. A generated
        # value carrying the full precision of its draw reads as
        # machine output and undermines the prose it sits in, so
        # numbers are rounded the way they would be charted.
        shown = str(value)
        try:
            fv = float(str(value).strip())
            shown = (str(int(round(fv))) if abs(fv) >= 10
                     else "{:.1f}".format(fv))
        except (TypeError, ValueError):
            pass
        if "transcription_error" in corrs:
            shown = _transpose(shown)
            truth["asserted_value"] = shown
        else:
            truth["asserted_value"] = shown
        unit = "" if "omitted_units" in corrs else (
            " " + fact.unit if fact.unit else "")
        if "copy_forward" in corrs:
            truth["is_current"] = False
            return ("{} {}{} (per prior note)".format(
                label, shown, unit), truth)
        return ("{} {}{}".format(label, shown, unit), truth)

    # conditions and medications
    if "negation_scope_trap" in corrs:
        # the finding IS present; the negation governs something
        # else entirely
        return (r.choice([
            "No improvement in {}.".format(label),
            "No change in {} since admission.".format(label),
            "{} with no resolution to date.".format(label),
            "No relief of {} with current therapy.".format(
                label)]), truth)
    if "negation_simple" in corrs:
        truth["asserts_present"] = False
        return (r.choice([
            "No {}.".format(label),
            "Denies {}.".format(label),
            "{} is not present.".format(label)]), truth)
    if "hedge" in corrs:
        truth["is_certain"] = False
        return (r.choice([
            "Possible {}.".format(label),
            "Cannot rule out {}.".format(label),
            "{} suspected but unconfirmed.".format(label)]), truth)
    if "temporal_history" in corrs:
        truth["is_current"] = False
        return (r.choice([
            "History of {}.".format(label),
            "Prior {}, resolved.".format(label),
            "{} in the past.".format(label)]), truth)
    if fact.kind == "med":
        return ("Continues {}.".format(label), truth)
    return (r.choice(["{} noted.".format(label),
                      "Findings consistent with {}.".format(label),
                      "{} present on exam.".format(label)]), truth)


def transcribe_row(row: Dict[str, Any], spec: TranscribeSpec,
                   row_index: int, prior: Optional[Dict] = None
                   ) -> Tuple[str, List[Dict[str, Any]], Dict]:
    """Return (note text, ledger, structured overrides).

    The overrides are how `both_disagree` and `structured_only`
    are honoured: the structured record keeps the true value while
    the prose says something else, or says nothing at all.
    """
    r = _rng(spec.seed, row_index, "note")
    by_section = {s: [] for s in SECTIONS}
    ledger: List[Dict[str, Any]] = []
    overrides: Dict[str, Any] = {}

    # a compound negation needs two absent findings to share one
    # operator, so collect the candidates first
    compound_pool = []

    for fact in spec.facts:
        value = row.get(fact.column, "")
        if fact.placement == "structured_only":
            continue
        if fact.placement == "text_only":
            # the whole point is that it can ONLY be found by
            # reading, so the column is emptied whether or not
            # this particular note ends up mentioning it
            overrides[fact.column] = ""
        present = _present(fact, value)

        chosen = []
        for c in fact.corruptions:
            if r.random() < spec.rates.get(c, 0.0):
                chosen.append(c)
        # negation only applies where the fact is genuinely ABSENT,
        # so a model that keyword-matches gets it backwards; and a
        # fact cannot be both negated and hedged
        if "negation_scope_trap" in chosen:
            # the trap asserts the finding IS present while a
            # negation sits nearby, so it is meaningless on a fact
            # that is genuinely absent — and it must never share a
            # clause with a real negation, or the ledger would
            # record two contradictory truths for one mention
            if present:
                chosen = [c for c in chosen
                          if c in ("negation_scope_trap",
                                   "abbreviation")]
            else:
                chosen = [c for c in chosen
                          if c != "negation_scope_trap"]
        if "negation_simple" in chosen or \
                "negation_compound" in chosen:
            if present:
                chosen = [c for c in chosen
                          if not c.startswith("negation")]
            else:
                chosen = [c for c in chosen
                          if c not in ("hedge",
                                       "temporal_history")]
        elif not present and fact.kind != "measurement":
            continue          # absent and not negated: say nothing

        if "negation_compound" in chosen:
            compound_pool.append(fact)
            continue

        text, truth = _phrase(fact, value, chosen, r)
        truth["placement"] = fact.placement
        truth["structured_value"] = value
        if fact.placement == "both_disagree":
            if fact.kind == "measurement":
                overrides[fact.column] = _transpose(str(value))
            else:
                overrides[fact.column] = "0" if present else "1"
            truth["disagrees_with_structured"] = True
            # the ledger must hold what the STRUCTURED RECORD ends
            # up saying, not the value we started from, or a
            # scorer cannot tell which source is wrong
            truth["structured_value"] = overrides[fact.column]
        if fact.placement == "text_only":
            overrides[fact.column] = ""
            truth["structured_value"] = ""
        by_section[fact.section].append(text)
        ledger.append(truth)

    if len(compound_pool) >= 2:
        a, b = compound_pool[0], compound_pool[1]
        by_section[a.section].append(
            "No {} or {}.".format(a.label, b.label))
        for f in (a, b):
            ledger.append({
                "fact": f.column, "label": f.label,
                "corruptions": ["negation_compound"],
                "asserts_present": False, "asserted_value": None,
                "is_current": True, "is_certain": True,
                "placement": f.placement,
                "structured_value": row.get(f.column, "")})
    else:
        for f in compound_pool:
            text, truth = _phrase(f, row.get(f.column, ""),
                                  ["negation_simple"], r)
            truth["placement"] = f.placement
            truth["structured_value"] = row.get(f.column, "")
            by_section[f.section].append(text)
            ledger.append(truth)

    lines = []
    for s in SECTIONS:
        body = list(by_section[s])
        if r.random() < spec.filler_rate:
            body.append(r.choice(FILLERS))
        if not body:
            continue
        r.shuffle(body)
        lines.append("{}: {}".format(SECTION_TITLES[s],
                                     " ".join(body)))
    return ("\n".join(lines), ledger, overrides)


def score_extraction(ledgers, extractions,
                     ) -> Dict[str, Any]:
    """Grade an extractor against the ledger, BY CORRUPTION TYPE.

    `extractions[i]` is what a model claims it found in note i: a
    list of {"fact": column, "present": bool, "value": optional}.

    A single overall recall hides everything that matters. Sliced
    by corruption, the report says which specific competence is
    missing — negation scope, temporal reasoning, unit inference —
    which is actionable in a way that a grade is not.
    """
    buckets: Dict[str, Dict[str, int]] = {}

    def bump(tag, key):
        b = buckets.setdefault(
            tag, {"n": 0, "found": 0, "missed": 0,
                  "false_positive": 0, "value_wrong": 0,
                  "stale_taken_as_current": 0,
                  "uncertain_taken_as_certain": 0})
        b[key] += 1
        return b

    for ledger, got in zip(ledgers, extractions):
        claimed = {}
        for g in got or []:
            claimed[g.get("fact")] = g
        for entry in ledger:
            tags = entry["corruptions"] or ["clean"]
            g = claimed.get(entry["fact"])
            said_present = bool(g and g.get("present", True))
            for t in tags:
                bump(t, "n")
                if entry["asserts_present"]:
                    bump(t, "found" if said_present else "missed")
                elif said_present:
                    # the text says the finding is ABSENT and the
                    # model claimed it anyway — the trap
                    bump(t, "false_positive")
                if (g and entry.get("asserted_value") is not None
                        and g.get("value") is not None
                        and str(g["value"]).strip()
                        != str(entry["asserted_value"]).strip()):
                    bump(t, "value_wrong")
                if (g and not entry["is_current"]
                        and g.get("current", True)):
                    bump(t, "stale_taken_as_current")
                if (g and not entry["is_certain"]
                        and g.get("certain", True)):
                    bump(t, "uncertain_taken_as_certain")

    report = {}
    for tag, b in sorted(buckets.items()):
        pos = b["found"] + b["missed"]
        neg = b["n"] - pos
        report[tag] = {
            "description": CORRUPTIONS.get(
                tag, "no corruption applied"),
            "mentions": b["n"],
            "recall": (round(b["found"] / pos, 3) if pos else None),
            "false_positive_rate": (
                round(b["false_positive"] / neg, 3)
                if neg else None),
            "value_errors": b["value_wrong"],
            "stale_taken_as_current": b["stale_taken_as_current"],
            "uncertain_taken_as_certain":
                b["uncertain_taken_as_certain"],
        }
    return report


def transcribe(rows: List[Dict[str, Any]], spec: TranscribeSpec,
               note_column: str = "clinical_note"
               ) -> Tuple[List[Dict[str, Any]],
                          List[List[Dict[str, Any]]]]:
    """Add a note column to every row and return the truth ledger.

    The ledger is the answer key: what each note actually asserts,
    which corruptions were applied, and where the prose and the
    structured record disagree.
    """
    out, ledgers = [], []
    for i, row in enumerate(rows):
        note, ledger, overrides = transcribe_row(row, spec, i)
        new = dict(row)
        new.update(overrides)
        new[note_column] = note
        out.append(new)
        ledgers.append(ledger)
    return out, ledgers
