"""Derive note facts from a real cohort's own columns.

WHY THIS EXISTS. `transcribe` needs ONE COLUMN PER FACT, and a
clinical extract's diagnoses, procedures and drugs arrive as
semicolon-joined SETS. So the facts a note should mention are not
columns at all until the tokens are expanded - and the expansion
has to be undone before the file is written, or the recipient
receives search scaffolding this project refuses to ship.

The part that is easy to get wrong is the UNDO. `transcribe`
honors `both_disagree` and `text_only` by overriding the fact's
own column, so an override on a derived indicator leaves the SET
STRING still asserting the thing the note denies - and the set is
what the recipient reads. A planted disagreement would then be
invisible, and the record would contradict its own note for a
reason nobody could see. `reconcile` writes the overrides back
into the set string and drops the indicators.

Tokens come from `sets.vocabulary`, so the k screen governs what a
note may mention exactly as it governs what the file may publish.
A note is a release.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from synthkit import sets as S
from synthkit.transcribe import CORRUPTIONS, FactSpec

# Clinical phrasing for the columns this project has met. A label
# is a statement about what a column MEANS, so it is declared
# rather than derived - and a column with no declared label and no
# word-shaped name is REPORTED unlabelled rather than given prose
# invented from a code.
MEASURE_LABELS: Dict[str, Tuple[str, str]] = {
    "systolic_blood_pressure": ("systolic blood pressure", "mmHg"),
    "systolic_blood_pressure_invasive":
        ("arterial systolic pressure", "mmHg"),
    "diastolic_blood_pressure": ("diastolic blood pressure", "mmHg"),
    "diastolic_blood_pressure_invasive":
        ("arterial diastolic pressure", "mmHg"),
    "mean_arterial_pressure": ("mean arterial pressure", "mmHg"),
    "mean_arterial_pressure_cuff":
        ("mean arterial pressure", "mmHg"),
    "mean_arterial_pressure_invasive":
        ("arterial mean pressure", "mmHg"),
    "heart_rate": ("heart rate", "bpm"),
    "respiratory_rate": ("respiratory rate", "breaths per minute"),
    "spo2": ("oxygen saturation", "%"),
    "temperature": ("temperature", "C"),
    "glasgow_coma_score": ("Glasgow coma score", ""),
    "body_mass_index_measured": ("body mass index", "kg/m2"),
    "height": ("height", "cm"),
    "weight": ("weight", "kg"),
    "urine_voided": ("urine output", "mL"),
    # names the 800-patient extract actually uses - the first real
    # run reported `temperature_oral` and `heart_rate_monitored`
    # as not-charted because this list named the generic forms
    "temperature_oral": ("oral temperature", "C"),
    "heart_rate_monitored": ("heart rate", "bpm"),
    "peripheral_pulse_rate": ("peripheral pulse", "bpm"),
    "mean_arterial_pressure_cuff_bmdi":
        ("mean arterial pressure", "mmHg"),
    "right_pupil_size": ("right pupil", "mm"),
    "left_pupil_size": ("left pupil", "mm"),
}

# A measurement needs ROWS, not a SHARE. This project has recorded
# the presence-share fault four times - a column whose present
# values are numbers is numeric however often it is present - and
# a 0.5 coverage gate committed it a fifth time: on the real
# extract's inpatient slice EVERY declared vital sits under it
# (spo2 0.194, respiratory_rate 0.343, systolic 0.449), so a note
# carried no vitals line at all. A vital recorded on a fifth of
# encounters is perfectly chartable; the note mentions it where it
# was taken and stays silent where it was not, which is what a
# real note does.
MIN_MEASURE_ROWS = 50

# Eligible as a CHARTED measurement, in the order a note would
# carry them. `span_days` and `age_at_visit` are deliberately
# absent: they are encounter attributes, and a vitals line reading
# "length of stay 3 days" is not prose anybody writes.
NOTE_MEASURES: List[str] = [
    "systolic_blood_pressure", "diastolic_blood_pressure",
    "systolic_blood_pressure_invasive",
    "diastolic_blood_pressure_invasive",
    "mean_arterial_pressure", "mean_arterial_pressure_cuff",
    "mean_arterial_pressure_invasive",
    "heart_rate", "respiratory_rate", "spo2", "temperature",
    "glasgow_coma_score", "body_mass_index_measured",
    "height", "weight", "urine_voided",
]

# Which set column carries which kind of fact, and where in a note
# that kind is written. A drug belongs in the plan and a diagnosis
# in the assessment; putting every fact in one section is the tell
# that gives generated prose away.
SET_KINDS: Dict[str, Tuple[str, str]] = {
    "conditions": ("condition", "assessment"),
    "procedures": ("condition", "plan"),
    "active_drugs": ("med", "plan"),
    "drug_routes": ("med", "plan"),
}

# Only corruptions the renderer actually applies to that kind.
# Recording an intent that never reaches the page would grade an
# extractor against text that does not exist, which the ledger is
# careful about and this must not undo.
MEASURE_CORRUPTIONS = ["transcription_error", "omitted_units",
                       "copy_forward", "abbreviation"]
NAMED_CORRUPTIONS = ["negation_simple", "negation_compound",
                     "hedge", "temporal_history",
                     "negation_scope_trap", "abbreviation"]


def _word_shaped(name: str) -> bool:
    """A column name a reader would accept as prose."""
    letters = sum(c.isalpha() for c in name)
    return letters >= 3 and not name[:1].isdigit()


def label_for_column(col: str) -> Optional[str]:
    if col in MEASURE_LABELS:
        return MEASURE_LABELS[col][0]
    if _word_shaped(col):
        return col.replace("_", " ").strip()
    return None


def unit_for_column(col: str) -> str:
    return MEASURE_LABELS.get(col, ("", ""))[1]


def label_for_token(token: str,
                    descriptions: Optional[Dict[str, str]] = None
                    ) -> str:
    """A token's clinical phrasing.

    Drug and route tokens in this extract are already names, so
    they need nothing. A diagnosis token is a CODE, and a note
    reading "history of E11.9" is not prose a clinician would
    write - so a description is used where one is supplied and the
    bare code is used where it is not, which is honest rather than
    invented.
    """
    tok = str(token)
    if descriptions:
        for key in (tok, tok.split("!")[-1]):
            if key in descriptions:
                return str(descriptions[key])
    if "!" in tok:
        tok = tok.split("!", 1)[1]
    return tok.replace("_", " ").strip()


def _assign(corruptions: List[str], i: int,
            per_fact: int = 2) -> List[str]:
    """Spread the taxonomy across facts, round robin.

    Every corruption must be reachable or the grader has a type it
    can never report on, and giving every fact every corruption
    would make a note unreadable - so each fact carries a couple
    and the rotation guarantees coverage across the set.
    """
    if not corruptions:
        return []
    n = len(corruptions)
    return [corruptions[(i * per_fact + j) % n]
            for j in range(min(per_fact, n))]


def build_facts(frame: pd.DataFrame,
                group_by: Optional[str] = None,
                k: int = 10,
                tokens_per_set: int = 8,
                descriptions: Optional[Dict[str, str]] = None,
                measures: int = 6,
                ) -> Tuple[pd.DataFrame, List[FactSpec], Dict[str, Any]]:
    """Expand what a note needs and declare the facts.

    Returns the frame with indicator columns added, the fact list,
    and the plan `reconcile` needs to undo the expansion.
    """
    out = frame.copy()
    facts: List[FactSpec] = []
    plan: Dict[str, Any] = {"sets": {}, "indicators": [],
                            "unlabelled": [], "skipped_sets": {},
                            "not_charted": [], "too_thin": [],
                            "over_measure_budget": []}

    groups = (out[group_by] if group_by and group_by in out.columns
              else pd.Series(np.arange(len(out)), index=out.index))

    # --- set columns become named facts -------------------------
    for col in list(out.columns):
        if col not in SET_KINDS or S.is_scaffolding(col):
            continue
        raw = out[col].astype(str)
        if not S.looks_like_set(raw):
            plan["skipped_sets"][col] = (
                "not shaped like a set in this frame")
            continue
        vocab = S.vocabulary(raw, groups, k=k)
        if not vocab or not vocab.get("tokens"):
            plan["skipped_sets"][col] = (
                "no token is held by at least {} patients".format(k))
            continue
        sep = vocab["separator"]
        kind, section = SET_KINDS[col]
        chosen = [str(t["value"])
                  for t in vocab["tokens"][:tokens_per_set]]
        ind_names = []
        for i, tok in enumerate(chosen):
            name = S.indicator_name(col, tok)
            out[name] = S.has_token(raw, sep, tok).astype(int)\
                .astype(str)
            ind_names.append(name)
            facts.append(FactSpec(
                column=name,
                label=label_for_token(tok, descriptions),
                kind=kind,
                placement="both_agree",
                corruptions=_assign(NAMED_CORRUPTIONS, len(facts)),
                section=section))
        plan["sets"][col] = {"sep": sep, "tokens": chosen,
                             "indicators": ind_names}
        plan["indicators"].extend(ind_names)

    # --- numeric columns become measurement facts ---------------
    taken = 0
    eligible = [c for c in NOTE_MEASURES if c in frame.columns]
    for col in frame.columns:
        if col in eligible or col in SET_KINDS or col == group_by:
            continue
        if S.is_scaffolding(col):
            continue
        vals = pd.to_numeric(frame[col], errors="coerce")
        if (int(vals.notna().sum()) >= MIN_MEASURE_ROWS
                and vals.nunique() >= 4):
            plan["not_charted"].append(col)
    for col in eligible:
        if taken >= measures:
            plan["over_measure_budget"].append(col)
            continue
        vals = pd.to_numeric(frame[col], errors="coerce")
        rows_with = int(vals.notna().sum())
        if rows_with < MIN_MEASURE_ROWS or vals.nunique() < 4:
            plan["too_thin"].append(
                "{} ({} row(s) carry a value)".format(
                    col, rows_with))
            continue
        lab = label_for_column(col)
        if lab is None:
            plan["unlabelled"].append(col)
            continue
        facts.append(FactSpec(
            column=col, label=lab, kind="measurement",
            unit=unit_for_column(col), placement="both_agree",
            corruptions=_assign(MEASURE_CORRUPTIONS, len(facts)),
            section="vitals"))
        taken += 1

    return out, facts, plan


def reconcile(frame: pd.DataFrame,
              plan: Dict[str, Any]) -> pd.DataFrame:
    """Write the overrides back into the set strings, then drop
    the indicators.

    A row's UNEXPANDED tokens must survive - only a handful of
    tokens per set become facts, and rebuilding the string from
    the indicators alone would silently delete the rest.
    """
    out = frame.copy()
    for col, info in plan.get("sets", {}).items():
        if col not in out.columns:
            continue
        sep = info["sep"]
        expanded = list(info["tokens"])
        names = info["indicators"]
        rebuilt = []
        for pos in range(len(out)):
            cur = str(out[col].iloc[pos])
            have = [t for t in cur.split(sep) if t.strip()]
            keep = [t for t in have if t not in expanded]
            for tok, name in zip(expanded, names):
                if name not in out.columns:
                    continue
                flag = str(out[name].iloc[pos]).strip()
                if flag in ("1", "True", "true"):
                    keep.append(tok)
            rebuilt.append(sep.join(keep))
        out[col] = rebuilt
    drop = [c for c in plan.get("indicators", [])
            if c in out.columns]
    return out.drop(columns=drop)


def default_rates(rate: float = 0.25) -> Dict[str, float]:
    """One rate per corruption, so every type is reachable."""
    return {c: float(rate) for c in CORRUPTIONS}
