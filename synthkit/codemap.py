"""A code-to-phrasing map, screened by what was already released.

WHY THIS EXISTS. A diagnosis token in a clinical extract is a
CODE, and the first real note run produced an assessment reading
"No I10 or E03.9" - which nobody writes. Descriptions turn that
into prose. Source tables in OMOP shape already carry them beside
the code, so the map is READ rather than invented.

AND THE SCREEN IS THE WHOLE DESIGN. A map built from the raw
source lists every code the source contains - including the ones
the k rule REFUSED TO PUBLISH - and a file naming those says "a
patient in this cohort has this rare condition", which is exactly
what the screen withheld. So the map is restricted to codes that
appear in the COHORT FILE being handed over: those are already
released, so naming them discloses nothing new, and anything else
is dropped and COUNTED so the screening is visible rather than
assumed.

A description is public reference data - an ICD-10 code means the
same thing in every hospital - so what is sensitive here is never
the phrasing. It is the LIST.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

CODE_COLUMNS = ("standard_concept_code", "concept_code",
                "source_value")
NAME_COLUMNS = ("standard_concept_name", "concept_name")


def _pick(columns: Iterable[str],
          wanted: Iterable[str]) -> Optional[str]:
    have = {str(c).strip().lower(): str(c) for c in columns}
    for w in wanted:
        if w in have:
            return have[w]
    return None


def code_columns(columns: Iterable[str]) -> List[str]:
    """EVERY column that carries a code, not the first one found.

    In OMOP the standard concept for a condition is SNOMED while
    the vocabulary a cohort was built on may be ICD-10 - which
    lives in the SOURCE value. Taking only the first code column
    described the standard code and left every ICD-10 token in the
    extract reading as a bare code.
    """
    out = []
    for c in columns:
        low = str(c).strip().lower()
        if low in CODE_COLUMNS or low.endswith("_source_value"):
            out.append(str(c))
    return out


def pairs_from(frame) -> Tuple[Dict[str, str], Optional[str]]:
    """code -> name from one table, or a reason it yielded none."""
    cols = list(frame.columns)
    codes = code_columns(cols)
    name = _pick(cols, NAME_COLUMNS)
    if not codes or name is None:
        return {}, ("no code/name pair - looked for {} or a "
                    "*_source_value beside {}"
                    .format("/".join(CODE_COLUMNS),
                            "/".join(NAME_COLUMNS)))
    out = {}
    names = frame[name].astype(str)
    for code in codes:
        for c, n in zip(frame[code].astype(str), names):
            c, n = c.strip(), " ".join(n.split())
            if not c or not n or n.lower() in ("nan", "none"):
                continue
            if c == n:
                continue      # a free-text source value is not a
                              # code, and mapping it to itself
                              # describes nothing
            out.setdefault(c, n)
    if not out:
        return {}, "the code and name columns are present but empty"
    return out, None


def cohort_codes(frame,
                 set_columns: Iterable[str] = (),
                 sep: str = ";") -> set:
    """Every token the released cohort file actually carries.

    Both the bare code and any `VOCAB!CODE` form, because a token
    may be written either way and a map that only answers one of
    them leaves half the notes reading as codes.
    """
    found = set()
    cols = [c for c in (set_columns or frame.columns)
            if c in frame.columns]
    for col in cols:
        for cell in frame[col].astype(str):
            if not cell or cell in ("nan", "None"):
                continue
            for tok in cell.split(sep):
                tok = tok.strip()
                if not tok:
                    continue
                found.add(tok)
                if "!" in tok:
                    found.add(tok.split("!", 1)[1])
    return found


def build(tables: Iterable[Any],
          released: Optional[set] = None
          ) -> Dict[str, Any]:
    """Merge the tables, then screen against what was released."""
    merged: Dict[str, str] = {}
    skipped: List[str] = []
    for label, frame in tables:
        got, why = pairs_from(frame)
        if why:
            skipped.append("{}: {}".format(label, why))
            continue
        for c, n in got.items():
            merged.setdefault(c, n)

    if released is None:
        return {"descriptions": merged, "screened": False,
                "tables_skipped": skipped,
                "note": ("NOT screened against a released cohort "
                         "- this map names every code the source "
                         "contains, including ones the k rule "
                         "refused to publish. Do not hand it to a "
                         "recipient.")}

    kept, withheld = {}, 0
    for c, n in merged.items():
        bare = c.split("!", 1)[1] if "!" in c else c
        if c in released or bare in released:
            kept[c] = n
            if bare != c:
                kept.setdefault(bare, n)
        else:
            withheld += 1
    return {"descriptions": kept, "screened": True,
            "codes_in_source": len(merged),
            "codes_kept": len(kept),
            "codes_withheld": withheld,
            "tables_skipped": skipped,
            "note": ("restricted to codes the released cohort "
                     "already carries - a code the k rule "
                     "withheld is not named here, because a list "
                     "of rare codes says a patient in this "
                     "cohort has one")}
