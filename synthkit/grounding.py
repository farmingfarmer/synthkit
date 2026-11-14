"""Is every statement supported by the record we handed over?

WHY THIS EXISTS. `transcribe.score_extraction` grades an extractor
BY CORRUPTION TYPE and is the right tool for that, so nothing here
recomputes it - a second implementation is how two halves come to
disagree. But it iterates the LEDGER, so a claim about a fact that
was never in the record at all is invisible to it: the loop never
reaches a fact it has no entry for. That claim is a hallucination,
and on synthetic data it is the one thing we can judge perfectly,
because WE WROTE THE RECORD and know with certainty what is in it.

A vendor's safety claim is usually that every statement traces
back to the chart and anything that does not is removed. That is
checkable here against a complete answer key rather than by a
reviewer reading a sample.

THE CATEGORIES ARE DISJOINT AND THEY TOTAL THE CLAIM COUNT. A
census whose parts do not add up to a number a reader can check
against is how a negative count once sat in a report balancing
perfectly; the identity is asserted rather than hoped for.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

CATEGORIES = (
    "both",            # the note and the record agree it is there
    "text_only",       # only the prose says so
    "record_only",     # only the structured field says so
    "contradicted",    # the prose explicitly DENIES it
    "ungrounded",      # neither source mentions it at all
)


def _norm(s: Any) -> str:
    return " ".join(str(s or "").strip().lower().split())


def _present_flag(v: Any) -> bool:
    return _norm(v) in ("1", "true", "yes", "y", "present")


def known_labels(facts: Sequence[Any]) -> Dict[str, str]:
    """label -> column, for every fact a note could carry."""
    out = {}
    for f in facts:
        lab = getattr(f, "label", None) or f.get("label")
        col = getattr(f, "column", None) or f.get("column")
        if lab and col:
            out[_norm(lab)] = col
    return out


def record_support(frame, facts: Sequence[Any],
                   plan: Optional[Dict[str, Any]] = None,
                   descriptions: Optional[Dict[str, str]] = None
                   ) -> List[set]:
    """Per row, the labels THE STRUCTURED RECORD supports.

    A set-derived fact has no column in the written file - the
    indicators were dropped - so its support is read from the SET
    STRING, which is what the recipient actually holds. Reading a
    dropped indicator instead would report every such fact as
    unsupported and call a faithful tool a liar.
    """
    plan = plan or {}
    tok_of: Dict[str, Any] = {}
    for col, info in plan.get("sets", {}).items():
        for tok, ind in zip(info.get("tokens", []),
                            info.get("indicators", [])):
            tok_of[ind] = (col, tok, info.get("sep", ";"))

    from synthkit.notefacts import label_for_token

    rows = frame.to_dict("records") if hasattr(frame, "to_dict") \
        else list(frame)
    out = []
    for row in rows:
        here = set()
        # every token the row actually holds, expanded or not
        for col, info in plan.get("sets", {}).items():
            sep = info.get("sep", ";")
            cell = str(row.get(col, ""))
            for tok in [t for t in cell.split(sep) if t.strip()]:
                here.add(_norm(label_for_token(tok, descriptions)))
                here.add(_norm(tok))
        for f in facts:
            lab = _norm(getattr(f, "label", None))
            col = getattr(f, "column", None)
            if col in tok_of:
                continue           # handled above, from the set
            val = row.get(col, "")
            if getattr(f, "kind", "") == "measurement":
                if _norm(val) not in ("", "nan", "none"):
                    here.add(lab)
            elif _present_flag(val):
                here.add(lab)
        out.append(here)
    return out


def text_support(ledgers: Iterable[Sequence[Dict[str, Any]]]
                 ) -> List[Dict[str, set]]:
    """Per row, what the PROSE asserts: present, and denied."""
    out = []
    for ledger in ledgers:
        says, denies = set(), set()
        for e in ledger or []:
            lab = _norm(e.get("label"))
            if e.get("asserts_present", True):
                says.add(lab)
            else:
                denies.add(lab)
        out.append({"says": says, "denies": denies})
    return out


def audit(ledgers, claims, frame=None, facts=(), plan=None,
          descriptions=None) -> Dict[str, Any]:
    """Census every claim against both sources.

    `claims[i]` is what a tool asserted about row i: a list of
    {"label": str, "present": bool}. A claim of ABSENCE is not an
    assertion about the patient's chart in the sense this audit
    cares about, so only positive claims are censused - counting
    "no diabetes" as a statement needing support would make
    silence indistinguishable from accuracy.
    """
    text = text_support(ledgers)
    rec = (record_support(frame, facts, plan, descriptions)
           if frame is not None else [set() for _ in text])
    if not (len(text) == len(rec) == len(claims)):
        raise ValueError(
            "one ledger, one record row and one claim list per "
            "row: got {}, {}, {}".format(
                len(text), len(rec), len(claims)))

    counts = {c: 0 for c in CATEGORIES}
    ungrounded_examples: List[Dict[str, Any]] = []
    contradicted_examples: List[Dict[str, Any]] = []
    total = 0
    for i, got in enumerate(claims):
        for g in got or []:
            if not g.get("present", True):
                continue
            lab = _norm(g.get("label"))
            total += 1
            in_text = lab in text[i]["says"]
            in_rec = lab in rec[i]
            denied = lab in text[i]["denies"]
            if in_text and in_rec:
                counts["both"] += 1
            elif in_text:
                counts["text_only"] += 1
            elif in_rec:
                counts["record_only"] += 1
            elif denied:
                counts["contradicted"] += 1
                if len(contradicted_examples) < 20:
                    contradicted_examples.append(
                        {"row": i, "label": lab})
            else:
                counts["ungrounded"] += 1
                if len(ungrounded_examples) < 20:
                    ungrounded_examples.append(
                        {"row": i, "label": lab})

    # the identity a reader can check against
    assert sum(counts.values()) == total, (
        "the census must total the claims")
    supported = counts["both"] + counts["text_only"] \
        + counts["record_only"]
    return {
        "claims_censused": total,
        "counts": counts,
        "supported": supported,
        "grounded_share": (round(supported / total, 4)
                           if total else None),
        "ungrounded_share": (round(counts["ungrounded"] / total, 4)
                             if total else None),
        "ungrounded_examples": ungrounded_examples,
        "contradicted_examples": contradicted_examples,
        "verdict": _verdict(total, counts),
        "note": ("recall and the per-corruption breakdown come "
                 "from transcribe.score_extraction, which this "
                 "deliberately does not recompute; what it adds "
                 "is the claim that matches NO fact, which that "
                 "loop cannot see"),
    }


def _verdict(total: int, counts: Dict[str, int]) -> str:
    """NOTHING CLAIMED is its own answer, not a clean one.

    A tool that asserts nothing cannot be ungrounded, and
    reporting that as a perfect grounding score would be the worst
    kind of pass.
    """
    if not total:
        return ("NOTHING CLAIMED - no positive assertion was "
                "made, so grounding is not measured here. That is "
                "an absent measurement and not a clean one.")
    bad = counts["ungrounded"] + counts["contradicted"]
    if bad == 0:
        return ("CLEAN - every positive claim traces to the note "
                "or the structured record")
    return ("{} of {} claims are unsupported: {} appear in "
            "neither source, {} are DENIED by the prose".format(
                bad, total, counts["ungrounded"],
                counts["contradicted"]))


def source_followed(ledgers, claims) -> Dict[str, Any]:
    """On a PLANTED DISAGREEMENT, which source did they believe?

    This is the hard part of chart abstraction and the thing no
    single recall number can report: where the prose and the field
    conflict by construction, a tool must pick one, and which one
    it picks is a competence rather than an error.
    """
    followed = {"text": 0, "structured": 0, "neither": 0}
    rows = 0
    for ledger, got in zip(ledgers, claims):
        claimed = {_norm(g.get("label")): g for g in (got or [])}
        for e in ledger or []:
            if not e.get("disagrees_with_structured"):
                continue
            rows += 1
            g = claimed.get(_norm(e.get("label")))
            said = bool(g and g.get("present", True))
            if said == bool(e.get("asserts_present", True)):
                followed["text"] += 1
            elif said == _present_flag(e.get("structured_value")):
                followed["structured"] += 1
            else:
                followed["neither"] += 1
    return {"disagreements": rows, "followed": followed,
            "note": ("absent is not the same fact as zero - with "
                     "no planted disagreement there is nothing "
                     "here to measure")
            if not rows else None}


def consistency(claims_a, claims_b) -> Dict[str, Any]:
    """Two runs over records identical in everything recorded.

    Needs no ground truth at all, which is what makes it usable
    where nothing can be graded: a tool that answers differently
    about the same chart has a defect that can be demonstrated
    without knowing which answer was right.
    """
    if len(claims_a) != len(claims_b):
        raise ValueError("one claim list per row on both sides")
    agree = differ = 0
    examples = []
    for i, (a, b) in enumerate(zip(claims_a, claims_b)):
        sa = {_norm(g.get("label")) for g in (a or [])
              if g.get("present", True)}
        sb = {_norm(g.get("label")) for g in (b or [])
              if g.get("present", True)}
        if sa == sb:
            agree += 1
        else:
            differ += 1
            if len(examples) < 20:
                examples.append({"row": i,
                                 "only_first": sorted(sa - sb),
                                 "only_second": sorted(sb - sa)})
    n = agree + differ
    return {"rows": n, "agree": agree, "differ": differ,
            "agreement": round(agree / n, 4) if n else None,
            "examples": examples,
            "verdict": ("CONSISTENT" if not differ else
                        "{} of {} rows answered differently for "
                        "a record identical in everything "
                        "recorded".format(differ, n))}
