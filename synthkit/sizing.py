"""WHAT THIS DATASET CAN HONESTLY GIVE AT A REQUESTED SIZE.

The ask, from the product walkthrough: a lever for output size, a
percentage of fidelity attached to each setting, and a floor below
which we do not hand the file over at all.

THE SHAPE OF THE ANSWER IS NOT THE SHAPE THAT WAS EXPECTED, and
saying so is the point of this module. The expectation was that
fidelity falls as the requested size grows - "I want 50,000 and then
at that point you have like 20% fidelity". Measured on the real
extract, 5x expansion HELD: direction 92.3% against 1x's 93.9%,
coverage 42/42, centre improved because more rows steady the bins.
The recorded conclusion is that contraction, not expansion, is where
patterns thin.

So two different things vary with size, in OPPOSITE directions, and
collapsing them into one number would be a lie whichever way it
leaned:

  PATTERN COVERAGE rises with size. A published pattern held by a
  small share of rows needs room to appear. Ask for 50 rows and the
  rare half of the vocabulary simply will not be drawn; ask for
  500,000 and everything appears in proportion. This is the number
  that behaves like "fidelity" in the ordinary sense, and it goes
  UP.

  EVIDENCE PER ROW falls with size. Ten synthetic patients per real
  patient does not carry ten patients' worth of independent
  evidence; it carries one patient's, repeated. Nothing is unfaithful
  about those rows - they are drawn from the same learned
  distribution - but a reader who treats 50,000 rows as 50,000
  independent observations is wrong, and the interface should say so
  before they do.

Both are reported. The FLOOR gates on coverage, because that is the
one where the file genuinely fails to contain what was promised.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# A pattern needs to appear enough times to be usable, not merely to
# appear once. Five is the smallest count at which a rate is worth
# quoting at all, and it is the same spirit as the k rule: one
# occurrence is an anecdote.
MIN_OCCURRENCES = 5

# "If we can get up to 70%, only those are the ones we'll give."
DEFAULT_FLOOR = 0.70


def pattern_coverage(probs: List[float], n_rows: int,
                     min_occurrences: int = MIN_OCCURRENCES) -> float:
    """Share of published patterns expected to appear usably often.

    A pattern on p of rows is expected n*p times in n rows. Below
    `min_occurrences` it is present in the vocabulary and absent from
    the file, which is the gap between what was promised and what was
    delivered."""
    live = [p for p in probs if p and p > 0]
    if not live or n_rows <= 0:
        return 0.0
    ok = sum(1 for p in live if n_rows * p >= min_occurrences)
    return ok / float(len(live))


def evidence_ratio(n_source_entities: int,
                   n_requested_entities: int) -> float:
    """Independent real evidence per synthetic entity, capped at 1.

    At or below the source size every synthetic entity has a real
    one behind it. Above it, the same learned distribution is being
    drawn from more times - which is legitimate, and is not more
    information."""
    if n_requested_entities <= 0 or n_source_entities <= 0:
        return 0.0
    return min(1.0, n_source_entities / float(n_requested_entities))


def advise(probs: List[float], n_source_entities: int,
           rows_per_entity: float, requested_entities: int,
           floor: float = DEFAULT_FLOOR) -> Dict[str, Any]:
    """The verdict for one requested size."""
    req = int(max(0, requested_entities))
    rows = int(round(req * max(rows_per_entity, 0.0)))
    cov = pattern_coverage(probs, rows)
    ev = evidence_ratio(n_source_entities, req)
    offered = cov >= floor
    if offered:
        why = ("{:.0%} of the published patterns have room to appear "
               "at this size".format(cov))
    else:
        why = ("only {:.0%} of the published patterns would appear "
               "often enough to be usable at this size, below the "
               "{:.0%} floor - the file would be missing what it "
               "claims to contain".format(cov, floor))
    return {"requested_entities": req, "requested_rows": rows,
            "pattern_coverage": round(cov, 4),
            "evidence_ratio": round(ev, 4),
            "offered": offered, "reason": why,
            "floor": floor}


def ladder(probs: List[float], n_source_entities: int,
           rows_per_entity: float,
           multiples: Optional[List[float]] = None,
           floor: float = DEFAULT_FLOOR) -> Dict[str, Any]:
    """The whole lever: every offered size, bounded by this dataset.

    This is the "curate what I can and can't do in terms of levers -
    here's low, here's high" step. The bounds are a property of THIS
    dataset, computed after its patterns are known, not a global
    setting."""
    muls = multiples or [0.25, 0.5, 1, 2, 5, 10, 20]
    rungs = [advise(probs, n_source_entities, rows_per_entity,
                    int(round(n_source_entities * m)), floor)
             for m in muls]
    for r, m in zip(rungs, muls):
        r["multiple"] = m
    ok = [r for r in rungs if r["offered"]]
    return {"rungs": rungs,
            "lowest_offered": (ok[0]["requested_entities"]
                               if ok else None),
            "highest_offered": (ok[-1]["requested_entities"]
                                if ok else None),
            "any_offered": bool(ok),
            "floor": floor,
            "note": ("These two move in OPPOSITE directions: "
                     "pattern coverage rises with size, because "
                     "rare patterns need room to appear, while "
                     "evidence per entity falls, because more "
                     "synthetic entities are drawn from the same "
                     "real ones. The floor gates on coverage - "
                     "that is where the file genuinely lacks what "
                     "it claims. The evidence figure is reported "
                     "so a larger file is not mistaken for more "
                     "information.")}
