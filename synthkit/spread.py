"""HOW MUCH OF A COLUMN'S VARIATION SURVIVED - one definition, one
set of words, shared by everything that reports it.

WHY THIS MODULE EXISTS. The quantity is sd_generated / sd_source: a
RATIO, where 1.0 is the target. Three separate tools computed it and
all three framed it as a "share of the source spread KEPT", which
cannot exceed 1.0 by its own wording - and a real column came back at
1.03, meaning the synthetic version was WIDER than reality. Each tool
then:

  * drew the bar at min(value, 1.0), so 1.03 rendered FULL - visually
    identical to a perfect match, contradicting its own label;
  * counted a miss as "the ratio is lower", which scores
    over-dispersion as an improvement;
  * and in one case printed "loses {1.0 - ratio}", which reads as
    "loses -3%".

The defect was reported against ONE of the three, fixed there, and
the other two were left carrying it - which is this repository's most
repeated mistake: fixing the instance rather than sweeping for the
pattern. A shared definition is the only version of the fix that a
third consumer cannot silently diverge from.

1.0 IS THE TARGET, NOT THE CEILING. Too wide is as wrong as too
narrow, and the words say which.
"""
from typing import Optional

# Within this of 1.0 either way, a column counts as about right.
# Chosen rather than derived: a measurement whose spread is within a
# tenth of the source's is not what anyone is worried about, and the
# number is stated wherever it is applied so a reader can disagree
# with it.
BAND = 0.10

ABOUT_RIGHT = "about right"
TOO_NARROW = "too narrow"
TOO_WIDE = "too wide"


def ratio(sd_source, sd_generated) -> Optional[float]:
    """sd_generated / sd_source, or None when it cannot be formed.

    A source sd of zero means the column does not vary at all in the
    real data, so there is no variation to reproduce and no ratio to
    report - NOT a divide-by-zero to swallow and NOT a 0% to print."""
    try:
        s = float(sd_source)
        g = float(sd_generated)
    except (TypeError, ValueError):
        return None
    if s <= 0.0:
        return None
    return g / s


def is_miss(r: Optional[float], band: float = BAND) -> bool:
    """A miss in EITHER direction. An unformable ratio is not a miss -
    it is an absence, and counting it as a failure would blame the
    generator for a column that never varied."""
    if r is None:
        return False
    return abs(r - 1.0) > band


def verdict(r: Optional[float], band: float = BAND) -> str:
    """The words a reader sees. Returned rather than formatted so the
    caller can place them; identical in every consumer."""
    if r is None:
        return "no variation to reproduce"
    if abs(r - 1.0) <= band:
        return ABOUT_RIGHT
    return TOO_NARROW if r < 1.0 else TOO_WIDE


def bar_fraction(r: Optional[float], full: float = 1.5) -> float:
    """How much of a bar to fill, on a scale where 1.0 sits at
    two-thirds and the bar can show OVERSHOOT.

    The clamped version - min(ratio, 1.0) - is what made 1.03 draw as
    a complete bar. Here the axis runs past 1.0, so a column that is
    too wide is visibly past the target line rather than pinned at
    it."""
    if r is None:
        return 0.0
    return max(0.0, min(r, full)) / full


def target_fraction(full: float = 1.5) -> float:
    """Where to draw the target line, in the same units as
    bar_fraction - so the two cannot drift apart."""
    return 1.0 / full
