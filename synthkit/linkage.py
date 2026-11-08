"""Linkage risk against information the recipient already holds.

THE ELEMENT THIS CLOSES IS OUR OWN. 45 CFR 164.514(b)(1) asks
whether the information could be used "alone OR IN COMBINATION
WITH OTHER REASONABLY AVAILABLE INFORMATION" to identify an
individual. `synthkit.determination` names that element and
cited two adversaries for it - one that sees generated rows, one
that assumes the weights leaked - and NEITHER holds any external
data. So the package was asking the regulation's question and
answering a narrower one, which is the shape of fault this
project refuses everywhere else.

WHAT A LINKAGE ATTACK EVEN MEANS AGAINST SYNTHETIC DATA. The
classic attack matches a released record to an external record
on shared quasi-identifiers and reads off the sensitive field.
Against synthetic output no released row belongs to anybody, so
a match is not an identification by itself - and treating it as
one would overstate the risk as badly as ignoring it understates
it. The question that survives the translation is narrower and
answerable:

    An adversary holds quasi-identifiers for a person they can
    name. They select the released rows matching that person's
    quasi-identifier combination. Does that selection tell them
    this person's sensitive value more accurately than the same
    procedure applied to someone who was NEVER in the cohort?

That difference is the EXCESS, and only the excess belongs to
the release - the same argument `attack.attribute_disclosure`
makes, with the adversary's external knowledge made explicit
rather than assumed away. A release that merely reproduces
population structure helps equally for members and
non-members, and population structure is revealed by any sample
of the population.

AND IT NEEDS A POSITIVE CONTROL OR IT IS DECORATION. A
"generator" that republishes the members' own records must be
CAUGHT by this measurement, or a clean excess means nothing.
This project has recorded a reassuring 0.500 that was worth
nothing because its control could not catch a verbatim
republish; `linkage_risk` returns the control arm beside the
honest one, computed the same way, and the caller is expected to
read them together.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# A selection this small is not a selection: with one or two
# matching rows the estimate is dominated by which rows happened
# to be drawn, and this project's own k rule says a group that
# small should not drive a published number either.
MIN_SELECTION = 5

# How much of a VERBATIM REPUBLISH's advantage the honest
# release may reproduce before it is called a leak. Relative,
# because the absolute ceiling depends on the selection size and
# the banding - both of which the caller chooses.
LEAK_SHARE = 0.5

# AND THE CONTROL MUST ITSELF LEAK ENOUGH FOR THE TEST TO HAVE
# POWER. If a VERBATIM REPUBLISH barely beats the non-member arm
# under the caller's bands, the arrangement cannot detect a leak
# at all - and reporting CLEAN from it would be a pass from a
# test that could not fail. Measured over six seeds on a cohort
# with ~10-row selections a republish reaches 0.026-0.104 sd, so
# this floor sits just under the observed minimum; a fixture
# with ~15-row selections produced 0.0067 and is correctly
# UNREADABLE.
CONTROL_MIN = 0.02


def _band(value, edges):
    """Which band a value falls in, or None."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    for i, e in enumerate(edges):
        if v < e:
            return i
    return len(edges)


def quasi_key(row: Dict[str, Any], quasi: Sequence[str],
              bands: Optional[Dict[str, Sequence[float]]] = None):
    """The combination an adversary matches on.

    NUMERIC QUASI-IDENTIFIERS ARE BANDED, because an adversary
    holding an exact age is a different adversary from one
    holding a birth year, and matching on an exact float would
    make almost every selection empty - which would read as
    SAFETY. An empty selection is a measurement that did not
    happen, not a release that protected anybody.
    """
    bands = bands or {}
    out = []
    for q in quasi:
        v = row.get(q)
        if q in bands:
            out.append(_band(v, bands[q]))
        else:
            out.append(None if v is None else str(v))
    return tuple(out)


def _numeric(rows, col):
    vals = []
    for r in rows:
        try:
            v = float(r.get(col))
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            vals.append(v)
    return np.asarray(vals, dtype=float)


def _guess(selection, sensitive):
    """What the adversary concludes from their selection."""
    v = _numeric(selection, sensitive)
    return float(np.mean(v)) if v.size else None


def linkage_risk(members: List[Dict], nonmembers: List[Dict],
                 released: List[Dict], quasi: Sequence[str],
                 sensitive: str,
                 bands: Optional[Dict] = None) -> Dict[str, Any]:
    """How much the release helps someone who already knows
    `quasi` about a person they can name.

    Returns the error an adversary makes about MEMBERS and about
    NON-MEMBERS, and the EXCESS between them. Lower error on
    members than on non-members is what the release contributed;
    equal error is population structure that any sample would
    reveal.

    The errors are in units of the sensitive column's own
    spread, so a number is comparable across columns and across
    cohorts. Reporting raw error would make a wide column look
    risky and a narrow one safe for reasons that have nothing to
    do with disclosure.
    """
    bands = bands or {}
    sel: Dict[Any, List[Dict]] = {}
    for r in released:
        sel.setdefault(quasi_key(r, quasi, bands), []).append(r)

    spread = _numeric(members + nonmembers, sensitive)
    sd = float(np.std(spread)) if spread.size else 0.0
    if sd <= 0:
        return {"usable": False,
                "why": "the sensitive column has no spread, so "
                       "an error in units of its spread is "
                       "undefined - and a column everybody "
                       "shares cannot single anybody out"}

    def arm(rows):
        errs, matched, thin = [], 0, 0
        for r in rows:
            try:
                truth = float(r.get(sensitive))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(truth):
                continue
            group = sel.get(quasi_key(r, quasi, bands), [])
            if len(group) < MIN_SELECTION:
                thin += 1
                continue
            guess = _guess(group, sensitive)
            if guess is None:
                thin += 1
                continue
            matched += 1
            errs.append(abs(guess - truth) / sd)
        return {
            "n_scored": matched,
            "n_thin": thin,
            "error_in_sd": (float(np.mean(errs)) if errs
                            else None),
        }

    mem, non = arm(members), arm(nonmembers)
    excess = None
    if mem["error_in_sd"] is not None \
            and non["error_in_sd"] is not None:
        # POSITIVE excess means the adversary does BETTER on
        # members - that is what the release contributed.
        excess = non["error_in_sd"] - mem["error_in_sd"]
    return {
        "usable": True,
        "quasi_identifiers": list(quasi),
        "sensitive": sensitive,
        "banded": sorted(bands),
        "members": mem, "nonmembers": non,
        "excess_in_sd": excess,
        "selections": len(sel),
        "min_selection": MIN_SELECTION,
        "note": "the error is in units of the sensitive "
                "column's own spread, so it compares across "
                "columns. POSITIVE excess means the adversary "
                "does better on people who WERE in the cohort, "
                "which is what the release contributed; near "
                "zero means they learned population structure "
                "that any sample of the population reveals. A "
                "selection under {} rows is skipped rather than "
                "scored - with one or two matching rows the "
                "estimate is whichever rows were drawn, and an "
                "empty selection is a measurement that did not "
                "happen, not a release that protected "
                "anybody.".format(MIN_SELECTION),
    }


def with_control(members: List[Dict], nonmembers: List[Dict],
                 released: List[Dict], quasi: Sequence[str],
                 sensitive: str,
                 bands: Optional[Dict] = None) -> Dict[str, Any]:
    """The honest arm and its POSITIVE CONTROL, together.

    The control hands the members' OWN RECORDS back as the
    "release". That generator leaks by construction, so this
    measurement must show a clearly larger excess for it - and
    if it does not, the honest arm's clean number is worth
    nothing. A control that cannot catch a verbatim republish is
    a check that cannot fail, which this project has been burned
    by on this exact surface.
    """
    honest = linkage_risk(members, nonmembers, released, quasi,
                          sensitive, bands)
    leaky = linkage_risk(members, nonmembers, members, quasi,
                         sensitive, bands)
    # CAUGHT REQUIRES THE CONTROL'S OWN EXCESS TO BE POSITIVE,
    # not merely larger than the honest arm's.
    #
    # The first cut tested only `control > honest`, so with a
    # negative honest excess a NEGATIVE control excess still
    # read as caught - and the output then said "the control
    # CAUGHT the republish but only by -0.0310 sd", a verdict
    # contradicting the number in its own sentence. A verbatim
    # republish must HELP the adversary; a republish that makes
    # them worse off has not been caught, it has shown the
    # arrangement cannot see anything.
    caught = None
    if honest.get("excess_in_sd") is not None \
            and leaky.get("excess_in_sd") is not None:
        caught = (leaky["excess_in_sd"] > honest["excess_in_sd"]
                  and leaky["excess_in_sd"] > 0.0)
    # THE BAR IS RELATIVE TO THE CONTROL, NOT A CONSTANT -
    # BECAUSE A CONSTANT WAS ABOVE ITS OWN CEILING.
    #
    # The first cut called it a leak above 0.10 sd of excess.
    # Measured over six seeds, a VERBATIM REPUBLISH - the worst
    # generator there is - produces an excess of only 0.026 to
    # 0.104, because a selection of at least five rows averages
    # the target's own value with four other people's. So a bar
    # at 0.10 would have missed most verbatim republishes: a bar
    # above its own ceiling, which is the exact error this
    # project found in a campaign tier demanding 0.700 over a
    # 0.698 ceiling and wrote a rule about.
    #
    # The control measures the ceiling on the same cohort with
    # the same quasi-identifiers, so the honest arm is judged
    # against it as a FRACTION. That is self-calibrating: a
    # coarser banding lowers both arms together.
    # THREE THINGS CAN GO WRONG AND THEY ARE NOT ONE THING.
    #
    #   the control did not beat the non-member arm  -> no
    #       control at all
    #   the control beat it but only barely           -> no
    #       POWER: a republish is undetectable here, so CLEAN
    #       would be a pass from a test that cannot fail
    #   the honest excess is NEGATIVE                 -> the
    #       adversary did WORSE on members; there is no leak to
    #       express as a share, and -818% is not a readout
    #
    # Collapsing them would be the blocked-versus-unchanged
    # fault; each takes a different action - fix the control,
    # use finer bands, or report clean.
    share = None
    powered = (caught is True
               and leaky["excess_in_sd"] is not None
               and leaky["excess_in_sd"] >= CONTROL_MIN)
    if powered and honest["excess_in_sd"] > 0:
        share = honest["excess_in_sd"] / leaky["excess_in_sd"]
    if caught is not True:
        verdict, why = "UNREADABLE", (
            "the CONTROL did not beat the non-member arm - a "
            "verbatim republish was not caught, so the honest "
            "number says nothing")
    elif not powered:
        verdict, why = "UNREADABLE", (
            "the control CAUGHT the republish but only by "
            "{:.4f} sd, under the {:.2f} floor - a republish is "
            "barely detectable under these bands, so this "
            "arrangement has no POWER and CLEAN would be a pass "
            "from a test that cannot fail. Use finer bands or a "
            "smaller minimum selection.".format(
                leaky["excess_in_sd"] or 0.0, CONTROL_MIN))
    elif honest["excess_in_sd"] <= 0:
        verdict, why = "CLEAN", (
            "the adversary did no better on members than on "
            "non-members, so there is no advantage to express "
            "as a share of the republish")
    elif share >= LEAK_SHARE:
        verdict, why = "LEAKS", (
            "the release reproduces {:.0%} of what a verbatim "
            "republish achieves on this cohort, at or above the "
            "{:.0%} bar".format(share, LEAK_SHARE))
    else:
        verdict, why = "CLEAN", (
            "the release reproduces {:.0%} of what a verbatim "
            "republish achieves on this cohort, under the {:.0%} "
            "bar".format(share, LEAK_SHARE))
    return {
        "honest": honest, "control": leaky,
        "control_caught": caught,
        "share_of_control": share,
        "leak_share_bar": LEAK_SHARE,
        "control_min": CONTROL_MIN,
        "powered": bool(powered),
        "verdict": verdict,
        "why": why,
        "note": "UNREADABLE means the CONTROL failed to show a "
                "larger excess than the honest arm - a verbatim "
                "republish was not caught, so the honest number "
                "says nothing and must not be reported as a "
                "pass. This project has recorded a reassuring "
                "0.500 that was worth exactly nothing for that "
                "reason. The verdict is a FRACTION of what the "
                "republish achieves on this same cohort, not an "
                "absolute excess: a republish reaches only "
                "0.026-0.104 sd here, because a selection of at "
                "least {} rows averages the target's own value "
                "with other people's, so an absolute bar of "
                "0.10 would have been above its own "
                "ceiling.".format(MIN_SELECTION),
    }
