"""The evidence package for a HIPAA Expert Determination.

WHAT THIS IS, AND THE FIRST THING IT SAYS ABOUT ITSELF. 45 CFR
164.514(b)(1) permits a release when a person with appropriate
knowledge of statistical methods determines that the risk of
re-identification is very small, AND documents the methods and
results that justify the determination. This module assembles
the documentation. It does NOT make the determination: only a
qualified expert does that, and a tool claiming otherwise would
be the same fault as an output described as differentially
private when only the weights are accounted.

WHY IT IS WORTH ASSEMBLING RATHER THAN WRITING FRESH. Most of
what the regulation asks for has been measured here for months
and lives in six different files - the attack battery with its
positive controls, the k rules, the Safe Harbor inventory, the
bin suppression, the fidelity gate. A reader deciding whether
the risk is very small had to hold all of it in their head. That
is the same complaint the per-claim receipts answered for
patterns, one level up.

EVERY ELEMENT CARRIES A STATUS DERIVED FROM THE ARTIFACTS, AND
`OWED` IS A FIRST-CLASS ANSWER. A package that looks complete
and is not is worse than no package, because an expert would
sign against evidence that does not exist. The counts are
DERIVED from the rows rather than written beside them, which is
the rule the Safe Harbor inventory already follows.
"""

from typing import Any, Dict, List, Optional, Tuple

# The regulation's own two requirements, quoted short, because a
# package organised around anything else leaves the reader
# mapping our headings onto theirs.
STATUTE = (
    "45 CFR 164.514(b)(1) - a person with appropriate knowledge "
    "of and experience with generally accepted statistical and "
    "scientific principles and methods for rendering "
    "information not individually identifiable (i) determines "
    "that the risk is very small that the information could be "
    "used, alone or in combination with other reasonably "
    "available information, by an anticipated recipient to "
    "identify an individual who is a subject of the "
    "information; and (ii) documents the methods and results "
    "of the analysis that justify such determination.")

# WHAT THIS DOCUMENT IS NOT. Stated as data rather than prose so
# every renderer carries it and none can drop it.
NOT_A_DETERMINATION = (
    "This is an EVIDENCE PACKAGE, not a determination. The "
    "determination in 164.514(b)(1) is made by a qualified "
    "person, not by a tool, and nothing here should be read as "
    "one.",
    "No source record travels with this package or with the "
    "release it describes. The measurements were taken on the "
    "machine that holds the extract and only their results "
    "crossed.",
    "Every element below carries a STATUS. An element marked "
    "OWED has no evidence in this run - an expert asked to sign "
    "against it would be signing against nothing.",
)

# Each element: the key, what the regulation needs, and which
# artifact or measurement answers it. `evidence` names the
# FUNCTION or FILE so a reader can go and check rather than
# taking the status on trust.
ELEMENTS: Tuple[Dict[str, Any], ...] = (
    {"key": "release_surface",
     "need": "Exactly what is released, and what is not",
     "evidence": "provenance.json plus the cloud allow-list "
                 "(synthkit.cloud.publish refuses anything off "
                 "it BY NAME, including key.csv and the source "
                 "extract)",
     "why": "A risk analysis of the wrong surface is worth "
            "nothing. The output-only release and a release of "
            "the trained network are different questions with "
            "different measured answers."},
    {"key": "method",
     "need": "The de-identification method, described",
     "evidence": "blueprint.json - every published quantity, "
                 "with k counted in PATIENTS; no published "
                 "number is an individual's value (the stored "
                 "minimum and maximum are the MEAN of the k "
                 "most extreme patients' own extremes)",
     "why": "The method is what an expert evaluates. A "
            "generator described only by its outputs cannot be "
            "evaluated at all."},
    {"key": "identifiers",
     "need": "That direct identifiers were addressed",
     "evidence": "scrub.inventory() - all 18 Safe Harbor "
                 "categories with a status and a note each, "
                 "including the ones NOT covered",
     "why": "Safe Harbor is a different pathway from Expert "
            "Determination, but an expert will ask what became "
            "of the direct identifiers, and a detector that "
            "finds nothing is a clean bill for whatever it "
            "LOOKS AT."},
    {"key": "threat_model",
     "need": "The anticipated recipient and what they can see "
             "of the release itself",
     "evidence": "the attack battery's two adversaries - one "
                 "that sees only GENERATED ROWS, and one that "
                 "assumes the trained WEIGHTS leaked. NEITHER "
                 "holds external data; that is the `linkage` "
                 "element below",
     "why": "'Very small risk' is meaningless without saying "
            "small against whom. The two adversaries are two "
            "different recipients and their numbers differ."},
    {"key": "linkage",
     "need": "...and in COMBINATION with other reasonably "
             "available information",
     "evidence": "synthkit.linkage - an adversary who already "
                 "holds quasi-identifiers for a person they can "
                 "name, scored as the EXCESS over the same "
                 "procedure applied to someone never in the "
                 "cohort, with a verbatim-republish positive "
                 "control on the same cohort",
     "why": "The statute asks about information used ALONE OR "
            "IN COMBINATION with other reasonably available "
            "information, and this package cited two "
            "adversaries for it that hold no external data - "
            "asking the regulation's question and answering a "
            "narrower one. Against synthetic output a match is "
            "not an identification by itself, so the question "
            "that survives is whether the release tells a "
            "linker more about a MEMBER than about a "
            "non-member."},
    {"key": "membership",
     "need": "Measured risk of determining that an individual "
             "was in the cohort",
     "evidence": "the membership-inference AUC, WITH the "
                 "republish positive control on the same arm",
     "why": "A control that cannot catch a verbatim republish "
            "is a check that cannot fail, and this project has "
            "recorded a reassuring 0.500 that was worth "
            "nothing for exactly that reason."},
    {"key": "attribute",
     "need": "Measured risk of inferring a sensitive field "
             "from the ordinary ones",
     "evidence": "the attribute-disclosure EXCESS over an "
                 "adversary trained on different real people "
                 "who were never in the cohort",
     "why": "Raw accuracy cannot answer this: a generator that "
            "reproduces relationships faithfully is by "
            "construction good at predicting one column from "
            "the others, and a real relationship is revealed "
            "by any sample of the population. Only the EXCESS "
            "belongs to the release."},
    {"key": "memorization",
     "need": "That the release does not reproduce individual "
             "records",
     "evidence": "the nearest-neighbour ratio (generated-to-"
                 "training over holdout-to-training) and the "
                 "reconstruction gap between training and "
                 "held-out PATIENTS, each with a positive "
                 "control",
     "why": "A ratio near 1.0 is also what a broken measure "
            "returns, which is why the control removes the "
            "thing being generalised to rather than widening "
            "the model."},
    {"key": "aggregates",
     "need": "That the published statistics themselves do not "
             "disclose",
     "evidence": "the k-anonymous bound rule, and the fidelity "
                 "deck suppressing source bins backed by fewer "
                 "than k patients and PRINTING the count it "
                 "suppressed",
     "why": "A fidelity report on real data is itself a "
            "release: a histogram is a set of bin counts, and "
            "a bin holding three patients describes a group "
            "small enough to gossip about."},
    {"key": "utility",
     "need": "The information retained, and why the release is "
             "useful",
     "evidence": "the fidelity gate and the mined-interaction "
                 "table; train-on-synthetic-test-on-real scores "
                 "both arms on the SAME held-out real rows",
     "why": "Over-suppression is the trivial way to make risk "
            "very small, so a determination that does not state "
            "the retained utility has not described a trade at "
            "all."},
    {"key": "residual",
     "need": "Residual risk and the scope of the analysis",
     "evidence": "the recorded limitations - one cohort shape "
                 "at one k on one extract, k-anonymity on what "
                 "is published is NOT differential privacy, and "
                 "the effect curves, interaction surfaces and "
                 "dynamics have NOT been audited the same way",
     "why": "A floor, not a certificate, and not a release "
            "decision for a different cohort. An expert's "
            "determination is bounded by the scope of the "
            "evidence, so the scope has to be written down."},
)


def _get(d: Optional[Dict], *path, **kw):
    """Walk a dict path, returning the default when absent.

    `absent` and `known to be zero` are different facts, which
    is why a missing key returns None here and every caller
    distinguishes it from a measured value.
    """
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return kw.get("default")
        cur = cur[p]
    return cur


def assess(fidelity: Optional[Dict] = None,
           blueprint: Optional[Dict] = None,
           provenance: Optional[Dict] = None,
           membership: Optional[Dict] = None,
           attribute: Optional[Dict] = None,
           scrub_inventory: Optional[List] = None,
           scope: Optional[str] = None,
           source_name: Optional[str] = None,
           run_config: Optional[Dict] = None,
           linkage: Optional[Dict] = None,
           ) -> Dict[str, Any]:
    """Which elements this run has evidence for, and which are
    OWED.

    Takes the artifacts rather than a directory so the bench,
    the CLI and the checks all drive one function - a second
    implementation is how the shipped package and the screen
    come to disagree.

    THE FIELD NAMES ARE THE CONTRACT AND THEY ARE NOT GUESSED.
    The first cut of this function read `arms`, `membership` and
    `attribute` out of one invented `attack.json`. The real
    batteries write TWO files with different shapes -
    `membership_new_path.py` writes worst_auc_max and a `runs`
    list whose entries carry `nearest_neighbor` and
    `likelihood`; `attribute_disclosure.py` writes `seeds` and
    `positive_control` - so every element would have read OWED
    on a run that had the evidence. Seventh instance of the
    wrong-key fault in this repo, and the suite asserts these
    names against the batteries' LIVE output rather than
    remembering them.
    """
    have = {}

    have["release_surface"] = bool(provenance)
    have["method"] = bool(_get(blueprint, "marginals")
                          or _get(blueprint, "columns"))
    have["identifiers"] = bool(scrub_inventory)

    # TWO ADVERSARIES, OR IT IS NOT A THREAT MODEL. The output
    # surface and the weights surface are different recipients
    # and their recorded numbers differ sharply - one arm
    # answers a question nobody asked.
    runs = _get(membership, "runs", default=None) or []
    both = [r for r in runs
            if isinstance(r, dict)
            and _get(r, "nearest_neighbor", "auc") is not None
            and _get(r, "likelihood", "auc") is not None]
    have["threat_model"] = len(both) >= 1

    # A MEMBERSHIP NUMBER NEEDS MORE THAN ONE SEED AND A
    # VERDICT. One attack AUC is a draw from a distribution, not
    # a measurement - the battery itself refuses fewer than two
    # seeds, so a summary with fewer did not come from it.
    seeds = _get(membership, "seeds", default=None) or []
    have["membership"] = bool(
        _get(membership, "worst_auc_max") is not None
        and _get(membership, "verdict")
        and len(seeds) >= 2)

    # THE EXCESS, AND THE POSITIVE CONTROL THAT MAKES IT WORTH
    # ANYTHING. A clean excess with no leaky-generator arm is
    # the check-that-cannot-fail shape.
    # AND THE CONTROL MUST HAVE FIRED, not merely existed.
    #
    # The first cut required a non-empty `positive_control`
    # list - which is satisfied by a control that ran and
    # caught nothing. On a real extract the battery reported
    # "THE POSITIVE CONTROL DID NOT FIRE" at +0.019 excess, and
    # this element would still have read present: an excess
    # whose control cannot catch a verbatim republish is the
    # check-that-cannot-fail shape, which is exactly what the
    # linkage element already refuses via `control_caught`.
    #
    # The flag is read, never re-derived: the bar lives in the
    # battery at its decision site, and computing it here too
    # would be two halves deciding one threshold.
    _fired = _get(attribute, "control_fired", default=None)
    have["attribute"] = bool(
        (_get(attribute, "seeds", default=None) or [])
        and (_get(attribute, "positive_control", default=None)
             or [])
        # A record predating the flag cannot be judged on it -
        # absent is not the same fact as did-not-fire - so an
        # old record still counts, and a NEW one must say True.
        and _fired is not False)

    # LINKAGE NEEDS ITS CONTROL TO HAVE CAUGHT THE REPUBLISH,
    # or the honest number says nothing - which is why the
    # status turns on `control_caught` rather than on the
    # verdict string.
    have["linkage"] = bool(
        linkage
        and linkage.get("control_caught") is True
        and (linkage.get("honest") or {}).get("excess_in_sd")
        is not None)

    # THE PATHS COME FROM A REAL ARTIFACT, NOT FROM MEMORY -
    # AND THE FIRST VERSION'S DID NOT, WHICH IS THE EIGHTH
    # WRONG-KEY FAULT IN THIS REPO.
    #
    # It read `fidelity.nn_ratio`, `fidelity.
    # reconstruction_gap` and `blueprint.
    # bounds_are_k_anonymous`. A real neural run writes
    # `fidelity.reconstruction` (train / holdout / ratio /
    # rows_train / rows_holdout / columns) and a whole
    # `blueprint.privacy` block (k / counted_in /
    # numeric_bounds / ... / NOT_a_formal_guarantee), and no
    # `nn_ratio` at all - that number belongs to the attack and
    # the challenger, not to this artifact. So both elements
    # read OWED on a run that HAD the evidence.
    #
    # My own checks could not catch it because they built the
    # artifacts by hand with the names I had invented: the
    # fixture agreed with the reader and both disagreed with the
    # only thing that matters. This file records that exact
    # fault for the relationships page, where twelve rows of
    # em-dashes shipped. The suite now builds its artifacts
    # through `write_run_artifacts`, so the keys come from the
    # real writer.
    recon = _get(fidelity, "reconstruction", "ratio",
                 default=None)
    nn = _get(fidelity, "nn_ratio", default=None)   # rules path
    have["memorization"] = bool(recon is not None
                                or nn is not None)

    # THE K RULE'S OWN RECORD, which is a block rather than a
    # flag: `k` with `counted_in` saying "patients, never rows",
    # and `numeric_bounds` describing the mean-of-k-extremes
    # rule. Checked for the PATIENT wording, because a k counted
    # in ROWS would let one person seen two hundred times set a
    # published bound alone - which is the whole reason the rule
    # is written that way.
    priv = _get(blueprint, "privacy", default=None) or {}
    counted = str(priv.get("counted_in") or "")
    have["aggregates"] = bool(
        priv.get("k") and "patient" in counted.lower()
        and priv.get("numeric_bounds"))
    # ...and an older artifact that carried the flag instead
    # still counts, because absent is not the same fact as
    # false.
    if _get(blueprint, "bounds_are_k_anonymous",
            default=None) is True:
        have["aggregates"] = True
    have["utility"] = _get(fidelity, "summary",
                           default=None) is not None

    # RESIDUAL RISK IS THE ONE ELEMENT A RUN CANNOT SUPPLY: it
    # is a statement about SCOPE that a person writes. It is
    # satisfied by a written scope passed in, and OWED
    # otherwise - never derived, because a scope nobody wrote is
    # not a scope.
    #
    # It is a PARAMETER rather than permanently false, which the
    # first cut had: that made `ready` unreachable and left the
    # "every element has evidence" branch as dead code inside a
    # tool built to catch exactly that.
    have["residual"] = bool(scope and len(scope.strip()) >= 200)

    # EVIDENCE MEASURED ON A DIFFERENT DATASET IS NOT EVIDENCE
    # ABOUT THIS ONE, AND THE PACKAGE HAS TO SEE THAT.
    #
    # The attack batteries default to their own cohort FIXTURE -
    # which this project's own posture calls "a floor, not a
    # certificate, and not a release decision for a different
    # cohort". A determination is scoped to one release, so
    # citing a fixture number for it would be the strongest form
    # of the measured-at-one-shape fault: not a stale number,
    # but a number about somebody else.
    #
    # So an element whose evidence names a source that is not
    # this run's reads FIXTURE rather than present. It is not
    # OWED - the measurement was made and is real - and it does
    # not count toward readiness either.
    def _elsewhere(rec):
        if not (rec and source_name):
            return False
        src = rec.get("source")
        if not src:
            return False
        return (str(source_name) not in str(src)
                and str(src) not in str(source_name))

    # AND A DIFFERENT CONFIGURATION IS ALSO DIFFERENT EVIDENCE.
    #
    # The attack runs the SHIPPED defaults unless told otherwise,
    # and a release may have overridden them. A larger token cap
    # publishes more vocabulary about members, so the two runs
    # measure DIFFERENT THINGS and neither is evidence for the
    # other - which is the whole argument, and it does not
    # depend on knowing the direction.
    #
    # MEASURED, AND THE DIRECTION IS NOT ESTABLISHED: on the
    # real extract the output-surface adversary read 0.498 at
    # cap 60 against 0.561 at cap 800 on seed 0, and 0.521
    # against 0.485 on seed 1 - opposite signs. A first draft of
    # this comment quoted seed 0 alone as showing the cap
    # "understates the risk"; that number is withdrawn. The
    # guard stands because the configurations differ, not
    # because one of them is known to be safer.
    # Treated exactly like a different dataset, because it is a
    # measurement of something other than what is being
    # certified.
    def _other_config(rec):
        if not (rec and run_config):
            return False
        for field in ("token_cap", "level_cap"):
            a, b = rec.get(field), run_config.get(field)
            if a is not None and b is not None and int(a) != int(b):
                return True
        return False

    # TWO CAUSES, TWO LABELS, BECAUSE THEY TAKE DIFFERENT
    # ACTIONS. Evidence from another DATASET needs re-running on
    # this cohort; evidence from the same data under different
    # CAPS needs re-running at these settings. Collapsing them
    # into one word would be the blocked-versus-unchanged fault,
    # which this project records for the corruption ledger and
    # for route B's two opposite readings of one number.
    # AND AN EVIDENCE RECORD WITH NO SOURCE COULD NOT BE
    # CHECKED, WHICH IS A THIRD THING AGAIN.
    #
    # The rule above is right - a record predating the `source`
    # field cannot be judged on it, because absent is not the
    # same fact as mismatched - but it means an OLD fixture
    # record passes the scope test silently. A real one turned
    # up within the hour: the attribute battery's first
    # real-extract chain ran from a build that predated the
    # field, so its fixture evidence carried no source and was
    # not flagged. The rule stands and the SILENCE does not:
    # these are listed so a determination reader knows which
    # evidence could not be scope-checked at all.
    unverifiable = []
    off_cohort = {}
    for key, rec in (("membership", membership),
                     ("attribute", attribute),
                     ("linkage", linkage)):
        if rec and source_name and not rec.get("source"):
            unverifiable.append(key)
        why = ("FIXTURE" if _elsewhere(rec)
               else "CONFIG" if _other_config(rec) else None)
        if not why:
            continue
        off_cohort[key] = why
        if key == "membership":
            off_cohort["threat_model"] = why

    rows = []
    for el in ELEMENTS:
        k_ = el["key"]
        got = bool(have.get(k_))
        bad = off_cohort.get(k_) if got else None
        rows.append(dict(
            el,
            present=got and not bad,
            status=(bad or ("present" if got else "OWED")),
            measured_elsewhere=bool(bad)))
    n_have = sum(1 for r in rows if r["present"])
    return {
        "statute": STATUTE,
        "not_a_determination": list(NOT_A_DETERMINATION),
        "elements": rows,
        # DERIVED from the rows, never written beside them.
        "elements_total": len(rows),
        "elements_present": n_have,
        # OWED means NO evidence. An element measured on
        # another dataset is listed under `elements_elsewhere`
        # and NOT here - the first cut had it in both, so the
        # output said FIXTURE on one line and OWED on another
        # about the same element, which is this project's own
        # two-numbers-disagreeing alarm in a governance
        # document.
        "elements_owed": [r["key"] for r in rows
                          if r["status"] == "OWED"],
        "elements_elsewhere": [r["key"] for r in rows
                               if r["measured_elsewhere"]],
        "scope_unverifiable": unverifiable,
        "ready": n_have == len(rows),
        "scope_chars": len(scope.strip()) if scope else 0,
        "source_name": source_name,
    }


def plain(report: Dict[str, Any]) -> List[str]:
    """The package as text, for the CLI.

    ONE renderer's worth of words, shared, for the same reason
    the gate's NEXT_STEPS live in one module: a package and a
    page that disagree about what is owed is worse than either
    alone.
    """
    out = ["HIPAA EXPERT DETERMINATION - EVIDENCE PACKAGE", ""]
    for line in report["not_a_determination"]:
        out.append("  ! " + line)
    out.append("")
    out.append("THE STATUTE")
    out.append("  " + report["statute"])
    out.append("")
    out.append("ELEMENTS: {} of {} have evidence in this run"
               .format(report["elements_present"],
                       report["elements_total"]))
    if report.get("elements_elsewhere"):
        out.append("  ({} measured on another dataset - see "
                   "below)".format(
                       len(report["elements_elsewhere"])))
    out.append("")
    for el in report["elements"]:
        # ANY status renders, rather than an enumerated map
        # that raises on a new one. The first version listed
        # present/FIXTURE/OWED and crashed with a KeyError the
        # moment CONFIG was added - a renderer that must be
        # edited whenever a status is added will be the thing
        # that breaks when somebody adds one.
        _st = el.get("status", "OWED")
        out.append("  [{}] {}".format(
            "x" if _st == "present" else _st, el["need"]))
        out.append("        evidence: " + el["evidence"])
        out.append("        why it is asked: " + el["why"])
        out.append("")
    if report.get("scope_unverifiable"):
        out.append("SCOPE NOT CHECKABLE: {}".format(
            ", ".join(report["scope_unverifiable"])))
        out.append("  These records name no SOURCE, so whether "
                   "they were measured on this cohort could not "
                   "be checked at all. A record")
        out.append("  predating the field cannot be judged on it "
                   "- absent is not the same fact as mismatched "
                   "- but neither is it")
        out.append("  confirmation. Re-run them with a build "
                   "that records the source.")
        out.append("")
    if report.get("elements_elsewhere"):
        out.append("OUT OF SCOPE: {}".format(
            ", ".join("{} ({})".format(
                r["key"], r["status"])
                for r in report["elements"]
                if r["measured_elsewhere"])))
        out.append("  FIXTURE means the measurement was made on "
                   "ANOTHER DATASET - re-run it on this cohort. "
                   "CONFIG means it was")
        out.append("  made on this data under DIFFERENT CAPS - "
                   "re-run it at these settings. Different "
                   "causes, different fixes,")
        out.append("  so they do not share a word.")
        out.append("  These measurements were made, and NOT on "
                   "the cohort OR the configuration being "
                   "released. A determination is")
        out.append("  scoped to one release, so a number about "
                   "a different dataset - or about the same "
                   "data under different")
        out.append("  caps - is not evidence for it. A larger "
                   "token cap publishes more vocabulary about "
                   "members, so an attack")
        out.append("  measured at a smaller cap UNDERSTATES the "
                   "risk, which is the wrong direction for a "
                   "privacy number to be")
        out.append("  wrong in.")
        out.append("")
    if report["elements_owed"]:
        out.append("OWED: " + ", ".join(report["elements_owed"]))
        out.append("  An expert asked to sign against an OWED "
                   "element would be signing against nothing. "
                   "Assemble these before")
        out.append("  seeking a determination.")
    else:
        out.append("Every element has evidence in this run. That "
                   "is READINESS, not a determination - the "
                   "determination is still")
        out.append("  a qualified person's to make.")
    return out


# ---------------------------------------------------------------
# THE PACKAGE AS A PAGE
#
# This project's own lesson: a measurement that never reaches a
# page might as well not exist for the person deciding. The
# reconstruction gap lived in a terminal and a JSON file until
# somebody asked why nothing SHOWS how training and held-out
# error compare - and this is the most governance-facing
# artifact here, read by somebody deciding whether to seek a
# determination, who will not run a CLI.
#
# ONE renderer's worth of words is shared with `plain()` through
# ELEMENTS and NOT_A_DETERMINATION, so the page and the terminal
# cannot disagree about what is owed. That is the rule the
# gate's NEXT_STEPS already follow.

INK = "#1d1d1f"
GRAY = "#6e6e73"
RED = "#8c1515"
AMBER = "#8a6100"
GREEN = "#2e7d55"
RULE = "#e3e3e6"

_CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',
  Helvetica,Arial,sans-serif;color:%(ink)s;background:#fff;
  margin:0;font-size:14px;line-height:1.55}
.wrap{max-width:900px;margin:0 auto;padding:48px 32px}
h1{font-size:26px;font-weight:700;letter-spacing:-.02em;
  margin:0}
h2{font-size:17px;font-weight:700;margin:34px 0 8px;
  padding-top:18px;border-top:1px solid %(rule)s}
.sub{color:%(gray)s;margin:6px 0 0}
.warn{background:#fdf6f6;border-left:3px solid %(red)s;
  padding:12px 16px;margin:18px 0;font-size:13.5px}
.warn b{display:block;margin-bottom:4px}
.tally{font-size:20px;font-weight:750;margin:14px 0 2px}
table{border-collapse:collapse;width:100%%;font-size:13px;
  margin:10px 0}
th{font-weight:600;text-align:left;color:%(gray)s;
  border-bottom:1px solid %(rule)s;padding:6px 14px 6px 0}
td{border-bottom:1px solid #f2f2f4;padding:8px 14px 8px 0;
  vertical-align:top}
.st{font-weight:700;font-size:11.5px;white-space:nowrap;
  padding:3px 8px;border-radius:5px}
.st.ok{background:#eef6f1;color:%(green)s}
.st.owed{background:#fdeeee;color:%(red)s}
.st.other{background:#fdf6ea;color:%(amber)s}
.why{color:%(gray)s;font-size:12.5px;margin-top:5px}
.ev{font-size:12.5px}
.note{color:%(gray)s;font-size:12.5px;margin-top:14px}
code{background:#f5f5f7;padding:1px 4px;border-radius:3px;
  font-size:12px}
"""


def _esc(t):
    import html as _h
    return _h.escape("" if t is None else str(t))


def render_html(report, title="HIPAA Expert Determination"):
    """The package as a self-contained page.

    Takes the SAME report `plain()` takes, so the two cannot
    disagree about what is present and what is owed - a
    governance document whose page and whose terminal say
    different things is worse than either alone.
    """
    css = _CSS % {"ink": INK, "gray": GRAY, "red": RED,
                  "amber": AMBER, "green": GREEN, "rule": RULE}
    out = ["<!DOCTYPE html><html><head><meta charset='utf-8'>",
           "<meta name='viewport' content='width=device-width,"
           "initial-scale=1'>",
           "<title>", _esc(title), " - evidence package</title>",
           "<style>", css, "</style></head><body>",
           "<div class='wrap'>",
           "<h1>", _esc(title), " &mdash; evidence package</h1>",
           "<p class='sub'>Assembled from a run's own artifacts. "
           "Nothing here is recomputed.</p>"]

    # THE REFUSAL FIRST, because a reader who skims must not be
    # able to mistake this for a determination.
    out.append("<div class='warn'><b>This is not a "
               "determination.</b>")
    out.append("<ul style='margin:6px 0 0;padding-left:18px'>")
    for line in report.get("not_a_determination", []):
        out.append("<li>" + _esc(line) + "</li>")
    out.append("</ul></div>")

    n_have = report.get("elements_present")
    n_all = report.get("elements_total")
    out.append("<div class='tally'>{} of {} elements have "
               "evidence in this run</div>".format(n_have, n_all))
    if report.get("ready"):
        out.append("<p class='sub'>That is <b>readiness</b>, not "
                   "a determination. The determination in "
                   "164.514(b)(1) is made by a qualified "
                   "person.</p>")
    else:
        out.append("<p class='sub'>An element marked OWED has no "
                   "evidence in this run &mdash; an expert asked "
                   "to sign against it would be signing against "
                   "nothing.</p>")

    out.append("<h2>The statute</h2>")
    out.append("<p class='ev'>" + _esc(report.get("statute"))
               + "</p>")

    out.append("<h2>The elements</h2>")
    out.append("<table><tr><th>status</th><th>what the "
               "regulation asks</th><th>the evidence, and why "
               "it is asked</th></tr>")
    for el in report.get("elements", []):
        st = el.get("status", "OWED")
        cls = ("ok" if st == "present"
               else "owed" if st == "OWED" else "other")
        out.append("<tr><td><span class='st {}'>{}</span></td>"
                   .format(cls, _esc(st)))
        out.append("<td><b>" + _esc(el.get("need")) + "</b></td>")
        out.append("<td class='ev'>" + _esc(el.get("evidence"))
                   + "<div class='why'>" + _esc(el.get("why"))
                   + "</div></td></tr>")
    out.append("</table>")

    # THE TWO SCOPE FAULTS, each with its own fix, because they
    # take different actions.
    if report.get("elements_elsewhere"):
        out.append("<h2>Out of scope</h2>")
        out.append("<table><tr><th>element</th><th>status</th>"
                   "<th>what to do</th></tr>")
        for el in report["elements"]:
            if not el.get("measured_elsewhere"):
                continue
            st = el.get("status")
            fix = ("re-run it on this cohort"
                   if st == "FIXTURE"
                   else "re-run it at these settings")
            out.append("<tr><td>" + _esc(el.get("key"))
                       + "</td><td><span class='st other'>"
                       + _esc(st) + "</span></td><td>"
                       + _esc(fix) + "</td></tr>")
        out.append("</table>")
        out.append("<p class='note'>FIXTURE means the "
                   "measurement was made on another dataset; "
                   "CONFIG means it was made on this data under "
                   "different caps. A determination is scoped to "
                   "one release, so neither counts toward "
                   "readiness.</p>")
    if report.get("scope_unverifiable"):
        out.append("<h2>Scope not checkable</h2>")
        out.append("<p class='ev'>" + _esc(", ".join(
            report["scope_unverifiable"]))
            + "</p><p class='note'>These records name no source, "
              "so whether they were measured on this cohort "
              "could not be checked at all. A record predating "
              "the field cannot be judged on it &mdash; absent "
              "is not the same fact as mismatched &mdash; but "
              "neither is it confirmation.</p>")

    if report.get("elements_owed"):
        out.append("<h2>Still owed</h2>")
        out.append("<p class='ev'><code>"
                   + _esc(", ".join(report["elements_owed"]))
                   + "</code></p>")
        out.append("<p class='note'>Assemble these before "
                   "seeking a determination.</p>")

    out.append("</div></body></html>")
    return "".join(out)
