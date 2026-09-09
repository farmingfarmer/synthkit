"""The M0 gate as ONE computation, shared by the script and the bench.

WHY THIS IS A MODULE AND NOT TWENTY LINES IN TWO PLACES. The gate
began as numbers in a chat log and went unreachable when the pair
count moved; `scripts/m0_gate.py` fixed that by making it a script.
Then the bench needed to show the same verdicts, and copying the
criteria into gui.py is the exact mechanism by which two halves of
this codebase have repeatedly come to disagree - screening tokens
twice in two files, two categorical draw paths each reading the
blueprint. The computation lives here once; the script and the UI
both call it.

The bar, restated from the counts it was set at (124/132 direction,
116/132 close, when the run related 132 pairs):

    direction   >= 93.9%   of the pairs the run relates
    close       >= 87.9%
    inverted    == 0       absolute - an inverted relationship reads
                           as a finding, whatever the denominator

plus coverage, set token shares and the set EMPTY rate at full
marks where the run measured them at all.
"""
from __future__ import annotations

from typing import Any, Dict, List

DIRECTION_MIN = 0.939
CLOSE_MIN = 0.879
SET_AT = 132


def assess(fid: Dict[str, Any]) -> Dict[str, Any]:
    """Judge a fidelity report against the gate.

    Returns {"criteria": [...], "met": bool, "pairs": int}; each
    criterion is {"name", "ok", "detail"}. Raises ValueError when the
    report relates no pairs - that is a finding, not a pass."""
    s = fid.get("summary") or {}
    pairs = int(s.get("pairs") or 0)
    if not pairs:
        raise ValueError(
            "this run related no pairs, so the gate cannot be read - "
            "that is a finding in itself, not a pass")

    def pct(num):
        return float(num or 0) / float(pairs)

    direction = pct(s.get("pairs_sign_ok"))
    close = pct(s.get("pairs_close"))
    inverted = int(s.get("pairs_inverted") or 0)

    crit: List[Dict[str, Any]] = []

    def add(name, ok, detail):
        crit.append({"name": name, "ok": bool(ok), "detail": detail})

    add("direction kept", direction >= DIRECTION_MIN,
        "{}/{} = {:.1%}  (need {:.1%}, which is the {}/{} the gate "
        "was set at)".format(s.get("pairs_sign_ok"), pairs, direction,
                             DIRECTION_MIN, 124, SET_AT))
    add("close", close >= CLOSE_MIN,
        "{}/{} = {:.1%}  (need {:.1%}, which is {}/{})".format(
            s.get("pairs_close"), pairs, close, CLOSE_MIN, 116,
            SET_AT))
    add("inverted", inverted == 0,
        "{} - absolute, and it stays absolute: an inverted "
        "relationship reads as a finding".format(inverted))

    cov_n, cov_d = s.get("coverage_ok"), s.get("columns")
    if cov_d:
        add("coverage", cov_n == cov_d, "{}/{}".format(cov_n, cov_d))
    tk_n, tk_d = s.get("set_tokens_ok"), s.get("set_tokens_compared")
    if tk_d:
        add("set token shares", tk_n == tk_d,
            "{}/{}".format(tk_n, tk_d))
    em_n, em_d = s.get("set_empty_ok"), s.get("set_empty_compared")
    if em_d:
        add("set EMPTY rate", em_n == em_d,
            "{}/{} - measured, and the token shares cannot see "
            "it".format(em_n, em_d))
    # SHAPE AND SURFACE CRITERIA - conditional, like the three
    # above: a run that measured none does not fail on absence.
    # Rank correlation is monotone, so a U-shaped relationship is
    # invisible to `close`; these two see what it cannot.
    sh_n, sh_d = s.get("shapes_ok"), s.get("shapes_compared")
    if sh_d:
        add("shapes tracked", sh_n == sh_d,
            "{}/{} effect curves within 0.35 sd - what `close` "
            "cannot see: a U-shape has no rank "
            "correlation".format(sh_n, sh_d))
    su_n, su_d = s.get("surfaces_ok"), s.get("surfaces_compared")
    if su_d:
        add("interaction surfaces", su_n == su_d,
            "{}/{} two-variable surfaces within 0.35 sd - "
            "higher-order structure is not published, so nothing "
            "above two-way is measured".format(su_n, su_d))

    return {"criteria": crit,
            "met": all(c["ok"] for c in crit),
            "pairs": pairs}


# STATION REFERENCES ARE TOKENS, RENDERED BY EACH CALLER. When
# guidance names a place in the bench ("draw the Dashboard"), the
# text carries [[station]] and the caller decides the form: the
# bench renders a miniature station button the operator can press,
# the CLI prints the station's name. One text, two renderings -
# the same reason this module exists at all.
STATIONS: Dict[str, Dict[str, str]] = {
    "dashboard": {"num": "VIEW", "label": "Dashboard",
                  "route": "map"},
    "fitsrc": {"num": "01", "label": "Source", "route": "measure"},
    "fitrun": {"num": "02", "label": "Fit", "route": "measure"},
    "fitver": {"num": "03", "label": "Verdict",
               "route": "measure"},
}


def plain(text: str) -> str:
    """Render [[station]] tokens as plain names, for terminals."""
    for key, st in STATIONS.items():
        text = text.replace(
            "[[{}]]".format(key),
            "the {} {} station".format(st["num"], st["label"]))
    return text


# WHAT EACH CRITERION MEANS IN THE DATA, pass or fail, and how to
# INSPECT it yourself - shown under every verdict row, because a
# PASS the operator cannot check for themselves is just a claim.
EXPLAIN: Dict[str, Dict[str, str]] = {
    "direction kept": {
        "explain": "Of the relationships the source really "
                   "contains, how many still POINT THE SAME WAY in "
                   "the synthetic file. A falling relationship that "
                   "generates as rising would read as a false "
                   "finding.",
        "inspect": "In [[dashboard]], press and hold a correlation "
                   "heatmap: a cell that changes colour family "
                   "under the crossfade is a direction loss. The "
                   "pattern cards draw the worst as curves that "
                   "disagree about which way they run."},
    "close": {
        "explain": "How many relationships keep their STRENGTH - "
                   "the generated correlation within 0.2 of the "
                   "source's. A pair can point the right way and "
                   "still arrive diluted.",
        "inspect": "In [[dashboard]], hold a heatmap: the cells "
                   "that visibly blink are the drifted pairs, and "
                   "the gate-issues section there lists EVERY "
                   "drifted pair by name with its source, "
                   "generated and drift."},
    "inverted": {
        "explain": "Relationships that come out OPPOSITE to the "
                   "source. Worse than missing, because an "
                   "inverted relationship reads as a finding. The "
                   "bar is zero, always.",
        "inspect": "Any inverted pair is named in the gate-issues "
                   "section of [[dashboard]] and shows as mirrored "
                   "curves in its pattern card."},
    "coverage": {
        "explain": "Each column should be PRESENT (non-missing) at "
                   "the source's own rate, within 0.05. A miss "
                   "here usually means a column was read as the "
                   "wrong type upstream.",
        "inspect": "In [[dashboard]], every column's stats table "
                   "shows missing % original beside synthetic - "
                   "scan for rows where the two disagree."},
    "set token shares": {
        "explain": "Each published list token - a condition, a "
                   "medication - should appear at its source "
                   "frequency, within 0.05.",
        "inspect": "In [[dashboard]], the paired bars on each set "
                   "column draw original beside synthetic share "
                   "for every published token."},
    "shapes tracked": {
        "explain": "Each published effect CURVE re-measured on "
                   "both tables - the bend itself, not a "
                   "correlation. A U-shape has rank correlation "
                   "near zero, so the close criterion cannot see "
                   "it; this one can.",
        "inspect": "In [[dashboard]], the pattern cards draw each "
                   "curve three ways - published, original, "
                   "synthetic - and state the miss in sd; DEPARTS "
                   "is the same 0.35 bar this criterion gates."},
    "interaction surfaces": {
        "explain": "Each published two-variable interaction "
                   "surface re-measured as a coarse grid on both "
                   "tables. Nothing above two-way is published by "
                   "the engine, so nothing above two-way is "
                   "measured - a metric for what the generator "
                   "cannot produce would only restate a known "
                   "limitation.",
        "inspect": "The gate-issues section of [[dashboard]] names "
                   "any departing surface with its column pair and "
                   "gap."},
    "set EMPTY rate": {
        "explain": "Visits holding an EMPTY list in the source "
                   "should be empty at the same rate in the "
                   "synthetic file. Token shares cannot see this.",
        "inspect": "In [[dashboard]], each set column section "
                   "states the empty rate for both sides."},
}


# WHAT A FAIL MEANS AND WHAT TO DO NEXT, per criterion - here, in
# the shared module, so the bench and the script print the same
# words. A gate that says NOT MET and stops talking leaves the
# operator to invent next steps under pressure; these are stated in
# order of cheapness, and "proceed with the failure stated" is a
# legitimate option on the list, because the gate is a floor and
# not a release decision.
NEXT_STEPS: Dict[str, Dict[str, Any]] = {
    "direction kept": {
        "means": "Fewer pairs kept their sign than the bar - "
                 "relationships are being lost, not just weakened.",
        "steps": [
            "Read the dropped/trimmed section of findings.txt: a "
            "relationship found and then trimmed to break a cycle "
            "is invisible in the output, and the trims are counted "
            "there.",
            "Re-run 2-3 seeds before believing a small margin - a "
            "single seed cannot resolve one, and this measurement "
            "moves several points seed to seed.",
            "Draw [[dashboard]] and read the pattern cards: a "
            "DEPARTS verdict names which relationships, in sd.",
            "If lags are off and the data is temporal, refit with "
            "lags - lagged pairs cannot be found without them."]},
    "close": {
        "means": "Direction survives but strength drifts on too "
                 "many pairs - the shapes are there, weaker or "
                 "stronger than the source.",
        "steps": [
            "Draw [[dashboard]]: each pattern card states the miss "
            "in standard deviations, so you can see WHICH pairs "
            "drift and by how much.",
            "Re-run 2-3 seeds - close moves several points on seed "
            "alone, and a margin inside that noise is a draw, not "
            "a verdict.",
            "Check the k-rule notes in findings.txt: a column whose "
            "magnitude sits beyond the published bound loses "
            "strength BY DESIGN, and that shortfall is privacy "
            "working, not a fault to chase.",
            "Proceed with the number stated, when the use needs "
            "direction rather than exact strength - direction "
            "PASS with zero inversions means no false findings, "
            "only fainter ones. Say '76.5% close' to whoever "
            "receives the file rather than rounding it away."]},
    "inverted": {
        "means": "STOP. A generated relationship runs OPPOSITE to "
                 "the source. Inverted is worse than missing, "
                 "because it reads as a finding.",
        "steps": [
            "Do not send this file to anyone.",
            "Note the pair from the fidelity report and re-run one "
            "different seed: reproducible inversion is a defect to "
            "report, single-seed inversion is still a stop for "
            "THIS file.",
            "The sweep gate for code changes holds at zero "
            "inversions; a run that shows one is exactly what the "
            "gate exists to catch."]},
    "coverage": {
        "means": "A column is present at the wrong rate - the "
                 "classic signature of a column read as the wrong "
                 "type, which fails silently everywhere else.",
        "steps": [
            "Run the types check on the source and read EVERY "
            "column's type and sentinel share - seconds, and one "
            "glance catches what five minutes of discovery "
            "cannot.",
            "Check the long/EAV note: a concept+value extract "
            "needs --long, and a date parsed as labels shows up "
            "here first.",
            "Fix the reading, refit. Do not tune anything else "
            "until coverage passes - every other number is "
            "measured through it."]},
    "shapes tracked": {
        "means": "A relationship's SHAPE drifted - the synthetic "
                 "curve bends differently from the source's, even "
                 "if the correlation survived.",
        "steps": [
            "Open the departing claim's pattern card in "
            "[[dashboard]]: the gray and cardinal curves show "
            "WHERE along the range the shapes disagree.",
            "Check whether the drift sits at a k-screened tail - "
            "a bend the privacy rule flattened is the rule "
            "working.",
            "Re-run a second seed: single-seed shape gaps move, "
            "and one measurement of a change is worth nothing "
            "here."]},
    "interaction surfaces": {
        "means": "A two-variable interaction did not survive - "
                 "the child responds to the PAIR differently in "
                 "the synthetic file.",
        "steps": [
            "Read the surface's pair and gap in the gate-issues "
            "section of [[dashboard]] - the pair names the two "
            "columns whose joint effect drifted.",
            "Check the trimmed-parents report in findings.txt: a "
            "surface whose parent was cut to break a cycle "
            "cannot fire at generation.",
            "Re-run a second seed before treating the gap as "
            "real - an interaction fix was once shipped on seed 0 "
            "and failed completely on seed 3."]},
    "set token shares": {
        "means": "Published list tokens (conditions, medications) "
                 "generate at the wrong frequencies.",
        "steps": [
            "Run `python scripts/peek.py RUNDIR empty` - it "
            "separates a count-column fault from a set-draw fault "
            "from the run directory alone, and the fix differs.",
            "Read the refused-token count in findings.txt: tokens "
            "under the k floor are removed on purpose, and the "
            "price is reported in tokens per row."]},
    "set EMPTY rate": {
        "means": "Visits that hold an empty list in the source do "
                 "not in the generated file, or the reverse - "
                 "token shares cannot see this.",
        "steps": [
            "Run `python scripts/peek.py RUNDIR empty`: if the "
            "size partner is zero at the generated empty rate the "
            "count column is the cause; at the source rate, the "
            "set draw is.",
            "Check zero_share on the partner count column - excess "
            "zeros there drag the set empty with them."]},
}


def next_steps(assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Guidance for each FAILED criterion, in the gate's own order.

    Returns [{"name", "means", "steps": [...]}] - empty when the
    gate is met, because advice under a green gate is noise."""
    out: List[Dict[str, Any]] = []
    for c in assessment.get("criteria") or []:
        if c.get("ok"):
            continue
        g = NEXT_STEPS.get(c.get("name"))
        if g:
            out.append({"name": c["name"], "means": g["means"],
                        "steps": list(g["steps"])})
    return out
