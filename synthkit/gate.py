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

    return {"criteria": crit,
            "met": all(c["ok"] for c in crit),
            "pairs": pairs}


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
            "Draw the Dashboard and read the pattern cards: a "
            "DEPARTS verdict names which relationships, in sd.",
            "If lags are off and the data is temporal, refit with "
            "lags - lagged pairs cannot be found without them."]},
    "close": {
        "means": "Direction survives but strength drifts on too "
                 "many pairs - the shapes are there, weaker or "
                 "stronger than the source.",
        "steps": [
            "Draw the Dashboard: each pattern card states the miss "
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
