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
