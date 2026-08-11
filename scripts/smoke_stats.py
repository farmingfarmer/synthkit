"""SYNTH_S6 smoke: statistical rigor proven — Wilson intervals
exact at the edges, Hanley-McNeil for AUROC, prescriptions that
scale with the gap, and interval-aware three-way verdicts flowing
through every campaign goal: DECISIVE when the CI clears the bar,
INCONCLUSIVE with an executable prescription when it straddles.

Run from the repo root:

    python scripts/smoke_stats.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synthkit.autosolver import autoclean, autosolver
from synthkit.campaign import compile_campaign, run_campaign
from synthkit.examples import reference_table
from synthkit.mlmetrics import (
    MetricInputError,
    auroc_interval,
    format_rate,
    required_n,
    wilson_interval,
)

PASS = 0


def check(label, cond):
    global PASS
    if not cond:
        print("FAIL: {}".format(label))
        sys.exit(1)
    PASS += 1
    print("ok  {:02d}  {}".format(PASS, label))


def main():
    # ---------- Wilson intervals ----------
    lo, hi = wilson_interval(5, 10)
    check("Wilson 5/10 brackets one half symmetrically-ish",
          0.22 < lo < 0.27 and 0.73 < hi < 0.78)
    lo, hi = wilson_interval(0, 20)
    check("k=0 pins the floor without lying about the ceiling",
          lo == 0.0 and 0.1 < hi < 0.2)
    lo, hi = wilson_interval(20, 20)
    check("k=n mirrors it at the top",
          hi == 1.0 and 0.8 < lo < 0.9)
    wide = wilson_interval(2, 19)
    narrow = wilson_interval(20, 190)
    check("same rate, tenfold n, decisively narrower interval",
          (narrow[1] - narrow[0]) < (wide[1] - wide[0]) / 2)
    check("the bake-off cell: 2/19 straddles a 0.10 bar",
          wide[0] < 0.10 < wide[1])
    check("n=0 answers with total ignorance",
          wilson_interval(0, 0) == (0.0, 1.0))
    try:
        wilson_interval(5, 3)
        check("impossible k refused", False)
    except MetricInputError:
        check("impossible k refused", True)
    check("format_rate reads like the report line",
          format_rate(2, 19).startswith("0.105 [")
          and "n=19" in format_rate(2, 19))

    # ---------- prescriptions ----------
    razor = required_n(0.105, 0.100)
    clear = required_n(0.263, 0.100)
    check("prescriptions scale with the gap: razor edges need "
          "armies, clear gaps need squads",
          razor > 5000 and clear < 50)
    check("a rate ON the bar prescribes nothing — no n resolves "
          "a tie",
          required_n(0.5, 0.5) is None)

    # ---------- AUROC intervals ----------
    lo, hi = auroc_interval(0.667, 56, 344)
    check("Hanley-McNeil brackets the bake-off AUROC sanely",
          0.55 < lo < 0.62 and 0.71 < hi < 0.78)
    tight = auroc_interval(0.667, 560, 3440)
    check("more of both classes tightens the AUROC interval",
          (tight[1] - tight[0]) < (hi - lo) / 2)
    check("a class-free AUROC interval admits ignorance",
          auroc_interval(0.7, 0, 100) == (0.0, 1.0))

    # ---------- verdicts through the ladder ----------
    camp = compile_campaign("clean", reference_table(rows=120),
                            bars={"fix_rate": 0.35,
                                  "detect_rate": 0.5})
    result = run_campaign(camp, autoclean, "autoclean")
    text = result.format_text()
    check("evidence lines carry intervals with n",
          "[ci " in text and "n=" in text)
    check("clear margins read DECISIVE",
          "DECISIVE" in text)

    camp = compile_campaign("clean", reference_table(rows=120),
                            bars={"fix_rate": 0.65,
                                  "detect_rate": 0.5})
    result = run_campaign(camp, autoclean, "autoclean")
    text = result.format_text()
    check("a bar inside the interval reads INCONCLUSIVE with an "
          "executable prescription",
          "INCONCLUSIVE" in text and "n~" in text
          and "to resolve" in text)

    spec = reference_table(rows=200, master_seed=11)
    spec.outcomes = [{
        "name": "readmitted", "kind": "logistic",
        "intercept": -3.2,
        "coefficients": {"los_days": 0.18, "age": 0.015}}]
    camp = compile_campaign("predict", spec,
                            bars={"auroc": 0.6, "gap_max": 0.4},
                            outcome="readmitted")
    result = run_campaign(camp, autosolver(), "baseline")
    text = result.format_text()
    check("AUROC evidence carries the class-split interval",
          "+/" in text and "[ci " in text)
    result2 = run_campaign(camp, autosolver(), "baseline")
    check("interval-aware ladders stay deterministic",
          result2.format_text() == text)

    print("\nAll {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
