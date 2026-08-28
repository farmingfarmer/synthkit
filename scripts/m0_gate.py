"""Read a finished run and say whether it clears the M0 gate.

    python scripts/m0_gate.py RUNDIR

WHY THIS EXISTS. The gate was a pair of absolute counts carried in
conversation - "close at least 116, direction back to about 124" -
measured when the run related 132 pairs. The denominator then moved
to 115 and to 114 as set columns changed what could be related, and
124 became unreachable BY CONSTRUCTION: it is larger than the number
of pairs the run now has. A target that cannot be met no longer
measures anything, and nobody noticed for two runs because the target
lived in a chat log rather than beside the numbers.

So the counts are restated as PROPORTIONS of whatever the run
related. This is a restatement of the original bar, not a new one:

    direction   124/132 = 93.9%  ->  >= 93.9%
    close       116/132 = 87.9%  ->  >= 87.9%
    inverted    0                ->  == 0

The third has always been absolute and stays that way. An inverted
relationship reads as a finding and is worse than a missing one, so
one of them fails the gate however large the denominator is.

THE OTHER THREE CRITERIA ARE NEW ONLY IN THAT THEY CAN NOW BE
MEASURED. Coverage, set-token shares and the set EMPTY rate were all
being reported already; they are named here so a run either clears
the gate or says which line it failed, rather than leaving a reader
to compare six numbers by eye.

Exit status is 0 when the gate is met and 1 when it is not, so this
can be run from a script without reading the text.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# The original bar, converted from the counts it was set at.
DIRECTION_MIN = 0.939
CLOSE_MIN = 0.879
SET_AT = 132


def _pct(num, den):
    return (float(num) / float(den)) if den else float("nan")


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/m0_gate.py RUNDIR")
    run = Path(sys.argv[1])
    fp = run / "fidelity.json"
    if not fp.exists():
        sys.exit("no fidelity.json in {} - run `synthkit fit` with "
                 "--generate first".format(run))
    fid = json.loads(fp.read_text(encoding="utf-8"))
    s = fid.get("summary") or {}

    pairs = int(s.get("pairs") or 0)
    if not pairs:
        sys.exit("this run related no pairs, so the gate cannot be "
                 "read - that is a finding in itself, not a pass")

    direction = _pct(s.get("pairs_sign_ok"), pairs)
    close = _pct(s.get("pairs_close"), pairs)
    inverted = int(s.get("pairs_inverted") or 0)

    lines = []

    def crit(name, ok, detail):
        lines.append((bool(ok), name, detail))

    crit("direction kept", direction >= DIRECTION_MIN,
         "{}/{} = {:.1%}  (need {:.1%}, which is the {}/{} the gate "
         "was set at)".format(s.get("pairs_sign_ok"), pairs, direction,
                              DIRECTION_MIN, 124, SET_AT))
    crit("close", close >= CLOSE_MIN,
         "{}/{} = {:.1%}  (need {:.1%}, which is {}/{})".format(
             s.get("pairs_close"), pairs, close, CLOSE_MIN, 116,
             SET_AT))
    crit("inverted", inverted == 0,
         "{} - absolute, and it stays absolute: an inverted "
         "relationship reads as a finding".format(inverted))

    cov_n, cov_d = s.get("coverage_ok"), s.get("columns")
    if cov_d:
        crit("coverage", cov_n == cov_d,
             "{}/{}".format(cov_n, cov_d))
    tk_n, tk_d = s.get("set_tokens_ok"), s.get("set_tokens_compared")
    if tk_d:
        crit("set token shares", tk_n == tk_d,
             "{}/{}".format(tk_n, tk_d))
    em_n, em_d = s.get("set_empty_ok"), s.get("set_empty_compared")
    if em_d:
        crit("set EMPTY rate", em_n == em_d,
             "{}/{} - measured, and the token shares cannot see "
             "it".format(em_n, em_d))

    print("M0 gate on {}".format(run))
    print()
    for ok, name, detail in lines:
        print("  {}  {:<18} {}".format("PASS" if ok else "FAIL",
                                       name, detail))
    print()
    failed = [n for ok, n, _ in lines if not ok]
    if failed:
        print("M0 NOT MET - {}".format(", ".join(failed)))
        print("The two proportions restate counts set when the run "
              "related {} pairs; this one relates {}.".format(
                  SET_AT, pairs))
        return 1
    print("M0 MET on all {} criteria.".format(len(lines)))
    print("One cohort, one seed. A gate is a floor, not a "
          "certificate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
