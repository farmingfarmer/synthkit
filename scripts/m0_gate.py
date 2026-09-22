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

# The computation lives in synthkit.gate - ONE place - because the
# bench shows the same verdicts and a copy in gui.py is how two
# halves of this codebase have repeatedly come to disagree.
import synthkit.gate as _gate

DIRECTION_MIN = _gate.DIRECTION_MIN
CLOSE_MIN = _gate.CLOSE_MIN
SET_AT = _gate.SET_AT


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/m0_gate.py RUNDIR")
    run = Path(sys.argv[1])
    fp = run / "fidelity.json"
    if not fp.exists():
        sys.exit("no fidelity.json in {} - run `synthkit fit` with "
                 "--generate first".format(run))
    fid = json.loads(fp.read_text(encoding="utf-8"))
    try:
        verdict = _gate.assess(fid)
    except ValueError as e:
        sys.exit(str(e))

    # The rendering lives in gate.report - `synthkit gate` prints
    # the same lines, one module, one text.
    lines, code = _gate.report(verdict, run)
    print("\n".join(lines))
    return code


if __name__ == "__main__":
    sys.exit(main())
