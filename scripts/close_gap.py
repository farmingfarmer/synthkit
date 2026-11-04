"""Which pairs fail `close`, and by WHAT MECHANISM.

WHY THIS EXISTS. `close` is the M0 criterion with the smallest
margin, and two attempts to improve it have been built and
REVERTED - both because they were aimed at a mechanism nobody
had measured on the run that was failing. The record is
explicit about the cost: a tier that lands one coupling can
cost more elsewhere than it gains, and only a control arm at
the same seed can tell.

So before anything is built, this says WHICH pairs miss and
groups them by the mechanism that would have to be fixed. It
reads artifacts a run already wrote - `fidelity.json` and
`blueprint.json` - and recomputes nothing, so it cannot
disagree with the gate.

THE GROUPS ARE THE CANDIDATE FIXES. A gap concentrated in
cross-set-token pairs is a draw problem; one concentrated in
pairs whose columns lose magnitude to the published bound is
the k rule doing its job and is NOT a defect to chase; one
spread evenly over ordinary numeric pairs is the sampler.
Telling them apart is the whole point, because the first two
have been confused for each other before.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import synthkit.sets as S                             # noqa: E402

WORDS = {"0": "zero", "1": "one", "2": "two", "3": "three",
         "4": "four", "5": "five", "6": "six", "7": "seven",
         "8": "eight", "9": "nine", "-": "minus", ".": "point",
         "+": "plus", "%": "percent", "/": "of"}


def say(x, nd=3):
    """Digits with their spelling, for the OCR channel."""
    s = ("{:+." + str(nd) + "f}").format(float(x))
    return "{} [{}]".format(s, " ".join(
        WORDS.get(c, c) for c in s))


def sayn(n):
    s = str(int(n))
    return "{} [{}]".format(s, " ".join(WORDS.get(c, c)
                                        for c in s))


def classify(row, bound_cols):
    """Which mechanism would have to be fixed for this pair.

    The ORDER matters: a pair can be several of these, and the
    first match is the one worth acting on. A bound-explained
    pair is named first because it is NOT a defect - the k rule
    removed the magnitude deliberately, and this project has
    already lost a day to reading that as a sampler bug.
    """
    a, b = row.get("child"), row.get("parent")
    sa, sb = S.is_scaffolding(a), S.is_scaffolding(b)
    tok_a = S.HAS in (a or "")
    tok_b = S.HAS in (b or "")
    if a in bound_cols or b in bound_cols:
        return "bound-explained (the k rule removed the magnitude)"
    if tok_a and tok_b:
        same = (a or "").split(S.HAS)[0] == (b or "").split(S.HAS)[0]
        return ("within-set token pair" if same
                else "CROSS-set token pair")
    if tok_a or tok_b:
        return "token against a plain column"
    if sa or sb:
        return "other scaffolding"
    return "ordinary numeric pair"


def main(argv):
    if not argv:
        print(__doc__)
        print("usage: close_gap.py RUNDIR")
        return 2
    run = Path(argv[0])
    fid_p = run / "fidelity.json"
    bp_p = run / "blueprint.json"
    for need in (fid_p, bp_p):
        if not need.exists():
            print("STOP: no {} in {}".format(need.name, run))
            return 1
    fid = json.loads(fid_p.read_text(encoding="utf-8"))
    bp = json.loads(bp_p.read_text(encoding="utf-8"))

    # THE KEY IS `relationships`, read from the file rather
    # than remembered. Writing the field name that sounds
    # right instead of the one the contract uses is a fault
    # this project recorded three times in one day.
    pf = fid.get("relationships") or {}
    rows = pf.get("pairs") or []
    if not rows:
        print("STOP: fidelity.json carries no per-pair rows - "
              "this run predates them, so the gap cannot be "
              "attributed. Re-run the fit to get them.")
        return 1

    # COLUMNS WHOSE MAGNITUDE THE k RULE TOOK. Reported by the
    # run itself, not inferred here - a diagnostic that decides
    # for itself which losses privacy explains would be free to
    # explain away anything.
    bound_cols = set()
    for c, v in (fid.get("spread") or {}).items():
        if isinstance(v, dict) and (v.get("beyond_bound_share")
                                    or 0) > 0.5:
            bound_cols.add(c)

    strong = [r for r in rows
              if r.get("kind") == "numeric"
              and abs(float(r.get("source") or 0)) >= 0.1]
    miss = [r for r in strong
            if abs(float(r.get("delta") or 0)) > 0.2]
    print("CLOSE GAP  {}".format(run.name))
    print("  pairs the source relates at 0.1 or more: {}"
          .format(sayn(len(strong))))
    print("  of those, within 0.2 (CLOSE): {}"
          .format(sayn(len(strong) - len(miss))))
    print("  MISSING close: {}".format(sayn(len(miss))))
    if strong:
        print("  which is {}".format(
            say(100.0 * (len(strong) - len(miss)) / len(strong),
                1)))

    groups = {}
    for r in miss:
        groups.setdefault(classify(r, bound_cols), []).append(r)
    print("")
    print("  BY MECHANISM - the group with the most pairs is "
          "where a fix would pay, and the bound-explained group "
          "is NOT a defect:")
    for g in sorted(groups, key=lambda k: -len(groups[k])):
        rs = groups[g]
        worst = max(abs(float(x.get("delta") or 0)) for x in rs)
        print("    {:<46} {:>3}  worst drift {}".format(
            g, len(rs), say(worst)))

    print("")
    print("  THE TEN WORST, with their mechanism:")
    for r in sorted(miss,
                    key=lambda x: -abs(float(x.get("delta") or 0))
                    )[:10]:
        print("    {:<22} <- {:<22} src {} gen {}".format(
            str(r.get("child"))[:22], str(r.get("parent"))[:22],
            say(float(r.get("source") or 0)),
            say(float(r.get("generated") or 0))))
        print("       {}".format(classify(r, bound_cols)))

    # HOW MANY PAIRS A FIX WOULD HAVE TO LAND TO MOVE THE GATE.
    # Without this the groups are interesting and not
    # actionable - eleven pairs in a group sounds like a lot
    # until you learn the criterion needs two.
    bar = 0.879
    need = 0
    while strong and (len(strong) - len(miss) + need) \
            < bar * len(strong):
        need += 1
    print("")
    print("  TO CLEAR THE 87.9% BAR this run needs {} more "
          "pair(s) inside 0.2.".format(sayn(need)))
    if need == 0:
        print("  (this run already clears it - the question is "
              "head-room, not a failure)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
