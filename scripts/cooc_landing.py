"""Did the published co-occurrence actually LAND in the output?

WHY THIS EXISTS. The real extract's remaining `close` misses are
dominated by CROSS-SET TOKEN pairs - a drug implying its route - and
the obvious suspects were both acquitted by reading the artifacts:
the pair IS detected, and it IS published. `sodium-chloride` against
`Flush` carries phi 0.88483 and a conditional of 0.837, which is the
source correlation to three decimals. So the contract has the
structure and the OUTPUT does not, which puts the fault in the draw
rather than in detection - and this project's rule is that the next
step is a diagnostic in the run directory, not a fourth guess.

So: for every published pair, measure P(b | a) in the GENERATED file
and compare it against the conditional the blueprint published. A
pair that lands says the draw solved it; a pair that does not names
the mechanism. The same move as `quota_landing`, for the same
reason - if the adopted pairs land while close does not move, the
drift is somewhere else entirely.

IT READS THE WRITTEN FILE, not an in-memory frame. The indicators
are derived from the set column the operator actually receives, so
this measures what a reader would find rather than what the engine
believed.

NO CELL IS PUBLISHED: counts, shares and token names only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sayn import annotate                               # noqa: E402
from synthkit import sets as S                          # noqa: E402


def _say(n):
    return annotate(str(n))


def verdict(landed, scored):
    """Whether the DRAW or the ADOPTION is the next question.

    A pure function on planted numbers, because a report whose
    whole content is the branch it chose must have both branches
    reachable - the dead-branch fault this project has already
    found in a linkage verdict and a release-dial report.
    """
    if not scored:
        return "nothing-scored"
    return "draw" if (landed / float(scored)) < 0.5 else "adoption"


def column_of(indicator):
    """`conditions__has__X` -> ("conditions", "X")."""
    if S.HAS not in indicator:
        return None, None
    col, tok = indicator.split(S.HAS, 1)
    return col, tok


def landing(run_dir, top=20, tol=0.15):
    run = Path(run_dir)
    bp = json.loads((run / "blueprint.json").read_text(
        encoding="utf-8"))
    cooc = bp.get("token_cooccurrence") or []
    if not cooc:
        return None, "this blueprint publishes NO co-occurrence - " \
                     "absent is not the same fact as unlanded"
    g = pd.read_csv(run / "generated.csv", dtype=str,
                    keep_default_na=False)

    sep = {}
    for name, spec in (bp.get("columns") or {}).items():
        m = spec.get("marginal") or {}
        if m.get("type") == "list":
            sep[name] = m.get("separator") or ";"

    cache = {}

    def ind(indicator):
        if indicator in cache:
            return cache[indicator]
        col, tok = column_of(indicator)
        if col is None or col not in g.columns:
            cache[indicator] = None
            return None
        v = S.has_token(g[col], sep.get(col, ";"), tok)
        cache[indicator] = np.nan_to_num(
            v.to_numpy(dtype=float)).astype(bool)
        return cache[indicator]

    rows = []
    for c in cooc:
        A, B = ind(c["a"]), ind(c["b"])
        if A is None or B is None:
            rows.append({"a": c["a"], "b": c["b"],
                         "published": c.get("cond"),
                         "achieved": None,
                         "why": "a column the output does not carry"})
            continue
        na = int(A.sum())
        if na == 0:
            rows.append({"a": c["a"], "b": c["b"],
                         "published": c.get("cond"),
                         "achieved": None,
                         "why": "the driving token was never emitted"})
            continue
        rows.append({"a": c["a"], "b": c["b"],
                     "published": float(c.get("cond") or 0.0),
                     "achieved": float((A & B).sum()) / float(na),
                     "why": None})

    scored = [r for r in rows if r["achieved"] is not None]
    unscored = [r for r in rows if r["achieved"] is None]
    for r in scored:
        r["miss"] = r["achieved"] - r["published"]
    landed = [r for r in scored if abs(r["miss"]) <= tol]
    return {
        "published": len(cooc),
        "scored": len(scored),
        "unscored": len(unscored),
        "landed": len(landed),
        "worst": sorted(scored, key=lambda r: -abs(r["miss"]))[:top],
        "unscored_rows": unscored[:top],
        "mean_abs_miss": (float(np.mean([abs(r["miss"])
                                         for r in scored]))
                          if scored else float("nan")),
    }, None


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("usage: cooc_landing.py RUNDIR [--top N] [--tol X]")
        return 0
    run = argv[0]
    top, tol = 20, 0.15
    i = 1
    while i < len(argv):
        if argv[i] == "--top" and i + 1 < len(argv):
            top = int(argv[i + 1])
            i += 2
            continue
        if argv[i] == "--tol" and i + 1 < len(argv):
            tol = float(argv[i + 1])
            i += 2
            continue
        i += 1
    p = Path(run)
    for f in ("blueprint.json", "generated.csv"):
        if not (p / f).exists():
            print("STOPPED: {} has no {}. This reads the WRITTEN "
                  "file, because the indicators are derived from "
                  "the set column the operator actually receives."
                  .format(run, f))
            return 2
    res, err = landing(run, top=top, tol=tol)
    if res is None:
        print("STOPPED: {}".format(err))
        return 3

    print("DID THE PUBLISHED CO-OCCURRENCE LAND?")
    print("  published pairs   {}".format(_say(res["published"])))
    print("  scored            {}".format(_say(res["scored"])))
    print("  not scorable      {}".format(_say(res["unscored"])))
    print("  LANDED within {}   {} of {}".format(
        _say(tol), _say(res["landed"]), _say(res["scored"])))
    print("  mean absolute miss {}".format(
        _say(round(res["mean_abs_miss"], 4))))
    print()
    if res["scored"]:
        if verdict(res["landed"], res["scored"]) == "draw":
            print("  MOST PUBLISHED CONDITIONALS DO NOT LAND. The")
            print("  contract carries the structure and the output")
            print("  does not, so the fault is in the DRAW rather")
            print("  than in detection - which is a different fix")
            print("  from raising a cap or loosening a screen.")
        else:
            print("  MOST PUBLISHED CONDITIONALS LAND. If `close`")
            print("  still misses on these pairs then the drift is")
            print("  NOT the adopted pairs, and the adoption gates")
            print("  are the next question rather than the solve.")
    print()
    print("WORST MISSES (published conditional against achieved):")
    for r in res["worst"]:
        print("  {:<34} <- {:<34}".format(
            str(r["b"])[-34:], str(r["a"])[-34:]))
        print("      published {}   achieved {}   miss {}".format(
            _say(round(r["published"], 4)),
            _say(round(r["achieved"], 4)),
            _say(round(r["miss"], 4))))
    if res["unscored_rows"]:
        print()
        print("NOT SCORABLE, by reason - an absent measurement is "
              "not a landed one:")
        for r in res["unscored_rows"]:
            print("  {:<30} {}".format(str(r["a"])[-30:], r["why"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
