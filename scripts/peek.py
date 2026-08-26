"""Read numbers out of a finished run, without typing a one-liner.

    python scripts/peek.py RUNDIR sets
    python scripts/peek.py RUNDIR tokens
    python scripts/peek.py RUNDIR pairs [SUBSTRING]
    python scripts/peek.py RUNDIR rel COLUMN
    python scripts/peek.py RUNDIR columns [SUBSTRING]

WHY THIS EXISTS. Seven diagnoses in a row were fetched with pasted
python -c one-liners a hundred characters wide, on a terminal that
has mangled pastes three times - and one of them was simply WRONG
(a generator expression whose loop clauses were in the wrong order,
which reads as an operator error and is not). The operator should
type a short command; the parsing belongs in a file that gets
reviewed like everything else.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _load(run, name):
    p = Path(run) / name
    if not p.exists():
        sys.exit("no {} in {}".format(name, run))
    return json.loads(p.read_text(encoding="utf-8"))


def sets_view(run):
    """Published vocabulary against the size budget, per set column.

    `shares sum` and `mean size` are the SAME quantity measured two
    ways when every token is published: the average number of tokens
    a row carries. They diverge exactly when the k floor removed
    tokens - and that gap is the inflation every surviving token has
    to absorb."""
    bp = _load(run, "blueprint.json")
    for c, spec in (bp.get("columns") or {}).items():
        m = spec.get("marginal") or {}
        if m.get("type") != "list":
            continue
        toks = m.get("tokens") or []
        sz = m.get("set_size") or {}
        vs, ps = sz.get("v") or [], sz.get("p") or []
        mean_size = sum(float(v) * float(p) for v, p in zip(vs, ps))
        ssum = sum(float(t["p"]) for t in toks)
        print("{}".format(c))
        print("   published {} | found {} | above k {}".format(
            len(toks), m.get("tokens_found"), m.get("tokens_above_k")))
        print("   shares sum {:.2f} | mean size {:.2f} | gap {:.2f}x"
              .format(ssum, mean_size,
                      (mean_size / ssum) if ssum > 0 else float("nan")))


def tokens_view(run):
    fid = _load(run, "fidelity.json")
    rows = [t for t in (fid.get("set_tokens") or [])
            if abs(t.get("delta") or 0) > 0.05]
    print("{} of {} tokens miss by more than 0.05".format(
        len(rows), len(fid.get("set_tokens") or [])))
    over = sum(1 for t in rows if t["delta"] > 0)
    print("   {} generated HIGH, {} generated LOW".format(
        over, len(rows) - over))
    for t in sorted(rows, key=lambda r: -abs(r["delta"]))[:15]:
        print("   {:<34} {:.4f} -> {:.4f}".format(
            str(t.get("column"))[-34:], t.get("share_source"),
            t.get("share_generated")))


def pairs_view(run, needle=None):
    fid = _load(run, "fidelity.json")
    rows = []
    for r in (fid.get("relationships") or {}).get("pairs") or []:
        s, g = r.get("source") or 0.0, r.get("generated") or 0.0
        if abs(s) < 0.1:
            continue
        if needle and needle not in (r.get("child", "")
                                     + r.get("parent", "")):
            continue
        rows.append((abs(s - g), r, s, g))
    rows.sort(key=lambda t: -t[0])
    print("{} pairs the source relates at 0.1+{}".format(
        len(rows), " matching {!r}".format(needle) if needle else ""))
    for _d, r, s, g in rows[:15]:
        print("   {:<28} <- {:<28} {:+.3f} -> {:+.3f}  {}".format(
            str(r.get("child"))[-28:], str(r.get("parent"))[-28:],
            s, g, r.get("measure")))


def rel_view(run, col):
    bp = _load(run, "blueprint.json")
    for r in bp.get("relationships") or []:
        if col not in [r.get("child")] + list(r.get("parents") or []):
            continue
        ev = r.get("evidence") or {}
        print("{} <- {}   skill {:.3f}".format(
            r.get("child"), r.get("parents"),
            float(ev.get("skill_out_of_sample") or 0)))
        for p, e in (ev.get("effect") or {}).items():
            print("   {:<26} imp {:<8} shape {:<18} of_class {}"
                  .format(p[:26],
                          round(float((ev.get("importance")
                                       or {}).get(p, 0)), 4),
                          str(e.get("shape")), e.get("of_class")))


def columns_view(run, needle=None):
    fid = _load(run, "fidelity.json")
    for c in fid.get("columns") or []:
        if needle and needle not in c.get("column", ""):
            continue
        print("{:<34} cov {} -> {}  sd {} -> {}  lag1 {} -> {}".format(
            str(c.get("column"))[-34:], c.get("coverage_source"),
            c.get("coverage_generated"), c.get("sd_source"),
            c.get("sd_generated"), c.get("lag1_source"),
            c.get("lag1_generated")))


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    run, what = sys.argv[1], sys.argv[2]
    arg = sys.argv[3] if len(sys.argv) > 3 else None
    if what == "sets":
        sets_view(run)
    elif what == "tokens":
        tokens_view(run)
    elif what == "pairs":
        pairs_view(run, arg)
    elif what == "rel":
        if not arg:
            sys.exit("rel needs a column name")
        rel_view(run, arg)
    elif what == "columns":
        columns_view(run, arg)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
