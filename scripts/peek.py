"""Read numbers out of a finished run, without typing a one-liner.

    python scripts/peek.py RUNDIR sets
    python scripts/peek.py RUNDIR tokens
    python scripts/peek.py RUNDIR pairs [SUBSTRING]
    python scripts/peek.py RUNDIR rel COLUMN
    python scripts/peek.py RUNDIR columns [SUBSTRING]
    python scripts/peek.py RUNDIR cap
    python scripts/peek.py RUNDIR empty

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


def cap_view(run):
    """What the SEARCH cap is excluding, in rows rather than ranks.

    `EXPAND_CAP` turns only the most common tokens into search
    columns, and only an expanded token can carry a relationship - on
    the real extract that leaves 1,026 of 1,050 `conditions` tokens
    unexaminable. Whether that costs anything depends entirely on how
    many ROWS a token just past the cap covers, and nothing reported
    that.

    It could not be answered on a fixture. There, every token past
    rank 12 sits at about 0.8% of rows - rank 24 at 0.88% and rank 27
    at 0.84%, which is 22 rows - so no relationship on an excluded
    token is detectable at ANY cap, and raising it to 40 changed
    nothing. That is a fact about the fixture's thin tail, not
    evidence the cap is free.

    On 55,428 rows a token at 0.5% covers 277, which is a different
    question with a different answer. This prints the answer."""
    bp = _load(run, "blueprint.json")
    try:
        from synthkit import sets as _S
        cap = _S.EXPAND_CAP
    except Exception:
        cap = 24
    n_rows = None
    prov = Path(run) / "provenance.json"
    if prov.exists():
        try:
            n_rows = (json.loads(prov.read_text(encoding="utf-8"))
                      .get("source", {}).get("rows_read"))
        except Exception:
            n_rows = None
    for c, spec in (bp.get("columns") or {}).items():
        m = (spec or {}).get("marginal") or {}
        if m.get("type") != "list":
            continue
        toks = m.get("tokens") or []
        if not toks:
            continue
        ps = [float(t.get("p") or 0.0) for t in toks]
        print("{}  ({} published, cap {})".format(c, len(ps), cap))

        def _at(rank):
            if rank <= 0 or rank > len(ps):
                return None
            return ps[rank - 1]

        for rank in (1, cap, cap + 1, 50, 100, 250, 500, 1000,
                     len(ps)):
            v = _at(rank)
            if v is None:
                continue
            rows = ("{:>8,}".format(int(round(v * n_rows)))
                    if n_rows else "       ?")
            print("   rank {:>5}  p {:.5f}   ~{} rows{}".format(
                rank, v, rows,
                "   <- last inside the cap" if rank == cap else
                ("   <- first EXCLUDED" if rank == cap + 1 else "")))
        inside = sum(ps[:cap])
        print("   the {} expanded tokens carry {:.1%} of the token "
              "mass; the other {} carry {:.1%}".format(
                  min(cap, len(ps)), inside / sum(ps) if sum(ps) else 0,
                  max(0, len(ps) - cap),
                  1 - (inside / sum(ps) if sum(ps) else 0)))
        print()


def empty_view(run):
    """Where a set column's EMPTY rows come from.

    A set column is empty on a row either because the source held
    nothing there, or because its SIZE PARTNER - a count column the
    blueprint declares equal to the set's size - came out zero. Those
    are different faults with different fixes, and the fidelity line
    ("empty 31.4% in source, 36.8% generated") cannot tell them
    apart.

    THIS COULD NOT BE ANSWERED ON A FIXTURE. Three shapes were built
    to reproduce a 5.4-point gap seen on a real extract - a size
    partner alone, a size partner with the measured sub-k tail, and a
    cycle with refinement sweeps on and off - and all three
    reproduced the source rate to within 1.1 points. Whatever causes
    it is in the real data, so the diagnostic goes where the data is.

    Reads only the run directory: nothing here needs the source."""
    bp = _load(run, "blueprint.json")
    import csv as _csv
    gen = Path(run) / "generated.csv"
    if not gen.exists():
        sys.exit("no generated.csv in {} - run with --generate"
                 .format(run))
    partners = {}
    for c in (bp.get("constraints") or []):
        if c.get("op") != "==":
            continue
        lhs, rhs = str(c.get("lhs")), str(c.get("rhs"))
        for a, b in ((lhs, rhs), (rhs, lhs)):
            if a.endswith("__n"):
                partners[a[:-3]] = b
    cols = {}
    for c, spec in (bp.get("columns") or {}).items():
        m = (spec or {}).get("marginal") or {}
        if m.get("type") == "list":
            cols[c] = m
    if not cols:
        print("no set columns in this blueprint")
        return
    want = set(cols) | set(v for v in partners.values())
    seen = dict((w, {"n": 0, "empty": 0, "zero": 0}) for w in want)
    with gen.open(encoding="utf-8", newline="") as fh:
        for row in _csv.DictReader(fh):
            for w in want:
                if w not in row:
                    continue
                v = (row[w] or "").strip()
                seen[w]["n"] += 1
                if v == "":
                    seen[w]["empty"] += 1
                try:
                    if float(v) == 0.0:
                        seen[w]["zero"] += 1
                except (TypeError, ValueError):
                    pass
    for c, m in sorted(cols.items()):
        pub = float(m.get("empty_in_source_share") or 0.0)
        unpub = float(m.get("unpublishable_row_share") or 0.0)
        st = seen.get(c) or {"n": 0, "empty": 0}
        got = (st["empty"] / float(st["n"])) if st["n"] else float("nan")
        print("{}".format(c))
        print("   published EMPTY in source {:.3f} | generated {:.3f}"
              "  ({:+.3f})".format(pub, got, got - pub))
        print("   rows holding ONLY sub-k tokens {:.3f} - these draw "
              "a token, so they push the other way".format(unpub))
        part = partners.get(c)
        if part:
            ps = seen.get(part) or {"n": 0, "zero": 0}
            pz = (ps["zero"] / float(ps["n"])) if ps["n"] else float("nan")
            print("   size partner `{}` is ZERO on {:.3f} of generated "
                  "rows".format(part, pz))
            print("   -> if that matches the generated empty rate, the "
                  "COUNT column is the cause, not the set draw")
        else:
            print("   no declared size partner - the sizes come from "
                  "the published distribution alone")
        print()


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
    elif what == "cap":
        cap_view(run)
    elif what == "empty":
        empty_view(run)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
