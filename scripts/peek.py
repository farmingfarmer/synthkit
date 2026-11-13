"""Read numbers out of a finished run, without typing a one-liner.

    python scripts/peek.py RUNDIR sets
    python scripts/peek.py RUNDIR tokens
    python scripts/peek.py RUNDIR pairs [SUBSTRING]
    python scripts/peek.py RUNDIR rel COLUMN
    python scripts/peek.py RUNDIR columns [SUBSTRING]
    python scripts/peek.py RUNDIR cap
    python scripts/peek.py RUNDIR empty
    python scripts/peek.py RUNDIR centre
    python scripts/peek.py RUNDIR surfaces
    python scripts/peek.py RUNDIR flat
    python scripts/peek.py RUNDIR sizes SRC.csv
    python scripts/peek.py RUNDIR marginal COLUMN SRC.csv
    python scripts/peek.py RUNDIR marginals SRC.csv
    python scripts/peek.py RUNDIR rerank
    python scripts/peek.py RUNDIR sweep
    python scripts/peek.py RUNDIR copies
    python scripts/peek.py RUNDIR dynamics
    python scripts/peek.py RUNDIR curves

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

# THE ROOT GOES ON THE PATH FOR EVERY SUBCOMMAND, not just one.
# This insert used to sit INSIDE the `empty` helper, so `peek
# surfaces` and the rest died with ModuleNotFoundError on a
# source tree with no install - the documented zipball flow on
# the data machine. A path fix buried in one function fixes one
# caller and leaves the others looking like a different bug.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


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


def centre_view(run):
    """The centre_miss diagnosis, read out of a finished run.

    The centre criterion failed on 16 of 34 numeric columns on the
    800-patient run with nothing to say why; the block was built to
    diagnose the remainder and this is its reader. Sorted worst
    first, with the two candidate mechanisms called out: source
    skew (the piecewise-CDF loss) and a missing tail shape (the
    published correction that would have countered it)."""
    print("centre_view on {}".format(run))
    fid = _load(run, "fidelity.json")
    cols = fid.get("columns") or []
    rows = [(c.get("column"), c.get("centre_miss"))
            for c in cols if c.get("centre_miss")]
    n_num = sum(1 for c in cols
                if c.get("mean_source") is not None)
    ok = (fid.get("summary") or {}).get("centre_ok")
    print("centre_ok {} of {} numeric columns; {} carry a "
          "centre_miss diagnosis".format(ok, n_num, len(rows)))
    if not rows:
        print("no centre_miss blocks - every centre landed inside "
              "a tenth of its column's spread")
        return
    print()
    print("{:<28} {:>7}  {:<24} {:>6}  {:<10} {}".format(
        "column", "miss_sd", "direction", "skew", "tail_pub",
        "integral"))
    for c, m in sorted(rows, key=lambda x: -x[1]["by_sd"]):
        print("{:<28} {:>7.2f}  {:<24} {:>6}  {:<10} {}".format(
            c[:28], m["by_sd"], m["direction"],
            ("{:.1f}".format(m["skew_source"])
             if m.get("skew_source") is not None else "-"),
            "yes" if m["tail_shape_published"] else "NO",
            "yes" if m.get("integral") else "no"))
    skewed = sum(1 for _c, m in rows
                 if abs(m.get("skew_source") or 0) >= 1.0)
    unshaped = sum(1 for _c, m in rows
                   if not m["tail_shape_published"])
    print()
    print("{} of {} missed columns are skewed (|skew| >= 1); "
          "{} had NO tail shape published - those are the "
          "piecewise-CDF suspects. A miss that is neither is a "
          "different mechanism, still to be found.".format(
              skewed, len(rows), unshaped))


def surfaces_view(run):
    """Why each published interaction surface lives or dies.

    The surface fix cured the fixture and moved NOTHING on the
    extract - same 2/6, identical gate line - so the mechanism
    there is different, and this reads it out of the run directory
    instead of a third guessed fixture: per surface, the pair, the
    blueprint parents, the fidelity gap, and every trimmed edge
    touching the child (with lag parents flagged, because lag
    columns exist only during discovery and a refinement that
    requires ALL original parents present skips such records
    entirely)."""
    print("surfaces_view on {}".format(run))
    bp = _load(run, "blueprint.json")
    fid = _load(run, "fidelity.json")
    gaps = {}
    for r in (fid.get("surfaces") or {}).get("surfaces") or []:
        gaps[(r["child"], tuple(sorted(r["pair"])))] = r
    dropped = ((fid.get("generation") or {})
               .get("edges_dropped") or [])
    by_child = {}
    for d in dropped:
        by_child.setdefault(d.get("child"), []).append(d)
    n_pub = 0
    for rel in (bp.get("relationships") or []):
        it = (rel.get("evidence") or {}).get("interaction")
        pair = list((it or {}).get("pair") or [])
        if len(pair) != 2:
            continue
        n_pub += 1
        child = rel.get("child")
        g = gaps.get((child, tuple(sorted(pair))))
        print()
        print("{} <- {} x {}".format(child, pair[0], pair[1]))
        if g:
            print("  gap {} sd, tracks {}".format(
                g.get("gap_sd"), g.get("tracks")))
        else:
            # SAY WHY, not just "?": on the real extract 66 of 72
            # published surfaces are unmeasurable because they
            # touch token-indicator columns that exist only inside
            # the search - a bare question mark read as a bug.
            from synthkit import sets as _sets
            if _sets.is_scaffolding(child) or any(
                    _sets.is_scaffolding(x) for x in pair):
                print("  not measured - touches a token-indicator "
                      "column, and the OUTPUT carries too few of "
                      "that token to fill a grid. It is NOT that "
                      "the column cannot be rebuilt: the "
                      "blueprint's own vocabulary rebuilds it, "
                      "which is why this bucket went from 67 to a "
                      "handful")
            else:
                print("  not measured - a column is absent from "
                      "the written files, or the overlapping "
                      "cells were too thin to bin")
        ps = rel.get("parents") or []
        missing = [x for x in pair if x not in ps]
        print("  blueprint parents: {}{}".format(
            ", ".join(ps),
            "  [PAIR MEMBER NOT IN PARENTS: {}]".format(
                ", ".join(missing)) if missing else ""))
        for d in by_child.get(child, []):
            lost = (d.get("parents_lost")
                    or [p for p in (d.get("parents") or [])
                        if p not in (d.get("kept_parents")
                                     or [])])
            lags = [p for p in lost if "_lag" in p]
            pair_lost = [p for p in lost if p in pair]
            print("  trimmed edge: lost {}{}{}".format(
                ", ".join(lost) or "(none)",
                "  [LAG PARENT LOST: {}]".format(
                    ", ".join(lags)) if lags else "",
                "  [SURFACE PAIR MEMBER LOST: {}]".format(
                    ", ".join(pair_lost)) if pair_lost else ""))
    if not n_pub:
        print("no interaction surfaces are published in this "
              "blueprint")
    print()
    # THE CATEGORIES MUST BE DISJOINT, OR THE RESIDUAL GOES
    # NEGATIVE AND THE TOTAL STILL ADDS UP. This counted every
    # surface touching a token indicator as unmeasurable - true
    # while `compare` refused those columns, and false since the
    # blueprint's own vocabulary is used to rebuild them. On a
    # real run it then read "73 published: 47 measured, 64 on
    # token-indicator columns, -38 otherwise unmeasured": the
    # arithmetic balanced because one term was NEGATIVE, which is
    # the totals rule satisfied by a number that cannot exist.
    #
    # A surface is MEASURED, or it is token-touching AND
    # unmeasured, or it is unmeasured for some other reason.
    # Measured wins, because a surface that WAS compared is not an
    # instrument gap whatever its columns are called.
    from synthkit import sets as _sets2
    # `gaps` is keyed by (child, sorted pair) - the same key the
    # measurement records, so a surface is matched exactly rather
    # than by its child alone.
    n_tok = 0
    for rel in (bp.get("relationships") or []):
        it2 = (rel.get("evidence") or {}).get("interaction")
        pr2 = list((it2 or {}).get("pair") or [])
        if len(pr2) != 2:
            continue
        child2 = rel.get("child") or ""
        if (child2, tuple(sorted(pr2))) in gaps:
            continue
        if _sets2.is_scaffolding(child2) or any(
                _sets2.is_scaffolding(x) for x in pr2):
            n_tok += 1
    _other = n_pub - len(gaps) - n_tok
    print("{} surface(s) published: {} measured, {} on "
          "token-indicator columns the OUTPUT carries too few of "
          "to bin, {} otherwise unmeasured. A "
          "pair member missing from the blueprint parents means "
          "the surface cannot fire in the FIRST pass; a lost lag "
          "parent means the cyclic refinement skips the record "
          "ENTIRELY.".format(
              n_pub, len(gaps), n_tok, _other))
    if _other < 0:
        print("  AND THAT RESIDUAL IS NEGATIVE, which cannot "
              "happen: the categories above have started "
              "overlapping and the total only balances because "
              "one term is impossible. Do not read these counts.")



def sizes_view(run, src):
    """Which numeric columns TRACK a set's size, and whether the
    engine holds them to it.

    The real extract's worst drift row reads
    `procedure_count <- procedures__n` at source 1.00, generated
    0.38 - a pair whose source correlation is PERFECT. The engine
    copies a count from its set when the two are EQUAL on >=99% of
    rows, so a column that tracks the size perfectly WITHOUT
    equalling it (a quantity rather than a token count, or a count
    taken before de-duplication) is never detected and is drawn a
    second time. Two independent draws of one quantity agree only
    by chance.

    Correlation and equality are not the same question, and the
    drift table asks one while the enforcement asks the other -
    which is exactly the kind of gap this file exists to close.
    The verdict per pair says which of the two mechanisms applies,
    so the next fix is chosen from a measurement rather than from
    a guess.

    Needs the SOURCE as well as the run: the question is whether
    something true of the source survived generation."""
    import pandas as pd
    from synthkit import sets as _sets

    gen_p = Path(run) / "generated.csv"
    if not gen_p.exists():
        sys.exit("no generated.csv in {}".format(run))
    if not src or not Path(src).exists():
        sys.exit("sizes needs the source csv: peek.py RUN sizes SRC.csv")
    # keep_default_na=False is how the pipeline reads a csv; the
    # default turns a present-and-empty set into NaN, which is a
    # different dataset.
    sdf = pd.read_csv(src, keep_default_na=False, low_memory=False)
    gdf = pd.read_csv(str(gen_p), keep_default_na=False,
                      low_memory=False)

    setcols = [c for c in sdf.columns
               if c in gdf.columns and _sets.looks_like_set(sdf[c])]
    print("set columns: {}".format(len(setcols)))
    if not setcols:
        print("  none - nothing to check")
        return
    rows = []
    for sc in setcols:
        sn = _sets.sizes_of(sdf[sc], ";")
        gn = _sets.sizes_of(gdf[sc], ";")
        for c in sdf.columns:
            if c == sc or c not in gdf.columns:
                continue
            sv = pd.to_numeric(sdf[c], errors="coerce")
            gv = pd.to_numeric(gdf[c], errors="coerce")
            sm = sn.notna() & sv.notna()
            if int(sm.sum()) < 100:
                continue
            try:
                sr = float(sv[sm].corr(sn[sm]))
            except Exception:
                continue
            if sr != sr or abs(sr) < 0.9:
                continue
            seq = float((sv[sm] == sn[sm]).mean())
            gm = gn.notna() & gv.notna()
            gr = (float(gv[gm].corr(gn[gm]))
                  if int(gm.sum()) >= 10 else float("nan"))
            geq = (float((gv[gm] == gn[gm]).mean())
                   if int(gm.sum()) >= 10 else 0.0)
            if seq >= 0.99 and geq >= 0.99:
                verdict = "IDENTITY KEPT"
            elif seq >= 0.99:
                verdict = "IDENTITY LOST"
            else:
                verdict = "TRACKS, NOT EQUAL"
            rows.append((verdict, c, sc, seq, sr, geq, gr))
    if not rows:
        print("  no numeric column tracks a set size at |r| >= 0.9")
        return
    order = {"IDENTITY LOST": 0, "TRACKS, NOT EQUAL": 1,
             "IDENTITY KEPT": 2}
    rows.sort(key=lambda r: (order.get(r[0], 3), -abs(r[4])))
    print("")
    print("  {:<18} {:<26} {:<14} {:>7} {:>7} {:>7} {:>7}".format(
        "verdict", "numeric column", "set", "src=", "src r",
        "gen=", "gen r"))
    for v, c, sc, seq, sr, geq, gr in rows:
        print("  {:<18} {:<26} {:<14} {:>6.0f}% {:>7.2f} "
              "{:>6.0f}% {:>7.2f}".format(
                  v, c[:26], sc[:14], seq * 100, sr, geq * 100, gr))
    print("")
    nlost = sum(1 for r in rows if r[0] == "IDENTITY LOST")
    ntrack = sum(1 for r in rows if r[0] == "TRACKS, NOT EQUAL")
    print("  {} identity LOST, {} track-but-not-equal, {} kept"
          .format(nlost, ntrack,
                  sum(1 for r in rows if r[0] == "IDENTITY KEPT")))
    if nlost:
        print("  IDENTITY LOST means the engine DETECTED an equality "
              "and did not hold it - an enforcement fault.")
    if ntrack:
        print("  TRACKS-BUT-NOT-EQUAL means the pair is perfectly "
              "related and NOT an equality, so the copy rule never "
              "fires: it is drawn twice. That is a DETECTION gap, "
              "and the fix is an affine rule, not a stricter one.")


def flat_view(run):
    """COLUMNS THE CONTRACT SAYS VARY AND THE FILE EMITS CONSTANT.

    Found because a real run printed scipy's ConstantInputWarning
    twice from inside spearmanr: an input array was constant, so the
    pair's rank correlation is undefined and the pair is SKIPPED. A
    skipped pair leaves the gate's denominator, so a column flattened
    to one value can cost fidelity and raise the percentage at the
    same time - which is the denominator trap wearing a defect.

    The duel page already names flattened columns, but only when BOTH
    engines ran; a single run had nothing that could say it. This
    reads the run directory alone, as every other view here does: the
    blueprint says what the column should hold, the written file says
    what it holds.

    A column the blueprint itself publishes as constant is NOT a
    defect and is counted separately, and so is a blueprint column
    absent from the file - set-token scaffolding is built for the
    search and dropped before the write. published == flat + varying
    + expected-constant + not-written, printed as an identity,
    because an omitted category is invisible in any other way."""
    import csv
    bp = _load(run, "blueprint.json")
    gp = Path(run) / "generated.csv"
    if not gp.exists():
        sys.exit("no generated.csv in {}".format(run))
    cols = bp.get("columns") or {}

    # DOES THE CONTRACT SAY THIS COLUMN VARIES? Read per marginal
    # kind rather than guessing from a single key - a `quantiles`
    # column carries no `levels` and a `list` carries neither.
    def declared_variety(spec):
        m = (spec or {}).get("marginal") or {}
        t = m.get("type")
        if t == "levels":
            return len(m.get("levels") or []), "levels"
        if t == "quantiles":
            v = m.get("v") or []
            return (2 if v and min(v) < max(v) else 1), "quantiles"
        if t == "list":
            return len(m.get("tokens") or []), "tokens"
        return None, t or "none"

    # one pass, and the per-column set STOPS GROWING at two: the
    # question is only whether the column varies, and a 47,000-row
    # file with high-cardinality columns would otherwise be held in
    # memory for no reason.
    with gp.open("r", encoding="utf-8", newline="") as fh:
        rd = csv.reader(fh)
        try:
            header = next(rd)
        except StopIteration:
            sys.exit("generated.csv in {} is empty".format(run))
        seen = [set() for _ in header]
        rows = 0
        for row in rd:
            rows += 1
            for i, v in enumerate(row[:len(header)]):
                if len(seen[i]) < 2:
                    seen[i].add(v)
    present = {c: seen[i] for i, c in enumerate(header)}

    flat, varying, expected, missing = [], [], [], []
    for c, spec in sorted(cols.items()):
        n, kind = declared_variety(spec)
        if c not in present:
            missing.append(c)
        elif len(present[c]) <= 1:
            if n is not None and n <= 1:
                expected.append(c)
            else:
                flat.append((c, n, kind,
                             sorted(present[c])[0] if present[c]
                             else ""))
        else:
            varying.append(c)

    print("generated.csv: {} rows, {} columns written"
          .format(rows, len(header)))
    print("blueprint publishes {} column(s)".format(len(cols)))
    print("")
    if flat:
        print("FLATTENED - the contract says these vary and the "
              "file holds ONE value. A pair on one of these is "
              "SKIPPED by the gate, not failed:")
        print("  {:<40} {:>9}  {:<10} {}"
              .format("column", "published", "kind", "the one value"))
        for c, n, kind, v in flat:
            print("  {:<40} {:>9}  {:<10} {!r}"
                  .format(str(c)[-40:], n, kind, str(v)[:24]))
    else:
        print("FLATTENED: none - every column the contract says "
              "varies does vary in the written file")
    print("")
    print("varying as published          {}".format(len(varying)))
    print("published AS constant         {}  (not a defect)"
          .format(len(expected)))
    print("not written to the file       {}  (set-token "
          "scaffolding is dropped before the write)"
          .format(len(missing)))
    print("FLATTENED                     {}".format(len(flat)))
    total = len(varying) + len(expected) + len(missing) + len(flat)
    print("                              ---")
    print("accounted for                 {} of {} published{}"
          .format(total, len(cols),
                  "" if total == len(cols) else "  <-- MISMATCH"))


def curves_view(run):
    """Which parents the contract names and publishes no curve for.

    WHY THIS EXISTS. A relationship can be published with a parent
    the sampler cannot use: `discover` skips a curve whose grid
    collapses, with a bare `continue`, and `blueprint` lists the
    parent anyway. Measured 2026-11-09 on a fixture, a set-token
    parent on 0.81% of rows drove its child by 7.2 sd, was found at
    importance 0.633, was published BY NAME at search cap 60 - and
    carried no curve, so the output held 1.4% of the relationship.
    Raising the cap did not help; a binary column taking a QUANTILE
    grid was the cause.

    A parent with no curve is the read-but-not-used fault in the one
    artifact generation reads, and nothing listed them.
    """
    bp = _load(run, "blueprint.json")
    if not bp:
        return
    rels = bp.get("relationships") or []
    rows = []
    for r in rels:
        eff = (r.get("evidence") or {}).get("effect") or {}
        for par in (r.get("parents") or []):
            rows.append((r.get("child"), par, par in eff,
                         "__has__" in str(par)))
    if not rows:
        print("this blueprint publishes no relationships, so there "
              "are no parents to ask about - that is a finding in "
              "itself, not an empty report")
        return
    miss = [x for x in rows if not x[2]]
    ind = [x for x in rows if x[3]]
    ind_miss = [x for x in rows if x[3] and not x[2]]
    print("PARENTS THE CONTRACT NAMES, AND WHETHER IT PUBLISHES A "
          "CURVE FOR THEM")
    print()
    print("  relationships published          {:>5}".format(len(rels)))
    print("  parent slots across them         {:>5}".format(len(rows)))
    print("  WITH a curve                     {:>5}".format(
        len(rows) - len(miss)))
    print("  with NO curve                    {:>5}".format(len(miss)))
    print("  of those, set-token indicators   {:>5}".format(
        len(ind_miss)))
    print()
    print("  indicator parent slots           {:>5}".format(len(ind)))
    if ind:
        print("  of those WITHOUT a curve         {:>5}  ({:.0f}%)".format(
            len(ind_miss), 100.0 * len(ind_miss) / len(ind)))
    else:
        print("  of those WITHOUT a curve           n/a - this "
              "blueprint has no indicator parent at all, so a "
              "curve fix for them cannot reach anything here")
    print()
    if miss:
        print("  every parent named without a curve "
              "(child <- parent):")
        for child, par, _h, _i in miss[:40]:
            print("    {:<34} <- {}".format(str(child)[:34], par))
        if len(miss) > 40:
            print("    ... and {} more".format(len(miss) - 40))
    else:
        print("  every published parent carries a curve")


# ONE IMPLEMENTATION, in the package, because the run REPORTS
# this now and a second copy here is how the two halves come to
# disagree - which this file records for token screening, the
# scaffolding predicate and both categorical draw paths.
from synthkit.marginals import (TOL as MARGINAL_TOL,   # noqa: E402
                                SEED as MARGINAL_SEED,
                                bands as _bands,
                                classify as classify_marginal,
                                total as _total,
                                shares as _shares)


def marginals_view(run, src_path):
    """Every numeric column at once, worst first.

    WHY THIS EXISTS SEPARATELY. The single-column view answers one
    question per round trip, and a round trip to the machine that
    holds the extract is minutes - so finding which columns are
    off their own published grid one at a time is the wrong shape
    of tool for the place it has to run.

    It also answers a question the single view cannot: whether a
    gap is ONE column or the engine. `procedure_quantity` was found
    by reading a gate table; nothing had asked whether the other
    thirty numerics are quietly in the same state.

    IT DELEGATES THE MEASUREMENT, and that is the load-bearing
    part. The first version re-derived the band distance here -
    `max(abs(a - b))` over the shares - so when the run block
    moved to a total variation distance over the same bands, this
    view and `fidelity.json` would have reported DIFFERENT numbers
    for the same column with nothing saying so. The view directly
    below this one carries a docstring about never reimplementing
    `_draw_numeric` for exactly that reason, and the distance got
    reimplemented three lines above it.
    """
    import pandas as pd
    from synthkit import marginals as _M

    bp = _load(run, "blueprint.json")
    genf = Path(run) / "generated.csv"
    if not genf.exists():
        sys.exit("STOPPED: this run has no generated.csv to read")
    g = pd.read_csv(genf, dtype=str, keep_default_na=False)
    src = pd.read_csv(src_path, dtype=str, keep_default_na=False)

    rec = None
    prov = Path(run) / "provenance.json"
    if prov.exists():
        try:
            rec = (json.loads(prov.read_text(encoding="utf-8"))
                   .get("source", {}).get("rows_read"))
        except Exception:
            rec = None
    if isinstance(rec, int) and rec != len(src):
        sys.exit("STOPPED: the run recorded {} source rows and this "
                 "source has {} - refusing rather than producing a "
                 "plausible number about the wrong file"
                 .format(rec, len(src)))

    blob = _M.measure(src, g, bp, limit=None)
    rows = blob["columns"]
    print("{} numeric column(s) compared, {} not comparable"
          .format(blob["columns_measured"], blob["not_comparable"]))
    print("  the distance is the SHARE OF MASS in the wrong band. "
          "`worst` is the largest single band, which is not the "
          "verdict - a small band cannot exceed the bar alone.")
    print("  grid-vs-source is the CONTRACT's error, grid-vs-output "
          "the SAMPLER's. They take opposite fixes.")
    print()
    print("  sorted by SOURCE-against-OUTPUT, which is what a "
          "recipient holds - the other two split the blame.")
    print()
    print("  {:<30} {:>9} {:>9} {:>11} {:>7} {:>7}  {}".format(
        "column", "grid~src", "grid~out", "SRC~OUT", "worst",
        "noise", "verdict"))
    for r in rows:
        print("  {:<30} {:>9.4f} {:>9.4f} {:>11.4f} {:>7.4f} "
              "{:>7.4f}  {}"
              .format(r["column"][:30], r["grid_vs_source"],
                      r["grid_vs_output"], r["source_vs_output"],
                      r["worst_band"], r.get("noise_floor") or 0.0,
                      r["verdict"]))
    print()
    # THE COLUMNS THAT COULD NOT BE READ. Printed, because an
    # absent measurement is not a clean one - and silently
    # dropping them would make a thin cohort look like a short
    # list of healthy columns.
    _np_rows = blob.get("no_power") or []
    if _np_rows:
        print("  NOT READ - at these row counts two samples of "
              "the SAME distribution differ by more than the "
              "{:.2f} bar, so any number would be noise:".format(
                  MARGINAL_TOL))
        for r in _np_rows:
            print("    {:<30} source {:>7} rows, output {:>7} "
                  "rows, noise floor {:.4f}".format(
                      r["column"][:30], r["rows_source"],
                      r["rows_output"], r["noise_floor"]))
        print()
    bad = [r for r in rows if r["verdict"] != "CLEAR"]
    print("  {} of {} column(s) are off their own published grid or "
          "their source by more than {:.2f}".format(
              len(bad), len(rows), MARGINAL_TOL))
    _held = blob["off_source"]
    print("  and {} of {} differ from the SOURCE in a recipient's "
          "hands by more than {:.2f}{}".format(
              len(_held), len(rows), MARGINAL_TOL,
              ": " + ", ".join(_held[:8]) if _held else ""))
    gen = blob["sampler_faults"]
    if gen:
        print("  {} of those are the SAMPLER's: {}".format(
            len(gen), ", ".join(gen[:8])))
    else:
        print("  NONE of them is the sampler's - every gap here is "
              "the published grid, which is a different fix.")


def marginal_view(run, col, src_path):
    """Source vs the PUBLISHED GRID vs the output, on one ruler.

    WHY THIS EXISTS. `procedure_quantity` on the real extract puts
    93% of rows on the single value 1 where the source puts 53%,
    and the whole body between 2 and 24 - 42% of the source's rows
    - is almost absent. That is the largest named fidelity gap on
    the file, and it has had two incompatible explanations on the
    record without anybody measuring which applies.

    The question splits cleanly in two and nothing was asking it
    that way: does the PUBLISHED grid already imply the wrong
    shape, or does the grid imply the right shape and something
    after the draw move it? Those take different fixes, and a
    third guess is not worth making - the same reason
    `cooc_landing` and `quota_landing` exist.

    The grid column is drawn through the engine's OWN
    `_draw_numeric`, never a reimplementation here: two halves
    that each decide what a published grid means is how they come
    to disagree, which this file records for token screening, the
    scaffolding predicate and both categorical draw paths.
    """
    import numpy as np
    import pandas as pd
    from synthkit.generate import _draw_numeric

    bp = _load(run, "blueprint.json")
    spec = ((bp.get("columns") or {}).get(col) or {})
    m = (spec.get("marginal") or {})
    if not spec:
        sys.exit("STOPPED: {} is not a column of this run's "
                 "blueprint".format(col))
    if m.get("type") != "quantiles":
        sys.exit("STOPPED: {} publishes a `{}` marginal, not "
                 "quantiles - this compares a NUMERIC grid against "
                 "the shape it produces".format(col, m.get("type")))

    genf = Path(run) / "generated.csv"
    if not genf.exists():
        sys.exit("STOPPED: this run has no generated.csv to read")
    g = pd.read_csv(genf, dtype=str, keep_default_na=False)
    src = pd.read_csv(src_path, dtype=str, keep_default_na=False)

    # THE SAME GUARD THE OTHER TOOLS CARRY. A run measured against
    # an extract that never produced it looks entirely reasonable,
    # and this project lost a session to exactly that.
    rec = None
    prov = Path(run) / "provenance.json"
    if prov.exists():
        try:
            rec = (json.loads(prov.read_text(encoding="utf-8"))
                   .get("source", {}).get("rows_read"))
        except Exception:
            rec = None
    if isinstance(rec, int) and rec != len(src):
        sys.exit("STOPPED: the run recorded {} source rows and this "
                 "source has {} - refusing rather than producing a "
                 "plausible number about the wrong file"
                 .format(rec, len(src)))

    if col not in g.columns:
        sys.exit("STOPPED: {} is in the blueprint but NOT in "
                 "generated.csv - search scaffolding is dropped "
                 "before the file is written, and nobody can read "
                 "a column that is not there".format(col))
    if col not in src.columns:
        sys.exit("STOPPED: {} is not a column of the source "
                 "handed in".format(col))

    sv = pd.to_numeric(src[col], errors="coerce").dropna().to_numpy(float)
    gv = pd.to_numeric(g[col], errors="coerce").dropna().to_numpy(float)
    if len(sv) < 50 or len(gv) < 50:
        sys.exit("STOPPED: too few present numeric values to "
                 "compare ({} source, {} generated)"
                 .format(len(sv), len(gv)))

    rng = np.random.default_rng(MARGINAL_SEED)
    grid = np.asarray(_draw_numeric(m, rng.random(len(gv))),
                      dtype=float)

    edges = _bands(sv)
    ss, gs, xs = (_shares(sv, edges), _shares(grid, edges),
                  _shares(gv, edges))
    d_gs = _total(gs, ss)
    d_gx = _total(gs, xs)
    # WHAT A RECIPIENT ACTUALLY HOLDS. The two gaps above split
    # the blame between the contract and the sampler, which is
    # what they are for - but neither is the thing a researcher
    # receives. A sampler that faithfully reproduces a wrong grid
    # reads CLEAN on grid-vs-output and is still wrong in their
    # hands, and a fix aimed at the SOURCE rather than the grid
    # improves this while making grid-vs-output look worse.
    d_sx = _total(ss, xs)
    verdict = classify_marginal(d_gs, d_gx)

    mode = float(pd.Series(sv).mode().iloc[0])
    print("{}   source {} present, generated {} present"
          .format(col, len(sv), len(gv)))
    print("  the GRID column is what this run's OWN published "
          "marginal produces through the engine's own draw -")
    print("  presence and relationships are not applied to it, so "
          "it is the column's published SHAPE alone.")
    print()
    print("  {:<22} {:>9} {:>9} {:>9}".format(
        "band", "source", "grid", "output"))
    lo = "-inf"
    for i, e in enumerate(edges):
        print("  {:<22} {:>9.4f} {:>9.4f} {:>9.4f}".format(
            "{} to {:g}".format(lo, e), ss[i], gs[i], xs[i]))
        lo = "{:g}".format(e)
    print("  {:<22} {:>9.4f} {:>9.4f} {:>9.4f}".format(
        "over {}".format(lo), ss[-1], gs[-1], xs[-1]))
    print()
    print("  {:<22} {:>9.4f} {:>9.4f} {:>9.4f}".format(
        "exactly {:g} (the mode)".format(mode),
        float(np.mean(sv == mode)), float(np.mean(grid == mode)),
        float(np.mean(gv == mode))))
    print("  {:<22} {:>9.4f} {:>9.4f} {:>9.4f}".format(
        "mean", float(np.mean(sv)), float(np.mean(grid)),
        float(np.mean(gv))))
    print("  {:<22} {:>9.4f} {:>9.4f} {:>9.4f}".format(
        "spread", float(np.std(sv)), float(np.std(grid)),
        float(np.std(gv))))
    print()
    print("  worst band gap, grid against source   {:.4f}".format(d_gs))
    print("  worst band gap, grid against output   {:.4f}".format(d_gx))
    print("  worst band gap, SOURCE against output {:.4f}  <- what "
          "a recipient holds".format(d_sx))
    print("  at a {:.2f} share bar: {}".format(MARGINAL_TOL, verdict))
    if verdict == "CONTRACT":
        print("  THE PUBLISHED GRID cannot represent the source's "
              "shape, and the output reproduces the grid. The fix "
              "is in the blueprint, not the sampler.")
    elif verdict == "GENERATION":
        print("  THE GRID MATCHES THE SOURCE and something after "
              "the draw moved it. The fix is in the sampler, not "
              "the contract.")
    elif verdict == "BOTH":
        print("  TWO FAULTS, not one: the grid misses the source "
              "AND the output misses the grid. Fixing either alone "
              "leaves a gap.")
    else:
        print("  this column's published shape and its output both "
              "track the source - whatever is wrong here, it is "
              "not the marginal.")
    print()
    print("  NOT measured here: whether the RELATIONSHIPS on this "
          "column survived. A column can have an exact marginal "
          "and the wrong value on every row.")


def _sweep_blocked(bp, col, cols):
    """True when NO relationship record for `col` has all of its
    parents among the columns the engine draws.

    That is the sweep's own `usable` test, read off the blueprint:
    it needs one record whose every parent is in `out`, and `out`
    holds the modelled columns.
    """
    recs = [r for r in (bp.get("relationships") or [])
            if r.get("child") == col]
    if not recs:
        return True
    return not any(all(p in cols for p in (r.get("parents") or []))
                   for r in recs)


def classify_rerank(bp):
    """(rows, would_be_reranked) for a blueprint.

    PURE over the blueprint, so the suite can hold it against what
    `generate` ACTUALLY re-ranks. A predictor that drifts from the
    engine is the two-halves fault, and this one exists precisely
    to decide whether a forty-five minute arm is worth running.
    """
    from synthkit.generate import _order
    cols = bp.get("columns") or {}
    res = _order(bp, refine=True)
    cyclic = res[5] if len(res) > 5 else {}
    parents = res[1] if len(res) > 1 else {}
    rows, would = [], []
    for c, spec in cols.items():
        spec = spec or {}
        m = spec.get("marginal") or {}
        if m.get("type") != "quantiles":
            continue
        cov = spec.get("coverage")
        child = bool(parents.get(c))
        cyc = c in cyclic
        pat = spec.get("level") == "patient"
        pres = bool(spec.get("presence"))
        if not child:
            why = "not a relationship child - its draw IS the marginal"
        elif cyc and not _sweep_blocked(bp, c, cols):
            why = "CYCLIC - the refinement sweep already owns it"
        elif cyc:
            # THE SWEEP NEEDS EVERY PARENT OF SOME RECORD to be a
            # column the engine actually draws, and a blueprint can
            # name one it never modelled - the token cap publishes
            # more indicators than it draws. Such a column gets no
            # re-rank from the sweep, and the acyclic path skips it
            # for being cyclic. Neither, so its marginal is
            # whatever curve plus noise produced.
            why = ("CYCLIC but the sweep CANNOT refine it - a "
                   "parent is not modelled - WOULD BE RE-RANKED")
            would.append(c)
        elif pat:
            why = "patient-level - a per-row re-rank would break constancy"
        elif cov is None or float(cov) >= 1.0:
            why = "fully covered - there is no mask to survive"
        else:
            why = "WOULD BE RE-RANKED"
            would.append(c)
        rows.append((c, cov, child, cyc, pres, why))
    return rows, would


def dynamics_view(run):
    """Which within-patient properties were lost, and how.

    `dynamics tracked` is the worst criterion on the real extract -
    35 of 92 - and it recorded a bare COUNT, so nobody could say
    which patient properties went or by how much. Losing a
    patient's course and INVENTING one are different faults with
    different fixes, and a magnitude alone cannot tell them apart.
    """
    fid = _load(run, "fidelity.json")
    summ = (fid or {}).get("summary") or {}
    rows = summ.get("dynamics_detail")
    if rows is None:
        print("this run PREDATES the per-column dynamics account - "
              "it carries only {} of {} tracked. Absent is not the "
              "same fact as nothing missed."
              .format(summ.get("dynamics_ok"),
                      summ.get("dynamics_compared")))
        return
    print("dynamics tracked {} of {}".format(
        summ.get("dynamics_ok"), summ.get("dynamics_compared")))
    if not rows:
        print("  nothing missed - every graded within-patient "
              "property came through inside the band")
        return
    own = [r for r in rows if not r.get("scaffolding")]
    scaf = [r for r in rows if r.get("scaffolding")]
    print("  {} miss(es) listed, {} on YOUR columns and {} on "
          "set-token scaffolding dropped before the file is "
          "written".format(len(rows), len(own), len(scaf)))
    print()
    print("  {:<34} {:<13} {:>8} {:>8} {:>8}  {}".format(
        "column", "measure", "source", "output", "delta", "how"))
    for r in (own + scaf):
        print("  {:<34} {:<13} {:>8.3f} {:>8.3f} {:>+8.3f}  {}{}"
              .format(r["column"][:34], r["measure"][:13],
                      r["source"], r["generated"], r["delta"],
                      r["direction"],
                      "  (scaffolding)" if r.get("scaffolding")
                      else ""))
    print()
    _fl = [r for r in rows if r["direction"] == "flattened"]
    print("  {} of {} are FLATTENED - the patient's course is "
          "weaker than the source's - and {} are OVERSTATED, which "
          "is structure the engine invented".format(
              len(_fl), len(rows), len(rows) - len(_fl)))
    print("  Those take different fixes: a flattened course wants "
          "persistence the draw is not carrying; an overstated one "
          "wants less.")


def copies_view(run):
    """Which columns are COPIED rather than drawn.

    An exact `==` constraint between two ordinary columns means one
    column IS the other, and generation carries that by copying -
    so the copied column is never drawn from its own published
    marginal at all. Everything measured about that marginal then
    describes a column nobody sampled: it will sit off its own
    grid by however much its SOURCE column does, and the refinement
    sweep will skip it for want of a marginal draw.

    Asked of the engine's OWN `_copy_partners`, never re-derived
    here - two halves that each decide what a constraint means is
    how they come to disagree.
    """
    from synthkit.generate import _copy_partners
    bp = _load(run, "blueprint.json")
    pairs = _copy_partners(bp)
    if not pairs:
        print("no column is copied from another on this blueprint - "
              "every numeric is drawn from its own marginal.")
        return
    print("{} column(s) are COPIED, not drawn:".format(len(pairs)))
    print()
    for rhs, lhs in sorted(pairs.items()):
        print("  {:<34} := {}".format(rhs[:34], lhs))
    print()
    print("  A copied column is NEVER drawn from its own published "
          "marginal, so a gap between that marginal and the output "
          "is not a sampler fault on THAT column - it is whatever "
          "its source column does. The refinement sweep also skips "
          "it, reporting `no marginal draw was kept for it`.")


def sweep_view(run):
    """What the refinement sweep actually did, per column.

    A cyclic column the sweep skips gets no re-rank from the
    acyclic path either - that path skips it for being cyclic - so
    its marginal is whatever curve plus noise produced. The run
    used to report a COUNT, and a count cannot say WHICH.
    """
    fid = _load(run, "fidelity.json")
    gen = (fid or {}).get("generation") or {}
    if "refinement_refined" not in gen:
        print("this run PREDATES the per-column sweep account - it "
              "carries only `refinements_applied` = {}. Absent is "
              "not the same fact as nothing skipped."
              .format(gen.get("refinements_applied")))
        return
    done = gen.get("refinement_refined") or []
    skip = gen.get("refinement_skipped") or {}
    print("cyclic columns: {} refined, {} SKIPPED (of {} reported "
          "cyclic)".format(len(done), len(skip),
                           gen.get("cyclic_columns")))
    print()
    if skip:
        print("  SKIPPED - nothing re-ranks these, so their "
              "marginal is whatever the curve plus noise made:")
        for c, why in sorted(skip.items()):
            print("    {:<34} {}".format(c[:34], why[:90]))
    else:
        print("  nothing was skipped - every cyclic column was "
              "refined, so a marginal that is off its grid is NOT "
              "explained by the sweep declining to run.")
    print()
    rr = gen.get("presence_reranked")
    if rr is None:
        print("  the presence-aware pass did not run on this "
              "build or was off")
    else:
        print("  the presence-aware pass re-ranked: {}"
              .format(", ".join(rr) if rr else "(nothing)"))


def rerank_view(run):
    """Which columns `--presence-rerank` would actually touch.

    WHY THIS EXISTS. The pass re-ranks a partially-covered ACYCLIC
    numeric child among the rows that survive its presence mask.
    Three of those words are conditions, and a paired real arm is
    about forty-five minutes of machine time - so whether the flag
    can move anything on a given run is worth knowing from the
    blueprint, which costs seconds.

    This project has twice run an arm that could not reach the
    thing it was testing and read the zero as evidence.
    """
    bp = _load(run, "blueprint.json")
    rows, would = classify_rerank(bp)
    cols = bp.get("columns") or {}
    print("{} numeric column(s) with a published grid"
          .format(len(rows)))
    print()
    print("  {:<34} {:>6} {:>6} {:>5} {}".format(
        "column", "cov", "child", "pres", "verdict"))
    for c, cov, child, cyc, pres, why in sorted(rows):
        print("  {:<34} {:>6} {:>6} {:>5} {}".format(
            c[:34], "-" if cov is None else "{:.2f}".format(float(cov)),
            "yes" if child else "no", "yes" if pres else "no", why))
    print()
    print("  the pass would re-rank {} column(s)".format(len(would)))
    if would:
        print("  namely: {}".format(", ".join(sorted(would)[:12])))
    else:
        # AN EMPTY ANSWER IS A FINDING, not a blank. A flag that
        # cannot touch anything on this run must not be measured
        # as though it had.
        print("  NOTHING - on this blueprint the flag cannot change "
              "a single value, so a paired arm would measure only "
              "the seed.")
    print()
    print("  `pres` says the column has an informative presence "
          "model. Where it does NOT, the mask is drawn flat and "
          "the surviving rows are already a fair sample - so the "
          "pass is correct but has little to do.")


def main():
    # `--help` IS A QUESTION, NOT A MISTAKE. This printed the
    # docstring and exited 2 for it, which is the code a shell
    # script reads as failure - and a help request that reports
    # failure is the sort of thing that makes a wrapper abort.
    # Asking is answered with exit 0; an incomplete command still
    # exits 2.
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__)
        return 0
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
    elif what == "centre":
        centre_view(run)
    elif what == "surfaces":
        surfaces_view(run)
    elif what == "flat":
        flat_view(run)
    elif what == "sizes":
        sizes_view(run, arg)
    elif what == "marginal":
        arg2 = sys.argv[4] if len(sys.argv) > 4 else None
        if not arg or not arg2:
            sys.exit("marginal needs a COLUMN and the source CSV")
        marginal_view(run, arg, arg2)
    elif what == "curves":
        curves_view(run)
    elif what == "rerank":
        rerank_view(run)
    elif what == "sweep":
        sweep_view(run)
    elif what == "copies":
        copies_view(run)
    elif what == "dynamics":
        dynamics_view(run)
    elif what == "marginals":
        if not arg:
            sys.exit("marginals needs the source CSV")
        marginals_view(run, arg)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
