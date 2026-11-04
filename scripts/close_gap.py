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


def surface(run, src_path):
    """Why the gate counts more pairs than a re-measure does.

    THE TWO SURFACES ARE NOT THE SAME, and until this was
    measured the difference was a hypothesis. The gate's
    `fidelity.json` was written by a run that expanded set
    tokens into `__has__` indicators with whatever cap that
    run was given; any later `compare()` re-derives those
    indicators through `prepare()`, which applies the module
    constant `EXPAND_CAP`. A pair whose indicator cannot be
    re-derived is SKIPPED, so the re-measure relates fewer
    pairs - and comparing one percentage against the other is
    comparing two questions.

    This counts both, so the claim rests on numbers rather
    than on reading the code.
    """
    import pandas as pd
    import synthkit.sets as _S
    from synthkit.discover import prepare
    bp = json.loads((Path(run) / "blueprint.json")
                    .read_text(encoding="utf-8"))
    fid = json.loads((Path(run) / "fidelity.json")
                     .read_text(encoding="utf-8"))
    df = pd.read_csv(src_path, dtype=str, keep_default_na=False,
                     encoding="utf-8-sig", low_memory=False)
    # THE SOURCE MUST BE THE ONE THIS RUN WAS BUILT FROM.
    # Nothing checked it, and a whole night of measurement
    # went against a different extract - 23,092 rows passed
    # where the run had recorded 55,428 - producing numbers
    # that looked perfectly reasonable and meant nothing.
    prov = Path(run) / "provenance.json"
    if prov.exists():
        try:
            _p = json.loads(prov.read_text(encoding="utf-8"))
        except ValueError:
            _p = {}
        _want = _p.get("rows_read")
        if isinstance(_p.get("source"), dict):
            _want = _want or _p["source"].get("rows_read")
        if _want and int(_want) != len(df):
            print("STOP: {} has {} rows but this run recorded "
                  "reading {}. That is not the extract this "
                  "run was built from - every number below "
                  "would be a comparison against a source "
                  "that never produced this output."
                  .format(Path(src_path).name, len(df),
                          int(_want)))
            return 2
    gb = bp.get("id_column") or bp.get("group_by") or "person_id"
    X = prepare(df, gb, ordinals=bp.get("ordinals") or None)[0]
    made = {c for c in X.columns if _S.HAS in c}

    wanted = set()
    for rel in (bp.get("relationships") or []):
        ch = rel.get("child")
        if ch and _S.HAS in str(ch):
            wanted.add(str(ch))
        for par in (rel.get("parents") or []):
            if _S.HAS in str(par):
                wanted.add(str(par))

    pub = {}
    for c, v in (bp.get("columns") or {}).items():
        mg = v.get("marginal") or {}
        if (mg.get("type") or "") == "list":
            pub[c] = len(mg.get("tokens") or [])

    print("WHY THE TWO SURFACES DIFFER  {}".format(
        Path(run).name))
    print("  the run RECORDED {} related pairs".format(
        (fid.get("relationships") or {}).get("compared")))
    print("  a re-measure through prepare() rebuilds {} token "
          "indicator column(s)".format(sayn(len(made))))
    print("  the blueprint's relationships NAME {} token "
          "indicator column(s)".format(sayn(len(wanted))))
    missing = sorted(wanted - made)
    print("  of those, {} CANNOT be rebuilt, so every pair "
          "touching them is skipped".format(sayn(len(missing))))
    print("  EXPAND_CAP is {} per set column, and the "
          "blueprint published:".format(sayn(_S.EXPAND_CAP)))
    for c in sorted(pub):
        print("      {:<24} {} token(s) published".format(
            c, sayn(pub[c])))
    if missing:
        print("  examples that cannot be rebuilt:")
        for m in missing[:6]:
            print("      {}".format(m))
    # AND THE SECOND HALF: IS THE TOKEN IN THE OUTPUT AT ALL?
    #
    # Rebuilding the indicator from the SOURCE is only half
    # the job - a pair is measured on both tables, so a token
    # the generated file never emits gives an all-zero
    # indicator there and the pair is skipped whatever the
    # cap does. Raising the cap recovered 57 pairs to 69 on
    # the real extract, not to 137, so the cap was A cause
    # and not the main one. This counts the other half.
    gen_p = Path(run) / "generated.csv"
    if gen_p.exists():
        g = pd.read_csv(gen_p, dtype=str,
                        keep_default_na=False,
                        encoding="utf-8-sig", low_memory=False)
        print("")
        print("  AND WHETHER THE OUTPUT CARRIES THE TOKEN AT "
              "ALL - a pair is measured on BOTH tables, so a "
              "token the generated file never emits is "
              "unmeasurable however the source is expanded:")
        for c in sorted(pub):
            if c not in g.columns or c not in df.columns:
                continue
            sep = ";"
            def toks(fr):
                out = set()
                for v in fr[c].fillna("").astype(str):
                    for t in v.split(sep):
                        t = t.strip()
                        if t:
                            out.add(t)
                return out
            st, gt = toks(df), toks(g)
            # THE OUTPUT COUNT CROSSES IN WORDS. It is the
            # number that decides whether a cap is binding,
            # and the digits channel off that screen turns a
            # 0 into an @.
            print("      {:<22} published {:>5}   in source "
                  "{:>5}   in OUTPUT {}".format(
                      c, pub[c], len(st), sayn(len(gt))))

        # AND WHY A PUBLISHED TOKEN CAN BE ABSENT.
        #
        # The independent per-token draw places round(p * n)
        # rows per token, which makes the marginal shares
        # exact by construction - and ROUNDS ANY TOKEN WHOSE
        # EXPECTED COUNT IS UNDER A HALF TO ZERO ROWS. That
        # is not sampling luck, it is deterministic erasure:
        # the same tokens vanish on every seed. This counts
        # how many published tokens that rule removes, which
        # separates "too rare to appear" from "a draw that
        # cannot represent it".
        nrows = len(g)
        print("")
        print("  AND WHY: the per-token draw places "
              "round(p x n) rows, so any published token with "
              "an expected count under a HALF gets ZERO rows "
              "- deterministically, on every seed:")
        tot_lost = 0
        for c in sorted(pub):
            mg = ((bp.get("columns") or {}).get(c)
                  or {}).get("marginal") or {}
            ps = [float(t.get("p") or 0.0)
                  for t in (mg.get("tokens") or [])]
            lost = [q for q in ps if q * nrows < 0.5]
            tot_lost += len(lost)
            print("      {:<22} {:>5} of {:>5} published "
                  "round to zero rows".format(
                      c, len(lost), len(ps)))
        print("      {:<22} {:>5} in total"
              .format("", tot_lost))

    # THE CONCLUSION, STATED ONLY IF THE NUMBERS SUPPORT IT.
    if missing and len(made) <= _S.EXPAND_CAP * max(1, len(pub)):
        print("")
        print("  ESTABLISHED: the re-measure rebuilds at most "
              "{} indicators per set column because "
              "`prepare()` applies EXPAND_CAP, while the run "
              "that wrote the blueprint expanded more. The "
              "two percentages are over DIFFERENT pair sets "
              "and must not be compared.".format(
                  sayn(_S.EXPAND_CAP)))
    elif not missing:
        print("")
        print("  The blueprint names no indicator that cannot "
              "be rebuilt, so the cap is NOT the cause here - "
              "whatever the difference is, it is somewhere "
              "else.")
    return 0


def main(argv):
    if not argv:
        print(__doc__)
        print("usage: close_gap.py RUNDIR")
        return 2
    if "--surface" in argv:
        rest = [x for x in argv if x != "--surface"]
        if len(rest) < 2:
            print("usage: close_gap.py RUNDIR --surface SRC.csv")
            return 2
        return surface(rest[0], rest[1])
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

    # FADE AND OVERSHOOT ARE DIFFERENT FAULTS, and one count
    # hides that. A relationship the generator LOST is a draw
    # or a trim problem; one it INVENTED beyond the source is
    # a tier over-enforcing, and this project already records
    # the sweeps over-correlating dense neighbours (0.989
    # against a source 0.967). Treating them as one number
    # sends a fix at the wrong half.
    over = [r for r in miss
            if abs(float(r.get("generated") or 0))
            > abs(float(r.get("source") or 0))]
    fade = [r for r in miss if r not in over]
    print("")
    print("  FADED (the generator lost it):     {}"
          .format(sayn(len(fade))))
    print("  OVERSHOT (the generator invented): {}"
          .format(sayn(len(over))))

    # AND WHETHER THE PAIR SITS IN A CYCLE, because that is
    # the suspect for overshoot: the refinement sweeps pull
    # cyclic neighbours together, which helps a ring whose
    # true adjacent correlation is high and overshoots one
    # whose source correlation is only moderate.
    kids = {}
    for rel in (bp.get("relationships") or []):
        for par in (rel.get("parents") or []):
            kids.setdefault(par, set()).add(rel.get("child"))

    def cyclic(r):
        a_, b_ = r.get("child"), r.get("parent")
        return (b_ in kids.get(a_, ())) and (a_ in kids.get(b_, ()))

    if over:
        n_cyc = sum(1 for r in over if cyclic(r))
        print("    of the overshot, {} sit in a CYCLE (both "
              "directions in the graph)".format(sayn(n_cyc)))
        print("    their source correlations: {}".format(
            ", ".join("{:+.2f}".format(float(r.get("source") or 0))
                      for r in over[:8])))
        print("    and what they became:       {}".format(
            ", ".join("{:+.2f}".format(
                float(r.get("generated") or 0))
                for r in over[:8])))
    # THE SAME QUESTION OF THE PAIRS THAT PASSED, which is the
    # control: if cyclic pairs overshoot generally, the ones
    # inside 0.2 should lean high too.
    good = [r for r in strong if r not in miss]
    gc = [r for r in good if cyclic(r)]
    if gc:
        lean = sum(1 for r in gc
                   if abs(float(r.get("generated") or 0))
                   > abs(float(r.get("source") or 0)))
        print("    CONTROL - of {} cyclic pairs that DID pass, "
              "{} also lean high".format(sayn(len(gc)),
                                         sayn(lean)))

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
