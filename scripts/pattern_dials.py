"""Every pattern found, and whether a dial can reach it.

THE ARCHITECTURE THIS SERVES. The goal is that a user sees every
pattern the tool found and chooses, PER PATTERN, how to distort
it. That is a better design than what exists, and tonight's
measurements say why: the dials today sit on COLUMNS - on
marginals in the blueprint - and generation frequently never
reads them. A column whose parents survive the cycle trim is
generated from THEIR CURVES, so a shift written into its
marginal is never read; on the real extract every well-covered
numeric is such a column, which makes the value dials inert
there.

A pattern-level dial is attached to the thing discovery
actually found and generation actually applies.

AND THE LAYER IS DECIDED BY EVIDENCE TOO. The dials that bite
on real data are the release controls - re-pointing and
suppression - because they act on the OUTPUT, after every tier
has run, as PERMUTATIONS. Nothing downstream can overwrite
them, each column's distribution is preserved exactly, and the
change is exactly invertible, which is what makes the cipher
possible. The marginal dials fail for precisely the opposite
reason.

So this inventory asks one question of every pattern: CAN A
DIAL REACH IT, and is that dial invertible? It is the substrate
for per-pattern control, and it is useful before any of that
exists, because it says plainly which patterns are currently
beyond reach.

WHAT IT REFUSES TO DO IS FLATTER THE COUNT. A pattern nobody
can dial is listed as UNREACHABLE by name. A pattern that
cannot be MEASURED at this cohort size is listed as
UNDECIDABLE rather than quietly absent - three-way structure
needs roughly 3,200 patients on the recorded power curve and
the real extract has 800, so "no three-way patterns found" is
a statement about the cohort, not about the data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit import sets as S                          # noqa: E402

# Which pattern kinds a dial can actually reach today. A kind
# not here is REFUSED BY NAME rather than silently accepted -
# an operator who dials an effect curve and gets a shrug has
# been told nothing.
DIALABLE = ("relationship", "effect_curve", "identity")

# Which dial, if any, reaches each kind of pattern - and whether
# that dial is a PERMUTATION, which is what makes it invertible
# through the cipher and what makes it immune to being
# overwritten by a later generation tier.
REACH = {
    "relationship": ("CHILD<-PARENT.strength", "blueprint",
                     False),
    "pair_correlation": ("--poison A~B=RHO", "output", True),
    "column_presence": ("COL.coverage", "blueprint", False),
    "column_centre": ("COL.shift", "blueprint", False),
    "column_spread": ("COL.scale", "blueprint", False),
    "within_patient": ("COL.persistence", "blueprint", False),
    "missing_clustering": ("COL.missing_clustering",
                           "blueprint", False),
    "effect_curve": (None, None, False),
    "interaction_surface": (None, None, False),
    "identity": (None, None, False),
    "set_token": (None, None, False),
    "whole_column": ("--suppress A~B", "output", True),
}


def pattern_id(kind, columns):
    """A stable id for a pattern, derived from its CONTENT.

    POSITIONAL IDS WOULD BE A TRAP. `p07` meaning the seventh
    row of a list shifts the moment discovery finds one more
    pattern, so a user who dialed p07 yesterday dials something
    else today with no error and no sign. The id is a short
    hash of the kind and the columns involved, so it names the
    same pattern across runs and a pattern that has genuinely
    gone simply is not found.
    """
    import hashlib
    cols = [c for c in columns if c]
    # DIRECTION IS PART OF THE PATTERN. Sorting the columns
    # gave `x <- y` and `y <- x` the SAME id - two different
    # patterns, one identifier, and a dial that would act on
    # whichever the lookup happened to find first. That is the
    # exact hazard content-derived ids exist to avoid, so only
    # a SYMMETRIC kind is sorted.
    if kind in ("identity",):
        cols = sorted(cols)
    key = "{}|{}".format(kind, "|".join(cols))
    return "p" + hashlib.sha1(
        key.encode("utf-8")).hexdigest()[:6]


def dial_pair(pat):
    """The (child, partner) a dial would rearrange, or None.

    Only pair-shaped patterns have one. A pattern with three
    columns names its first two, because the rearrangement acts
    on a child against one partner at a time - and saying that
    here is better than letting a three-column pattern look
    dialable and quietly act on part of itself.
    """
    if pat["kind"] not in DIALABLE:
        return None
    cols = [c for c in pat["columns"] if c]
    if len(cols) < 2:
        return None
    return (cols[0], cols[1])


def patterns(bp):
    """Every pattern the blueprint carries, one row each."""
    out = []
    cols = bp.get("columns") or {}
    rels = bp.get("relationships") or []

    for r in rels:
        ch = r.get("child")
        pars = r.get("parents") or []
        ev = r.get("evidence") or {}
        # SCAFFOLDING WAS EXCLUDED FROM THE COLUMN LOOP AND NOT
        # FROM THIS ONE. A `__has__` indicator participates in
        # the real blueprint's graph - the repo's own history
        # records a numeric driven by a single token at r2
        # 0.733 - so a relationship naming one would have been
        # listed as a pattern the operator owns, when it is
        # search machinery that never reaches the written file.
        # One exclusion written in one place is one applied in
        # one place.
        if S.is_scaffolding(ch) or any(S.is_scaffolding(q)
                                       for q in pars):
            continue
        out.append({
            "kind": "relationship",
            "what": "{} <- {}".format(ch, ", ".join(pars[:3])),
            "columns": [ch] + list(pars),
            "strength": ev.get("skill_out_of_sample"),
        })
        for pcol, eff in (ev.get("effect") or {}).items():
            out.append({
                "kind": "effect_curve",
                "what": "{} responds to {} ({})".format(
                    ch, pcol, (eff or {}).get("shape", "?")),
                "columns": [ch, pcol],
                "strength": (eff or {}).get("effect_size"),
            })
        inter = ev.get("interaction")
        if inter:
            pair = inter.get("pair") or []
            out.append({
                "kind": "interaction_surface",
                "what": "{} from {} jointly".format(
                    ch, " x ".join(pair[:2])),
                "columns": [ch] + list(pair),
                "strength": inter.get("gain"),
            })

    for c, v in cols.items():
        if S.is_scaffolding(c):
            continue
        dyn = v.get("dynamics") or {}
        cov = v.get("coverage")
        if cov is not None and float(cov) < 0.999:
            out.append({"kind": "column_presence",
                        "what": "{} present on {:.0%}".format(
                            c, float(cov)),
                        "columns": [c], "strength": float(cov)})
        if abs(float(dyn.get("within_lag1") or 0.0)) > 0.05:
            out.append({"kind": "within_patient",
                        "what": "{} tracks itself visit to "
                                "visit".format(c),
                        "columns": [c],
                        "strength": dyn.get("within_lag1")})
        if abs(float(dyn.get("missing_clustering")
                     or 0.0)) > 0.05:
            out.append({"kind": "missing_clustering",
                        "what": "{}'s absence comes in "
                                "runs".format(c),
                        "columns": [c],
                        "strength": dyn.get("missing_clustering")})

    for con in (bp.get("constraints") or []):
        if con.get("op") == "==" and \
                float(con.get("holds_in_source") or 0) >= 0.999:
            out.append({
                "kind": "identity",
                "what": "{} == {} on every row".format(
                    con.get("lhs"), con.get("rhs")),
                "columns": [con.get("lhs"), con.get("rhs")],
                "strength": 1.0})
    for p in out:
        p["id"] = pattern_id(p["kind"], p["columns"])
        p["dialable"] = dial_pair(p) is not None
    return out


def reachable_band(frame, child, parent):
    """The rank correlations this pair can actually reach.

    A pair's achievable range is bounded by its TIE STRUCTURE,
    and the bound can be far inside [-1, +1]: on the real
    extract span_days holds five distinct values over 56,065
    rows and no arrangement reaches +/-0.4 against
    condition_count. A per-pattern dial that cannot say this
    reports a miss for an ask that never existed.

    Measured, not derived: both columns sorted the same way is
    the ceiling, opposed is the floor.
    """
    import numpy as np
    import pandas as pd
    a_ = pd.to_numeric(frame[parent], errors="coerce")
    b_ = pd.to_numeric(frame[child], errors="coerce")
    m = a_.notna() & b_.notna()
    if int(m.sum()) < 20:
        return None
    av = np.sort(a_[m].to_numpy())
    bv = np.sort(b_[m].to_numpy())
    hi = float(pd.Series(av).corr(pd.Series(bv),
                                  method="spearman"))
    lo = float(pd.Series(av).corr(pd.Series(bv[::-1]),
                                  method="spearman"))
    return (lo, hi)


def dial(run, spec, hold_others=True, apply_it=False, seed=0):
    """Dial ONE pattern by its id, as an output permutation."""
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(ROOT / "scripts"))
    from fidelity_deck import read_frame
    from repair_inversions import repair

    run = Path(run)
    bp_path = run / "blueprint.json"
    gen_path = run / "generated.csv"
    for need in (bp_path, gen_path):
        if not need.exists():
            print("STOP: no {} in {}".format(need.name, run))
            return 1
    if "=" not in spec:
        print("STOP: --dial wants ID=TARGET, got {}".format(spec))
        return 2
    pid, _, want_s = spec.partition("=")
    try:
        want = float(want_s)
    except ValueError:
        print("STOP: {} is not a number".format(want_s))
        return 2

    bp = json.loads(bp_path.read_text(encoding="utf-8"))
    pats = patterns(bp)
    hit = [q for q in pats if q["id"] == pid.strip()]
    if not hit:
        print("STOP: no pattern with id {}. Run without --dial "
              "to list them.".format(pid))
        return 1
    pat = hit[0]
    pair = dial_pair(pat)
    if pair is None:
        # REFUSED BY NAME. An operator who dials a pattern no
        # dial reaches deserves to be told which pattern and
        # why, not handed a shrug or a silent no-op.
        print("STOP: pattern {} is a {} - no dial reaches that "
              "kind today. {}".format(pid, pat["kind"],
                                      pat["what"]))
        return 1
    child, parent = pair

    gen = read_frame(gen_path)
    for c in (child, parent):
        if c not in gen.columns:
            print("STOP: {} is not in generated.csv - this "
                  "pattern names a column the written file "
                  "does not carry.".format(c))
            return 1

    def _sp(f):
        a_ = pd.to_numeric(f[parent], errors="coerce")
        b_ = pd.to_numeric(f[child], errors="coerce")
        m = a_.notna() & b_.notna()
        return float(a_[m].corr(b_[m], method="spearman"))

    before = _sp(gen)
    band = reachable_band(gen, child, parent)
    print("pattern {}  {}".format(pid, pat["what"]))
    print("  rearranging {} against {}".format(child, parent))
    print("  now {:+.3f}   asked {:+.3f}".format(before, want))
    if band:
        print("  reachable band for this pair [{:+.3f}, "
              "{:+.3f}]".format(band[0], band[1]))
        if want < band[0] - 1e-9 or want > band[1] + 1e-9:
            print("  THE ASK IS OUTSIDE THAT BAND: no "
                  "arrangement of these values reaches it. The "
                  "dial will go as far as the pair allows.")

    # HOLD AT THE FILE'S OWN VALUES, not the source's. The
    # operator is changing one pattern of THIS output; the
    # other patterns should stay as this file has them, which
    # is what passing the generated frame as its own reference
    # means.
    out, _notes = repair(gen, gen, [(child, parent)], seed=seed,
                         targets={(child, parent): want},
                         hold_others=hold_others)
    after = _sp(out)
    print("  landed {:+.3f}".format(after))
    print("  mode: {}".format(
        "neighbours HELD - the ask may be geometrically "
        "unavailable while they are pinned" if hold_others
        else "neighbours free to move - the ask lands, and "
             "the drift is reported below"))

    moved = []
    for c in out.columns:
        if c in (child,):
            continue
        a0 = pd.to_numeric(gen[c], errors="coerce")
        if not a0.notna().any():
            continue
        g0 = pd.to_numeric(gen[child], errors="coerce")
        g1 = pd.to_numeric(out[child], errors="coerce")
        m0 = a0.notna() & g0.notna()
        m1 = a0.notna() & g1.notna()
        if int(m0.sum()) < 20 or int(m1.sum()) < 20:
            continue
        r0 = float(a0[m0].corr(g0[m0], method="spearman"))
        r1 = float(a0[m1].corr(g1[m1], method="spearman"))
        if np.isfinite(r0) and np.isfinite(r1) \
                and abs(r1 - r0) > 0.1 and c != parent:
            moved.append((c, r0, r1))
    if moved:
        print("  COLLATERAL - other couplings of {} that "
              "moved:".format(child))
        for c, r0, r1 in moved[:6]:
            print("      {:<28} {:+.3f} -> {:+.3f}".format(
                c, r0, r1))
    else:
        print("  no other coupling of {} moved by more than "
              "0.10".format(child))

    same = np.array_equal(
        np.sort(pd.to_numeric(gen[child],
                              errors="coerce").dropna()),
        np.sort(pd.to_numeric(out[child],
                              errors="coerce").dropna()))
    print("  the column holds the IDENTICAL multiset: {} - so "
          "this is a rearrangement, the marginal cannot have "
          "moved, and the change is invertible".format(
              "yes" if same else "NO"))
    if apply_it:
        dest = run / "generated_dialled.csv"
        out.to_csv(dest, index=False)
        print("  wrote {}".format(dest))
    else:
        print("  nothing written - pass --apply to write "
              "generated_dialled.csv")
    return 0


def report(run):
    run = Path(run)
    bp_path = run / "blueprint.json"
    if not bp_path.exists():
        print("STOP: no blueprint.json in {}".format(run))
        return 1
    bp = json.loads(bp_path.read_text(encoding="utf-8"))
    pats = patterns(bp)

    by_kind = {}
    for p in pats:
        by_kind.setdefault(p["kind"], []).append(p)

    print("PATTERNS FOUND, AND WHETHER A DIAL REACHES THEM")
    print("  {}".format(run))
    print("")
    print("  {:<22} {:>6}  {:<26} {:<10} {}".format(
        "pattern kind", "found", "dial that reaches it",
        "layer", "invertible"))
    reach = unreach = 0
    for kind in sorted(by_kind):
        dial, layer, inv = REACH.get(kind, (None, None, False))
        n = len(by_kind[kind])
        if dial:
            reach += n
        else:
            unreach += n
        print("  {:<22} {:>6}  {:<26} {:<10} {}".format(
            kind, n, dial or "- none -", layer or "-",
            "yes" if inv else ("no" if dial else "-")))

    inv_n = sum(len(v) for k, v in by_kind.items()
                if REACH.get(k, (None, None, False))[2])
    print("")
    print("  {} patterns reachable by some dial, {} NOT "
          "reachable at all.".format(reach, unreach))
    print("  {} reachable by an INVERTIBLE dial."
          .format(inv_n))
    if inv_n == 0:
        # THE GAP, IN ONE SENTENCE. The invertible dials exist -
        # re-pointing and suppression are permutations on the
        # output, they always bite, and the cipher can undo
        # them exactly. They are simply not ATTACHED to
        # anything discovery found: an operator names a pair by
        # hand. Every dial that IS attached to a discovered
        # pattern is a blueprint dial, which is the layer
        # measured to be inert on the real extract. So the
        # per-pattern control the product wants does not exist
        # yet in either direction, and saying that plainly is
        # more useful than a count of seven.
        print("")
        print("  NOT ONE PATTERN IS REACHABLE BY AN INVERTIBLE "
              "DIAL. The invertible dials exist - re-pointing "
              "and suppression are permutations on the output, "
              "they always bite, and the cipher undoes them "
              "exactly - but they are attached to a pair an "
              "operator names BY HAND, not to anything "
              "discovery found. Every dial that IS attached to "
              "a discovered pattern is a blueprint dial, which "
              "is the layer measured inert on the real "
              "extract. That gap is the work, and it is the "
              "same gap from both ends.")
    print("")
    # THE UNREACHABLE ONES ARE THE ROADMAP, so they are named
    # rather than summed. A count of what works is an advert; a
    # list of what does not is a plan.
    for kind in sorted(by_kind):
        if REACH.get(kind, (None,))[0]:
            continue
        print("  NO DIAL REACHES {} ({}):".format(
            kind, len(by_kind[kind])))
        for p in by_kind[kind][:4]:
            print("      {}".format(p["what"]))
        if len(by_kind[kind]) > 4:
            print("      ... and {} more".format(
                len(by_kind[kind]) - 4))
    print("")
    print("  AND WHAT IS NOT LISTED IS NOT NECESSARILY ABSENT. "
          "Three-way structure needs roughly 3,200 patients on "
          "this project's own power curve; below that it is "
          "UNDECIDABLE rather than absent, and nothing here "
          "searched for it. A pattern this inventory does not "
          "name was either not looked for or not decidable at "
          "this cohort size.")
    print("  THE LAYER COLUMN IS THE ONE THAT MATTERS FOR "
          "CONTROL: a `blueprint` dial is written into the "
          "recipe and can be silently overwritten by a later "
          "generation tier - measured, on the real extract, "
          "for every value dial. An `output` dial is a "
          "permutation applied after everything else, which is "
          "why it always bites and why it can be inverted.")
    return 0


def main(argv):
    if not argv:
        print(__doc__)
        print("usage:")
        print("  pattern_dials.py RUNDIR")
        print("  pattern_dials.py RUNDIR --dial ID=TARGET "
              "[--let-neighbours-move] [--apply]")
        return 2
    run = argv[0]
    rest = argv[1:]
    spec = None
    for i, tok in enumerate(rest):
        if tok == "--dial" and i + 1 < len(rest):
            spec = rest[i + 1]
    if spec is None:
        return report(run)
    return dial(run, spec,
                hold_others="--let-neighbours-move" not in rest,
                apply_it="--apply" in rest)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
