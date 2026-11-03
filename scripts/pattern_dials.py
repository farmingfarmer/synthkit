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
    return out


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
        print("usage: pattern_dials.py RUNDIR")
        return 2
    return report(argv[0])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
