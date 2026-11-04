"""Exercise the per-pattern dials on a REAL run, reporting in words.

WHY THIS EXISTS. The per-pattern dials were built and measured on
fixtures: eight set tokens, one planted surface, a four-column
frame. The real extract carries 1,827 set tokens and 7 published
surfaces across 142 columns, and this file's own rule is that a
cap, a solve or a draw measured at one scale is not measured.
Running them there needs a driver, for two reasons.

FIRST, A PATTERN ID MUST NEVER BE TRANSCRIBED. The data machine is
read by screen OCR and this project has measured what that does to
characters: `0123456789` came back as `9123456789`, `0.37` as
`@.37`. A pattern id is a six-character hex string, so typing one
back by hand is a coin flip. The driver SELECTS its own targets
from the blueprint and never puts an id in front of a human.

SECOND, THE VERDICT HAS TO SURVIVE THE CHANNEL. Every number that
decides something is printed as a WORD beside its digits, and
every pass/fail is a word the machine computed, never a comparison
a reader makes against a digit they squinted at.

It chooses targets that can actually move - a token with a real
tie to its partner, a surface with a real gain - because a dial
measured on a pattern that was already flat lands at zero from
zero and proves nothing.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402

import pattern_dials as P                             # noqa: E402
import synthkit.sets as S                             # noqa: E402

WORDS = {"0": "zero", "1": "one", "2": "two", "3": "three",
         "4": "four", "5": "five", "6": "six", "7": "seven",
         "8": "eight", "9": "nine", "-": "minus", ".": "point",
         "+": "plus"}


def say(x, nd=3):
    """A number and its spelling, because digits do not survive.

    `0123456789` read back as `9123456789` off that screen while
    all ten spelled words came back perfect. Anything that decides
    something crosses in words.
    """
    s = "{:+.{}f}".format(float(x), nd)
    return "{} [{}]".format(s, " ".join(WORDS.get(c, c)
                                        for c in s))


def verdict(label, ok):
    print("  {:<46} {}".format(label, "YES" if ok else "NO"))
    return bool(ok)


def pick_set_token(bp, gen):
    """The published token with the strongest tie to a numeric.

    A DIAL MEASURED ON A PATTERN THAT IS ALREADY FLAT proves
    nothing - it lands at zero from zero. So the target is chosen
    by having something to move, and the chosen tie is printed so
    the reader can see it was not noise.
    """
    nums = [c for c in gen.columns
            if not S.is_scaffolding(c)
            and pd.to_numeric(gen[c], errors="coerce").notna().mean()
            > 0.5
            and pd.to_numeric(gen[c], errors="coerce").nunique() > 8]
    best = None
    for col, v in (bp.get("columns") or {}).items():
        mg = v.get("marginal") or {}
        if (mg.get("type") or "") != "list" or col not in gen.columns:
            continue
        strs = gen[col].fillna("").astype(str)
        for t in (mg.get("tokens") or [])[:40]:
            tv = t.get("value")
            if not tv:
                continue
            h = pd.Series(S.has_token(strs, ";", tv)
                          .to_numpy(dtype=float))
            if h.mean() < 0.02 or h.mean() > 0.9:
                continue
            for nc in nums[:25]:
                z = pd.to_numeric(gen[nc], errors="coerce")
                m = h.notna() & z.notna()
                if int(m.sum()) < 500:
                    continue
                r = float(h[m].corr(z[m], method="spearman"))
                if np.isfinite(r) and (best is None
                                       or abs(r) > abs(best[0])):
                    best = (r, col, tv, nc)
    return best


def pick_surface(bp, gen):
    """The published surface whose child carries the most gain."""
    best = None
    for q in P.patterns(bp):
        if q["kind"] != "interaction_surface":
            continue
        cols = [c for c in q["columns"] if c]
        if len(cols) < 3 or any(c not in gen.columns for c in cols):
            continue
        y = pd.to_numeric(gen[cols[0]], errors="coerce")
        a = pd.to_numeric(gen[cols[1]], errors="coerce")
        b = pd.to_numeric(gen[cols[2]], errors="coerce")
        live = y.notna() & a.notna() & b.notna()
        if int(live.sum()) < 500:
            continue
        try:
            qa = pd.qcut(a[live], 3, labels=False,
                         duplicates="drop").to_numpy()
            qb = pd.qcut(b[live], 3, labels=False,
                         duplicates="drop").to_numpy()
        except ValueError:
            continue
        g = P._grid_gain(y[live].to_numpy(), qa, qb)
        if np.isfinite(g) and (best is None or g > best[0]):
            best = (g, q)
    return best


def _drift(txt):
    """The larger of the two reported main-effect drifts."""
    # SLICED TO WHAT FOLLOWS THE WORD, because the tool
    # prints the main effects and their drift on ONE line:
    # "main effects after +0.113 and +0.088 - drift -0.062 /
    # +0.035". Taking every signed token on that line reported
    # 0.113 - the main-effect VALUE - as the drift, which is a
    # measurement script measuring something other than the
    # thing it names. Found by the check, not by reading it.
    for line in txt.splitlines():
        if "drift" in line:
            line = line.split("drift", 1)[1]
            bits = [b for b in line.replace("/", " ").split()
                    if b.startswith(("+", "-"))]
            vals = []
            for b in bits:
                try:
                    vals.append(abs(float(b)))
                except ValueError:
                    pass
            if vals:
                return max(vals)
    return 0.0


def main(argv):
    run = Path(argv[0]) if argv else Path(
        open(ROOT / "_run.txt").read().strip())
    bp = json.loads((run / "blueprint.json").read_text(
        encoding="utf-8"))
    gen = pd.read_csv(run / "generated.csv",
                      keep_default_na=False, low_memory=False)
    print("REAL RUN  {}".format(run.name))
    print("  rows {}   columns {}".format(len(gen),
                                          len(gen.columns)))
    pats = P.patterns(bp)
    kinds = {}
    for q in pats:
        kinds[q["kind"]] = kinds.get(q["kind"], 0) + 1
    print("  patterns {}  ({})".format(
        len(pats), ", ".join("{} {}".format(v, k)
                             for k, v in sorted(kinds.items()))))
    ok = True

    # ---- SET TOKEN, AT REAL WIDTH
    print("")
    print("SET TOKEN DIAL")
    hit = pick_set_token(bp, gen)
    if not hit:
        print("  no token with a measurable tie - NOT TESTED")
    else:
        r0, col, tok, part = hit
        want = -r0 if abs(r0) > 0.25 else (
            0.4 if r0 < 0 else -0.4)
        print("  {} / {} against {}".format(col, tok, part))
        print("  tie now {}".format(say(r0)))
        # A WEAK STARTING TIE MAKES THIS A WEAKER TEST than it
        # looks - dialling a token that is already near zero
        # measures the arrangement, not the removal of a real
        # association. Said out loud rather than buried.
        if abs(r0) < 0.15:
            print("  NOTE: that tie is weak, so this exercises "
                  "the arrangement rather than the removal of "
                  "a strong association.")
        print("  asking  {}".format(say(want)))
        tf, err = P.dial_set_token(gen, col, tok, part, want,
                                   seed=0)
        if tf is None:
            print("  REFUSED: {}".format(err))
            ok = False
        else:
            movers = [col] + [c for c in P.size_partners(bp, col)
                              if c in gen.columns]
            out = gen.copy()
            for c in movers:
                out[c] = gen[c].to_numpy()[tf]
            h = pd.Series(S.has_token(
                out[col].fillna("").astype(str), ";", tok
            ).to_numpy(dtype=float))
            z = pd.to_numeric(out[part], errors="coerce")
            got = float(h.corr(z, method="spearman"))
            print("  landed  {}".format(say(got)))
            ok &= verdict("landed within 0.05 of the ask",
                          abs(got - want) < 0.05)
            # EVERY PUBLISHED TOKEN'S SHARE MUST BE UNTOUCHED.
            worst = 0.0
            for t in ((bp["columns"][col]["marginal"]
                       .get("tokens")) or []):
                tv = t.get("value")
                if not tv:
                    continue
                a0 = float(S.has_token(
                    gen[col].fillna("").astype(str), ";", tv).mean())
                a1 = float(S.has_token(
                    out[col].fillna("").astype(str), ";", tv).mean())
                worst = max(worst, abs(a0 - a1))
            print("  worst token share drift {}".format(
                say(worst, 4)))
            ok &= verdict("every published share untouched",
                          worst < 1e-9)
            ok &= verdict("column holds the identical multiset",
                          sorted(gen[col].astype(str))
                          == sorted(out[col].astype(str)))
            # THE DECLARED SIZE IDENTITY - the one that broke.
            print("  size partners that travelled: {}".format(
                ", ".join(movers[1:]) or "none declared"))
            for pcol in movers[1:]:
                sz = S.sizes_of(out[col].fillna("").astype(str), ";")
                bad = float((sz != pd.to_numeric(
                    out[pcol], errors="coerce")).mean())
                was = float((S.sizes_of(
                    gen[col].fillna("").astype(str), ";")
                    != pd.to_numeric(gen[pcol],
                                     errors="coerce")).mean())
                print("  identity {} == len({}) broken on {} "
                      "of rows, was {}".format(
                          pcol, col, say(bad, 4), say(was, 4)))
                ok &= verdict("identity no worse than before",
                              bad <= was + 1e-9)

    # ---- INTERACTION SURFACE, ON A PUBLISHED ONE
    print("")
    print("INTERACTION SURFACE DIAL")
    sh = pick_surface(bp, gen)
    if not sh:
        print("  no measurable published surface - NOT TESTED")
    else:
        g0, q = sh
        cols = [c for c in q["columns"] if c]
        print("  {} from {} x {}".format(cols[0], cols[1],
                                         cols[2]))
        print("  gain now {}".format(say(g0, 4)))
        import contextlib
        import io as _io
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = P.dial(run, "{}=0.0".format(q["id"]),
                        apply_it=False)
        txt = buf.getvalue()
        for line in txt.splitlines():
            t = line.strip()
            if t.startswith(("landed gain", "main effects",
                             "reachable band", "the column "
                             "holds")):
                print("  {}".format(t))
        ok &= verdict("the dial ran clean", rc == 0)
        ok &= verdict("it reports the drift it caused",
                      "drift" in txt)
        ok &= verdict("the multiset is identical",
                      "IDENTICAL multiset: yes" in txt)
        # A LARGE MAINS DRIFT IS A FINDING, NOT A FAILURE,
        # and the first version of this driver graded it as a
        # failure - printing SOMETHING-FAILED directly beneath
        # the tool's own correct report of what it had done.
        # A verdict must not contradict the note under it.
        # Whether the mains can be held while a surface is
        # flattened is a property of the DATA; what the tool
        # owes is to SAY so, and that is what is graded.
        if "THE MAIN EFFECTS MOVED" in txt:
            print("  FINDING: this surface could not be "
                  "flattened without moving its main effects "
                  "more than 0.06. That is a cost of the ask "
                  "on this data, and the tool stated it.")
        ok &= verdict("it states plainly when the mains could "
                      "not be held",
                      ("THE MAIN EFFECTS MOVED" in txt)
                      == (abs(_drift(txt)) > 0.06))

    # ---- THE VALUE DIAL, ON THE REAL BLUEPRINT
    #
    # THIS IS THE CLAIM THAT WAS MADE ABOUT THIS MACHINE. The
    # record says every value dial read NO-OP on the real
    # extract and named span_days, and the mechanism was
    # mis-attributed to the column being a relationship child.
    # It was the CYCLIC REFINEMENT SWEEP re-ranking onto a
    # marginal pool captured before the dial. The fix moves
    # the pool with the dial, and it was verified on a
    # four-column fixture. A fixture is not this blueprint.
    #
    # Generation is run at a reduced patient count - the
    # cyclic structure lives in the blueprint, not in the row
    # count, so the dial either reads the moved pool or it
    # does not. The count is printed so nobody reads this as
    # a full-extract run.
    if "--skip-generate" not in argv:
        print("")
        print("VALUE DIAL ON THE REAL BLUEPRINT")
        import copy
        import time
        from synthkit.generate import generate
        rels = bp.get("relationships") or []
        kids = {}
        for r in rels:
            kids.setdefault(r.get("child"), []).extend(
                r.get("parents") or [])
        cyclic = set()
        for ch, pars in kids.items():
            for pcol in pars:
                if ch in (kids.get(pcol) or []):
                    cyclic.add(ch)
        # A COLUMN WITH A DECLARED IDENTITY IS NOT DIALABLE
        # BY ITS MARGINAL, AND GRADING IT AS ONE WOULD BE A
        # FALSE ALARM. The first run of this driver picked
        # `nc` - a set-size partner - and reported the dial
        # as a NO-OP: asked +2.342, moved +1.628, spread
        # 1.125 -> 1.771. That is not the dial failing. `nc`
        # IS the length of its set, so generation OVERWRITES
        # it after every column exists, which is the identity
        # tier doing exactly what this file says it must do.
        # Reporting it as a dial fault would send the next
        # person hunting a bug that is a guarantee.
        declared = set()
        for q in (bp.get("constraints") or []):
            if (q.get("op") or "") == "==":
                for side in (q.get("lhs"), q.get("rhs")):
                    if side:
                        declared.add(side)
        cand = [c for c, v in (bp.get("columns") or {}).items()
                if (v.get("marginal") or {}).get("type")
                == "quantiles"
                and not S.is_scaffolding(c)
                and c in cyclic
                and c in gen.columns
                and c not in declared]
        skipped = [c for c, v in (bp.get("columns") or {}).items()
                   if (v.get("marginal") or {}).get("type")
                   == "quantiles"
                   and not S.is_scaffolding(c)
                   and c in cyclic
                   and c in gen.columns
                   and c in declared]
        if skipped:
            print("  skipped {} column(s) with a DECLARED "
                  "IDENTITY - generation computes them from "
                  "another column, so a marginal dial on one "
                  "is overridden by design: {}".format(
                      len(skipped), ", ".join(skipped[:5])))
        # PREFER THE COLUMN THE RECORD NAMES, so the claim is
        # tested where it was made rather than somewhere
        # convenient.
        cand.sort(key=lambda c: (c != "span_days", c))
        ctrl = [c for c, v in (bp.get("columns") or {}).items()
                if (v.get("marginal") or {}).get("type")
                == "quantiles"
                and not S.is_scaffolding(c)
                and c not in cyclic
                and c in gen.columns]
        print("  cyclic numeric columns: {}".format(len(cand)))
        print("  acyclic numeric columns (controls): {}"
              .format(len(ctrl)))
        if not cand:
            print("  no cyclic numeric column - NOT TESTED")
        else:
            col = cand[0]
            npat = 120
            sd0 = float(pd.to_numeric(
                gen[col], errors="coerce").std())
            ask = round(2.0 * sd0, 4) if sd0 > 0 else 10.0
            print("  dialling {} by shift {}".format(
                col, say(ask)))
            print("  (generating {} patients, not the full "
                  "cohort - the cyclic structure is in the "
                  "blueprint, not the row count)".format(npat))
            t0 = time.time()
            b0 = copy.deepcopy(bp)
            g0 = generate(b0, n_patients=npat, seed=5)
            b1 = copy.deepcopy(bp)
            b1["columns"][col].setdefault("dials", {})[
                "shift"] = ask
            g1 = generate(b1, n_patients=npat, seed=5)
            took = time.time() - t0
            m0 = float(pd.to_numeric(
                g0[col], errors="coerce").mean())
            m1 = float(pd.to_numeric(
                g1[col], errors="coerce").mean())
            print("  two generations took {:.0f}s".format(took))
            print("  mean  {}  ->  {}".format(say(m0), say(m1)))
            print("  moved {}   asked {}".format(
                say(m1 - m0), say(ask)))
            landed = abs((m1 - m0) - ask) < max(0.1 * abs(ask),
                                                0.5)
            ok &= verdict("the value dial BITES on a cyclic "
                          "column", landed)
            if not landed and abs(m1 - m0) < max(
                    0.05 * abs(ask), 0.25):
                print("  THIS IS THE RECORDED NO-OP, STILL "
                      "PRESENT: the dial reached the "
                      "blueprint and the column did not move.")
            # THE NEIGHBOURING PROPERTY, in the same test.
            s0 = float(pd.to_numeric(
                g0[col], errors="coerce").std())
            s1 = float(pd.to_numeric(
                g1[col], errors="coerce").std())
            print("  spread {} -> {}".format(say(s0), say(s1)))
            ok &= verdict("...and the SPREAD did not move with "
                          "it",
                          abs(s1 - s0) < max(0.05 * abs(s0),
                                             1e-6))
            # AND NO OTHER COLUMN MOVED.
            moved = []
            for c in g0.columns:
                if c == col:
                    continue
                a0 = pd.to_numeric(g0[c], errors="coerce")
                a1 = pd.to_numeric(g1[c], errors="coerce")
                if not a0.notna().any():
                    continue
                d0, d1 = float(a0.mean()), float(a1.mean())
                sc = float(a0.std()) or 1.0
                if abs(d1 - d0) > 0.05 * sc:
                    moved.append(c)
            print("  other columns whose mean moved: {}".format(
                len(moved)))
            if moved:
                print("    {}".format(", ".join(moved[:6])))
            ok &= verdict("...and it moved nothing else",
                          not moved)

    print("")
    print("OVERALL {}".format("ALL-CLEAN" if ok
                              else "SOMETHING-FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
