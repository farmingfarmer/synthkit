"""Smoke: rearranging one pattern to order, and what it costs.

TWO THINGS AT ONCE, because they are one piece of machinery.

FIRST, `repair_inversions.py` HAD NO CHECKS AT ALL. It rewrites
a generated file with --apply and was used on the deliverable;
nothing automated ever exercised it. That alone is worth
closing.

SECOND, it is the right engine for PER-PATTERN DIALS. The
product goal is a user seeing every pattern found and choosing
how to distort each one. The dials that bite on real data are
permutations applied to the OUTPUT - blueprint dials are
silently overwritten by later generation tiers - and this
module already rearranges a child jointly against all its
partners, which is what stops a single-pattern dial from
randomising everything else. Reusing it rather than writing a
second rearrangement is also what keeps the two from
disagreeing.

AND THE CENTRAL CONSTRAINT, MEASURED. Dialing one pattern while
pinning its neighbours is sometimes GEOMETRICALLY IMPOSSIBLE:
where a common driver feeds two children, their couplings to it
are linked. On exactly that shape - a drives both b and c:

    target   holding others     letting them move
    -0.50    a~b -0.209         a~b -0.490
    -0.80    a~b -0.297         a~b -0.806

Letting them move lands the ask. Holding them lands about 40%
of it AND drags the held pair anyway, because the solve splits
the difference. So the operator chooses, and the report says
what the choice cost. One default hiding this would be a dial
that silently under-delivers.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def fixture(seed=4, n=2000):
    """ONE DRIVER FEEDING TWO CHILDREN, which is the shape that
    makes the constraint visible. With independent children,
    holding one neighbour costs nothing and the whole trade-off
    this suite is about cannot be reached."""
    r = np.random.RandomState(seed)
    a = r.normal(50, 10, n)
    b = 1.8 * a + r.normal(0, 6, n)
    c = -1.2 * a + r.normal(0, 8, n)
    src = pd.DataFrame({"a": np.round(a, 2),
                        "b": np.round(b, 2),
                        "c": np.round(c, 2)})
    # SHUFFLE THE CHILD, NOT THE ROWS. `sample(frac=1.0)`
    # permutes every column TOGETHER, so every correlation
    # survives untouched - the first version of this fixture
    # did that and the "restore" check passed by restoring
    # something that had never been lost. The suite's own
    # fixture-validation check caught it. Shuffling b alone is
    # what actually destroys b's couplings while leaving a and
    # c exactly as they were.
    gen = src.copy()
    gen["b"] = (src["b"].sample(frac=1.0, random_state=1)
                .to_numpy())
    return src, gen


def sp(f, x, y):
    return float(pd.to_numeric(f[x], errors="coerce").corr(
        pd.to_numeric(f[y], errors="coerce"), method="spearman"))


def main():
    from repair_inversions import repair

    src, gen = fixture()
    s_ab, s_bc = sp(src, "a", "b"), sp(src, "b", "c")
    check("the fixture has ONE DRIVER feeding TWO children, so "
          "holding a neighbour actually costs something: "
          "source a~b {:+.3f}, b~c {:+.3f}".format(s_ab, s_bc),
          abs(s_ab) > 0.5 and abs(s_bc) > 0.5)
    check("...and the shuffled frame really has lost the "
          "pattern, so a restore has something to recover: "
          "a~b {:+.3f}".format(sp(gen, "a", "b")),
          abs(sp(gen, "a", "b")) < 0.1)

    # ---- THE ORIGINAL BEHAVIOUR, which had no checks at all
    fix, _ = repair(src, gen, [("b", "a")], seed=0)
    check("with no target, repair RESTORES the source coupling "
          "- {:+.3f} against a source {:+.3f}".format(
              sp(fix, "a", "b"), s_ab),
          abs(sp(fix, "a", "b") - s_ab) < 0.1)
    check("...and the child keeps the EXACT multiset it had, "
          "so the marginal cannot move and the change is "
          "invertible from the arrangement alone",
          np.array_equal(
              np.sort(pd.to_numeric(fix["b"]).to_numpy()),
              np.sort(pd.to_numeric(gen["b"]).to_numpy())))
    check("...and every other column is byte-identical - only "
          "the child was rearranged",
          all(np.array_equal(fix[c].to_numpy(),
                             gen[c].to_numpy())
              for c in ("a", "c")))

    # ---- THE DIAL LANDS WHEN NEIGHBOURS MAY MOVE
    free, _ = repair(src, gen, [("b", "a")], seed=0,
                     targets={("b", "a"): -0.5},
                     hold_others=False)
    got_free = sp(free, "a", "b")
    check("a per-pattern dial LANDS when neighbours may move - "
          "asked -0.50, got {:+.3f}".format(got_free),
          abs(got_free + 0.5) < 0.1)
    check("...still as a permutation, so a dialed file is a "
          "rearrangement of the generated one and nothing else",
          np.array_equal(
              np.sort(pd.to_numeric(free["b"]).to_numpy()),
              np.sort(pd.to_numeric(gen["b"]).to_numpy())))

    # ---- AND FALLS SHORT WHEN THEY ARE HELD. This is the
    # finding, not a defect: the two couplings are linked
    # through the common driver.
    held, _ = repair(src, gen, [("b", "a")], seed=0,
                     targets={("b", "a"): -0.5},
                     hold_others=True)
    got_held = sp(held, "a", "b")
    check("...and FALLS SHORT when they are held - asked "
          "-0.50, got {:+.3f}. Not a weak dial: the couplings "
          "are linked through the common driver, so the ask is "
          "geometrically unavailable".format(got_held),
          abs(got_held) < abs(got_free) - 0.15)
    check("...and the two modes genuinely differ on the "
          "NEIGHBOUR too - held {:+.3f} against free {:+.3f}, "
          "so the choice is real rather than cosmetic".format(
              sp(held, "b", "c"), sp(free, "b", "c")),
          abs(sp(held, "b", "c") - sp(free, "b", "c")) > 0.3)

    # ---- A NAMED PATTERN IS NEVER DROPPED FOR BEING WEAK
    weak, _ = repair(src, gen, [("b", "a")], seed=0,
                     floor=0.99,
                     targets={("b", "a"): -0.5},
                     hold_others=False)
    check("a dialed pair is kept even when its source "
          "strength is under the floor - the floor screens "
          "noise partners, and a pattern the operator NAMED is "
          "not noise. got {:+.3f}".format(sp(weak, "a", "b")),
          abs(sp(weak, "a", "b") + 0.5) < 0.15)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
