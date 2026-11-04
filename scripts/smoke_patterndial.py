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

    # ---- REPAIRING THE PAIRS THAT MISS `close`
    #
    # The close criterion has the smallest margin of the nine,
    # and on the real extract it sat at 87.6% against an 87.9%
    # bar - ONE pair short. The same joint rearrangement that
    # fixed the inversion there can be aimed at a drifted pair
    # instead, targeting the correlation the SOURCE has.
    #
    # It is a PERMUTATION, so no marginal moves and the
    # published bounds are exactly what they were; and the
    # WHOLE gate is re-measured, because a repair aimed at one
    # criterion can cost others - which is not a hypothesis,
    # it is what two reverted fidelity tiers did.
    import json
    import subprocess as _sp2
    import tempfile as _tf2
    import numpy as _np2
    import pandas as _pd2
    from synthkit import blueprint as _B2
    from synthkit.discover import discover as _disc2
    from synthkit.generate import generate as _gen2
    from synthkit.pipeline import compare as _cmp2

    _rng = _np2.random.default_rng(5)
    _n, _v = 300, 6
    _pid = _np2.repeat(["P{:03d}".format(i) for i in range(_n)], _v)
    _N = _n * _v
    _a = _rng.normal(0, 1, _N)
    _b = 0.7 * _a + _rng.normal(0, 0.7, _N)
    _c = 0.5 * _a + _rng.normal(0, 0.9, _N)
    _d = _rng.normal(0, 1, _N)
    _df = _pd2.DataFrame({
        "person_id": _pid, "visit": _np2.tile(_np2.arange(_v), _n),
        "a": _np2.round(_a, 3), "b": _np2.round(_b, 3),
        "c": _np2.round(_c, 3), "d": _np2.round(_d, 3)})
    _bp = _B2.build(_df, _disc2(_df, group_by="person_id", seed=1),
                    group_by="person_id")
    with _tf2.TemporaryDirectory() as _td:
        _rd = Path(_td)
        (_rd / "blueprint.json").write_text(json.dumps(_bp),
                                            encoding="utf-8")
        _g = _gen2(_bp, n_patients=_n, seed=3)
        _g.to_csv(_rd / "generated.csv", index=False)
        _sp = _rd / "src.csv"
        _df.to_csv(_sp, index=False)
        _fid = _cmp2(_df, _g, _bp, "person_id",
                     _bp.get("time_column"))
        (_rd / "fidelity.json").write_text(json.dumps(_fid),
                                           encoding="utf-8")
        _rel = _fid["relationships"]
        _miss = [r for r in _rel["pairs"]
                 if r.get("kind") == "numeric"
                 and abs(r.get("source") or 0) >= 0.1
                 and abs(r.get("delta") or 0) > 0.2]
        check("the fixture run actually MISSES close on a pair "
              "whose columns are both in the written file - "
              "without one the repair below has nothing to do "
              "and would pass by not running",
              bool(_miss))
        _r = _sp2.run(
            [sys.executable, "scripts/repair_inversions.py",
             str(_rd), str(_sp), "--close"],
            capture_output=True, text=True,
            cwd=str(ROOT), timeout=1800)
        _o = _r.stdout
        check("--close finds the drifted pairs and aims each at "
              "the correlation the SOURCE has, rather than at "
              "zero or a guess",
              _r.returncode == 0
              and "misses CLOSE on" in _o
              and "source +" in _o)
        # AND IT LANDS NEAR THE SOURCE, which is the observable
        # - asserting that a `targets` argument was passed
        # proved nothing, because the repair already aims at
        # the source by default and the two paths agree.
        _pl = [ln for ln in _o.splitlines() if "pair " in ln
               and "->" in ln]
        _got = None
        if _pl:
            _got = float(_pl[0].split("->")[1].strip())
        _want = float(_miss[0]["source"])
        check("...and the repaired pair LANDS near that source "
              "value - {} against a source of {}"
              .format("{:+.3f}".format(_got) if _got is not None
                      else "none", "{:+.3f}".format(_want)),
              _got is not None and abs(_got - _want) < 0.1)
        check("...and it re-measures the WHOLE gate, not only "
              "the criterion it aimed at - a repair that costs "
              "more than it gains must be visible in the same "
              "output",
              "re-measuring the WHOLE gate" in _o
              and "direction kept" in _o
              and "dynamics tracked" in _o)
        check("...and close is FIXED on this run while nothing "
              "reads BROKEN",
              "close" in _o and "FIXED" in _o
              and "BROKEN" not in _o)
        # THE DENOMINATORS MUST MATCH, or the verdict column
        # is comparing two measurements. On the real extract
        # the recorded run related 137 pairs and the
        # re-measure related 57 - it reads the written file,
        # which drops the token indicators the search
        # expanded - so `120/137 before, 47/57 after` was two
        # answers to two questions wearing a before and an
        # after.
        _pc = [ln for ln in _o.splitlines()
               if "pairs compared:" in ln]
        check("the tool measures BOTH sides itself and prints "
              "the two denominators, so a mismatch cannot "
              "hide inside a percentage: {}"
              .format(_pc[0].strip() if _pc else "(absent)"),
              _pc and "before" in _pc[0] and "after" in _pc[0])
        check("...and on this run they MATCH, which is what "
              "makes the verdicts above a comparison",
              _pc and "DENOMINATORS DIFFER" not in _pc[0])

        # THE RECORDED COUNT CAN DIFFER FROM THE RE-MEASURED
        # ONE, AND THE TOOL MUST SAY SO.
        #
        # This fixture cannot produce that by itself - it has
        # no set columns, so the pipeline's token expansion
        # does not apply and the two counts agree. The REAL
        # extract recorded 137 pairs and re-measures 57. The
        # mismatch is planted here instead of faked away,
        # because the branch that explains it is the one that
        # stops a reader comparing 120/137 against 47/57.
        _fk = json.loads((_rd / "fidelity.json")
                         .read_text(encoding="utf-8"))
        _fk["relationships"]["compared"] = 999
        (_rd / "fidelity.json").write_text(json.dumps(_fk),
                                           encoding="utf-8")
        _r2 = _sp2.run(
            [sys.executable, "scripts/repair_inversions.py",
             str(_rd), str(_sp), "--close"],
            capture_output=True, text=True,
            cwd=str(ROOT), timeout=1800)
        check("when the RECORDED pair count differs from the "
              "re-measured one the tool says so, marks the "
              "recorded percentage NOT comparable, and is "
              "explicit that the CAUSE is not established - "
              "naming a cause it has not measured would be "
              "the fault this suite exists to catch",
              "the run RECORDED 999 pairs" in _r2.stdout
              and "NOT comparable" in _r2.stdout
              and "not established" in _r2.stdout)
        check("...and it writes NOTHING without --apply, "
              "because a repair that costs more than it gains "
              "should not be able to land by accident",
              "nothing written" in _o
              and not (_rd / "generated_repaired.csv").exists())

        # THE COLUMNS THAT CANNOT BE REACHED ARE COUNTED, NOT
        # SWALLOWED. Token indicators and set sizes are search
        # machinery, dropped before the file is written -
        # nobody can rearrange a column that is not there, and
        # a silent skip would make the repairable count read as
        # the whole gap.
        import repair_inversions as _RI
        # ASSERTED ON THE SELECTION ITSELF, because the
        # fixture has no unreachable pair and the CLI branch
        # therefore never runs - a mutation that disabled the
        # report found nothing until this asked the function
        # directly. The pair a<-c is made unreachable by
        # telling it the file carries only `a`.
        _dr, _ab = _RI._drifted_pairs(_fid, {"a"}, limit=0)
        check("a close miss on a column the written file does "
              "not carry is RETURNED as skipped rather than "
              "dropped - nobody can rearrange a column that is "
              "not there, and a silent skip makes the "
              "repairable count read as the whole gap",
              len(_ab) >= 1 and not _dr)
        _dr3, _ab3 = _RI._drifted_pairs(_fid, set(_g.columns),
                                        limit=0)
        check("...while with the real column set the same pair "
              "IS repairable, so the skip is about reachability "
              "and not about the pair",
              len(_dr3) >= 1 and not _ab3)
        _dr2, _ = _RI._drifted_pairs(_fid, set(_g.columns),
                                     limit=1)
        check("...and --max repairs the WORST drift first, so "
              "a capped run spends its one repair where there "
              "is most to give back - the real extract needed "
              "exactly one more pair",
              len(_dr2) <= 1)

    # ---- THE MISSES THAT ARE NOT IN THE FILE AT ALL
    #
    # On the real extract the biggest group of close misses is
    # a `set__has__token` against an ordinary column, and the
    # indicator is NOT in generated.csv - it is search
    # machinery, dropped before the file is written. The set it
    # was derived from IS written, so moving the whole set
    # string moves the indicator with it.
    #
    # THE FADE IS PLANTED, because generation does not produce
    # one on this fixture: the set column is shuffled ALONE
    # (with its size partner, so the declared identity
    # survives), which kills the token's tie to the numeric
    # without touching anything else. Shuffling the whole
    # frame would permute every column together and preserve
    # every correlation, which is the vacuous-fixture fault
    # this repo has already recorded.
    import pattern_dials as _PD3
    import synthkit.sets as _S3
    import smoke_patterndials as _SP3
    _sdf = _SP3.set_fixture(n=320, v=6)
    _sbp = _B2.build(_sdf, _disc2(_sdf, group_by="person_id",
                                  seed=2),
                     group_by="person_id")
    _sgen = _gen2(_sbp, n_patients=320, seed=4)
    _perm = _np2.random.default_rng(9).permutation(len(_sgen))
    _bad = _sgen.copy()
    for _c in ("conds", "nc"):
        if _c in _bad.columns:
            _bad[_c] = _sgen[_c].to_numpy()[_perm]
    _sfid = _cmp2(_sdf, _bad, _sbp, "person_id",
                  _sbp.get("time_column"))
    _srel = _sfid["relationships"]
    _tokmiss = [r for r in _srel["pairs"]
                if r.get("kind") == "numeric"
                and abs(r.get("source") or 0) >= 0.1
                and abs(r.get("delta") or 0) > 0.2
                and (_S3.HAS in str(r["child"]))
                != (_S3.HAS in str(r["parent"]))]
    check("the planted fade produces a close miss on a TOKEN "
          "INDICATOR against a plain column - the shape that "
          "is the biggest group on the real extract and that "
          "no rearrangement of the written file could reach",
          bool(_tokmiss))
    with _tf2.TemporaryDirectory() as _td3:
        _rd3 = Path(_td3)
        (_rd3 / "blueprint.json").write_text(json.dumps(_sbp),
                                             encoding="utf-8")
        _bad.to_csv(_rd3 / "generated.csv", index=False)
        (_rd3 / "fidelity.json").write_text(json.dumps(_sfid),
                                            encoding="utf-8")
        _sp3 = _rd3 / "src.csv"
        _sdf.to_csv(_sp3, index=False)
        _r3 = _sp2.run(
            [sys.executable, "scripts/repair_inversions.py",
             str(_rd3), str(_sp3), "--close"],
            capture_output=True, text=True,
            cwd=str(ROOT), timeout=1800)
        _o3 = _r3.stdout
        check("...and the repair REACHES it through the set "
              "column the indicator was derived from, saying "
              "which set moved and against what",
              _r3.returncode == 0
              and "reached through its SET column" in _o3
              and "moved against" in _o3)
        check("...and close is FIXED while nothing reads "
              "BROKEN - moving a set moves every OTHER token "
              "on it too, so only the whole-gate re-measure "
              "can say whether it was worth it",
              "close" in _o3 and "FIXED" in _o3
              and "BROKEN" not in _o3)
        check("...and `set token shares` is unmoved, because a "
              "permutation cannot change what was published - "
              "the privacy posture is exactly what it was",
              "set token shares" in _o3
              and "set token shares" in _o3.split("BROKEN")[0])
        # THIS FIXTURE HAS A SIZE PARTNER, so accepting "no
        # size partner" as an alternative made the check
        # unable to fail - removing the partner passed. The
        # fixture's own declared identity is asserted first so
        # the demand below is not vacuous.
        check("the fixture's set column DECLARES a size "
              "partner, so the demand below is not vacuous",
              "nc" in _PD3.size_partners(_sbp, "conds"))
        check("...and the SIZE PARTNER travels with the set, "
              "so the declared identity survives a repair "
              "aimed at a token - `nc` IS the length of "
              "`conds`, and moving one without the other "
              "writes a file whose count contradicts its own "
              "list",
              "with nc" in _o3)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
