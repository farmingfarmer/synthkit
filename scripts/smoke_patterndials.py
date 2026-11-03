"""Smoke: every pattern found, and whether a dial reaches it.

WHY THIS EXISTS. The product goal is that a user sees every
pattern the tool found and chooses, per pattern, how to distort
it. Before any of that can be built, somebody has to be able to
say which patterns exist and which are currently beyond reach -
and the answer has to be uncomfortable where the truth is.

TWO MEASUREMENTS FROM TONIGHT SHAPE IT. A dial written into the
BLUEPRINT can be silently overwritten by a later generation
tier: a column whose parents survive the cycle trim is drawn
from their curves, so a shift on its marginal is never read, and
on the real extract every well-covered numeric is such a column.
A dial applied to the OUTPUT as a permutation always bites,
preserves the column's distribution exactly, and is invertible
through the cipher.

So the inventory reports the LAYER of each dial, and counts how
many patterns are reachable by an invertible one. On any
blueprint today that count is ZERO - the invertible dials are
attached to a pair an operator names by hand, not to anything
discovery found. Naming that is more useful than a total.
"""
import io
import json
import sys
import tempfile
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


def fixture(seed=3, n=240, v=7):
    """Carries one of every pattern kind the inventory can name.

    An inventory tested against a blueprint with two patterns in
    it proves nothing about its coverage - so this plants a
    relationship with effect curves, a column with real
    missingness, within-patient serial correlation, an exact
    identity twin, a set column and a bystander.
    """
    r = np.random.RandomState(seed)
    g = np.repeat(np.arange(n), v)
    base = r.normal(0, 1, n)[g]
    x = np.empty(len(g))
    for i in range(n):
        sl = slice(i * v, (i + 1) * v)
        e = r.normal(0, 4, v)
        w = np.empty(v)
        w[0] = e[0]
        for t in range(1, v):
            w[t] = 0.7 * w[t - 1] + e[t]
        x[sl] = 50 + 10 * base[sl] + w
    sp = r.normal(7, 2, len(g))
    sp[r.random_sample(len(g)) < 0.35] = np.nan
    toks = ["E11", "I10", "J45"]
    _per_pat = [sorted(r.choice(toks, size=r.randint(1, 3),
                                replace=False))
                for _ in range(n)]
    _sticky = []
    for i in _per_pat:
        for _ in range(v):
            _sticky.append(";".join(i))
    return pd.DataFrame({
        "person_id": ["P{:04d}".format(i) for i in g],
        "x": np.round(x, 2),
        "y": np.round(2.5 * x + r.normal(0, 6, len(g)), 2),
        "twin": np.round(x, 2),
        "z": np.round(r.normal(20, 5, len(g)), 2),
        "sparse": np.round(sp, 2),
        # A NUMERIC DRIVEN BY ONE TOKEN, so a relationship
        # naming a `__has__` indicator actually exists. Without
        # it the scaffolding exclusion has no path to reach -
        # the indicators carry coverage 1.0 and zero dynamics,
        # so they produce no pattern row of their own and the
        # check passed whether the exclusion was there or not.
        "severity": np.round(
            [8.0 if "E11" in c else 2.0 for c in _sticky]
            + r.normal(0, 0.4, len(g)), 2),
        # PATIENT-STICKY conditions, so the token indicators
        # carry WITHIN-PATIENT persistence and therefore
        # produce pattern rows of their own. With conditions
        # drawn independently per visit, every `__has__`
        # column had coverage 1.0 and no dynamics, so it
        # generated no pattern row at all - and the check that
        # scaffolding is excluded passed whether the exclusion
        # was there or not. A fixture must reach the fault.
        "conditions": _sticky})


def main():
    from synthkit import blueprint as B
    from synthkit.discover import discover
    import pattern_dials as P

    df = fixture()
    bp = B.build(df, discover(df, group_by="person_id", seed=1),
                 group_by="person_id")
    pats = P.patterns(bp)
    kinds = {p["kind"] for p in pats}

    check("the fixture carries several KINDS of pattern - an "
          "inventory tested on a blueprint with one kind in it "
          "says nothing about its coverage. found: {}".format(
              ", ".join(sorted(kinds))),
          len(kinds) >= 4)
    check("...including a RELATIONSHIP and the EFFECT CURVES "
          "under it, which are different patterns and are "
          "counted separately",
          "relationship" in kinds and "effect_curve" in kinds)
    check("...and an exact IDENTITY, which is a pattern a user "
          "would certainly want to distort and which no dial "
          "reaches",
          "identity" in kinds)

    # ---- SCAFFOLDING IS NOT A PATTERN THE USER OWNS
    scaf = [p for p in pats
            if any(c and "__" in c for c in p["columns"])]
    check("no pattern is reported on scaffolding columns - a "
          "`__has__` indicator is search machinery, not "
          "something an operator chose to collect",
          not scaf)

    # ---- THE REACH TABLE IS HONEST IN BOTH DIRECTIONS
    reachable = [k for k in kinds
                 if P.REACH.get(k, (None,))[0]]
    unreachable = [k for k in kinds
                   if not P.REACH.get(k, (None,))[0]]
    check("some kinds are reachable and some are NOT, and both "
          "are named - a table where everything is reachable "
          "is an advert. reachable: {} / unreachable: {}"
          .format(len(reachable), len(unreachable)),
          reachable and unreachable)

    inv = [k for k in kinds
           if P.REACH.get(k, (None, None, False))[2]]
    check("NOT ONE discovered pattern is reachable by an "
          "INVERTIBLE dial - the invertible dials are attached "
          "to a pair named by hand, and every dial attached to "
          "a discovered pattern is a blueprint dial, the layer "
          "measured inert on the real extract",
          not inv)

    # ---- THE REPORT SAYS IT
    with tempfile.TemporaryDirectory() as td:
        run = Path(td)
        (run / "blueprint.json").write_text(json.dumps(bp),
                                            encoding="utf-8")
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = P.report(run)
        out = buf.getvalue()
    check("the report runs on a run directory and exits clean",
          rc == 0)
    check("...and states the invertibility gap in words, not "
          "only as a zero somebody has to notice",
          "NOT ONE PATTERN IS REACHABLE BY AN INVERTIBLE "
          "DIAL" in out)
    check("...and NAMES the unreachable patterns rather than "
          "summing them, because the list is the roadmap and "
          "the count is not",
          "NO DIAL REACHES effect_curve" in out)
    check("...and says that what is absent may be UNDECIDABLE "
          "rather than absent - three-way structure needs "
          "roughly 3,200 patients and this cohort has fewer",
          "UNDECIDABLE" in out and "3,200" in out)
    check("...and explains the LAYER column, which is the one "
          "that decides whether a dial can be overwritten",
          "can be silently overwritten" in out)

    # ---- IDS ARE CONTENT-DERIVED, WHICH IS THE WHOLE POINT
    ids = [q["id"] for q in pats]
    check("every pattern has a DISTINCT id - {} patterns, {} "
          "ids. A collision would dial whichever the lookup "
          "found first, silently".format(len(ids),
                                         len(set(ids))),
          len(ids) == len(set(ids)))
    again = P.patterns(bp)
    check("...and the ids are STABLE across a re-read, because "
          "they hash the pattern's CONTENT. Positional ids "
          "would shift the moment discovery finds one more "
          "pattern and dial something else with no error",
          [q["id"] for q in again] == ids)
    fwd = [q for q in pats
           if q["kind"] == "relationship" and " <- " in q["what"]]
    if len(fwd) >= 2:
        check("...and DIRECTION changes the id - x <- y and "
              "y <- x are different patterns, and sorting the "
              "columns gave them the same id until it was "
              "caught",
              len({q["id"] for q in fwd}) == len(fwd))

    # ---- THE DIAL, END TO END
    import contextlib
    import numpy as np
    from synthkit.generate import generate
    from fidelity_deck import read_frame
    with tempfile.TemporaryDirectory() as td2:
        run2 = Path(td2)
        (run2 / "blueprint.json").write_text(json.dumps(bp),
                                             encoding="utf-8")
        gen = generate(bp, n_patients=240, seed=7)
        gen.to_csv(run2 / "generated.csv", index=False)
        target = next(q for q in pats
                      if q["dialable"]
                      and q["kind"] == "relationship")

        def _run(*extra):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = P.dial(run2,
                            "{}=-0.3".format(target["id"]),
                            *extra)
            return rc, buf.getvalue()

        rc, free = _run(False, False)
        got = float([ln for ln in free.splitlines()
                     if "landed" in ln][0].split()[-1])
        check("a pattern dialled BY ID lands when neighbours "
              "may move - asked -0.300, landed {:+.3f}".format(
                  got),
              rc == 0 and abs(got + 0.3) < 0.1)
        check("...and the run reports the REACHABLE BAND, so a "
              "miss can be told apart from an ask that never "
              "existed",
              "reachable band for this pair" in free)
        check("...and names the COLLATERAL - the other "
              "couplings of that column which moved",
              "COLLATERAL" in free or "no other coupling" in free)
        check("...and states that the column holds the "
              "identical multiset, which is what makes the "
              "change a rearrangement and invertible",
              "IDENTICAL multiset: yes" in free)
        check("...and writes NOTHING without --apply, because "
              "a tool that rewrites a released file by default "
              "is not a tool anyone should run twice",
              not (run2 / "generated_dialled.csv").exists()
              and "nothing written" in free)

        rc2, _held = _run(True, True)
        out = read_frame(run2 / "generated_dialled.csv")
        ch = P.dial_pair(target)[0]
        # BOTH SIDES READ FROM DISK. Comparing the written
        # file against the IN-MEMORY frame compares two
        # different things: a CSV round trip moves the last
        # bits, so the multiset was identical to tolerance and
        # unequal exactly. `dialback verify` reads both sides
        # from disk for precisely this reason, and the claim
        # being made - that the released file is a
        # rearrangement of the generated one - is a claim about
        # two FILES.
        _on_disk = read_frame(run2 / "generated.csv")
        check("...and with --apply it writes a file that is a "
              "PERMUTATION of the generated one - both read "
              "from disk, the dialled column holds the same "
              "multiset",
              rc2 == 0
              and np.array_equal(
                  np.sort(pd.to_numeric(out[ch],
                                        errors="coerce")
                          .dropna().to_numpy()),
                  np.sort(pd.to_numeric(_on_disk[ch],
                                        errors="coerce")
                          .dropna().to_numpy())))

        # ---- REFUSALS, BY NAME
        # A CHECK GUARDED BY `if` CAN VANISH INSTEAD OF
        # FAILING. This was written as `if und:` - so widening
        # DIALABLE until nothing was undialable made the check
        # SKIP rather than go red, and the mutation that should
        # have caught it found nothing. The fixture is asserted
        # to contain an undialable pattern FIRST, then the
        # refusal is tested unconditionally.
        und = next((q for q in pats if not q["dialable"]), None)
        check("the fixture still contains a pattern no dial "
              "reaches - without one the refusal check below "
              "would skip rather than fail, which is how a "
              "widened DIALABLE list would pass unnoticed",
              und is not None)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc3 = P.dial(run2, "{}=-0.3".format(
                (und or {}).get("id", "pzzzzzz")))
        check("a pattern NO dial reaches is refused by name, "
              "naming the kind - an operator handed a shrug "
              "has been told nothing",
              rc3 != 0 and und is not None
              and und["kind"] in buf.getvalue())
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc4 = P.dial(run2, "pzzzzzz=-0.3")
        check("...and an unknown id is refused with a pointer "
              "to the listing, not a traceback",
              rc4 != 0 and "no pattern with id" in buf.getvalue())

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
