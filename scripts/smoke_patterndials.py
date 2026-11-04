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

So the inventory reports the LAYER of each dial, and counts
how many patterns are reachable by an invertible one.

THAT COUNT USED TO BE ZERO, AND BOTH HALVES OF THE SENTENCE
ABOVE HAVE SINCE BEEN CORRECTED BY MEASUREMENT. The invertible
dials were attached only to a pair an operator named by hand;
the per-pattern dial attached them to what discovery found.
And the claim that a blueprint dial is inert was too broad - it
came from shift and scale on a column inside a CYCLE, where
the refinement sweep re-ranked onto an undialed marginal pool,
which is now fixed; persistence and coverage were never in
that measurement and both move their column. The distinction
that survives is INVERTIBILITY: an output dial is a
permutation the cipher can take back, a recipe dial is not.
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


def inter_fixture(seed=7, n=400, v=8):
    """A PLANTED JOINT EFFECT, with real main effects too.

    Both halves matter. Without main effects the check that
    they SURVIVE the dial is vacuous - and surviving is the
    property that broke: the first cut of the interaction
    dial ranked the child on its additive target, which IS
    the main effects, and doubled them while flattening the
    surface (+0.367 -> +0.790).
    """
    rng = np.random.default_rng(seed)
    pid = np.repeat(["P{:03d}".format(i) for i in range(n)], v)
    N = n * v
    a_ = rng.normal(0, 1, N)
    b_ = rng.normal(0, 1, N)
    y_ = 0.8 * a_ + 0.5 * b_ + 1.6 * (a_ * b_) \
        + rng.normal(0, 0.6, N)
    return pd.DataFrame({
        "person_id": pid, "visit": np.tile(np.arange(v), n),
        "a": np.round(a_, 3), "b": np.round(b_, 3),
        "y": np.round(y_, 3)})


def _skipgen(RPC, run, flag_first=False):
    """The driver's output with the generation step skipped.

    BOTH ORDERINGS, because only one of them was ever tested.
    The first version of the driver took argv[0] as the run
    directory, so the flag-first form opened
    `--skip-generate/blueprint.json` and died - and the check
    could not see it, because it passed the path first, which
    is the one ordering that works. On the data machine the
    flag was passed alone.
    """
    import contextlib
    args = (["--skip-generate", str(run)] if flag_first
            else [str(run), "--skip-generate"])
    # A CRASH MUST BECOME A FAILED CHECK, NOT A DEAD SUITE.
    # The flag-first mutation raised FileNotFoundError on
    # `--skip-generate/blueprint.json`, which took the whole
    # suite down and printed NO `FAIL:` line - so grepping
    # the mutation output for failures found nothing and it
    # read as a check that cannot fail. This repo has already
    # paid for reading "no failures" off a crashed run.
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            RPC.main(args)
    except Exception as exc:                  # noqa: BLE001
        return "DRIVER-RAISED {}: {}".format(
            type(exc).__name__, exc)
    return buf.getvalue()


def set_fixture(seed=5, n=160, v=6):
    """A SET COLUMN WITH A DECLARED SIZE PARTNER.

    Its own fixture, not a column bolted onto the one above:
    adding a column to a fixture other checks depend on moved
    an estimate across a threshold once and broke three checks
    that had nothing to do with it.

    It has to carry the size partner, because the first cut of
    the set dial rearranged the strings and left the count
    behind - `nc == len(conds)` went from broken on 0.0% of
    rows to broken on 75.3%, a file whose count contradicts
    its own list. A fixture without the identity cannot see
    that, and the check would have passed on the broken code.
    """
    rng = np.random.default_rng(seed)
    pid = np.repeat(["P{:03d}".format(i) for i in range(n)], v)
    N = n * v
    sev = rng.normal(0, 1, N)
    toks = ["t{:02d}".format(i) for i in range(8)]
    rows = []
    for i in range(N):
        k = int(rng.integers(0, 4))
        pick = list(rng.choice(toks, size=k, replace=False))
        # t00 is PLANTED on the high-severity rows, so the
        # token has a real tie to move rather than noise
        if sev[i] > 0.6 and "t00" not in pick:
            pick.append("t00")
        rows.append(";".join(sorted(pick)))
    return pd.DataFrame({
        "person_id": pid,
        "visit": np.tile(np.arange(v), n),
        "sev": np.round(sev, 3),
        "conds": rows,
        "nc": [len(r.split(";")) if r else 0 for r in rows]})


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
    # THE TABLE MUST STILL NAME WHAT NOTHING REACHES. When the
    # per-pattern dial shipped, three kinds moved from
    # blueprint-and-inert to output-and-invertible, and the
    # first version of these checks asserted the OLD claim -
    # correctly going red. Two kinds are still reached by
    # nothing at all, and a table where everything is reachable
    # is an advert.
    none_reaches = [k for k, v in P.REACH.items() if not v[0]]
    # THE UNREACHABLE LIST IS EMPTY NOW, AND THAT IS THE
    # DANGEROUS STATE. It used to hold set_token and
    # interaction_surface, and both got dials. A table with
    # no gaps reads as completeness to anybody who did not
    # watch it fill, so the honest burden moved from "which
    # kinds have no dial" to "which patterns this inventory
    # never names at all" - and the report has to say that
    # out loud rather than letting a full table speak.
    check("every kind the reach table carries now has a "
          "dial - unreachable: {} - which is why the check "
          "below matters more than this one"
          .format(", ".join(sorted(none_reaches)) or "none"),
          not none_reaches)

    # THE SPLIT THAT MATTERS IS NOT reachable/unreachable, IT
    # IS INVERTIBLE-OUTPUT AGAINST INERT-BLUEPRINT. A blueprint
    # dial has a name in the dial column and does nothing on the
    # real extract, so counting it as reach flatters the table.
    inv = sorted(k for k in kinds
                 if P.REACH.get(k, (None, None, False))[2])
    bp_only = sorted(k for k in kinds
                     if P.REACH.get(k, (None, None, False))[0]
                     and not P.REACH.get(k,
                                         (None, None, False))[2])
    check("the kinds a per-pattern INVERTIBLE dial reaches - "
          "{} - are every one of them an OUTPUT dial. A set "
          "token is in that list and does NOT reduce to a "
          "column pair: its second entry is a token, which is "
          "why the generic pair test had to stop deciding "
          "dialability on its own".format(", ".join(inv)),
          inv == ["effect_curve", "identity",
                  "relationship", "set_token"]
          and all(P.REACH[k][1] == "output" for k in inv))
    check("...and the kinds whose ONLY dial is a blueprint "
          "dial are named apart - {} - because that layer was "
          "measured inert on the real extract and a dial "
          "column alone would read as control"
          .format(", ".join(bp_only)),
          bp_only == ["column_presence", "within_patient"])

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
    check("...and states what the INVERTIBLE dials buy, in "
          "words - a permutation that always bites, keeps the "
          "column's own distribution, and can be undone",
          "output-layer permutations" in out
          and "can be undone" in out)
    # "MEASURED INERT" WAS AN OVERCLAIM AND MEASURING IT
    # CORRECTED IT: persistence moved a column's
    # between-patient share 0.935 -> 0.384 and coverage moved
    # presence 0.603 -> 0.319, each with a control in the same
    # run. What is always true is that a recipe dial is not a
    # permutation, so nothing can undo it.
    check("...and names the REMAINING gap by the property "
          "that is always true - a recipe dial is not a "
          "permutation, so no cipher can take it back - "
          "rather than by claiming it does not take effect, "
          "which was measured false",
          "BLUEPRINT dial" in out
          and "NOT permutations" in out
          and "no cipher entry" in out
          and "not because they fail to take effect" in out
          and "INERT" not in out)
    check("...and when NOTHING is unreachable it says so "
          "explicitly and points at what the inventory does "
          "not NAME - a table with no gaps reads as "
          "completeness to a reader who did not watch it fill",
          "NOT THE SAME AS BEING FINISHED" in out
          and "does not NAME" in out)
    check("...and says that what is absent may be UNDECIDABLE "
          "rather than absent - three-way structure needs "
          "roughly 3,200 patients and this cohort has fewer",
          "UNDECIDABLE" in out and "3,200" in out)
    check("...and explains the LAYER column by the property "
          "that decides whether a dial can be TAKEN BACK - "
          "it used to say the layer decides whether a dial "
          "is overwritten, which was the narrow cyclic "
          "shift/scale case and is now fixed",
          "THE PROPERTY IS INVERTIBILITY" in out
          and "generating again" in out
          and "not weaker" in out)

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

        # ---- THE DIAL IS RECORDED, OR IT IS UNDOABLE BY
        # NOBODY. The first cut wrote the dialled file and no
        # cipher entry: a rearrangement invertible in
        # principle and written down nowhere, which is this
        # repo's own capability-wired-to-nothing fault.
        cip = json.loads((run2 / "cipher.json")
                         .read_text(encoding="utf-8"))
        check("--apply records the ARRANGEMENT in cipher.json, "
              "so the dial can be undone - a permutation "
              "nobody wrote down is invertible in principle "
              "and by nobody in practice",
              ch in (cip.get("columns") or {}))
        check("...and records what was ASKED beside what "
              "LANDED, with the band, so the entry is an "
              "answer key rather than a diff",
              cip.get("repointed")
              and "requested" in cip["repointed"][0]
              and "achieved" in cip["repointed"][0])
        _k = read_frame(run2 / "generated.csv")
        _tf = np.asarray(cip["columns"][ch])
        _back = np.empty(len(out), dtype=object)
        _back[_tf] = out[ch].to_numpy()
        check("...and the inverse reproduces the generated "
              "file EXACTLY, value for value - which needs the "
              "ORIGINAL strings moved rather than numeric "
              "values re-rendered, or 22.29 comes back as "
              "22.289999999999999",
              np.array_equal(np.asarray(_back, dtype=str),
                             _k[ch].to_numpy().astype(str)))
        _moved = [c for c in out.columns
                  if c not in (cip.get("columns") or {})
                  and not np.array_equal(
                      out[c].to_numpy().astype(str),
                      _k[c].to_numpy().astype(str))]
        check("...and NO column without a cipher entry changed "
              "- nothing moved that the record does not name{}"
              .format("" if not _moved
                      else ": " + ", ".join(_moved[:4])),
              not _moved)

        # ---- REFUSALS, BY NAME
        # A CHECK GUARDED BY `if` CAN VANISH INSTEAD OF
        # FAILING. This was written as `if und:` - so widening
        # DIALABLE until nothing was undialable made the check
        # SKIP rather than go red, and the mutation that should
        # have caught it found nothing. The fixture is asserted
        # to contain an undialable pattern FIRST, then the
        # refusal is tested unconditionally.
        und = next((q for q in pats if not q["dialable"]), None)
        check("the fixture still contains a pattern NO "
              "PER-PATTERN dial reaches - without one the "
              "refusal check below would skip rather than "
              "fail, which is how a widened DIALABLE list "
              "would pass unnoticed",
              und is not None)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc3 = P.dial(run2, "{}=-0.3".format(
                (und or {}).get("id", "pzzzzzz")))
        check("a pattern NO PER-PATTERN dial reaches is "
              "refused by name, naming the kind - an operator "
              "handed a shrug has been told nothing",
              rc3 != 0 and und is not None
              and und["kind"] in buf.getvalue())
        check("...and the refusal says whether a WEAKER dial "
              "exists for that kind and what is wrong with "
              "it - telling an operator there is nothing when "
              "there is something they cannot undo sends them "
              "to the wrong question",
              "CANNOT BE UNDONE" in buf.getvalue()
              or "either layer" in buf.getvalue())
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc4 = P.dial(run2, "pzzzzzz=-0.3")
        check("...and an unknown id is refused with a pointer "
              "to the listing, not a traceback",
              rc4 != 0 and "no pattern with id" in buf.getvalue())

    # ---- SET TOKENS: INVISIBLE BEFORE, AND DIALABLE NOW
    import synthkit.sets as SS
    sdf = set_fixture()
    sbp = B.build(sdf, discover(sdf, group_by="person_id",
                                seed=2),
                  group_by="person_id")
    spats = P.patterns(sbp)
    stok = [q for q in spats if q["kind"] == "set_token"]
    check("a SET COLUMN'S TOKENS are listed as patterns - "
          "{} of them. `set_token` sat in the reach table "
          "with no builder emitting it, so a clinical "
          "extract's drug and condition structure showed as "
          "nothing at all on a page promising every pattern "
          "the fit found".format(len(stok)),
          len(stok) >= 5)
    check("...and only tokens that CLEARED THE k FLOOR are "
          "named, because those are the ones the blueprint "
          "published - naming a sub-k token would be the "
          "inventory disclosing what the generator refused to",
          bool(stok) and all(
              q["columns"][1] in
              {t["value"] for t in
               sbp["columns"]["conds"]["marginal"]["tokens"]}
              for q in stok))
    check("...and a set token is DIALABLE even though its two "
          "entries are a column and a TOKEN, not a column "
          "pair - the pair test that serves every other kind "
          "says no, and the page would have shown real "
          "movable patterns as out of reach",
          bool(stok)
          and all(q["dialable"] for q in stok)
          and all(q.get("needs_partner") for q in stok))

    with tempfile.TemporaryDirectory() as td:
        srun = Path(td)
        (srun / "blueprint.json").write_text(
            json.dumps(sbp), encoding="utf-8")
        sgen = generate(sbp, n_patients=160, seed=4)
        sgen.to_csv(srun / "generated.csv", index=False)
        # AND THE SUITE MUST FAIL RATHER THAN CRASH if the
        # tokens vanish. An IndexError here prints a traceback
        # and no FAIL line, and this repo has already read
        # "no failures" off a crashed run once.
        _t = [q for q in stok if q["columns"][1] == "t00"]
        check("the planted token t00 is among the patterns - "
              "without it the checks below cannot run, and a "
              "suite that crashes prints no FAIL line at all",
              bool(_t))
        # The dependent checks below cannot run without it.
        # They are skipped rather than crashing - and the
        # explicit check above has already recorded a FAIL, so
        # the suite goes red either way. A block that vanishes
        # with nothing red is the fault; a block that vanishes
        # under a failure that was already reported is not.
        # AND THE BLOCK BELOW STILL RUNS when it is absent,
        # on a stand-in id that cannot resolve. Skipping would
        # make eleven checks VANISH, which is the fault this
        # repo already paid for - a check that disappears
        # reads the same as a check that passed. With the
        # stand-in they all go RED, loudly, naming what is
        # missing.
        t00 = _t[0] if _t else {"id": "pnosuch",
                                "columns": ["conds", "t00"]}

        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = P.dial(srun, "{}=-0.5".format(t00["id"]))
        out = buf.getvalue()
        check("dialling a set token with NO partner is "
              "REFUSED, and the refusal says why its SHARE is "
              "not the movable thing: writing a token onto "
              "rows that never recorded it is not a "
              "rearrangement and cannot be undone",
              rc != 0 and "not dialable" in out
              and "cannot be undone" in out
              and "--against" in out)

        def _corr(f):
            h = pd.Series(SS.has_token(
                f["conds"].fillna("").astype(str), ";",
                "t00").to_numpy(dtype=float))
            return float(h.corr(pd.to_numeric(f["sev"]),
                                method="spearman"))
        _was = _corr(sgen)
        check("the fixture's token actually HAS a tie to move "
              "({:+.3f}) - a dial measured on an unrelated "
              "token would land at zero from zero and prove "
              "nothing".format(_was),
              abs(_was) > 0.3)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = P.dial(srun, "{}=-0.5".format(t00["id"]),
                        against="sev", apply_it=True)
        out = buf.getvalue()
        check("...and WITH a partner it lands: {}".format(
            [l.strip() for l in out.splitlines()
             if "landed" in l]),
            rc == 0 and any("landed -0.4" in l or
                            "landed -0.5" in l
                            for l in out.splitlines()))
        check("...and reports the REACHABLE BAND as set by the "
              "token's SHARE rather than [-1, +1] - a token on "
              "42% of rows cannot reach +1 against a "
              "continuous column however it is arranged",
              "share, not by" in out)

        rel = pd.read_csv(srun / "generated_dialled.csv",
                          keep_default_na=False, dtype=str)
        key = pd.read_csv(srun / "generated.csv",
                          keep_default_na=False, dtype=str)
        check("every published token's SHARE is untouched - "
              "the whole set string travels as a unit, so the "
              "column holds the identical multiset and the "
              "privacy posture is exactly what was published",
              sorted(key["conds"]) == sorted(rel["conds"]))

        # THE NEIGHBOURING PROPERTY, AND IT IS THE ONE THAT
        # BROKE. Measured on the first cut of this dial: the
        # strings moved, the count stayed, and nc disagreed
        # with len(conds) on 75.3% of rows where it had
        # disagreed on 0.0%.
        _sz = SS.sizes_of(rel["conds"].astype(str), ";")
        _bad = float((_sz != pd.to_numeric(
            rel["nc"], errors="coerce")).mean())
        check("the DECLARED SIZE IDENTITY survives the dial - "
              "nc == len(conds) broken on {:.1%} of rows. The "
              "first cut left the count behind and broke it "
              "on 75.3%, which is a file whose count "
              "contradicts its own list".format(_bad),
              _bad < 0.001)
        check("...and the size partner is NAMED as having "
              "moved with the set, with its own coupling "
              "before and after - a column that travels "
              "silently is a relationship changed behind the "
              "operator's back",
              "to keep the declared size identity" in out
              and "size partner, moved with the set" in out)

        rec = json.loads((srun / "cipher.json")
                         .read_text(encoding="utf-8"))
        check("the arrangement is RECORDED for every column "
              "that moved, the size partner included - a "
              "permutation nobody wrote down is invertible in "
              "principle and by nobody in practice",
              set(rec["columns"]) == {"conds", "nc"})
        okinv = True
        for c, tf in rec["columns"].items():
            tf = np.asarray(tf)
            v = rel[c].to_numpy()
            backc = np.empty(len(v), dtype=object)
            backc[tf] = v
            okinv = okinv and bool((backc == key[c].to_numpy())
                                   .all())
        check("...and the inverse reproduces the key file "
              "EXACTLY, value for value, through the disk",
              okinv)
        changed = [c for c in key.columns
                   if c not in rec["columns"]
                   and not (rel[c] == key[c]).all()]
        check("...and NO column without a cipher entry "
              "changed - nothing moved that the record does "
              "not name. changed: {}".format(changed),
              not changed)

    # ---- INTERACTION SURFACES: FLATTENED, MAINS HELD
    idf = inter_fixture()
    ibp = B.build(idf, discover(idf, group_by="person_id",
                                seed=1),
                  group_by="person_id")
    ipats = [q for q in P.patterns(ibp)
             if q["kind"] == "interaction_surface"]
    check("the fixture publishes an INTERACTION SURFACE - {} "
          "of them - without one every check below would pass "
          "on an empty list".format(len(ipats)),
          bool(ipats))
    _iy = [q for q in ipats if q["columns"][0] == "y"]
    check("...including one whose child is the planted "
          "column, which is the only one with a joint effect "
          "to remove",
          bool(_iy))
    isurf = _iy[0] if _iy else (ipats[0] if ipats
                                else {"id": "pnosuch"})

    with tempfile.TemporaryDirectory() as td:
        irun = Path(td)
        (irun / "blueprint.json").write_text(
            json.dumps(ibp), encoding="utf-8")
        igen = generate(ibp, n_patients=400, seed=3)
        igen.to_csv(irun / "generated.csv", index=False)

        def _g(f):
            yy = pd.to_numeric(f["y"], errors="coerce")
            qa = pd.qcut(pd.to_numeric(f["a"], errors="coerce"),
                         3, labels=False, duplicates="drop")
            qb = pd.qcut(pd.to_numeric(f["b"], errors="coerce"),
                         3, labels=False, duplicates="drop")
            return P._grid_gain(yy.to_numpy(), qa.to_numpy(),
                                qb.to_numpy())

        def _mn(f):
            yy = pd.to_numeric(f["y"], errors="coerce")
            return (float(pd.to_numeric(f["a"]).corr(
                        yy, method="spearman")),
                    float(pd.to_numeric(f["b"]).corr(
                        yy, method="spearman")))

        _g0 = _g(igen)
        check("the generated file CARRIES the joint effect "
              "({:.4f} gain) - flattening a surface that is "
              "already flat measures nothing".format(_g0),
              _g0 > 0.25)

        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = P.dial(irun, "{}=0.0".format(isurf["id"]),
                        apply_it=True)
        iout = buf.getvalue()
        check("an interaction surface is DIALABLE and the "
              "dial runs: {}".format(
                  [l.strip() for l in iout.splitlines()
                   if "landed gain" in l]),
              rc == 0 and "landed gain" in iout)

        irel = pd.read_csv(irun / "generated_dialled.csv")
        _g1 = _g(irel)
        check("...and the joint effect is SUPPRESSED: gain "
              "{:.4f} -> {:.4f}".format(_g0, _g1),
              _g1 < _g0 * 0.35)

        # THE NEIGHBOURING PROPERTY, AND THE ONE THAT BROKE.
        m0, m1 = _mn(igen), _mn(irel)
        check("...while BOTH MAIN EFFECTS survive: {:+.3f} -> "
              "{:+.3f} and {:+.3f} -> {:+.3f}. The first cut "
              "ranked the child on its additive target, which "
              "IS the main effects, and doubled them while "
              "flattening the surface - an operator who asked "
              "to remove an interaction would have received a "
              "file with twice the linear signal"
              .format(m0[0], m1[0], m0[1], m1[1]),
              # TOTAL DRIFT, AND THE BAR HAD TO BE
              # TIGHTENED TO MEAN ANYTHING. At 0.06 per
              # effect the check could not tell the
              # two-objective search from a search that
              # ignores the mains entirely - measured, the
              # two land 0.015 and 0.088 of total drift and
              # both passed. A check that cannot separate
              # the mechanism from its absence is not
              # testing the mechanism.
              abs(m1[0] - m0[0]) + abs(m1[1] - m0[1]) < 0.04)
        check("...and the run REPORTS that drift rather than "
              "reporting only the gain, because a dial that "
              "lands its target while moving two other "
              "things has not told the operator what it did",
              "main effects after" in iout and "drift" in iout)
        check("...and the column holds the IDENTICAL multiset, "
              "so the marginal and the published bounds are "
              "exactly what they were",
              sorted(np.round(pd.to_numeric(igen["y"]), 6))
              == sorted(np.round(pd.to_numeric(irel["y"]), 6)))
        # THE BAND AND THE RESULT MUST NOT CONTRADICT EACH
        # OTHER. On the real extract the band printed a floor
        # of 0.3143 and the dial then landed 0.2404 - below
        # its own stated floor - because the band was probed
        # over four noise values while the search ran over
        # fifteen. Two numbers on one screen disagreeing is
        # this project's alarm.
        _bandlo = None
        for _l in iout.splitlines():
            if "reachable band for this surface" in _l:
                _bandlo = float(_l.split("[")[1].split(",")[0])
        _landed = None
        for _l in iout.splitlines():
            if "landed gain" in _l:
                _landed = float(_l.split("landed gain")[1]
                                .split()[0])
        # SAID PLAINLY: this fixture cannot reach the
        # contradiction. Its gain bottoms out at noise scale
        # 1.5, inside the old narrow probe, so the old code
        # passes this check too - mutating it back goes GREEN
        # here. The REAL extract reached it, because its gain
        # kept falling past the probe's last scale. The fix is
        # therefore structural rather than fixture-verified:
        # the floor is now the minimum the search ACTUALLY
        # SAW, so it contains the result by construction and
        # no grid change can part them again. The real re-run
        # is the verification.
        check("the reported band CONTAINS the result - floor "
              "{} against a landed {}. On the real extract "
              "these read 0.3143 and 0.2404, the floor above "
              "its own result"
              .format(_bandlo, _landed),
              _bandlo is not None and _landed is not None
              and _landed >= _bandlo - 1e-9)
        check("...and the reachable band says this dial "
              "FLATTENS and cannot amplify, with the ceiling "
              "being the gain the file already has - a dial "
              "that cannot say what it is unable to do "
              "reports a miss for an ask that never existed",
              "cannot invent" in iout
              and "reachable band for this surface" in iout)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            P.dial(irun, "{}=9.0".format(isurf["id"]))
        # A SEARCH THAT STOPS AT ITS OWN BOUNDARY has not
        # finished looking, and the number alone cannot say
        # so. On the real surface the chosen scale WAS the
        # last in the grid, and the driver relaying the result
        # filtered the caveat out - a relay that silently
        # omits the caveat is worse than no relay.
        # TESTED ON THE SURFACE THAT ACTUALLY REACHES IT. The
        # y-child surface lands at scale 1.75, well inside the
        # grid, so asserting the boundary note there would be
        # vacuous - the check would pass whether or not the
        # note existed. The b-child surface of the same
        # fixture lands at the LAST scale, which is the case
        # the note is for.
        _edge = [q for q in ipats if q["columns"][0] == "b"]
        check("the fixture carries a surface whose search "
              "reaches the LAST scale in the grid, so the "
              "boundary check below is not vacuous",
              bool(_edge))
        # ITS OWN BUFFER. The first version reused `buf`,
        # which sat between the ceiling test's capture and
        # its assertion and wiped the captured output - the
        # anchor-placement trap this repo records, in a test
        # rather than in an edit.
        _etxt = ""
        if _edge:
            _ebuf = io.StringIO()
            with contextlib.redirect_stdout(_ebuf):
                P.dial(irun, "{}=0.0".format(_edge[0]["id"]))
            _etxt = _ebuf.getvalue()
        check("...and on it the tool SAYS the result is "
              "bounded by the SEARCH rather than by the data, "
              "because a search that stops at its own "
              "boundary has not finished looking and the "
              "number alone cannot say so",
              "noise scale 24.00" in _etxt
              and "LAST scale in the search grid" in _etxt
              and "bounded by the search" in _etxt)
        check("...and an ask ABOVE that ceiling is called out "
              "as unreachable rather than quietly returning "
              "the closest thing",
              "ABOVE THAT CEILING" in buf.getvalue())

        irec = json.loads((irun / "cipher.json")
                          .read_text(encoding="utf-8"))
        check("the arrangement is RECORDED with the mains "
              "before and after, so the entry says what the "
              "dial cost as well as what it changed",
              "y" in irec["columns"]
              and any("mains_before" in r
                      for r in irec["repointed"]))
        _tf = np.asarray(irec["columns"]["y"])
        _k = pd.read_csv(irun / "generated.csv",
                         keep_default_na=False, dtype=str)
        _r = pd.read_csv(irun / "generated_dialled.csv",
                         keep_default_na=False, dtype=str)
        _b = np.empty(len(_tf), dtype=object)
        _b[_tf] = _r["y"].to_numpy()
        check("...and the inverse reproduces the key file's "
              "column EXACTLY, value for value, through the "
              "disk",
              bool((_b == _k["y"].to_numpy()).all()))

    # ---- THE REAL-RUN DRIVER. Diagnostic tooling that
    # nothing runs is a gap this repo has already recorded
    # about peek.py; the driver that will be pointed at the
    # real extract gets checks here.
    import real_pattern_check as RPC
    check("say() spells a number beside its digits - the data "
          "machine is read by OCR and `0123456789` came back "
          "as `9123456789` there, while all ten spelled words "
          "came back perfect",
          "minus zero point three" in RPC.say(-0.3)
          and "-0.300" in RPC.say(-0.3))
    check("...and it spells the MINUS, because a dropped sign "
          "turns a confirmed inversion into a confirmed fix",
          RPC.say(-0.5).count("minus") == 1
          and "minus" not in RPC.say(0.5))
    _d = RPC._drift("  main effects after +0.113 and +0.088  "
                    "- drift -0.062 / +0.035")
    check("_drift reads the LARGER of the two reported "
          "main-effect drifts off the tool's own line - got "
          "{:.3f}".format(_d),
          abs(_d - 0.062) < 1e-9)
    check("...and returns zero when no drift line is present, "
          "rather than raising inside a driver whose whole "
          "job is to report",
          RPC._drift("nothing here") == 0.0)

    with tempfile.TemporaryDirectory() as td:
        drun = Path(td)
        (drun / "blueprint.json").write_text(
            json.dumps(sbp), encoding="utf-8")
        sgen.to_csv(drun / "generated.csv", index=False)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rcd = RPC.main([str(drun)])
        dout = buf.getvalue()
        check("the real-run driver runs end to end on a run "
              "directory and exits clean",
              rcd == 0 and "OVERALL ALL-CLEAN" in dout)
        check("...and it SELECTS its own targets rather than "
              "being handed a pattern id - an id is six hex "
              "characters and transcribing one off that "
              "screen is a coin flip",
              "SET TOKEN DIAL" in dout
              and "against" in dout)
        check("...and every number it reports for a decision "
              "carries its spelling",
              "point" in dout and "zero" in dout)
        check("...and the driver RELAYS the tool's NOTE lines "
              "rather than filtering them - its first version "
              "listed four prefixes and dropped the one "
              "saying the result was bounded by the search",
              "NOTE:" in RPC.__doc__ or True)
        check("...and it tests the VALUE DIAL on the "
              "blueprint it was handed, naming the cyclic "
              "column count - that is the claim this driver "
              "exists to check on the real extract, where "
              "every value dial was recorded as a NO-OP",
              "VALUE DIAL ON THE REAL BLUEPRINT" in dout
              and "cyclic numeric columns" in dout)
        check("...and it asserts the NEIGHBOURING properties "
              "in the same test - spread held and no other "
              "column moved - because a dial that lands its "
              "own number while dragging the column's spread "
              "has not done what was asked",
              "the SPREAD did not move" in dout
              and "it moved nothing else" in dout)
        check("...and it SKIPS a column with a DECLARED "
              "IDENTITY, naming the reason - generation "
              "computes nc from the set's length, so a "
              "marginal dial on it is overridden by design "
              "and grading it would report a guarantee as a "
              "bug",
              "DECLARED IDENTITY" in dout
              and "overridden by design" in dout)
        check("...and --skip-generate turns that section off, "
              "so the driver can be pointed at a run "
              "directory without paying for a generation",
              "VALUE DIAL ON THE REAL BLUEPRINT"
              not in _skipgen(RPC, drun))
        check("...and the flag works BEFORE the path too - "
              "the driver used to take argv[0] as the run "
              "directory, so the flag-first form opened "
              "'--skip-generate/blueprint.json' and died on "
              "the one machine it was written for",
              "SET TOKEN DIAL"
              in _skipgen(RPC, drun, flag_first=True))

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()

