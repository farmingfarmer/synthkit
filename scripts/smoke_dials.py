"""Smoke: the dials are reachable, and they report what they did.

WHY THIS EXISTS. `resolve` honored eight dials and `smoke_generate`
proved each changed the output, but there was no way to reach one: no
flag, no interface, no documentation. The only route was hand-editing
`blueprint.json` ON THE MACHINE HOLDING THE EXTRACT, which is the
worst place to edit anything, because a typo there is invisible from
where the code is written.

TWO CHECKS HERE ARE LOAD-BEARING.

A MISSPELLED DIAL MUST BE AN ERROR. `age.shitf=5` that is quietly
ignored looks exactly like a dial with no effect, and this project's
own rule is that a flag read but not used is worse than no flag.

AND THE ANSWER MUST BE RE-MEASURED, not read back out of the file it
was written into - which proves the write happened and nothing else.
That check caught a real defect on its first run: `shift` and `scale`
were applied as `(x + shift) * scale`, so a shift of 12 beside a scale
of 1.5 arrived as +35.8 and the scale dragged the center with it.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit import dials as D                        # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def main():
    # ---- parsing -------------------------------------------------
    got = D.parse(["age.shift=5", "lab.scale=1.5",
                   "patients.count=200"])
    check("a dial is parsed into the column and the name it names",
          got == {"age": {"shift": 5.0}, "lab": {"scale": 1.5},
                  "patients": {"count": 200.0}})

    for bad, why in (("ageshift=5", "no dot"),
                     ("age.shift", "no value"),
                     ("age.shift=high", "not a number")):
        try:
            D.parse([bad])
            raised = False
        except ValueError as e:
            raised = "--dial" in str(e) or "number" in str(e)
        check("{!r} is refused with a sentence rather than a "
              "traceback - this runs where a traceback costs a round "
              "trip ({})".format(bad, why), raised)

    # ---- applying ------------------------------------------------
    bp = {"columns": {"age": {"kind": "numeric", "dials": {}},
                      "site": {"kind": "categorical", "dials": {}}},
          "patients": {"count": 100, "dials": {}}}
    problems = D.apply(bp, {"age": {"shift": 5.0}})
    check("a good dial reaches the blueprint's own dial block",
          not problems
          and bp["columns"]["age"]["dials"]["shift"] == 5.0)

    # THE LOAD-BEARING ONE.
    problems = D.apply(bp, {"aeg": {"shift": 5.0}})
    check("a column that does not exist is REPORTED, not ignored - a "
          "misspelling that is silently dropped looks exactly like a "
          "dial with no effect", len(problems) == 1)
    check("...and the message suggests what was meant, because the "
          "operator cannot see the column list from where they are",
          "age" in problems[0])

    problems = D.apply(bp, {"age": {"shitf": 5.0}})
    check("a dial name that does not exist is reported too, and the "
          "message lists the ones that do",
          len(problems) == 1 and "shift" in problems[0])

    # ---- verifying, against generated data -----------------------
    import numpy as np
    import pandas as pd
    from synthkit import blueprint as B
    from synthkit.discover import discover
    from synthkit.generate import generate

    r = np.random.RandomState(3)
    n_pat, n_vis = 260, 8
    g = np.repeat(np.arange(n_pat), n_vis)
    x = r.normal(50, 10, len(g))
    df = pd.DataFrame({
        "person_id": ["P{:04d}".format(i) for i in g],
        "x": np.round(x, 2),
        "y": np.round(2.5 * x + r.normal(0, 6, len(g)), 2)})
    cat = discover(df, group_by="person_id", seed=1)
    base_bp = B.build(df, cat, group_by="person_id")
    src_sd = float(df["x"].std())

    overlay = {"x": {"shift": 20.0}}
    tuned = json.loads(json.dumps(base_bp))
    check("applying to a real blueprint reports no problem",
          not D.apply(tuned, overlay))
    gen = generate(tuned, n_patients=n_pat, seed=7)
    rows = D.verify(tuned, overlay, df, gen, "person_id")
    shift_row = [q for q in rows if q["dial"] == "x.shift"][0]
    check("the shift is RE-MEASURED on the generated frame - "
          "requested {:g}, achieved {:.2f}".format(
              shift_row["requested"], shift_row["achieved"]),
          shift_row["achieved"] is not None)
    check("...and it is reported as hit, because it was",
          shift_row["hit"] is True)
    check("...and stated in sds too, which is what says whether the "
          "move was large FOR THIS COLUMN",
          abs(shift_row["achieved_in_sds"] - 20.0 / src_sd) < 0.25)

    # ---- A VALUE DIAL CAN LAND PERFECTLY AND STILL DELETE THE
    # COLUMN'S RELATIONSHIPS. Found by the dial matrix, measured
    # before it was reported: a child is drawn from a curve
    # published over the parent's OBSERVED range and np.interp
    # holds flat outside it, so shifting a parent far enough
    # stops its children tracking it. Swept on this fixture,
    # parent~child Spearman over three seeds:
    #     0.0 sd  0.968     2.0 sd  0.832
    #     1.0 sd  0.965     3.0 sd  0.313     5.0 sd  -0.027
    # The dial reports a clean hit throughout. That silent
    # difference is exactly what this module exists to refuse.
    def _pair(fr):
        a_ = pd.to_numeric(fr["x"], errors="coerce")
        b_ = pd.to_numeric(fr["y"], errors="coerce")
        m_ = a_.notna() & b_.notna()
        return float(a_[m_].corr(b_[m_], method="spearman"))

    big_ov = {"x": {"shift": 3.0 * src_sd}}
    big_bp = json.loads(json.dumps(base_bp))
    D.apply(big_bp, big_ov)
    big_gen = generate(big_bp, n_patients=n_pat, seed=7)
    big_row = [q for q in D.verify(base_bp, big_ov, df, big_gen,
                                   "person_id")
               if q["dial"] == "x.shift"][0]
    check("THE PHENOMENON ITSELF: a 3 sd shift on a parent "
          "flattens its relationship to its child, from {:.2f} "
          "to {:.2f}".format(_pair(gen), _pair(big_gen)),
          _pair(big_gen) < _pair(gen) - 0.3)
    check("...and the dial still reports a clean HIT, which is "
          "why the collateral has to be reported beside it "
          "rather than left to be noticed",
          big_row.get("hit") is True)
    check("...so the row carries the collateral as DATA - the "
          "share outside the published grid and the children "
          "that will have flattened",
          isinstance(big_row.get("collateral"), dict)
          and big_row["collateral"]["outside_published_range"] > 0.2
          and "y" in big_row["collateral"]["children"])
    check("...and says it in a sentence an operator can act on",
          "OUTSIDE its published grid" in (big_row.get("note")
                                           or "")
          and "flattened" in (big_row.get("note") or ""))
    small_ov = {"x": {"shift": 0.5 * src_sd}}
    small_bp = json.loads(json.dumps(base_bp))
    D.apply(small_bp, small_ov)
    small_gen = generate(small_bp, n_patients=n_pat, seed=7)
    small_row = [q for q in D.verify(base_bp, small_ov, df,
                                     small_gen, "person_id")
                 if q["dial"] == "x.shift"][0]
    check("...while a SAFE shift reports no collateral at all - "
          "it marks a real property rather than printing a "
          "warning on everything",
          small_row.get("collateral") is None
          and not small_row.get("note"))

    # ---- A VALUE DIAL ON A CURVE-DRAWN COLUMN DOES NOTHING,
    # AND USED TO SAY NOTHING ABOUT WHY. Found by taking the
    # dial matrix to the real extract, where EVERY value dial
    # on span_days read NO-OP - and every well-covered numeric
    # there is a relationship child. Reproduced here on four
    # columns, shift 40:
    #     c    no parents                 +40.00  applied
    #     b    child, marginal-drawn      +40.00  applied
    #     a    child, curve-drawn           0.00  NO-OP
    #     out  child, curve-drawn           0.00  NO-OP
    # The dial reaches the blueprint, `apply` reports no
    # problem, and the number does not move. Which children are
    # curve-drawn is decided by cycle-breaking and trim order,
    # so an operator cannot read it off the blueprint - which
    # is why the MISS has to explain itself.
    _r9 = np.random.RandomState(5)
    _n9, _v9 = 300, 8
    _g9 = np.repeat(np.arange(_n9), _v9)
    _a9 = np.round(_r9.normal(50, 10, len(_g9)), 2)
    _b9 = np.round(_r9.normal(10, 3, len(_g9)), 2)
    _c9 = np.round(_r9.normal(-5, 2, len(_g9)), 2)
    _o9 = np.round(1.5 * _a9 + 2.0 * _b9 + 0.5 * _c9
                   + _r9.normal(0, 4, len(_g9)), 2)
    _df9 = pd.DataFrame({
        "person_id": ["P{:04d}".format(i) for i in _g9],
        "a": _a9, "b": _b9, "c": _c9, "out": _o9})
    _bp9 = B.build(_df9, discover(_df9, group_by="person_id",
                                  seed=1),
                   group_by="person_id")
    _kids9 = {q.get("child")
              for q in (_bp9.get("relationships") or [])}
    check("the no-op fixture actually CONTAINS a curve-drawn "
          "child - a check about a dial that cannot apply is "
          "vacuous without one",
          "out" in _kids9)

    def _row9(col):
        _ov = {col: {"shift": 40.0}}
        _b = json.loads(json.dumps(_bp9))
        D.apply(_b, _ov)
        _gg = generate(_b, n_patients=_n9, seed=7)
        return [q for q in D.verify(_bp9, _ov, _df9, _gg,
                                    "person_id")
                if q["dial"] == col + ".shift"][0]

    _miss = _row9("out")
    _hits = _row9("c")
    check("a value dial on a column with NO parents applies "
          "exactly - requested 40, achieved {:.2f}".format(
              _hits.get("achieved") or 0),
          _hits.get("hit") is True)
    # THESE TWO CHECKS PINNED A DEFECT, AND THE DEFECT IS
    # FIXED - so they went red, which is what a
    # characterization test is for. They asserted that the
    # same dial on a CURVE-DRAWN CHILD did nothing. That was
    # never the real rule: measured across all four cells with
    # a control in each run, an acyclic curve-drawn child read
    # the dial perfectly and only a CYCLIC one missed, because
    # the refinement sweep re-ranks onto a marginal pool
    # captured before the dial. The pool moves with the dial
    # now, and the assertion is the opposite of what it was.
    check("...and the SAME dial on a CURVE-DRAWN CHILD now "
          "applies too - requested 40, achieved {:.2f}. This "
          "assertion is inverted from the one it replaces: it "
          "used to pin the no-op as a known defect"
          .format(_miss.get("achieved") or 0),
          _miss.get("hit") is True)
    check("...and a landed dial carries no excuse note, so an "
          "explanation marks a real cause rather than "
          "printing on every row",
          "relationship CHILD" not in (_hits.get("note") or "")
          and "relationship CHILD" not in (_miss.get("note")
                                           or ""))

    # AND THE CELL THAT ACTUALLY BROKE IS PINNED BY NAME. The
    # fixture above has to CONTAIN a cyclic column or the
    # inverted assertion is vacuous - the acyclic case passed
    # before the fix and would pass after it, so a fixture
    # with no cycle proves nothing about the change.
    _cyc9 = set()
    for _q in (_bp9.get("relationships") or []):
        for _p in (_q.get("parents") or []):
            if any(_z.get("child") == _p
                   and "out" in (_z.get("parents") or [])
                   for _z in (_bp9.get("relationships") or [])):
                _cyc9.add(_q.get("child"))
    check("the fixture contains a column inside a CYCLE - the "
          "only cell that ever missed - so the inverted "
          "assertion above is not passing on the acyclic case "
          "that always worked",
          bool(_cyc9))

    # THE PROPERTY THE FIX RESTS ON, asserted rather than
    # described: a shift moves the marginal pool AS A WHOLE, so
    # the dialled column holds the undialed multiset translated
    # by exactly the shift. If the sweep had drifted, or
    # re-ranked onto a different pool, the sorted values would
    # not line up.
    _plain = generate(json.loads(json.dumps(_bp9)),
                      n_patients=_n9, seed=7)
    _bs = json.loads(json.dumps(_bp9))
    D.apply(_bs, {"out": {"shift": 40.0}})
    _shift = generate(_bs, n_patients=_n9, seed=7)
    _u = np.sort(np.asarray(_plain["out"], dtype=float))
    _d = np.sort(np.asarray(_shift["out"], dtype=float))
    _worst = float(np.nanmax(np.abs(_d - _u - 40.0))) \
        if len(_u) == len(_d) else 1e9
    check("a shift TRANSLATES the whole multiset - the sorted "
          "dialled values equal the sorted undialed values "
          "plus the shift, worst gap {:.4f}. The sweep still "
          "re-ranks onto exactly the multiset the marginal "
          "produced, which is what stops it drifting"
          .format(_worst),
          _worst < 0.01)

    # THE OTHER BLUEPRINT DIALS BITE, AND THAT IS ASSERTED
    # RATHER THAN ASSUMED. The bench and the pattern report
    # both used to tell a reader the recipe layer "was
    # measured inert" - a claim carried over from shift/scale
    # on a cyclic column, which was the only cell that ever
    # missed and is now fixed. persistence and coverage were
    # never in that measurement. If either of them ever DOES
    # go inert, the pages are lying again, so the claim gets a
    # check instead of a comment.
    _r8 = np.random.RandomState(11)
    _n8, _v8 = 200, 8
    _g8 = np.repeat(np.arange(_n8), _v8)
    _lv8 = np.repeat(_r8.normal(0, 1, _n8), _v8)
    _st8 = np.round(100 + 15 * _lv8 + _r8.normal(0, 4, len(_g8)), 2)
    _sp8 = np.round(_r8.normal(5, 1, len(_g8)), 2)
    _sp8 = np.where(_r8.rand(len(_g8)) < 0.6, _sp8, np.nan)
    _df8 = pd.DataFrame({
        "person_id": ["Q{:04d}".format(i) for i in _g8],
        "sticky": _st8, "sparse": _sp8,
        "plain": np.round(_r8.normal(7, 2, len(_g8)), 2)})
    _bp8 = B.build(_df8, discover(_df8, group_by="person_id",
                                  seed=2),
                   group_by="person_id")

    def _gen8(col, **dl):
        _b = json.loads(json.dumps(_bp8))
        if dl:
            D.apply(_b, {col: dl})
        return generate(_b, n_patients=_n8, seed=9)

    def _between(g, col):
        _d = pd.DataFrame({
            "p": g["person_id"],
            "v": pd.to_numeric(g[col], errors="coerce")})
        _t = _d["v"].var()
        if not _t or not np.isfinite(_t) or _t <= 0:
            return float("nan")
        return float(_d.groupby("p")["v"].mean().var() / _t)

    _b_off = _between(_gen8("sticky"), "sticky")
    _b_on = _between(_gen8("sticky", persistence=0.4), "sticky")
    check("the fixture's sticky column IS sticky before the "
          "dial ({:.3f} of its variance between patients) - a "
          "persistence dial measured on a column that "
          "re-rolls every row is the vacuous "
          "between-patient check this repo already paid for"
          .format(_b_off),
          _b_off > 0.5)
    check("...and the PERSISTENCE dial bites: between-patient "
          "share {:.3f} -> {:.3f} on an ask of 0.4x. The "
          "recipe layer is not inert, which is why the pages "
          "no longer say it is"
          .format(_b_off, _b_on),
          _b_on < _b_off - 0.1)

    def _cov8(g):
        return float(pd.to_numeric(g["sparse"],
                                   errors="coerce").notna().mean())
    _c_off = _cov8(_gen8("sparse"))
    _c_on = _cov8(_gen8("sparse", coverage=0.30))
    check("...and the COVERAGE dial lands on its ASK rather "
          "than merely moving - presence {:.3f} -> {:.3f} "
          "against a requested 0.30".format(_c_off, _c_on),
          abs(_c_on - 0.30) < 0.05 and _c_off > 0.5)

    # A dial that genuinely does not arrive must read as a MISS. The
    # report is worthless if it cannot say no.
    faked = [{"dial": "x.shift", "requested": 20.0,
              "achieved": 0.4, "hit": False}]
    text = D.render(faked)
    check("a dial that did NOT arrive renders as MISS - a report that "
          "cannot say no is decoration",
          "MISS" in text and "20" in text)
    check("...and the report explains that a miss is not always a "
          "fault, so nobody hunts a sampler bug that was the privacy "
          "rule",
          "bound" in text and "not always a fault" in text)

    # ---- the CLI actually exposes all of this --------------------
    out = subprocess.run(
        [sys.executable, "-c",
         "from synthkit.cli import main; main(['--help'])"],
        capture_output=True, text=True, cwd=str(ROOT))
    helptext = (out.stdout or "") + (out.stderr or "")
    for verb in ("fit", "types", "dials"):
        check("`synthkit {}` is a real subcommand - the fitted path "
              "was reachable only as a script under scripts/, which "
              "pip does not install".format(verb),
              "{}".format(verb) in helptext)

    fit_help = subprocess.run(
        [sys.executable, "-c",
         "from synthkit.cli import main; main(['fit','--help'])"],
        capture_output=True, text=True, cwd=str(ROOT))
    ftext = (fit_help.stdout or "") + (fit_help.stderr or "")
    check("`synthkit fit` carries the pipeline's own flags rather "
          "than a second copy of them",
          "--src" in ftext and "--lags" in ftext
          and "--dial" in ftext and "--types-only" in ftext)

    # The shim must keep working: docs/WINDOWS.md names it, and that
    # ritual is pasted by hand.
    shim = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_discovery.py"),
         "--help"], capture_output=True, text=True, cwd=str(ROOT))
    check("the old `scripts/run_discovery.py` path still runs - "
          "WINDOWS.md names it and an operator may hold an older "
          "copy of the guide",
          shim.returncode == 0 and "--src" in (shim.stdout or ""))

    with tempfile.TemporaryDirectory() as td:
        bp_path = Path(td) / "blueprint.json"
        bp_path.write_text(json.dumps(base_bp), encoding="utf-8")
        listing = subprocess.run(
            [sys.executable, "-c",
             "from synthkit.cli import main; "
             "main(['dials', r'{}'])".format(bp_path)],
            capture_output=True, text=True, cwd=str(ROOT))
        lt = listing.stdout or ""
        check("`synthkit dials` names the tunable columns, so nobody "
              "has to open the JSON to find out what exists",
              listing.returncode == 0 and "x" in lt and "shift" in lt)
        check("...and tells the reader how to set one",
              "--dial" in lt)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    relationship_dial_checks()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


def relationship_dial_checks():
    """The edge dial, held to the contract its first cut broke.

    The record-level version was measured killing an INNOCENT
    NEIGHBOR edge (span~drug collapsed to 0.007 when span<-proc
    was dialed to 0) while the named pair survived through the
    reverse record at 0.907. Per-edge routing fixed both; these
    checks hold the whole contract: unit branches, then a real
    two-fit measurement - both directions to 0 kills the pair
    AND spares the neighbor, and strength=1.0 is bit-identical
    to an undialed run."""
    import hashlib
    import numpy as np
    import pandas as pd

    # ---- unit branches ----
    bp = {"columns": {}, "relationships": [
        {"child": "span", "parents": ["proc", "drug"]},
        {"child": "proc", "parents": ["span"]}]}
    probs = D.apply(bp, D.parse(["span<-proc.strength=0"]))
    check("an edge dial routes to edge_strength on the ONE named "
          "parent, never the record",
          not probs
          and bp["relationships"][0]["dials"]["edge_strength"]
          ["proc"] == 0.0
          and "strength" not in bp["relationships"][0]["dials"]
          or "edge_strength" in bp["relationships"][0]["dials"])
    probs2 = D.apply(bp, D.parse(["spam<-proc.strength=0"]))
    check("...an unknown edge is an error with suggestions, not "
          "a shrug",
          probs2 and "did you mean" in probs2[0])
    src = pd.DataFrame({"span": [1, 2, 3, 4] * 30,
                        "proc": [1, 2, 3, 4] * 30,
                        "drug": [4, 3, 2, 1] * 30})
    gen = pd.DataFrame({"span": [1, 2, 3, 4] * 30,
                        "proc": [2, 1, 4, 3] * 30,
                        "drug": [4, 3, 2, 1] * 30})
    rows = D.verify(bp, {"span<-proc": {"strength": 0.0}},
                    src, gen)
    check("...a single edge at 0 with a LIVE reverse edge is "
          "informational, naming the reverse edge instead of "
          "judging a target it cannot hit",
          rows[0]["hit"] is None
          and "reverse" in (rows[0].get("note") or "").lower())
    txt = D.render(rows)
    check("...and render shows it as informational with the note, "
          "never as MISS - a note explaining the number must not "
          "sit under a verdict contradicting it",
          "MISS" not in txt and "note:" in txt)

    # ---- the two-fit contract, on a small planted triangle ----
    r = np.random.RandomState(0)
    n_pat, vis = 120, 6
    n = n_pat * vis
    lat = np.repeat(r.normal(0, 1, n_pat), vis) + r.normal(
        0, .4, n)
    y3 = 4 + 1.1 * lat + r.normal(0, 0.9, n)
    y2 = 6 + 1.3 * lat + 0.45 * (y3 - y3.mean()) + r.normal(
        0, 1.0, n)
    y1 = 3 + 1.0 * (y2 - y2.mean()) + 0.75 * (y3 - y3.mean()) \
        + r.normal(0, 1.1, n)
    df = pd.DataFrame({
        "person_id": ["P{:03d}".format(i)
                      for i in np.repeat(np.arange(n_pat), vis)],
        "span": np.round(y1, 2), "proc": np.round(y2, 2),
        "drug": np.round(y3, 2)})
    with tempfile.TemporaryDirectory() as td:
        csv = Path(td) / "t.csv"
        df.to_csv(csv, index=False)
        root = Path(__file__).resolve().parent.parent

        def fit(out, *dials_args):
            cmd = [sys.executable, "-m", "synthkit.cli", "fit",
                   "--src", str(csv), "--out", str(Path(td) / out),
                   "--group-by", "person_id", "--seed", "11",
                   "--generate"]
            for d_ in dials_args:
                cmd += ["--dial", d_]
            q = subprocess.run(cmd, capture_output=True,
                               text=True, cwd=str(root))
            assert q.returncode == 0, (q.stdout + q.stderr)[-800:]
            return pd.read_csv(Path(td) / out / "generated.csv")

        g_base = fit("base")
        g_noop = fit("noop", "span<-proc.strength=1.0")
        g_kill = fit("kill", "span<-proc.strength=0",
                     "proc<-span.strength=0")

        def c(f, a, b):
            return abs(float(f[a].corr(f[b], method="spearman")))
        check("both directions at 0 REMOVE the pair (|r| {:.2f} "
              "-> {:.2f}) while the neighbor edge survives "
              "({:.2f} -> {:.2f}) - the exact combination the "
              "record-level cut failed".format(
                  c(g_base, "span", "proc"),
                  c(g_kill, "span", "proc"),
                  c(g_base, "span", "drug"),
                  c(g_kill, "span", "drug")),
              c(g_kill, "span", "proc") < 0.15
              and c(g_kill, "span", "drug")
              > 0.5 * c(g_base, "span", "drug"))
        h = lambda f: hashlib.sha256(
            f.to_csv(index=False).encode()).hexdigest()
        check("...and strength=1.0 is bit-identical to an "
              "undialed run - a no-op dial must be a true no-op",
              h(g_noop) == h(g_base))


if __name__ == "__main__":
    main()
