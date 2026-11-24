"""Which pass drops a published curve - the instrument, pinned.

WHY THIS SUITE EXISTS. On the real extract the published curve for
`height <- year_of_birth` spans 6.5473 raw over 27,336 rows and the
rank correlation of that curve against the GENERATED height is
0.0405: the output does not follow it at all. Four mechanisms were
measured and refuted - the cyclic sweep's rank map AMPLIFIES, the
parents' correlation SURVIVES, the published partials DO
reconstruct both marginals on a fixture with the same collinearity,
and the sweep does not skip the column.

`generate` moves a column at a dozen points after it is drawn, so
the question is which one. `TRACE_CURVES` takes the same
correlation at every pass boundary and the trajectory names the
pass. It is READ-ONLY, which is the property this suite exists to
hold: an instrument that changes the thing it measures is worse
than no instrument.
"""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from synthkit.discover import discover                 # noqa: E402
import synthkit.blueprint as B                         # noqa: E402
import synthkit.generate as G                          # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
import curve_trace as CT                               # noqa: E402
import curve_why as CW                                 # noqa: E402
from curve_why import applied                          # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def digest(df):
    return hashlib.sha256(
        df.to_csv(index=False).encode()).hexdigest()[:16]


def per_visit():
    r = np.random.RandomState(5)
    npat, vis = 700, 6
    n = npat * vis
    yob = np.repeat(r.uniform(0, 1, npat), vis)
    age = np.clip(1.0 - yob + r.normal(0, 0.10, n), 0, 1)
    y = (8.0 * (1.0 - 4.0 * (age - 0.5) ** 2) + 6.0 * yob
         + r.normal(0, 1.0, n))
    return pd.DataFrame({
        "person_id": np.repeat(np.arange(npat), vis).astype(str),
        "yob": np.round(yob, 4), "age": np.round(age, 4),
        "y": np.round(y, 4)})


def patient_level():
    """A PATIENT-LEVEL child with one patient-level parent and one
    that VARIES WITHIN the patient - the shape of `height <-
    year_of_birth` beside `height <- age_at_visit`."""
    r = np.random.RandomState(9)
    npat, vis = 800, 8
    n = npat * vis
    plvl = np.repeat(r.uniform(0, 1, npat), vis)
    wv = np.clip(plvl + r.normal(0, 0.25, n), 0, 1)
    hpat = 10.0 * (1.0 - 4.0 * (r.uniform(0, 1, npat) - 0.5) ** 2)
    h = np.repeat(hpat, vis) + 6.0 * plvl
    return pd.DataFrame({
        "person_id": np.repeat(np.arange(npat), vis).astype(str),
        "plvl": np.round(plvl, 4), "wv": np.round(wv, 4),
        "h": np.round(h, 4)})


def _numf(fr, c):
    return pd.to_numeric(fr[c].replace("", None), errors="coerce")


# `a` is perfectly ANTI-correlated with `b` and only partly with
# `c`, so a computation that took the first co-parent rather than
# the worst would name `c`.
_cop = pd.DataFrame({"a": [1, 2, 3, 4, 5] * 20,
                     "b": [5, 4, 3, 2, 1] * 20,
                     "c": [1, 1, 2, 2, 3] * 20})


def token_driven():
    """A set column whose TOKENS drive a numeric, so the blueprint
    carries relationships with INDICATOR parents.

    Needed because most co-parents on a real blueprint are
    `__has__` indicators, and a fixture whose tokens carry no
    structure produces none - which this project already records
    as the reason a cap experiment measured nothing.
    """
    r = np.random.RandomState(3)
    npat, vis = 400, 6
    n = npat * vis
    toks = ["t%02d" % i for i in range(6)]
    sets, h0, h1 = [], [], []
    for _ in range(n):
        k = r.randint(1, 4)
        pick = list(r.choice(toks, size=k, replace=False))
        sets.append(";".join(sorted(pick)))
        h0.append(1 if "t00" in pick else 0)
        h1.append(1 if "t01" in pick else 0)
    z = r.normal(0, 1, n)
    y = (4.0 * np.array(h0) + 3.0 * np.array(h1) + 1.5 * z
         + r.normal(0, 0.6, n))
    return pd.DataFrame({
        "person_id": np.repeat(np.arange(npat), vis).astype(str),
        "drugs": sets, "z": np.round(z, 4),
        "y": np.round(y, 4)})


def main():
    df = per_visit()
    bp = B.build(df, discover(df, group_by="person_id"),
                 group_by="person_id")

    # ------------------------------------------------------------
    # READ-ONLY, AND OFF BY DEFAULT. An instrument that changes the
    # thing it measures is worse than no instrument, and a probe
    # that costs something on every run is a tax on the shipping
    # path.
    # ------------------------------------------------------------
    check("the trace is OFF by default, so no run pays for it "
          "unless somebody asks", G.TRACE_CURVES is False)
    G.TRACE_CURVES = False
    off = G.generate(bp, seed=11)
    rep_off = {}
    G.generate(bp, seed=11, report=rep_off)
    G.TRACE_CURVES = True
    rep = {}
    on = G.generate(bp, seed=11, report=rep)
    G.TRACE_CURVES = False
    check("...and with it OFF the report carries no trace at all - "
          "absent rather than an empty dict, so a reader cannot "
          "mistake 'not measured' for 'nothing happened'",
          "curve_trace" not in rep_off)
    check("the traced run is BYTE-IDENTICAL to the untraced one, "
          "which is the whole claim: the probe reads and never "
          "writes ({} both ways)".format(digest(off)),
          digest(off) == digest(on))

    tr = rep.get("curve_trace") or {}
    check("...and it traced every published numeric curve, so a "
          "pass naming no curve is a finding rather than an empty "
          "instrument ({} curves)".format(len(tr)), len(tr) >= 4)
    # THE ENTRY GREW A THIRD FIELD - the comparable-row count -
    # so the stage is read by POSITION rather than unpacked.
    stages = [e[0] for e in list(tr.values())[0]]
    check("every probe fires in order, from the draw through to "
          "the arithmetic identities, so a trajectory can name "
          "WHICH pass moved a curve: {}".format(
              ", ".join(stages[:3]) + " ... " + stages[-1]),
          stages[0] == "after-draw"
          and "after-sweeps" in stages
          and "after-masks" in stages
          and stages[-1] == "after-identities"
          and len(stages) >= 8)
    check("...and the same stages are recorded for every curve, or "
          "two trajectories could not be compared",
          all([e[0] for e in v] == stages for v in tr.values()))
    check("a curve the engine DOES carry reads a substantial "
          "correlation at every pass - without one of those this "
          "suite could not tell a dropped curve from an instrument "
          "that reads zero everywhere",
          any(all(rho is not None and abs(rho) > 0.3
                  for rho in (e[1] for e in v)) for v in tr.values()))

    # NO VALUE IS PUBLISHED. The trace carries correlations and
    # column names only - the same rule the marginal measure
    # follows for bin counts.
    import json
    blob = json.dumps(tr)
    check("the trace publishes correlations and column names only, "
          "never a value or a row count - a diagnostic that leaks "
          "the data it describes is a worse problem than the one "
          "it solves",
          "rows" not in blob and "values" not in blob
          and all(isinstance(rho, (float, int)) or rho is None
                  for v in tr.values() for rho in (e[1] for e in v)))

    # ------------------------------------------------------------
    # IT CAN SHOW A DROP, AND NAMES WHERE. A patient-level child
    # takes ONE value per patient, so a curve on a parent that
    # VARIES within the patient cannot survive - the patient's
    # order comes from their MEAN of the applied values, and
    # within-patient variation is averaged out of it by
    # construction. That is `height <- age_at_visit`.
    # ------------------------------------------------------------
    pl = patient_level()
    bp2 = B.build(pl, discover(pl, group_by="person_id"),
                  group_by="person_id")
    check("the fixture's child really is PATIENT-LEVEL, or this "
          "measures nothing - one distinct value per patient, and "
          "the blueprint says so",
          int(pl.groupby("person_id")["h"].nunique().max()) == 1
          and ((bp2.get("columns") or {}).get("h")
               or {}).get("level") == "patient")
    G.TRACE_CURVES = True
    rep2 = {}
    G.generate(bp2, seed=11, report=rep2)
    G.TRACE_CURVES = False
    tr2 = rep2.get("curve_trace") or {}
    wv = tr2.get("h <- wv")
    plv = tr2.get("h <- plvl")
    check("a patient-level child carries its PATIENT-LEVEL "
          "parent's curve - that parent's effect is constant "
          "within the patient, so the mean preserves it" + (
              " (rho {})".format(plv[0][1]) if plv else ""),
          plv is not None and abs(plv[0][1]) > 0.2)
    check("...and CANNOT carry a curve on a parent that varies "
          "WITHIN the patient: near zero from the first probe, "
          "which places the loss inside the draw loop where the "
          "collapse runs" + (
              " (rho {})".format(wv[0][1]) if wv else ""),
          wv is not None and abs(wv[0][1]) < 0.15)
    check("...and that is STRUCTURAL rather than a pass to fix - "
          "it stays near zero through every later pass, so no "
          "trajectory names a culprit and nothing downstream is "
          "to blame",
          wv is not None
          and all(rho is None or abs(rho) < 0.15 for rho in (e[1] for e in wv)))

    # ================================================================
    # THE DRIVER. `locate` is a PURE FUNCTION on the trajectory and
    # is asserted on PLANTED input, because a tool whose entire
    # output is the branch it chose must have every branch
    # exercised - this project has found dead branches in a linkage
    # verdict and in a release-dial report by not doing that.
    # ================================================================
    L = CT.locate
    check("a curve the output follows at the end reads CARRIED",
          L([("a", 0.8), ("b", 0.7), ("c", 0.6)])[0] == "CARRIED")
    check("...and a curve below the floor at the first probe AND "
          "the last reads NEVER-APPLIED naming the draw - no pass "
          "put it there and none took it away, which is a "
          "different finding and a different fix",
          L([("a", 0.04), ("b", 0.03), ("c", 0.02)])
          == ("NEVER-APPLIED", "a", None))

    # THE END IS THE PRIMARY FACT. The first version of this
    # verdict returned NEVER-APPLIED on the first probe alone - and
    # `_order`'s cycle trim REMOVES an unplaced parent from the
    # first pass BY DESIGN, leaving the refinement sweeps to put it
    # back once every column has a value. So a curve the sweeps
    # restored was being reported as never applied, which is the
    # opposite of the truth: that pass was doing its job. Found by
    # reading the trim, not by any check, and the real-extract
    # count of 133 was taken with the broken rule.
    check("...and a curve ABSENT at the draw and present at the "
          "end is CARRIED, naming the pass that first brought it "
          "over the floor - the cycle trim drops an unplaced "
          "parent by design and the sweeps restore it, so judging "
          "on the first probe alone reports a working pass as a "
          "loss",
          L([("a", 0.03), ("b", 0.72), ("c", 0.70)])
          == ("CARRIED", "b", None))
    check("...and a curve applied at the draw and gone at the end "
          "reads DROPPED, naming the pass with the LARGEST single "
          "fall rather than the last one",
          L([("a", 0.8), ("b", 0.05), ("c", 0.04)])[:2]
          == ("DROPPED", "b")
          and L([("a", 0.8), ("b", 0.75), ("c", 0.02)])[:2]
          == ("DROPPED", "c"))
    check("...and a curve that stops being MEASURABLE is DROPPED "
          "at the pass where that began, not silently carried",
          L([("a", 0.8), ("b", None), ("c", None)])[:2]
          == ("DROPPED", "b"))
    check("...and nothing measurable at all is NOT-MEASURABLE "
          "rather than clean - an absent measurement is not a "
          "clean one",
          L([("a", None), ("b", 0.8)])[0] == "NOT-MEASURABLE"
          and L([])[0] == "NOT-MEASURABLE")

    # A REVERSAL IS NOT A SURVIVAL. The probe correlates the
    # curve's OWN delta against the child, so a correct
    # application reads POSITIVE whichever way the curve runs - and
    # a negative reading is the output moving opposite to the
    # published shape, which this project calls worse than a loss
    # because it reads as a finding.
    check("a curve the output follows BACKWARDS reads REVERSED, "
          "naming where the sign turned over - a verdict of "
          "CARRIED there would be the kinder number and the wrong "
          "one",
          L([("a", 0.8), ("b", -0.7), ("c", -0.6)])[:2]
          == ("REVERSED", "b")
          and L([("a", -0.8), ("b", -0.7)])[:2] == ("REVERSED", "a"))
    check("...and a curve faded PAST zero is DROPPED rather than "
          "REVERSED, because the gate's own rule is that an "
          "inversion needs MAGNITUDE - a relationship pushed to "
          "noise is a loss, and the two take different fixes",
          L([("a", 0.8), ("b", 0.3), ("c", -0.02)])[0] == "DROPPED")

    # ONE BAR, NOT TWO. "Did the curve reach the column" is already
    # decided by `curve_why.applied`, and two halves each deciding
    # what applied means is how they come to disagree - this
    # project says that about token screening, the scaffolding
    # predicate and both categorical draw paths.
    check("the driver's floor IS `curve_why.applied` rather than a "
          "second bar of its own",
          CT.applied is applied
          and all(L([("a", r)])[0] == "NEVER-APPLIED"
                  for r in (0.0, 0.05, 0.099))
          and L([("a", 0.11)])[0] != "NEVER-APPLIED")

    # A REPORT THAT LISTS CATEGORIES MUST STATE A TOTAL A READER
    # CAN CHECK AGAINST THEM, or a missing category is invisible in
    # every other way - and a census term that double-counted once
    # printed a NEGATIVE residual on a real run.
    rows = [{"verdict": v} for v in
            ("CARRIED", "CARRIED", "DROPPED", "REVERSED",
             "NEVER-APPLIED", "NOT-MEASURABLE")]
    c = CT.census(rows)
    check("the census is an IDENTITY a reader can check: the "
          "verdicts add to the number traced",
          c["traced"] == 6
          and sum(c[v] for v in CT.VERDICTS) == c["traced"])

    # ---- THE REFUSALS, each by exit code rather than by wording -
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        def drive(*args):
            return subprocess.run(
                [sys.executable,
                 str(ROOT / "scripts" / "curve_trace.py")]
                + [str(x) for x in args],
                cwd=str(ROOT), capture_output=True, text=True,
                timeout=1800)

        bare = td / "bare"
        bare.mkdir()
        r = drive(bare)
        check("a directory with no blueprint is REFUSED by name "
              "rather than traced into nothing",
              r.returncode == 2 and "blueprint.json" in r.stdout)

        # A TRACE AT A DIFFERENT SEED EXPLAINS A DIFFERENT DRAW, so
        # the seed is READ and never chosen. Absent is not zero.
        nb = td / "noseed"
        nb.mkdir()
        (nb / "blueprint.json").write_text("{}", encoding="utf-8")
        r = drive(nb)
        check("...and a blueprint with no provenance beside it is "
              "refused, because the run's own seed is then unknown",
              r.returncode == 2 and "seed" in r.stdout)
        (nb / "provenance.json").write_text(
            json.dumps({"settings": {"group_by": "person_id"}}),
            encoding="utf-8")
        r = drive(nb)
        check("...and a provenance that records NO seed is refused "
              "too - absent is not zero, and a guessed seed would "
              "trace a different draw",
              r.returncode == 2 and "no seed" in r.stdout)

        # ---- END TO END ON A RUN THE REAL CLI WROTE -------------
        #
        # A FIXTURE THAT SHARES THE READER'S INVENTED KEYS IS NOT A
        # TEST. This driver reads `settings.generate_args` and
        # `settings.seed`; a hand-written provenance would agree
        # with whatever names I used here and both would disagree
        # with the only thing that matters. Twelve rows of
        # em-dashes once shipped behind forty green checks for
        # exactly that reason.
        # THE TOKEN FIXTURE, so this one run also carries
        # SCAFFOLDING - which is what makes the surface question
        # below answerable through the driver rather than beside
        # it.
        fsrc = td / "src.csv"
        token_driven().to_csv(fsrc, index=False)
        run = td / "run"
        rv = subprocess.run(
            [sys.executable, "-m", "synthkit.cli", "fit",
             "--src", str(fsrc), "--out", str(run),
             "--group-by", "person_id", "--seed", "11",
             "--refine-sweeps", "1", "--no-identities",
             "--generate"],
            cwd=str(ROOT), capture_output=True, text=True,
            timeout=3600)
        check("the real CLI writes a run this driver can read",
              rv.returncode == 0
              and (run / "blueprint.json").exists()
              and (run / "provenance.json").exists())

        prov = json.loads((run / "provenance.json").read_text(
            encoding="utf-8"))
        use, missing, unknown = CT.recover_args(prov)
        check("...and the driver RECOVERS the keywords that run "
              "called generate with, so the trace is of the same "
              "call rather than of the defaults: refine_sweeps {} "
              "and enforce_identities {}".format(
                  use.get("refine_sweeps"),
                  use.get("enforce_identities")),
              use.get("refine_sweeps") == 1
              and use.get("enforce_identities") is False
              and not missing)

        # ABSENT IS NOT THE DEFAULT. A run made before the record
        # existed carries none of them, and the driver must say in
        # how many ways the trace may differ rather than quietly
        # using the defaults.
        old_prov = {"settings": {"seed": 11}}
        use0, missing0, _ = CT.recover_args(old_prov)
        check("...and a run that recorded NONE of them leaves the "
              "driver naming every one as not-recoverable rather "
              "than assuming the default: {} named".format(
                  len(missing0)),
              not use0 and len(missing0) >= 7)

        # A RECORDED KEYWORD THIS BUILD DOES NOT ACCEPT is dropped
        # and named - an older run can carry an argument since
        # removed, and passing it would be a crash where the honest
        # answer is a narrower trace.
        # ASSERT THE OUTCOME, NOT THE REPORT. The first version of
        # this check read the `unknown` list - which is computed
        # from the record and not from what is PASSED, so a
        # mutation that splatted every recorded keyword into the
        # engine broke nothing. The observable is that the run
        # still TRACES: an unknown keyword reaching `generate` is a
        # TypeError.
        stale = td / "stale"
        stale.mkdir()
        (stale / "blueprint.json").write_text(
            (run / "blueprint.json").read_text(encoding="utf-8"),
            encoding="utf-8")
        sp = json.loads((run / "provenance.json").read_text(
            encoding="utf-8"))
        sp["settings"]["generate_args"]["seed_offset"] = 3
        (stale / "provenance.json").write_text(
            json.dumps(sp), encoding="utf-8")
        r = drive(stale, "--patients", "60")
        check("...and a run recording a keyword this build does "
              "NOT accept still traces - the keyword is dropped "
              "and named rather than splatted into a TypeError",
              r.returncode == 0 and "seed_offset" in r.stdout
              and "does not accept" in r.stdout)

        r = drive(run)
        out = r.stdout
        check("the driver runs end to end on that run and its "
              "census BALANCES - a term that double-counted once "
              "printed a negative residual on a real run",
              r.returncode == 0 and "balances" in out
              and "DOES NOT ADD UP" not in out)
        check("...and it PRINTS THE CURVES rather than only "
              "counting them, because a count cannot answer a "
              "question about which",
              " <- " in out and "draw " in out and "end " in out)
        check("...and it reports the figure for the columns a "
              "researcher actually receives apart from the "
              "scaffolding, which four fifths of the graded "
              "surface is on the real extract",
              "researcher actually receives" in out)
        check("...and it says it used the run's OWN seed, so a "
              "reader cannot mistake a trace of another draw for "
              "an explanation of this one",
              "OWN seed 11" in out)

        # THE DRIVER MUST USE THE PROBED FRAME, asserted THROUGH
        # `trace_run` rather than beside it. A check that called
        # the computation directly on both frames left a mutation
        # putting the driver back on the RETURNED frame breaking
        # nothing - the property was asserted and the DECISION was
        # not, which is the same fault this suite already caught
        # once on the unknown-keyword list.
        _b4, _t4, _f4 = CT.trace_run(run, 11, use, patients=60)
        check("...and `trace_run` hands the driver the frame the "
              "PROBES read rather than the one the caller gets, "
              "which is why the collinearity question can be "
              "answered for a curve whose co-parent is "
              "scaffolding",
              any("__has__" in c for c in _f4.columns))

        # ---- THE PARTIAL-VERSUS-MARGINAL CAVEAT -----------------
        #
        # A published curve is a PARTIAL effect and this probe
        # reads a MARGINAL correlation, so on collinear parents a
        # correctly applied curve can read near zero or negative.
        # That is the distinction that invalidated this project's
        # earlier CONTRACT/SAMPLER split, and a trace that claimed
        # 133 curves were never applied without it would be making
        # the same mistake at scale.
        check("the collinearity question is asked with the SAME "
              "computation and the SAME bar as `curve_why`, not a "
              "second copy - two halves each deciding what "
              "collinear means is how they come to disagree",
              CT.max_coparent is CW.max_coparent
              and CT.COLLINEAR_R is CW.COLLINEAR_R)
        check("...and that computation takes the WORST co-parent "
              "rather than the first, and returns nothing where "
              "there is no co-parent - absent is not zero, and a "
              "zero would read as 'the parents are independent' "
              "where nobody could tell",
              CW.max_coparent(_cop, ["a", "b", "c"], "a",
                              _numf)[1] == "b"
              and CW.max_coparent(_cop, ["a"], "a",
                                  _numf) == (None, None))
        check("...and the two verdicts that CLAIM the curve is not "
              "followed are withheld on a collinear parent, while "
              "DROPPED is NOT - a dropped curve was applied at the "
              "draw and lost later, and the co-parent structure "
              "does not change between passes, so it cannot "
              "explain that",
              CT.not_established("NEVER-APPLIED", 0.95)
              and CT.not_established("REVERSED", 0.95)
              and not CT.not_established("DROPPED", 0.99)
              and not CT.not_established("CARRIED", 0.99)
              and not CT.not_established("NEVER-APPLIED", 0.5)
              and not CT.not_established("NEVER-APPLIED", None))
        check("...and the driver prints the co-parent reading PER "
              "VERDICT with the carried bucket named as the "
              "control, because a reading on the failures alone "
              "says nothing - exactly that caveat was vacuous on "
              "the real extract and one control refuted it",
              "per verdict" in out and "control" in out
              and "median" in out)

    # ================================================================
    # THE CO-PARENT QUESTION MUST BE ASKED OF THE FRAME THE PROBES
    # READ. Scaffolding is dropped before `generate` returns, and
    # most co-parents on a real blueprint are `__has__` indicators -
    # so asking the RETURNED frame answered for only 7 of 131
    # curves on the real extract, which reads as a thin sample
    # rather than as a question asked of the wrong surface.
    # ================================================================
    tdf = token_driven()
    bp3 = B.build(tdf, discover(tdf, group_by="person_id"),
                  group_by="person_id")
    G.TRACE_CURVES = True
    rep3 = {}
    ret3 = G.generate(bp3, seed=7, report=rep3)
    G.TRACE_CURVES = False
    tr3 = rep3.get("curve_trace") or {}
    pf3 = rep3.get("curve_trace_frame")
    check("the fixture really does carry a relationship with an "
          "INDICATOR parent, without which this measures nothing - "
          "a fixture whose tokens carry no structure produces none",
          any("__has__" in k for k in tr3))
    gone = [c for c in (pf3.columns if pf3 is not None else [])
            if c not in ret3.columns]
    check("...and the probed frame carries the scaffolding the "
          "RETURNED frame has dropped, which is the whole "
          "difference: {}".format(len(gone)),
          pf3 is not None and len(gone) >= 3
          and any("__has__" in c for c in gone))

    def measurable(frame):
        kids = {}
        for rel in (bp3.get("relationships") or []):
            kids[rel.get("child")] = rel.get("parents") or []
        got = 0
        for key in tr3:
            ch, _, par = key.partition(" <- ")
            if ch not in frame.columns or par not in frame.columns:
                continue
            r_, _c = CW.max_coparent(frame, kids.get(ch) or [], par,
                                     _numf)
            if r_ is not None:
                got += 1
        return got
    on_probed, on_returned = measurable(pf3), measurable(ret3)
    check("...and the collinearity question is ANSWERABLE for more "
          "curves on the probed frame than on the returned one, "
          "which is the observable the fix exists for: {} against "
          "{}".format(on_probed, on_returned),
          on_probed > on_returned)


    # AND THE RETURNED FRAME IS STILL UNTOUCHED on this shape too -
    # the stashed frame is a COPY, so a caller cannot reach back
    # into what was generated.
    G.TRACE_CURVES = False
    plain3 = G.generate(bp3, seed=7)
    check("...and stashing it leaves the generated output "
          "BYTE-IDENTICAL on a frame that HAS scaffolding, so the "
          "copy is a copy",
          digest(plain3) == digest(ret3))

    # ================================================================
    # A PARENT A SURFACE CONSUMES. `_apply_numeric` applies the
    # published surface and then marks BOTH pair members done, so
    # their individual curves are never applied - by design. A
    # diagnostic that did not know this reports such a curve as
    # never applied, which is misleading in the worst direction:
    # the dependence IS carried, by a coarser object. Two of the
    # nine own-column never-applied curves on the real extract are
    # this shape, `height <- year_of_birth` among them.
    # ================================================================
    from synthkit.generate import surface_pair as _spair
    check("the driver asks the ENGINE's own question about which "
          "parents a surface consumes, not a copy of it - a second "
          "predicate is how the engine and the diagnostic come to "
          "disagree about what generation did",
          CT.surface_pair is _spair)

    _good = {"child": "y", "parents": ["a", "b", "c"],
             "evidence": {"interaction": {
                 "pair": ["a", "b"], "grid_a": [1, 2, 3]}}}
    avail = {"y", "a", "b", "c"}
    check("...and it names the pair where the surface CAN be "
          "applied",
          _spair(_good, avail) == ["a", "b"])
    check("...and refuses where the surface cannot be applied at "
          "all - no interaction, no grid, a pair that is not a "
          "pair, a member the claim does not parent, or a member "
          "the frame does not carry - because each of those is a "
          "surface the engine SKIPS, leaving the individual "
          "curves to run",
          _spair({"child": "y", "parents": ["a", "b"],
                  "evidence": {}}, avail) is None
          and _spair({"child": "y", "parents": ["a", "b"],
                      "evidence": {"interaction": {
                          "pair": ["a", "b"]}}}, avail) is None
          and _spair({"child": "y", "parents": ["a", "b"],
                      "evidence": {"interaction": {
                          "pair": ["a"], "grid_a": [1]}}},
                     avail) is None
          and _spair({"child": "y", "parents": ["a"],
                      "evidence": {"interaction": {
                          "pair": ["a", "b"], "grid_a": [1]}}},
                     avail) is None
          and _spair(_good, {"y", "a", "c"}) is None)
    check("...and the note fires on a failing verdict and NOT on "
          "CARRIED, where saying the surface was used instead "
          "would contradict the reading directly above it",
          CT.surface_note("NEVER-APPLIED", True)
          and CT.surface_note("REVERSED", True)
          and CT.surface_note("DROPPED", True)
          and not CT.surface_note("CARRIED", True)
          and not CT.surface_note("NEVER-APPLIED", False))
    check("...and the count is reported BESIDE each verdict rather "
          "than removed from it, because dropping these from the "
          "failure count would turn a reading into a better one "
          "by changing what is counted",
          "SURFACE consumes" in out and "BESIDE" in out
          and "own-column" in out)

    # ================================================================
    # PUBLISHED == TRACED + PRESENCE-ONLY + ABSENT + THIN. The
    # first real read traced 278 of 281 published curves and said
    # nothing about the other three, so every proportion in the
    # report had an unstated denominator.
    # ================================================================
    _bp5 = {"relationships": [
        {"child": "y", "parents": ["a", "b", "c", "d"],
         "evidence": {"effect": {
             # traced
             "a": {"grid": [1, 2], "response": [0.0, 1.0]},
             # a KIND, not a gap: the probe cannot evaluate it
             "b": {"grid": [1, 2], "response": [0.0, 1.0],
                   "shape": "presence-only"},
             # a column the probed frame does not carry
             "c": {"grid": [1, 2], "response": [0.0, 1.0]},
             # no response at all, so it was never published as a
             # curve and must not enter the total
             "d": {"grid": [1, 2]}}}}]}
    pc = CT.publish_census(_bp5, 1, {"y", "a", "b"})
    check("the published total counts only what the contract "
          "actually published as a CURVE - a parent with a grid "
          "and no response is not one and must not inflate the "
          "denominator",
          pc["published"] == 3)
    check("...and the four terms ADD UP to it, which is the "
          "identity a reader can check - this project has printed "
          "a NEGATIVE residual on a real run by not holding one",
          pc["traced"] + pc["presence_only"] + pc["absent_columns"]
          + pc["thin_or_flat"] == pc["published"])
    check("...and presence-only is counted as its own KIND rather "
          "than as a thin reading, because the child depends on "
          "whether the parent was MEASURED and `_curve_delta` "
          "returns zeros for it without the pending mask the "
          "probe does not have: presence-only {}, absent {}, thin "
          "{}".format(pc["presence_only"], pc["absent_columns"],
                      pc["thin_or_flat"]),
          pc["presence_only"] == 1 and pc["absent_columns"] == 1
          and pc["thin_or_flat"] == 0)
    check("...and a residual that cannot be negative is CALLED "
          "OUT when it is, because a total that balances does not "
          "make a double-counted term right - which is exactly "
          "how a real run came to print a negative count and look "
          "internally consistent",
          CT.impossible_residual(
              CT.publish_census(_bp5, 3, {"y", "a", "b"}))
          and not CT.impossible_residual(
              CT.publish_census(_bp5, 1, {"y", "a", "b"}))
          and not CT.impossible_residual({}))
    check("...and the driver STATES the published total beside "
          "what it traced, so no proportion in the report has an "
          "unstated denominator",
          "contract PUBLISHES" in out and "the four add to" in out
          and "DOES NOT ADD UP" not in out)

    # ----------------------------------------------------------
    # DID THE PROBE'S OWN SAMPLE COLLAPSE AT THE PASS BLAMED?
    #
    # `m` inside the probe is the rows where BOTH the curve's
    # delta and the child are finite, and `_curve_delta` is not
    # finite where the PARENT is absent - so a presence mask on
    # the parent re-measures the correlation on the parent's
    # surviving rows rather than on all of them. A fall there can
    # be the RULER shrinking rather than the child moving, and
    # the two take completely different fixes.
    #
    # THE REAL EXTRACT IS WHY THIS EXISTS. `procedure_count <-
    # mean_arterial_pressure_invasive` falls 0.172 to 0.007 at
    # after-masks, and the CHILD is present on 100% of rows while
    # the parent is present on 0.20% - so no mask touches the
    # child at all and the whole fall is the sample. It was the
    # largest unexplained own-column loss on the file.
    #
    # PLANTED, because the branch needs the mask's fall to be the
    # LARGEST one and no fixture here can be relied on to produce
    # that ordering: on a sparse-parent fixture built for it the
    # blame landed on after-sweeps instead.
    # ----------------------------------------------------------
    coll = CT.sample_at(
        [("after-draw", 0.42, 12000),
         ("after-sweeps", 0.40, 12000),
         ("after-masks", 0.01, 42)], "after-masks")
    check("a probe whose comparable rows fall from 12,000 to 42 "
          "at the pass blamed is reported as a COLLAPSED sample, "
          "so a reader is not sent after a child that cannot have "
          "moved",
          coll and coll["collapsed"] is True
          and coll["rows_before"] == 12000
          and coll["rows_after"] == 42)
    same = CT.sample_at(
        [("after-draw", 0.42, 12000),
         ("after-sweeps", 0.20, 12000)], "after-sweeps")
    check("...and a fall measured on the SAME rows is NOT - that "
          "one is the engine, and calling it a sample artifact "
          "would excuse a real loss",
          same and same["collapsed"] is False)
    half = CT.sample_at(
        [("a", 0.4, 1000), ("b", 0.2, 600)], "b")
    check("...and a sample that merely thins does not clear the "
          "bar, which is deliberately loose at a half rather "
          "than arguing about cases nobody can separate",
          half and half["collapsed"] is False)

    # ABSENT IS NOT "IT DID NOT COLLAPSE". A run made before the
    # probe recorded counts cannot answer the question at all,
    # and reporting False there would be a claim nobody measured.
    old = CT.sample_at([("a", 0.4), ("b", 0.2)], "b")
    check("a trajectory with NO counts returns None rather than "
          "False - an older run cannot answer this, which is a "
          "different fact from the sample holding",
          old is None)
    check("...and so does a stage that is not in the trajectory, "
          "or the first one, which has nothing before it",
          CT.sample_at([("a", 0.4, 9), ("b", 0.2, 9)], "zz") is None
          and CT.sample_at([("a", 0.4, 9), ("b", 0.2, 9)], "a")
          is None)

    # AND THE PROBE ACTUALLY RECORDS THE COUNT, asserted through
    # a real traced generate rather than on the function alone -
    # a detector reading a field nobody writes is the
    # read-but-not-used fault.
    check("the engine's own probe records the comparable-row "
          "count beside every reading, without which none of the "
          "above can fire on a real run",
          any(len(e) > 2 and e[2] is not None
              for traj in ((rep.get("curve_trace") or {})).values()
              for e in traj))

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
