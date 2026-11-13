"""A paired arm is the only thing that has overturned a claim here.

It is also the easiest table to fake. This suite holds
`arm_compare.py` to the three ways the record says it has already
gone wrong: two engines compared as one, a denominator that moved
while the percentages were read, and a control arm run on an older
tree that differed by every default in between.

The decisions are PURE FUNCTIONS over planted provenance, so the
checks do not need two real runs of the extract to exercise them -
which is the thing that made a per-arm behavioural audit hard.
"""
import io
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import arm_compare as A                                 # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def plant(d, name, source="tidy.csv", rows=100, seed=7,
          engine=None, build="b1", settings=None, stack=None,
          fid=None):
    run = Path(d) / name
    run.mkdir(parents=True, exist_ok=True)
    sett = {"seed": seed, "group_by": "pid"}
    sett.update(settings or {})
    if engine:
        sett["engine"] = engine
    prov = {"source": {"name": source, "rows_read": rows,
                       "patients": 10},
            "settings": sett, "build": build,
            "stack": stack or {"python": "3.14.6",
                               "pandas": "3.0.5"}}
    (run / "provenance.json").write_text(
        json.dumps(prov), encoding="utf-8")
    base = {"summary": {"coverage_ok": 40,
                        "coverage_compared": 40,
                        "dynamics_ok": 20,
                        "dynamics_compared": 80,
                        "set_tokens_ok": 70,
                        "set_tokens_compared": 72,
                        # THE NAMES THE REAL WRITER USES, asserted
                        # against it below rather than invented
                        # here - a fixture that shares the
                        # reader's invented names agrees with the
                        # reader and both disagree with the only
                        # thing that matters.
                        "pairs": 100, "pairs_sign_ok": 95,
                        "pairs_close": 90, "pairs_inverted": 0,
                        "pairs_own": 30, "pairs_own_close": 25,
                        "pairs_own_inverted": 0},
            "marginals": {"columns_measured": 31,
                          "off_source": ["a", "b"],
                          "worst_source_vs_output": 0.30,
                          "sampler_faults": ["a"]},
            "tvd": {"accuracy": 0.9},
            "generation": {}}
    if fid:
        for k, v in fid.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                base[k].update(v)
            else:
                base[k] = v
    (run / "fidelity.json").write_text(
        json.dumps(base), encoding="utf-8")
    return str(run)


def run_cli(a, b, testing="a flag"):
    p = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "arm_compare.py"),
         a, b, "--testing", testing],
        capture_output=True, text=True, timeout=120)
    return p.returncode, p.stdout + p.stderr


def main():
    # ---- THE DECISION, PURE ----------------------------------
    base = {"source": "t.csv", "rows": 100, "patients": 10,
            "seed": 7, "engine": "rules", "build": "b1",
            "settings": {"seed": 7}, "stack": {}, "dials": {},
            "marginal_metric": "m"}

    same = dict(base)
    fatal, readable = A.differences(base, same)
    check("two identical runs produce no difference at all, so a "
          "pass here is not the function finding nothing it can "
          "see", not fatal and not readable)

    other_engine = dict(base, engine="neural")
    fatal, _ = A.differences(base, other_engine)
    check("TWO ENGINES is fatal - the record's own worst case, "
          "where a neural run was compared against five rules "
          "arms and the difference was read as a regression",
          any(k == "engine" for k, _, _ in fatal))

    other_seed = dict(base, seed=11)
    fatal, _ = A.differences(base, other_seed)
    check("a different SEED is fatal - a generation change read "
          "at one seed is worth nothing, and two seeds are two "
          "questions",
          any(k == "seed" for k, _, _ in fatal))

    other_src = dict(base, source="u.csv")
    fatal, _ = A.differences(base, other_src)
    check("a different SOURCE is fatal, which is the fault that "
          "had a whole session's numbers measured against an "
          "extract that never produced the run",
          any(k == "source" for k, _, _ in fatal))

    other_build = dict(base, build="b2")
    fatal, readable = A.differences(base, other_build)
    check("a different BUILD is READABLE rather than fatal - an "
          "old-against-new control arm is cross-build by "
          "definition, so refusing it would refuse the comparison "
          "this tool is for",
          not fatal and any(k == "build" for k, _, _ in readable))

    # A FIELD ABSENT ON ONE SIDE IS A DIFFERENCE. An older run
    # that never recorded a setting cannot be assumed to have had
    # today's default - which is exactly how a control arm came to
    # differ by a default nobody could see.
    _a = dict(base, settings={"seed": 7, "copy_presence": True})
    _b = dict(base, settings={"seed": 7})
    fatal, readable = A.differences(_a, _b)
    check("a setting recorded on ONE side only is reported as a "
          "difference, not treated as a match - an older run's "
          "silence is not today's default",
          any(k == "settings.copy_presence"
              for k, _, _ in readable))

    # THE DIALS ARE THE THING THE ARM IS USUALLY ABOUT, and they
    # were in no recorded field at all until provenance carried
    # them. Two runs could differ in what the sampler DID and
    # agree in every recorded field.
    _a = dict(base, dials={"generate": {"TAIL_SPREAD": False}})
    _b = dict(base, dials={"generate": {"TAIL_SPREAD": True}})
    _, readable = A.differences(_a, _b)
    check("a dial that differs is reported by name, so the arm's "
          "own subject is visible in the table rather than only "
          "in the caller's words",
          any(k == "dials.generate.TAIL_SPREAD"
              for k, _, _ in readable))

    _a = dict(base, dials={"generate": {"TAIL_SPREAD": False}})
    _b = dict(base, dials={})
    _, readable = A.differences(_a, _b)
    check("...and a run that records NO dials at all is called "
          "out as such, because an older run's silence is not "
          "today's defaults and that is exactly what made a "
          "cross-build arm invisible",
          any(k == "dials recorded at all"
              for k, _, _ in readable))

    # THE RULER CAN MOVE TOO, and it looks exactly like a number.
    _a = dict(base, marginal_metric="worst single band")
    _b = dict(base, marginal_metric="total variation distance")
    _, readable = A.differences(_a, _b)
    check("two runs whose MARGINAL METRIC differs are flagged - "
          "the ruler was changed once already and two readings "
          "taken on different ones are indistinguishable by eye",
          any("RULER" in k for k, _, _ in readable))

    _a = dict(base, marginal_metric="total variation distance")
    _b = dict(base, marginal_metric=None)
    _, readable = A.differences(_a, _b)
    check("...and a run that recorded no metric at all is "
          "flagged rather than assumed to match",
          any("RULER" in k for k, _, _ in readable))

    _a = dict(base, stack={"pandas": "3.0.5"})
    _b = dict(base, stack={"pandas": "2.3.3"})
    _, readable = A.differences(_a, _b)
    check("...and a different LIBRARY stack likewise, because "
          "four dial verdicts on this project differ by pandas "
          "version alone",
          any(k == "stack.pandas" for k, _, _ in readable))

    # ---- THROUGH THE CLI -------------------------------------
    with TemporaryDirectory() as d:
        ra = plant(d, "a")
        rb = plant(d, "b")
        rc, out = run_cli(ra, rb)
        check("two matching runs compare cleanly through the CLI, "
              "exit 0", rc == 0)
        check("...and say so, rather than leaving a reader to "
              "infer that nothing differed",
              "No difference visible in provenance" in out)
        check("...and the arm's own question is printed at the "
              "top, because a table with no stated question is "
              "read as whichever difference is noticed first",
              "ARM: a flag" in out)

        rn = plant(d, "n", engine="neural")
        rc, out = run_cli(ra, rn)
        check("the engine mismatch REFUSES with exit 2 rather "
              "than printing a table", rc == 2)
        check("...and names the engine as the reason",
                "engine differs" in out)
        check("...and prints no criterion table at all, so there "
              "is nothing to quote out of a refused comparison",
              "criterion" not in out)

        # A MOVING DENOMINATOR IS MARKED ON ITS OWN ROW. The
        # record carries three conclusions drawn from a
        # percentage whose denominator had moved.
        rm = plant(d, "m", fid={"summary": {
            "pairs": 125, "pairs_sign_ok": 118,
            "pairs_close": 100, "pairs_inverted": 0,
            "pairs_own": 30, "pairs_own_close": 25,
            "pairs_own_inverted": 0}})
        rc, out = run_cli(ra, rm)
        check("a run relating a different number of pairs is "
              "compared, not refused - the counts are still a "
              "comparison even when the proportions are not",
              rc == 0)
        check("...and the row SAYS the denominator moved, which "
              "is the only thing that stops the proportions being "
              "read",
              "denominator MOVED" in out)

        # WHICH COLUMN, not how many.
        rq = plant(d, "q", fid={"marginals": {
            "off_source": ["b", "zz"],
            "columns_measured": 31,
            "worst_source_vs_output": 0.2,
            "sampler_faults": []}})
        rc, out = run_cli(ra, rq)
        check("the marginal columns are named by side - a count "
              "cannot say which column a change fixed and which "
              "it broke",
              "B fixed: a" in out and "B broke: zz" in out)

        # A ROW NEITHER RUN CARRIES IS SAID, NOT SKIPPED. A key
        # read under the wrong name looks exactly like a criterion
        # the runs do not have, and the first real arm printed a
        # finished-looking table with no close, no direction and
        # no INVERTED for precisely that reason.
        _thin = {"summary": {"coverage_ok": 40,
                             "coverage_compared": 40}}
        r1 = plant(d, "t1", fid={"summary": None})
        (Path(r1) / "fidelity.json").write_text(
            json.dumps({"summary": _thin["summary"],
                        "generation": {}}), encoding="utf-8")
        r2 = plant(d, "t2", fid={"summary": None})
        (Path(r2) / "fidelity.json").write_text(
            json.dumps({"summary": _thin["summary"],
                        "generation": {}}), encoding="utf-8")
        rc, out = run_cli(r1, r2)
        check("a criterion NEITHER run carries is printed as "
              "such, because a key read under the wrong name is "
              "indistinguishable from a criterion that was never "
              "measured",
              "NEITHER RUN CARRIES" in out)
        check("...and the rows are NAMED, so a reader can see "
              "that close and INVERTED were among them rather "
              "than reading a table that looks finished",
              "close" in out.split("NEITHER RUN CARRIES")[1]
              and "INVERTED" in out.split(
                  "NEITHER RUN CARRIES")[1])

        # A TIER RECORDED ON ONE SIDE ONLY.
        rt = plant(d, "t", fid={"generation": {
            "copy_presence_modeled": [{"column": "x"}]}})
        rc, out = run_cli(ra, rt)
        check("a sampler tier recorded on one side only is called "
              "out, because a tier that cannot prove it executed "
              "has not been deployed",
              "ONE SIDE ONLY" in out
              and "copy_presence_modeled" in out)

        # --testing IS REQUIRED.
        p = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "arm_compare.py"),
             ra, rb], capture_output=True, text=True, timeout=120)
        check("the tool REFUSES without --testing, so no arm "
              "table exists without a stated question",
              p.returncode != 0
              and "testing" in (p.stdout + p.stderr))

        # AN UNFINISHED RUN IS NAMED, not read as empty.
        bad = Path(d) / "half"
        bad.mkdir()
        (bad / "provenance.json").write_text("{}", encoding="utf-8")
        rc, out = run_cli(ra, str(bad))
        check("a run with no fidelity.json is refused by name "
              "rather than compared as a run of zeros",
              rc != 0 and "not a finished run" in out)

    # EVERY KEY THE TOOL READS MUST EXIST IN A FIDELITY DICT THE
    # REAL WRITER PRODUCED. The first version read `fid["pairs"]`
    # for the pair counts - that block's own internal names - and
    # `fid["pairs"]` is a dict whose `pairs` key is a list of
    # rows, so all five came back None and every pair row was
    # SILENTLY SKIPPED. The first real arm printed coverage,
    # tokens, dynamics and the marginals with no close, no
    # direction and no INVERTED: the criteria it was for. The
    # fixture above could not catch it, because the fixture used
    # the same invented names as the reader.
    import numpy as _np
    import pandas as _pd
    from synthkit import blueprint as _B
    from synthkit import discover as _D
    from synthkit import pipeline as _PL

    _rng = _np.random.default_rng(5)
    _n = 600
    _pid = ["p{:03d}".format(i // 3) for i in range(_n)]
    _x = _rng.normal(10.0, 2.0, _n)
    _frame = _pd.DataFrame({
        "pid": _pid, "x": _x,
        "y": 2.0 * _x + _rng.normal(0, 1.0, _n),
        "z": _rng.normal(0, 1.0, _n)})
    _cat = _D.discover(_frame, group_by="pid")
    _bp = _B.build(_frame, _cat, group_by="pid")
    _fid = _PL.compare(_frame, _frame.copy(), _bp, "pid", None)
    _r = A.readings
    _need = {
        "related": "pairs", "direction": "pairs_sign_ok",
        "close": "pairs_close", "inverted": "pairs_inverted",
        "own_inverted": "pairs_own_inverted",
        "own_related": "pairs_own",
        "own_close": "pairs_own_close"}
    _summ = _fid.get("summary") or {}
    _absent = sorted(k for k in _need.values() if k not in _summ)
    check("the real writer produces a summary with pair counts in "
          "it, so this check is not passing on an empty dict ({} "
          "keys)".format(len(_summ)),
          len(_summ) >= 10)
    check("...and EVERY pair key the tool reads is a key the real "
          "writer produces - the fixture above cannot catch a "
          "wrong name because it shares it: {}".format(
              ", ".join(_absent) or "none absent"),
          not _absent)

    # NO ROW APPEARS TWICE. A duplicated entry printed
    # "own-column pairs" on two lines of the first real arm table
    # - harmless to the arithmetic and exactly the kind of thing a
    # reader stops to re-read, and a static list is where a
    # copy-paste lands.
    _labels = [r[0] for r in A.ROWS]
    check("every criterion row is listed once - a duplicate "
          "printed `own-column pairs` twice on the first real arm "
          "table ({} rows, {} distinct)".format(
              len(_labels), len(set(_labels))),
          len(_labels) == len(set(_labels)))
    _keys = [r[1] for r in A.ROWS]
    check("...and every row's key is one `readings` produces, so "
          "a row cannot be silently unreadable",
          not [k for k in _keys
               if k not in A.readings.__code__.co_consts
               and k not in ("related", "direction", "close",
                             "inverted", "own_inverted",
                             "own_related", "own_close",
                             "coverage_ok", "coverage_n",
                             "dynamics_ok", "dynamics_n",
                             "token_ok", "token_n", "marg_n",
                             "marg_off", "marg_sampler")])

    # NOTHING IS RECOMPUTED. Every figure is read, because a
    # second implementation of the gate is how a report and a
    # verdict come to disagree.
    src = io.open(ROOT / "scripts" / "arm_compare.py",
                  encoding="utf-8").read()
    check("the tool imports no gate and no measure - it reads the "
          "two run directories and computes no criterion of its "
          "own",
          "import synthkit.gate" not in src
          and "from synthkit" not in src
          and "assess(" not in src)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
