"""Smoke: the dials are reachable, and they report what they did.

WHY THIS EXISTS. `resolve` honoured eight dials and `smoke_generate`
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
of 1.5 arrived as +35.8 and the scale dragged the centre with it.
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
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
