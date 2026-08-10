"""Smoke: the one command a person actually types, run as a person
would type it.

This is the only suite that exercises the CLI as a SUBPROCESS. Every
other check imports the modules directly, which cannot catch an
argument that does not parse, an import that only resolves on this
machine, or a file that never gets written. The path this tests is the
one that runs where it cannot be debugged.

Failure messages are checked too. A traceback on a Windows terminal
five hundred miles away costs a round trip; a sentence does not.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLI = str(ROOT / "scripts" / "run_discovery.py")

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def run(args):
    return subprocess.run([sys.executable, "-W", "ignore", CLI] + args,
                          capture_output=True, text=True, cwd=str(ROOT))


def build(path, n_pat=90, n_vis=6, seed=4):
    r = np.random.RandomState(seed)
    rows = []
    vid = 0
    for p in range(n_pat):
        sex = r.choice(["F", "M"])
        for v in range(n_vis):
            vid += 1
            x = r.uniform(0, 10)
            rows.append({
                "person_id": "P{:04d}".format(p),
                "visit_id": vid,
                "visit_start_date": "2021-{:02d}-{:02d}".format(
                    (v % 12) + 1, (v % 27) + 1),
                "sex": sex,
                "x": round(x, 4),
                "y": round(2.5 * x + r.normal(0, 0.8), 4),
                "lab": (round(r.normal(40, 6), 3)
                        if r.random_sample() < 0.5 else ""),
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="synthkit_cli_"))
    src = tmp / "tidy.csv"
    build(src)

    # ---- the failures a person will actually hit -----------------
    r = run(["--src", str(tmp / "nope.csv"), "--out", str(tmp / "a")])
    check("a missing input file is a SENTENCE, not a traceback - a "
          "stack trace on a terminal that cannot be debugged costs a "
          "round trip",
          r.returncode == 2 and "no such file" in r.stderr
          and "Traceback" not in r.stderr)

    r = run(["--src", str(src), "--out", str(tmp / "b"),
             "--group-by", "not_a_column"])
    check("a wrong --group-by names the column AND lists what is "
          "there, so the fix does not need another round trip",
          r.returncode == 2 and "not_a_column" in r.stderr
          and "person_id" in r.stderr)

    # ---- the real run --------------------------------------------
    out = tmp / "run"
    r = run(["--src", str(src), "--out", str(out), "--generate",
             "--patients", "80"])
    check("the command completes", r.returncode == 0)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])

    for name in ("catalogue.json", "blueprint.json", "findings.txt",
                 "generated.csv", "fidelity.json"):
        check("it writes {}".format(name), (out / name).exists())

    check("progress is reported with a clock, so a run that takes "
          "minutes does not look hung",
          "columns," in r.stdout or "searching" in r.stdout)
    check("nothing is written outside the directory the user chose",
          sorted(p.name for p in tmp.iterdir())
          == ["b", "run", "tidy.csv"])

    bp = json.loads((out / "blueprint.json").read_text())
    check("the blueprint validates - it was just built, so a problem "
          "here is a bug and not an edit",
          bp.get("blueprint_version") is not None)
    check("the identifier was dropped rather than modelled",
          "visit_id" in (bp["excluded"]["identifiers"]))
    check("dynamics were measured into it, so the generator can "
          "reproduce missingness runs and steadiness",
          "dynamics" in bp["columns"]["lab"])

    txt = (out / "findings.txt").read_text()
    check("findings.txt states the relationship in words a reader can "
          "act on, naming the columns", "y <- x" in txt or "x <- y" in txt)
    check("...and says the evidence was measured on held-out patients",
          "held-out patients" in txt)
    check("...and tells the reader how to change it",
          "blueprint.json" in txt and "strength 0" in txt)
    check("a relationship true in BOTH directions is shown once and "
          "named, not printed twice as if it were two findings",
          txt.count("y <- x") + txt.count("x <- y") == 1)

    g = pd.read_csv(out / "generated.csv")
    check("the generated table has the source's columns and none of "
          "its identifiers",
          "x" in g.columns and "y" in g.columns
          and "visit_id" not in g.columns)
    check("...and the requested number of patients",
          g["person_id"].nunique() == 80)
    s = pd.DataFrame({"a": g["x"], "b": g["y"]}).dropna()
    check("...and the relationship actually survived into the file "
          "that gets handed on", float(s.a.corr(s.b)) > 0.6)

    fid = json.loads((out / "fidelity.json").read_text())["summary"]
    check("fidelity is reported as counts over columns, so a wide "
          "extract is readable at a glance",
          fid["columns"] > 0 and fid["coverage_ok"] <= fid["columns"])
    check("...and most columns keep their coverage",
          fid["coverage_ok"] >= 0.8 * fid["columns"])

    # ---- the quick first pass ------------------------------------
    out2 = tmp / "quick"
    r2 = run(["--src", str(src), "--out", str(out2), "--max-rows",
              "120"])
    check("--max-rows gives a cheap first pass before committing to a "
          "full extract", r2.returncode == 0
          and (out2 / "findings.txt").exists())

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
