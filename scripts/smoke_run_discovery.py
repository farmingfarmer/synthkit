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
                # ~6% IMPOSSIBLE DAYS - counted on vid, the per-ROW
                # counter, not on v, which only runs 0..5 and would
                # make one row in six bad. That is 16.7%, over the 0.9
                # bar, so the column stops being a date at all and
                # there is nothing left to warn about. Real extracts carry typing
                # errors, those rows become MISSING, and a coverage
                # loss that size passes the 0.05 bar silently - which
                # is the direction a whole column was destroyed in
                # once. The run has to say so out loud.
                "collected_date": ("2021-02-30" if vid % 16 == 0
                                   else "2021-{:02d}-{:02d}".format(
                                       (v % 12) + 1, (v % 27) + 1)),
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

    check("the run echoes the settings it actually received, so a "
          "flag that never arrived is visible in the log instead of "
          "being mistaken for a broken fix",
          "invocation:" in r.stdout and "--src" in r.stdout)
    check("...and says plainly when no --time-col was given, which is "
          "the case that looked identical to the flag being ignored",
          "the axis will be DETECTED" in r.stdout)
    check("progress is reported with a clock, so a run that takes "
          "minutes does not look hung",
          "columns," in r.stdout or "searching" in r.stdout)
    check("the run NAMES the columns it read as dates and the format "
          "it read them under - a date silently typed as something "
          "else is how 88% of one column became a sentinel",
          "read as dates:" in r.stdout
          and "visit_start_date (%Y-%m-%d)" in r.stdout)
    check("...and WARNS when values did not parse, because those rows "
          "become missing and a coverage loss that size clears the "
          "0.05 bar without anyone seeing it",
          "WARNING collected_date:" in r.stdout
          and "became MISSING" in r.stdout)
    check("...and does NOT warn about a column whose dates are all "
          "valid, or the warning means nothing",
          "WARNING visit_start_date:" not in r.stdout)
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

    # RELATIONSHIP FIDELITY. Every other number in this file checks
    # one column at a time, and a table can pass all of them while
    # carrying no structure between columns - the classic way a
    # synthetic generator looks right and is useless. Worse still is
    # an INVERTED relationship, which reads as a finding.
    fidall = json.loads((out / "fidelity.json").read_text())
    rel = fidall.get("relationships") or {}
    check("the run measures whether RELATIONSHIPS survived, not only "
          "whether each column looks right on its own",
          rel.get("compared", 0) >= 1)
    check("...and none came out INVERTED - the opposite sign to the "
          "source is worse than a missing relationship{}".format(
              "" if not rel.get("inverted") else
              " -- " + "; ".join(
                  "{}~{} {:+.2f}->{:+.2f}".format(
                      r["child"], r["parent"], r["source"],
                      r["generated"])
                  for r in rel["inverted"][:4])),
          not rel.get("inverted"))
    check("...most keep their direction",
          rel["sign_kept"] >= 0.8 * rel["compared"])
    check("the run PRINTS the relationship result, so it is visible "
          "without opening a JSON file",
          "RELATIONSHIPS:" in r.stdout)

    fid = fidall["summary"]
    check("fidelity is reported as counts over columns, so a wide "
          "extract is readable at a glance",
          fid["columns"] > 0 and fid["coverage_ok"] <= fid["columns"])
    check("...and most columns keep their coverage",
          fid["coverage_ok"] >= 0.8 * fid["columns"])

    # ---- what will NOT reach the generated data ------------------
    # A catalogue holds relationships in both directions and can hold
    # loops; a sampler cannot. On the real extract 12 of 22 were
    # dropped and nothing told the reader which - so a relationship
    # shown as a finding could be absent from the file they were
    # handed, with no way to know.
    check("findings.txt states what the sampler had to discard",
          "WILL NOT REACH THE GENERATED DATA" in txt)
    check("...and separates a RESTATEMENT, which loses nothing, from "
          "a parent REMOVED to break a loop, which loses that "
          "parent's contribution, from a column left with NO parent "
          "at all",
          "(nothing was dropped)" in txt
          or any(w in txt for w in ("RESTATEMENTS", "SOME PARENTS "
                                    "REMOVED", "NO parent")))
    # THE COUNTS MUST ADD UP. On the real extract the section said
    # "2 were mirrors" while the run reported 12 dropped: ten had some
    # parents removed to break a loop, a category the report had no
    # section for at all, so ten relationships lost a parent and the
    # reader was never told. A total that a reader can check against
    # the parts is what makes an omission visible.
    import re as _re
    if "WILL NOT REACH" in txt:
        tail = txt.split("WILL NOT REACH")[1]
        tot = _re.search(r"(\d+) were dropped", tail)
        parts = [int(x) for x in _re.findall(
            r"^(\d+) (?:RESTATEMENTS|had SOME|left their|were dropped "
            r"for)", tail, _re.M)]
        check("the dropped-edge section states a TOTAL and the "
              "categories beneath it sum to exactly that - a category "
              "with no section is invisible any other way",
              tot is not None and parts
              and sum(parts) == int(tot.group(1)))

    rep_d = json.loads(
        (out / "fidelity.json").read_text())["generation"]
    for d in rep_d.get("edges_dropped") or []:
        check("every dropped edge says whether its child kept any "
              "other parent, which is what decides if it matters",
              "child_keeps_parents" in d and "skill" in d)
        break
    else:
        check("nothing was dropped on this fixture, and the report "
              "says so rather than omitting the section",
              "(nothing was dropped)" in txt)

    # ---- arithmetic is sorted out of the findings ----------------
    # On the real extract the top two entries were
    # `returned_within_30d <- days_to_next_visit` at 100% and
    # `age_at_visit <- year_of_birth` at 98% - a threshold and a
    # subtraction, both computed by the wrangler - which pushed
    # haematocrit-from-haemoglobin at 96%, the one nobody planted,
    # onto the second page.
    src2 = tmp / "tidy2.csv"
    d = pd.read_csv(src)
    d["x_doubled"] = d["x"] * 2.0          # pure arithmetic
    d.to_csv(src2, index=False)
    out3 = tmp / "arith"
    r3 = run(["--src", str(src2), "--out", str(out3)])
    check("a run with an arithmetic column completes",
          r3.returncode == 0)
    t3 = (out3 / "findings.txt").read_text()
    check("a column that is another one doubled is filed as "
          "NEAR-DETERMINISTIC rather than reported as a discovery",
          "NEAR-DETERMINISTIC" in t3)
    head = t3.split("NEAR-DETERMINISTIC")[0]
    check("...so it does not sit above the real relationship in the "
          "findings section",
          "x_doubled" not in head)
    check("...and it is still PRINTED, because sorting is not hiding",
          "x_doubled" in t3)
    check("...and the cut is named as a rule of thumb with the "
          "numbers behind it, not stated as a law",
          "rule of thumb" in t3 and "96%" in t3)

    # ---- --time-col must actually be used ------------------------
    # It was carried into the blueprint and then ignored when the lag
    # features were built - which is what every temporal statistic is
    # measured on - so passing it changed nothing and the message
    # still named the detected column. On the real extract detection
    # chose visit_END_date, which misorders overlapping stays.
    out6 = tmp / "tcol"
    r6 = run(["--src", str(src), "--out", str(out6), "--lags",
              "--time-col", "visit_start_date"])
    check("an explicit --time-col is honoured and SAID to be, naming "
          "what detection would have chosen instead",
          r6.returncode == 0
          and ("AS REQUESTED" in r6.stdout
               or "as requested" in r6.stdout))
    r7 = run(["--src", str(src), "--out", str(tmp / "z2"), "--lags",
              "--time-col", "sex"])
    check("...and a column that cannot order visits is refused with a "
          "sentence rather than silently sorted by it",
          r7.returncode == 2 and "sequence" in r7.stderr)
    r8 = run(["--src", str(src), "--out", str(tmp / "z3"), "--lags",
              "--time-col", "no_such_col"])
    check("...and one that does not exist is refused too",
          r8.returncode == 2 and "no_such_col" in r8.stderr)

    r4 = run(["--src", str(src), "--out", str(tmp / "z"),
              "--exclude", "not_there"])
    check("--exclude naming a column that does not exist is refused "
          "with a sentence",
          r4.returncode == 2 and "not_there" in r4.stderr)
    out5 = tmp / "excl"
    r5 = run(["--src", str(src), "--out", str(out5), "--exclude",
              "lab"])
    check("--exclude actually removes the column from everything "
          "downstream", r5.returncode == 0
          and "lab" not in json.loads(
              (out5 / "blueprint.json").read_text())["columns"])

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
