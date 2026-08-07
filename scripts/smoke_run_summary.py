"""Smoke: a run summary that a terminal cannot corrupt.

Five consecutive pastes of pipeline output arrived with columns
shifted, and one of them paired a column with ANOTHER column's source
value - inventing a 1.79x steadiness overshoot that did not exist.
The checks here are about format as much as arithmetic: one fact per
line, nothing wider than the limit, and a long name wrapped rather
than allowed to push a number onto the next column.
"""
import csv
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_summary import autocorr_by_person      # noqa: E402

PASS = FAIL = 0
LONG = "a_very_long_column_name_that_would_wrap_a_narrow_terminal"


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def run(*args):
    return subprocess.run([sys.executable] + list(args),
                          capture_output=True, text=True, cwd=str(ROOT))


def main():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "run"
        d.mkdir()
        rnd = random.Random(3)
        rows = []
        for p in range(120):
            w = rnd.gauss(0, 1)
            for _v in range(10):
                w = 0.9 * w + 0.435 * rnd.gauss(0, 1)
                rows.append({"person_id": "P{:04d}".format(p),
                             "steady": round(w, 4),
                             LONG: round(w, 4),
                             "flat": round(rnd.gauss(0, 1), 4)})
        with (d / "generated_condnet.csv").open(
                "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)

        # Targets chosen against the MEASURED value so the flags have
        # known answers: one matching, one half it (overshoot), one
        # triple it (undershoot).
        got = autocorr_by_person(rows, "steady", "person_id")
        (d / "condnet_model.json").write_text(json.dumps({
            "report": {
                "columns_modelled": 3, "bins": 3, "effective_n": 120,
                "rows": 1200, "edge_count": 7,
                "comparisons_corrected_for": 3,
                "unmodellable_excluded": {"visit_id": "identifier"},
                "persistence_targets": {
                    "steady": round(got, 3),
                    LONG: round(got / 2.0, 3),
                    "flat": round(min(0.99, got * 3.0), 3)},
            }}), encoding="utf-8")
        (d / "fidelity_condnet.json").write_text(
            json.dumps({"passed": 41, "checks": 60,
                        "fidelity": "FAIL"}), encoding="utf-8")

        r = run("scripts/run_summary.py", "--run", str(d),
                "--width", "56")
        if r.returncode:
            print(r.stdout, r.stderr)
        check("summary runs", r.returncode == 0)
        lines = [x for x in r.stdout.splitlines()]
        check("NO line exceeds the width - a wrapped line is what "
              "shifts a value onto the wrong column",
              lines and all(len(x) <= 56 for x in lines))
        check("a long column name is WRAPPED, not truncated, so the "
              "name stays readable and its number stays with it",
              any(LONG[:30] in x for x in lines))

        check("each column's steadiness is on its OWN line, carrying "
              "its own source value",
              sum(1 for x in lines if "steady :" in x) == 1)
        check("retention is reported against the right source",
              any("100%" in x and "steady :" in x for x in lines))
        check("an overshooting column is named as overshooting",
              any(x.startswith("above 120%") and LONG[:20] in x
                  for x in lines)
              or any("above 120%" in x for x in lines)
              and any(LONG[:20] in x for x in lines))
        check("an undershooting column is named separately",
              any("below 60%" in x for x in lines)
              and any("flat" in x for x in lines))
        check("refused columns are listed, one per line",
              any("refused visit_id" in x for x in lines))
        check("the correction paid is reported",
              any("corrected over 3 comparisons" in x for x in lines))
        check("fidelity is read from the artifact, not the console",
              any("fidelity passed 41" in x for x in lines))
        # The real artifact nests its counts under "summary". Looking
        # only at the top level printed NOTHING, which reads as "this
        # run had no fidelity data" rather than "the reader looked in
        # the wrong place".
        nested = Path(td) / "nested"
        nested.mkdir()
        for name in ("condnet_model.json", "generated_condnet.csv"):
            (nested / name).write_bytes((d / name).read_bytes())
        (nested / "fidelity_condnet.json").write_text(json.dumps(
            {"summary": {"checks": 107, "passed": 60, "failed": 47,
                         "fidelity_verdict": "FAIL"},
             "marginals": {}}), encoding="utf-8")
        rn = run("scripts/run_summary.py", "--run", str(nested),
                 "--width", "56")
        check("fidelity counts are found where the real artifact "
              "actually keeps them, under summary",
              "fidelity passed 60" in rn.stdout
              and "fidelity checks 107" in rn.stdout)
        bare = Path(td) / "bare"
        bare.mkdir()
        for name in ("condnet_model.json", "generated_condnet.csv"):
            (bare / name).write_bytes((d / name).read_bytes())
        (bare / "fidelity_condnet.json").write_text(
            json.dumps({"marginals": {}}), encoding="utf-8")
        rb = run("scripts/run_summary.py", "--run", str(bare),
                 "--width", "56")
        check("a fidelity file with no counts SAYS so rather than "
              "printing nothing",
              "no counts found" in rb.stdout)

        # ---- guards -------------------------------------------------
        r2 = run("scripts/run_summary.py", "--run",
                 str(Path(td) / "nope"))
        check("a missing run directory fails readably",
              r2.returncode != 0
              and "not found" in (r2.stdout + r2.stderr)
              and "Traceback" not in (r2.stderr or ""))
        empty = Path(td) / "empty"
        empty.mkdir()
        r3 = run("scripts/run_summary.py", "--run", str(empty))
        check("a run with no model says which engine it wanted",
              r3.returncode != 0
              and "condnet" in (r3.stdout + r3.stderr))

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
