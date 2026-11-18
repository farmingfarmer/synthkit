"""The data dictionary describes the file we actually send.

WHY IT IS GENERATED. A dictionary written beside a dataset drifts
from it the first time anything changes and the reader cannot
tell. Every row is read from the blueprint and from the CSV, so a
moved column or a thinned vocabulary appears without anybody
remembering.

The two checks that carry the most weight: no SCAFFOLDING column
may appear - a dictionary describing columns the recipient does
not receive is worse than none - and the privacy sentences must
be QUOTED from the blueprint rather than re-worded, because a
privacy claim restated in a second place is how two halves come
to say different things about one guarantee.
"""
import json
import subprocess
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


def fixture(path):
    """A set column, a count that is its length on only some rows,
    and a sparse numeric - so the token census, the count-vs-set
    disagreement and the coverage columns all have something to
    report."""
    r = np.random.RandomState(6)
    npat, vis = 350, 6
    n = npat * vis
    toks = ["t%02d" % i for i in range(8)]
    sets, counts = [], []
    for _ in range(n):
        k = r.randint(1, 5)
        pick = sorted(set(r.choice(toks, size=k, replace=False)))
        sets.append(";".join(pick))
        # the count is taken BEFORE de-duplication on the real
        # extract, so it exceeds the set length on some rows
        counts.append(len(pick) + (1 if r.rand() < 0.4 else 0))
    val = np.round(r.normal(50, 9, n), 2)
    keep = r.rand(n) < 0.3
    pd.DataFrame({
        "person_id": np.repeat(np.arange(npat), vis).astype(str),
        "dx": sets,
        "dx_count": counts,
        "lab": ["" if not k else str(v) for k, v in zip(keep, val)],
    }).to_csv(path, index=False)


def main():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "src.csv"
        fixture(src)
        run = td / "run"
        rv = subprocess.run(
            [sys.executable, "-m", "synthkit.cli", "fit",
             "--src", str(src), "--out", str(run),
             "--group-by", "person_id", "--seed", "3",
             "--generate"],
            cwd=str(ROOT), capture_output=True, text=True,
            timeout=3600)
        check("the real CLI writes a run the dictionary can read",
              rv.returncode == 0
              and (run / "generated.csv").exists())

        r = subprocess.run(
            [sys.executable,
             str(ROOT / "scripts" / "data_dictionary.py"), str(run)],
            cwd=str(ROOT), capture_output=True, text=True,
            timeout=900)
        check("...and the dictionary is generated from it",
              r.returncode == 0
              and (run / "data_dictionary.html").exists()
              and (run / "data_dictionary.csv").exists())

        doc = (run / "data_dictionary.html").read_text(
            encoding="utf-8")
        gen = pd.read_csv(run / "generated.csv", dtype=str,
                          keep_default_na=False)
        rows = pd.read_csv(run / "data_dictionary.csv", dtype=str,
                           keep_default_na=False)
        check("EVERY column of the file it describes has a row - a "
              "dictionary that silently omits a field is worse "
              "than none ({} of {})".format(
                  len(rows), len(gen.columns)),
              sorted(rows["column"]) == sorted(gen.columns))

        bp = json.loads((run / "blueprint.json").read_text(
            encoding="utf-8"))
        from synthkit.sets import is_scaffolding
        scaff = [c for c in (bp.get("columns") or {})
                 if is_scaffolding(c)]
        # A GUARD ON THE ENGINE, NOT ON THE RENDERER, and said so
        # rather than dressed up. The dictionary iterates the
        # FILE, so scaffolding cannot appear unless the engine
        # stopped dropping it before writing - which is the thing
        # actually worth catching here, and would reach a
        # recipient as columns the extract never had.
        check("the FILE carries no scaffolding, so the dictionary "
              "cannot describe a column the recipient does not "
              "receive - a guard on the engine's drop rather than "
              "on this renderer ({} exist in the contract)"
              .format(len(scaff)),
              bool(scaff)
              and not any(c in set(gen.columns) for c in scaff)
              and not any(c in set(rows["column"]) for c in scaff))

        # QUOTED, NOT PARAPHRASED.
        priv = bp.get("privacy") or {}
        check("the privacy sentences are QUOTED from the blueprint "
              "verbatim - re-wording a guarantee in a second place "
              "is how two halves come to say different things "
              "about it",
              priv.get("numeric_bounds")
              and priv["numeric_bounds"][:70] in doc
              and priv.get("NOT_a_formal_guarantee")
              and priv["NOT_a_formal_guarantee"][:60] in doc)

        # COVERAGE IS MEASURED, not copied from the contract.
        lab = rows[rows["column"] == "lab"]
        check("coverage is reported BOTH as published and as "
              "MEASURED in the file, so a column that drifted is "
              "visible rather than taken on trust",
              len(lab) == 1
              and lab.iloc[0]["coverage_published"] != ""
              and lab.iloc[0]["coverage_in_this_file"] != "")

        check("...and the set column reports what the contract "
              "PUBLISHED against what the file actually carries, "
              "which is where the k rule's cost shows up",
              any("published," in d and "distinct in this file" in d
                  for d in rows["detail"]))

        check("...and a count column that disagrees with its set's "
              "length is NAMED with the share, because a tool "
              "reading both fields will see the disagreement",
              "names the length of" in doc)

        for head, _ in __import__("data_dictionary").STRUCTURAL_NOTES:
            pass
        import data_dictionary as DD
        check("...and the structural absences a reviewer must be "
              "told about are stated - no principal diagnosis, no "
              "POA, no discharge disposition - and marked as "
              "REVIEWED statements rather than derived ones",
              all(h in doc for h, _ in DD.STRUCTURAL_NOTES)
              and "Reviewed statements" in doc
              and "principal diagnosis" in doc.lower())

        check("...and a directory with no generated file is "
              "REFUSED by name rather than producing an empty "
              "dictionary",
              subprocess.run(
                  [sys.executable,
                   str(ROOT / "scripts" / "data_dictionary.py"),
                   str(td)], cwd=str(ROOT), capture_output=True,
                  text=True).returncode == 2)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
