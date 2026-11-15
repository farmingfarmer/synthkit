"""Smoke: the stand-in readers, and the TRADE that makes them a
ruler.

THE CENTRAL CHECK is that the pair trades. A keyword reader walks
into every plain negation - "no diabetes" contains "diabetes" -
and gets the SCOPE TRAP right by luck, because there the finding
really is present. A negation-aware reader fixes the first and
breaks the second, because the trap puts a negation in the clause
without letting it reach the finding. If both readers failed the
same way there would be nothing to grade and the taxonomy would
be decoration.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd                               # noqa: E402

from stand_in_reader import read_note, MODES       # noqa: E402

PASS = FAIL = 0
SI = str(ROOT / "scripts" / "stand_in_reader.py")
F = [{"label": "type 2 diabetes mellitus",
      "column": "conditions__has__E11.9"}]


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def claims(txt, mode):
    return bool(read_note(txt, F, mode))


def main():
    plain = "Assessment: type 2 diabetes mellitus, stable."
    neg = "Assessment: no type 2 diabetes mellitus on review."
    trap = ("Assessment: no improvement in type 2 diabetes "
            "mellitus.")
    absent = "Assessment: patient is well."

    check("every reader finds a plainly stated finding, without "
          "which nothing below means anything",
          all(claims(plain, m) for m in MODES))
    check("and none invents one that is not mentioned at all",
          not any(claims(absent, m) for m in MODES))

    # --- THE TRADE
    check("the keyword reader walks into a plain NEGATION - 'no "
          "diabetes' contains 'diabetes'",
          claims(neg, "keyword"))
    check("...and the negation-aware reader does not",
          not claims(neg, "negation-aware"))
    check("but the keyword reader gets the SCOPE TRAP right, "
          "where the negation does not reach the finding and the "
          "finding IS present",
          claims(trap, "keyword"))
    check("...and the negation-aware reader gets it WRONG, which "
          "is the trade that makes this a competence rather than "
          "a mark",
          not claims(trap, "negation-aware"))

    # --- GREEDY over-claims, which is the GROUNDING audit's
    # business rather than the per-type scorer's
    two = [{"label": "arterial systolic pressure",
            "column": "systolic_blood_pressure_invasive"},
           {"label": "arterial mean pressure",
            "column": "mean_arterial_pressure_invasive"}]
    only_one = "Vitals: arterial systolic pressure 128 mmHg."
    g = read_note(only_one, two, "greedy")
    k = read_note(only_one, two, "keyword")
    check("the greedy reader claims BOTH pressure facts from a "
          "note that states ONE, because they share a word - "
          "which is the over-claiming the grounding audit "
          "exists for: greedy {} against keyword {}".format(
              len(g), len(k)),
          len(g) == 2 and len(k) == 1)

    # --- the CLI
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        key = {"facts": F,
               "ledger": [[{"label": F[0]["label"],
                            "fact": F[0]["column"],
                            "asserts_present": True,
                            "corruptions": []}],
                          []]}
        (td / "k.json").write_text(json.dumps(key),
                                   encoding="utf-8")
        pd.DataFrame({"clinical_note": [plain, absent]}).to_csv(
            td / "n.csv", index=False)

        def run(*a):
            r = subprocess.run(
                [sys.executable, SI] + [str(x) for x in a],
                capture_output=True, text=True, timeout=300)
            return r.returncode, (r.stdout or "") + (r.stderr or "")

        rc, out = run("--notes", td / "n.csv", "--ledger",
                      td / "k.json", "--mode", "keyword",
                      "-o", td / "c.json")
        got = json.loads((td / "c.json").read_text(
            encoding="utf-8"))
        check("the CLI writes one claim list per NOTE, in order",
              rc == 0 and len(got) == 2 and len(got[0]) == 1
              and got[1] == [])

        pd.DataFrame({"clinical_note": [absent, absent]}).to_csv(
            td / "none.csv", index=False)
        rc, out = run("--notes", td / "none.csv", "--ledger",
                      td / "k.json", "--mode", "keyword",
                      "-o", td / "c2.json")
        check("a reader that finds NOTHING says so - that is its "
              "own answer and not a clean one",
              rc == 0 and "NOTHING CLAIMED" in out)

        pd.DataFrame({"clinical_note": [plain]}).to_csv(
            td / "short.csv", index=False)
        rc, _ = run("--notes", td / "short.csv", "--ledger",
                    td / "k.json", "--mode", "keyword",
                    "-o", td / "c3.json")
        check("a note file of the wrong length is refused - these "
              "would not be the same run", rc != 0)

        (td / "bad.json").write_text(json.dumps({"x": 1}),
                                     encoding="utf-8")
        rc, _ = run("--notes", td / "n.csv", "--ledger",
                    td / "bad.json", "--mode", "keyword",
                    "-o", td / "c4.json")
        check("a file that is not a ledger is refused", rc == 2)

    print()
    if FAIL:
        print("{} of {} checks failed.".format(FAIL, PASS + FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
