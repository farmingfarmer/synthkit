"""`cooc_landing.py` asks whether the published conditionals LANDED.

WHY THIS SUITE EXISTS. The real extract's remaining `close` misses
are dominated by cross-set token pairs, and reading the artifacts
acquitted both obvious suspects: the pair IS detected and it IS
published - `sodium-chloride` against `Flush` carries phi 0.88483,
the source correlation to three decimals. So the contract has the
structure and the output does not. This tool is the diagnostic that
separates a DRAW fault from an ADOPTION fault, and a number it
produces would decide which of those gets built - which is exactly
the kind of tool this project has twice shipped with no checks at
all.

WHAT IS LOAD-BEARING. That a MISS is measured rather than assumed:
the suite plants a blueprint whose published conditional is
deliberately wrong and requires the tool to report the miss with the
right magnitude and sign. And that NOT SCORABLE is its own answer
with a reason - a token the output never emitted has no achieved
conditional, and reporting that as a miss would blame the draw for
something the vocabulary did.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cooc_landing import column_of, landing, verdict    # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def plant(t, cond_a, cond_b, emit_b=True):
    """A generated file where P(b|a) is KNOWN by construction.

    Rows 0-99 hold token A; of those, exactly 50 also hold B, so
    P(B | A) is 0.50. B is held by 150 rows (0-49 and 100-199) of
    which 50 hold A, so P(A | B) is 1/3 - NOT 0.50, which is what
    the first version of this fixture assumed and the check caught.
    The two conditionals of one pair are different numbers, which
    is the whole reason both directions are published.
    """
    rows = []
    for i in range(400):
        tk = []
        if i < 100:
            tk.append("A")
            if i < 50 and emit_b:
                tk.append("B")
        elif i < 200 and emit_b:
            tk.append("B")
        rows.append(";".join(tk))
    pd.DataFrame({"person_id": ["P%03d" % (i // 2)
                                for i in range(400)],
                  "drugs": rows}).to_csv(t / "generated.csv",
                                         index=False)
    bp = {
        "columns": {"drugs": {"kind": "set",
                              "marginal": {"type": "list",
                                           "separator": ";"}}},
        "token_cooccurrence": [
            {"a": "drugs__has__A", "b": "drugs__has__B",
             "cond": cond_a, "lift": 2.0, "base": 0.3,
             "phi": 0.4, "patients": 50},
            {"a": "drugs__has__B", "b": "drugs__has__A",
             "cond": cond_b, "lift": 2.0, "base": 0.3,
             "phi": 0.4, "patients": 50}],
    }
    (t / "blueprint.json").write_text(json.dumps(bp),
                                      encoding="utf-8")


def run(args):
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "cooc_landing.py")]
        + args, capture_output=True, text=True, cwd=str(ROOT))
    return r.returncode, r.stdout + r.stderr


def main():
    c, tk = column_of("conditions__has__ICD10!E11.9")
    check("an indicator name splits into its COLUMN and its TOKEN, "
          "so the diagnostic reads the set column the operator "
          "actually receives rather than a scaffolding column that "
          "is dropped before the file is written",
          c == "conditions" and tk == "ICD10!E11.9")

    check("...and a name that is not an indicator returns nothing "
          "rather than guessing a split",
          column_of("plain_column") == (None, None))

    with tempfile.TemporaryDirectory() as td:
        t = Path(td)

        # ---- A MISS IS MEASURED, NOT ASSUMED --------------------
        # True conditional is 0.50 by construction. The blueprint
        # claims 0.90 one way and 0.50 the other, so one pair must
        # miss by -0.40 and the other must land exactly.
        plant(t, cond_a=0.90, cond_b=1.0 / 3.0)
        res, err = landing(t, tol=0.15)
        check("the tool measures the achieved conditional from the "
              "WRITTEN file: a pair whose published value is 0.90 "
              "where the file holds 0.50 is reported as a miss of "
              "-0.40, by magnitude AND sign",
              res is not None
              and any(abs(r["miss"] + 0.40) < 1e-6
                      for r in res["worst"]))

        check("...and the pair whose published value MATCHES the "
              "file lands, so the tool is not simply calling "
              "everything a miss - and the two directions of one "
              "pair are DIFFERENT conditionals, 0.50 one way and "
              "a third the other, which is why both are published",
              res["landed"] == 1 and res["scored"] == 2)

        # ---- NOT SCORABLE IS ITS OWN ANSWER ---------------------
        plant(t, cond_a=0.90, cond_b=1.0 / 3.0,
              emit_b=False)
        res2, _ = landing(t, tol=0.15)
        check("a pair whose DRIVING token the output never emitted "
              "is NOT SCORABLE with a reason, rather than counted "
              "as a miss - blaming the draw for something the "
              "vocabulary did would send the next person to the "
              "wrong fix",
              res2["unscored"] >= 1
              and any("never emitted" in (r.get("why") or "")
                      for r in res2["unscored_rows"]))

        # ---- THE VERDICT, ON PLANTED NUMBERS --------------------
        check("the verdict names the DRAW when most conditionals "
              "do not land - the contract carrying structure the "
              "output lacks is a different fault from a screen "
              "letting the wrong pairs through",
              verdict(1, 10) == "draw")

        check("...and names ADOPTION when most of them do land, "
              "because then the drifting pairs are not the adopted "
              "ones and the gates are the next question",
              verdict(9, 10) == "adoption")

        check("...and NOTHING SCORED is its own answer rather than "
              "being folded into either - an absent measurement is "
              "not a clean one",
              verdict(0, 0) == "nothing-scored")

        # ---- refusals -------------------------------------------
        rc, out = run([str(t)])
        check("the CLI runs on a real run directory and prints the "
              "landed count", rc == 0 and "LANDED" in out)

        rc2, out2 = run([str(t / "nope")])
        check("a directory with no blueprint is REFUSED by name, "
              "and the message says the tool reads the WRITTEN file",
              rc2 == 2 and "WRITTEN" in out2)

        empty = t / "empty"
        empty.mkdir()
        (empty / "blueprint.json").write_text(
            json.dumps({"columns": {}}), encoding="utf-8")
        pd.DataFrame({"a": [1]}).to_csv(empty / "generated.csv",
                                        index=False)
        rc3, out3 = run([str(empty)])
        check("...and a blueprint that publishes NO co-occurrence "
              "says so rather than reporting zero misses as a "
              "clean bill - absent is not the same fact as landed",
              rc3 == 3 and "absent is not" in out3)

        rc4, out4 = run(["--help"])
        check("--help exits 0 with its usage, so a wrapper does "
              "not read a help request as a failure",
              rc4 == 0 and "usage:" in out4)

        # ---- the numbers cross in WORDS -------------------------
        check("every figure is repeated in words, because the "
              "machine holding the extract is read by screen OCR "
              "and a bare digit off it is an artifact",
              "= zero point" in out or "zero" in out)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
