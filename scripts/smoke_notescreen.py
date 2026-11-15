"""Smoke: a note mentions only what the blueprint published.

A NOTE IS A RELEASE, so the k rule governs prose exactly as it
governs a column - and the check must compare against the
BLUEPRINT rather than against the fact list, because the facts
are derived from the published vocabulary and comparing them with
themselves would pass however the derivation drifted. The
mutation that makes it circular is below.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from note_screen import screen                    # noqa: E402

PASS = FAIL = 0
NS = str(ROOT / "scripts" / "note_screen.py")


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def bp_with(tokens):
    return {"columns": {"conditions": {"marginal": {
        "type": "list", "separator": ";",
        "tokens": [{"value": t, "p": 0.1} for t in tokens]}}}}


def key_with(cols):
    return {"facts": [{"column": c, "label": "lab " + c}
                      for c in cols]}


def run(*a):
    r = subprocess.run([sys.executable, NS] + [str(x) for x in a],
                       capture_output=True, text=True, timeout=300)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def main():
    pub = ["ICD10-CM!A", "ICD10-CM!B", "ICD10-CM!C"]
    bp = bp_with(pub)

    good = key_with(["conditions__has__ICD10-CM!A",
                     "conditions__has__ICD10-CM!B"])
    r = screen(bp, good)
    check("a note set mentioning only published tokens is CLEAN, "
          "over a NON-ZERO denominator: {} mentioned".format(
              r["tokens_mentioned"]),
          r["tokens_mentioned"] == 2 and not r["leaked"]
          and r["verdict"].startswith("CLEAN"))

    bad = key_with(["conditions__has__ICD10-CM!A",
                    "conditions__has__ICD10-CM!WITHHELD"])
    r = screen(bp, bad)
    check("a token the blueprint did NOT publish is caught and "
          "NAMED, because a count a reader cannot act on is not "
          "a finding",
          len(r["leaked"]) == 1
          and r["leaked"][0]["token"] == "ICD10-CM!WITHHELD"
          and "undone the k screen" in r["verdict"])

    r = screen(bp, {"facts": []})
    check("NO TOKEN MENTIONED is its own answer and NOT a clean "
          "one - a screen over nothing is the worst kind of "
          "clean",
          r["tokens_mentioned"] == 0
          and "not measured" in r["verdict"]
          and not r["verdict"].startswith("CLEAN"))

    # a set column the blueprint says nothing about is a DIFFERENT
    # fault from a token outside a known list, and is flagged
    # rather than skipped
    r = screen(bp, key_with(["procedures__has__71260"]))
    check("a column with no published token list is flagged "
          "rather than passed over in silence",
          len(r["leaked"]) == 1
          and "no token list" in r["leaked"][0]["why"])

    # a non-indicator fact is not a token claim at all
    r = screen(bp, key_with(["spo2"]))
    check("a plain measurement fact is not counted as a token "
          "mention", r["tokens_mentioned"] == 0)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "bp.json").write_text(json.dumps(bp),
                                    encoding="utf-8")
        (td / "ok.json").write_text(json.dumps(good),
                                    encoding="utf-8")
        (td / "bad.json").write_text(json.dumps(bad),
                                     encoding="utf-8")
        rc, out = run("--ledger", td / "ok.json",
                      "--blueprint", td / "bp.json")
        check("the clean case exits 0 through the real CLI",
              rc == 0 and "CLEAN" in out)
        rc, out = run("--ledger", td / "bad.json",
                      "--blueprint", td / "bp.json")
        check("...and a leak exits NON-ZERO, so a pipeline can "
              "stop on it rather than printing a warning nobody "
              "reads", rc != 0 and "LEAKED" in out)
        (td / "nope.json").write_text(json.dumps({"x": 1}),
                                      encoding="utf-8")
        rc, out = run("--ledger", td / "nope.json",
                      "--blueprint", td / "bp.json")
        check("a file that is not a ledger is refused", rc == 2)

    print()
    if FAIL:
        print("{} of {} checks failed.".format(FAIL, PASS + FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
