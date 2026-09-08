"""Smoke: a language model in the note-extraction chair, graded by
corruption type, with unusable replies counted rather than fatal."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from synthkit.noteextract import (          # noqa: E402
    NoteExtractor, ScriptedBackend, facts_from_specs, SYSTEM)
from synthkit.transcribe import (           # noqa: E402
    CORRUPTIONS, TranscribeSpec, transcribe, score_extraction)

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


class Canned:
    """Returns whatever we hand it — for exercising the parser."""

    def __init__(self, reply):
        self.reply = reply
        self.seen = []

    def complete(self, prompt, system="", max_tokens=0,
                 temperature=0.0):
        self.seen.append((prompt, system))
        return self.reply


def main():
    from narrative_showdown import FACTS, TERMS, cohort
    facts = facts_from_specs(FACTS)
    check("only facts that reach the prose are asked about — a "
          "structured-only fact is never put to the model",
          all(f["key"] != "structured_only" for f in facts)
          and len(facts) == len([f for f in FACTS
                                 if f.placement
                                 != "structured_only"]))

    # ---------- the contract the model is held to ----------
    for word in ("present", "current", "certain", "JSON array"):
        check("the prompt asks explicitly about `{}`"
              .format(word), word in SYSTEM)
    # Defining the fields the model must fill is necessary — it
    # cannot answer "certain" without being told what certain
    # means. Naming the TRAPS would be a different thing entirely,
    # and would make the exam meaningless.
    TRAPS = ("negation_compound", "negation_scope_trap",
             "copy_forward", "transcription_error",
             "compound", "scope", "carried forward",
             "transposed")
    check("the prompt is blind to the TRAPS — it defines the "
          "fields it asks for but never says which corruptions "
          "were planted",
          not any(t in SYSTEM.lower() for t in TRAPS))
    check("...while still defining what it means by present, "
          "current and certain, or the answers would be "
          "unanswerable",
          "negated" in SYSTEM and "hedged" in SYSTEM
          and "historical" in SYSTEM)

    # ---------- parsing and its failure modes ----------
    good = json.dumps([{"fact": facts[0]["key"], "present": False,
                        "current": True, "certain": False,
                        "value": None}])
    ex = NoteExtractor(Canned(good), facts)
    out = ex.extract("any note")
    check("a well-formed reply parses into the grading contract",
          out and out[0]["present"] is False
          and out[0]["certain"] is False)
    ex2 = NoteExtractor(Canned("Sure! Here is what I found:"),
                        facts)
    check("an unusable reply is COUNTED, not crashed on",
          ex2.extract("n") == []
          and ex2.reliability()["malformed"] == 1)
    ex3 = NoteExtractor(Canned(json.dumps(
        [{"fact": "not_a_real_fact", "present": True}])), facts)
    ex3.extract("n")
    check("an invented fact key is rejected and counted as its "
          "own failure mode",
          ex3.reliability()["hallucinated_fact_keys"] == 1)
    ex4 = NoteExtractor(Canned(
        "here you go ```json\n[{\"fact\": \"" + facts[0]["key"]
        + "\", \"present\": \"yes\"}]\n``` hope that helps"),
        facts)
    o4 = ex4.extract("n")
    check("a reply wrapped in prose and fences is salvaged, and "
          "loose booleans are read",
          o4 and o4[0]["present"] is True
          and ex4.reliability()["malformed"] == 0)
    ex5 = NoteExtractor(Canned("[]"), facts)
    ex5.extract("n")
    check("an empty array is distinguished from an unusable reply",
          ex5.reliability()["empty_replies"] == 1
          and ex5.reliability()["malformed"] == 0)

    # ---------- the intervention seam ----------
    canned = Canned("[]")
    ex6 = NoteExtractor(canned, facts,
                        extra_system="ALWAYS check negation scope.")
    ex6.extract("n")
    check("prompt hardening reaches the system prompt as an "
          "explicit, diffable addition",
          "ALWAYS check negation scope." in canned.seen[0][1]
          and canned.seen[0][1].startswith(SYSTEM[:40]))

    # ---------- grading three readers ----------
    rows, _ = cohort(200, 2, 5)
    rates = {c: 0.30 for c in CORRUPTIONS}
    rates["abbreviation"] = 0.5
    spec = TranscribeSpec(FACTS, rates=rates, seed=5)
    noted, ledgers = transcribe(rows, spec)
    notes = [n["clinical_note"] for n in noted]

    reps = {}
    for behavior in ("naive", "careful", "malformed"):
        b = ScriptedBackend(facts, behavior, TERMS)
        e = NoteExtractor(b, facts, name=behavior)
        reps[behavior] = (score_extraction(
            ledgers, e.extract_all(notes)), e.reliability())

    naive_r, naive_rel = reps["naive"]
    careful_r, _ = reps["careful"]
    bad_r, bad_rel = reps["malformed"]

    check("a keyword reader scores perfect recall on clean "
          "mentions and reverses every negated one",
          naive_r["clean"]["recall"] == 1.0
          and naive_r["negation_simple"][
              "false_positive_rate"] > 0.9)
    check("...and takes historical mentions as current",
          naive_r["temporal_history"][
              "stale_taken_as_current"] > 0)
    check("a clause-scoped reader fixes negation but is destroyed "
          "by the scope trap",
          careful_r["negation_simple"][
              "false_positive_rate"] < 0.1
          and careful_r["negation_scope_trap"]["recall"] < 0.2)
    check("the two readers fail DIFFERENTLY — the finding a "
          "single overall score would erase",
          naive_r["negation_scope_trap"]["recall"]
          > careful_r["negation_scope_trap"]["recall"])
    check("a model that cannot hold the answer format is reported "
          "as unreliable rather than as merely inaccurate",
          bad_rel["malformed_rate"] > 0.9
          and "procurement finding" in bad_rel["reading"])
    check("reliability is reported alongside accuracy for every "
          "reader",
          all("malformed_rate" in r[1] for r in reps.values())
          and naive_rel["malformed_rate"] == 0.0)

    # ---------- the runner ----------
    with tempfile.TemporaryDirectory() as td:
        res = subprocess.run(
            [sys.executable, "scripts/note_vendor_run.py",
             "--backend", "stub", "--behavior", "careful",
             "--notes", "60", "-o", td, "--report"],
            capture_output=True, text=True, cwd=str(ROOT))
        check("the runner seats a reader and writes a report",
              res.returncode == 0
              and (Path(td) / "note_vendor_report.json").exists())
        P = json.loads((Path(td) / "note_vendor_report.json")
                       .read_text(encoding="utf-8"))
        check("the report carries both halves a procurement "
              "conversation needs",
              "reliability" in P
              and len(P["extraction_by_corruption"]) >= 5)
        check("the report explains what each corruption tests",
              "what each corruption tests" in res.stdout)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
