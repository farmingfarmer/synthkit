"""Smoke: is every statement supported by the record we handed over?

THE CHECK THAT JUSTIFIES THE MODULE is the one that measures
`transcribe.score_extraction` producing an IDENTICAL report for an
honest reader and one that hallucinates a diagnosis on every row.
That loop iterates the LEDGER, so a claim it has no entry for is
invisible to it - and that claim is precisely a hallucination.
Without that check this module would be a second implementation of
grading, which is how two halves come to disagree.

Every one of the five categories is exercised, because a census
with an unreachable branch is a census that cannot report it.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd                          # noqa: E402

from smoke_notefacts import cohort, RARE     # noqa: E402
from synthkit import grounding as G          # noqa: E402
from synthkit.notefacts import (             # noqa: E402
    build_facts, reconcile, default_rates)
from synthkit.transcribe import (            # noqa: E402
    TranscribeSpec, transcribe, score_extraction)

PASS = FAIL = 0
DESC = {c: "condition " + c for c in "BCDEFG"}


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def built(seed=7, rate=0.3, tweak=None):
    df = cohort(seed=seed)
    frame, facts, plan = build_facts(
        df, group_by="person_id", k=10, tokens_per_set=4,
        descriptions=DESC)
    if tweak:
        tweak(facts, plan)
    noted, led = transcribe(
        frame.to_dict("records"),
        TranscribeSpec(facts=facts, rates=default_rates(rate),
                       seed=5))
    final = reconcile(pd.DataFrame(noted), plan)
    return final, facts, plan, led


def asserted(led):
    """What a perfect reader of the prose would claim."""
    return [[{"label": e["label"], "fact": e["fact"],
              "present": True}
             for e in L if e["asserts_present"]] for L in led]


def run_cli(cmd, *argv):
    import subprocess
    r = subprocess.run(
        [sys.executable, "-m", "synthkit.cli", cmd]
        + [str(a) for a in argv],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300)
    return r.returncode, r.stdout + r.stderr


def cli_checks():
    import json
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        run = td / "run"
        run.mkdir()
        cohort().to_csv(run / "generated.csv", index=False)
        desc = td / "d.json"
        json.dump(DESC, open(desc, "w"))
        rc, _ = run_cli("notes", "--run", run, "-o", td / "o",
                        "--descriptions", desc, "--rate", "0.3")
        check("the notes command produced a ledger to audit",
              rc == 0 and (td / "o" / "ledger.json").exists())
        key = json.load(open(td / "o" / "ledger.json"))
        n = len(key["ledger"])

        flat, nested = [], []
        for i, L in enumerate(key["ledger"]):
            row = [{"label": e["label"], "fact": e["fact"],
                    "present": True}
                   for e in L if e["asserts_present"]]
            nested.append(row)
            for g in row:
                flat.append(dict(g, row=i))
            flat.append({"row": i, "label": "acute kidney injury",
                         "present": True})
            nested[-1] = row + [{"label": "acute kidney injury",
                                 "present": True}]
        json.dump(flat, open(td / "flat.json", "w"))
        json.dump(nested, open(td / "nested.json", "w"))

        rc, out_flat = run_cli(
            "grounded", "--ledger", td / "o" / "ledger.json",
            "--claims", td / "flat.json",
            "--notes", td / "o" / "notes.csv",
            "--descriptions", desc, "-o", td / "r1.json")
        check("the audit runs on a FLAT claim list carrying `row`",
              rc == 0)
        rc, out_nest = run_cli(
            "grounded", "--ledger", td / "o" / "ledger.json",
            "--claims", td / "nested.json",
            "--notes", td / "o" / "notes.csv",
            "--descriptions", desc, "-o", td / "r2.json")
        check("...and on one list per row", rc == 0)
        r1 = json.load(open(td / "r1.json"))
        r2 = json.load(open(td / "r2.json"))
        check("both shapes reach the SAME verdict - a vendor's "
              "output shape is not a grounding question",
              r1["grounding"]["counts"]
              == r2["grounding"]["counts"])
        check("and the hallucination is caught through the real "
              "CLI: {} ungrounded".format(
                  r1["grounding"]["counts"]["ungrounded"]),
              r1["grounding"]["counts"]["ungrounded"] == n)

        # WITHOUT the record, a claim read off a code set would be
        # called a hallucination - so the run SAYS the record was
        # not consulted rather than reporting a worse number as
        # though it were the same measurement
        rc, nonotes = run_cli(
            "grounded", "--ledger", td / "o" / "ledger.json",
            "--claims", td / "flat.json")
        check("with no --notes the run warns that the structured "
              "record could not be consulted",
              rc == 0 and "could not be consulted" in nonotes)

        # --- refusals, by exit code
        bad = td / "bad.json"
        json.dump({"nothing": 1}, open(bad, "w"))
        # --- CONSISTENCY THROUGH THE CLI. No ground truth is
        # involved, which is what makes it usable where nothing
        # can be graded.
        rc, same = run_cli(
            "grounded", "--ledger", td / "o" / "ledger.json",
            "--claims", td / "nested.json",
            "--claims-b", td / "nested.json",
            "--notes", td / "o" / "notes.csv", "-o", td / "c1.json")
        check("two identical readings are CONSISTENT through the "
              "real CLI",
              rc == 0 and "CONSISTENT" in same)
        honest = [[g for g in row
                   if g["label"] != "acute kidney injury"]
                  for row in nested]
        json.dump(honest, open(td / "honest.json", "w"))
        rc, diff = run_cli(
            "grounded", "--ledger", td / "o" / "ledger.json",
            "--claims", td / "nested.json",
            "--claims-b", td / "honest.json",
            "--notes", td / "o" / "notes.csv", "-o", td / "c2.json")
        c2 = json.load(open(td / "c2.json")).get("consistency")
        differ = (c2 or {}).get("differ")
        check("...and two readings that differ are named with the "
              "rows and the labels: {} differ".format(differ),
              rc == 0 and differ == n
              and "acute kidney injury" in diff)
        r1_con = json.load(open(td / "r1.json")).get(
            "consistency", "KEY-ABSENT")
        check("a run without --claims-b records no consistency "
              "result rather than an empty one that reads as "
              "agreement (got {!r})".format(r1_con),
              r1_con is None)
        stub = td / "stub.json"
        json.dump(nested[:3], open(stub, "w"))
        rc, _ = run_cli(
            "grounded", "--ledger", td / "o" / "ledger.json",
            "--claims", td / "nested.json",
            "--claims-b", stub)
        check("a second reading of the wrong length is refused",
              rc != 0)

        rc, _ = run_cli("grounded", "--ledger", bad,
                        "--claims", td / "flat.json")
        check("a file that is not a ledger is refused", rc != 0)
        short = td / "short.json"
        json.dump(nested[:3], open(short, "w"))
        rc, _ = run_cli("grounded", "--ledger",
                        td / "o" / "ledger.json",
                        "--claims", short)
        check("a claim list of the wrong length is refused rather "
              "than graded on the first few rows", rc != 0)
        other = td / "other"
        other.mkdir()
        cohort(seed=99).head(20).to_csv(
            other / "generated.csv", index=False)
        run_cli("notes", "--run", other, "-o", td / "o2",
                "--rate", "0")
        rc, _ = run_cli("grounded", "--ledger",
                        td / "o" / "ledger.json",
                        "--claims", td / "flat.json",
                        "--notes", td / "o2" / "notes.csv")
        check("a notes file from a DIFFERENT run is refused - "
              "these are not the same cohort", rc != 0)


def main():
    final, facts, plan, led = built()
    perfect = asserted(led)

    a = G.audit(led, perfect, frame=final, facts=facts, plan=plan,
                descriptions=DESC)
    check("the fixture gives the audit something to censuse, "
          "without which every check below passes on zero: {} "
          "claims".format(a["claims_censused"]),
          a["claims_censused"] > 200)
    check("a reader that claims exactly what the prose asserts is "
          "CLEAN", a["verdict"].startswith("CLEAN"))
    check("...and every one of its claims is supported by BOTH "
          "sources", a["counts"]["both"] == a["claims_censused"])

    # --- THE JUSTIFICATION, MEASURED RATHER THAN ARGUED
    liar = [c + [{"label": "acute kidney injury",
                  "fact": "conditions__has__ICD10-CM!ZZZ",
                  "present": True}] for c in perfect]
    s_ok = score_extraction(
        led, [[{"fact": g["fact"], "present": True} for g in c]
              for c in perfect])
    s_bad = score_extraction(
        led, [[{"fact": g["fact"], "present": True} for g in c]
              for c in liar])
    check("score_extraction reports IDENTICALLY for an honest "
          "reader and one hallucinating a diagnosis on every row - "
          "which is why this module is not a second scorer",
          s_ok == s_bad)

    b = G.audit(led, liar, frame=final, facts=facts, plan=plan,
                descriptions=DESC)
    check("...and the audit CATCHES it: {} ungrounded".format(
        b["counts"]["ungrounded"]),
          b["counts"]["ungrounded"] == len(led)
          and b["counts"]["ungrounded"] > 0)
    check("it names examples rather than reporting a count a "
          "reader cannot act on",
          b["ungrounded_examples"]
          and b["ungrounded_examples"][0]["label"]
          == "acute kidney injury")
    check("the verdict is not CLEAN",
          not b["verdict"].startswith("CLEAN"))

    # --- CONTRADICTED is its own category, disjoint from
    # ungrounded: a finding the prose DENIES is a different fault
    # from one nothing mentions, and they take different actions
    denied = [[{"label": e["label"], "fact": e["fact"],
                "present": True}
               for e in L if not e["asserts_present"]]
              for L in led]
    c = G.audit(led, denied, frame=final, facts=facts, plan=plan,
                descriptions=DESC)
    check("claiming a finding the prose DENIES is contradicted, "
          "not ungrounded: {}".format(c["counts"]),
          c["counts"]["contradicted"] > 100
          and c["counts"]["ungrounded"] == 0)

    # --- record_only. A TOKEN THAT BECAME NO FACT IS STILL IN THE
    # RECORD, and crediting only expanded facts accused a vendor
    # who read a genuine diagnosis off the set string.
    lab = RARE.split("!")[1]
    rec_claims = [[{"label": lab, "present": True}]
                  if RARE in str(v) else []
                  for v in final["conditions"]]
    d = G.audit(led, rec_claims, frame=final, facts=facts,
                plan=plan, descriptions=DESC)
    check("a real token that became NO fact still counts as "
          "support - calling that a hallucination is a false "
          "accusation: {} record_only".format(
              d["counts"]["record_only"]),
          d["counts"]["record_only"] > 50
          and d["counts"]["ungrounded"] == 0
          and d["verdict"].startswith("CLEAN"))

    # --- text_only
    def make_text_only(facts_, plan_):
        for f in facts_:
            if f.kind == "condition":
                f.placement = "text_only"
                return
    f2, fac2, pl2, led2 = built(tweak=make_text_only)
    only_lab = [f.label for f in fac2
                if f.placement == "text_only"][0]
    t_claims = [[{"label": only_lab, "present": True}]
                if any(e["label"] == only_lab
                       and e["asserts_present"] for e in L)
                else [] for L in led2]
    e = G.audit(led2, t_claims, frame=f2, facts=fac2, plan=pl2,
                descriptions=DESC)
    check("a text_only fact is supported by the PROSE alone, so "
          "claiming it proves the tool read the note: {} "
          "text_only".format(e["counts"]["text_only"]),
          e["counts"]["text_only"] > 20
          and e["counts"]["both"] == 0
          and e["counts"]["ungrounded"] == 0)

    # --- the identity a reader can check against
    for name, rep in (("perfect", a), ("liar", b),
                      ("denied", c), ("record", d),
                      ("text", e)):
        check("the {} census totals its claims exactly".format(
            name),
              sum(rep["counts"].values()) == rep["claims_censused"])

    # --- NOTHING CLAIMED is its own answer
    none = G.audit(led, [[] for _ in led], frame=final,
                   facts=facts, plan=plan, descriptions=DESC)
    check("a tool that asserts nothing cannot be ungrounded, and "
          "that is NOT reported as clean",
          none["claims_censused"] == 0
          and "NOTHING CLAIMED" in none["verdict"]
          and not none["verdict"].startswith("CLEAN"))

    # --- a claim of ABSENCE is not an assertion needing support
    neg = [[{"label": "acute kidney injury", "present": False}]
           for _ in led]
    check("a negative claim is not censused - counting 'no "
          "diabetes' as a statement needing support would make "
          "silence indistinguishable from accuracy",
          G.audit(led, neg, frame=final, facts=facts, plan=plan,
                  descriptions=DESC)["claims_censused"] == 0)

    # --- consistency needs no ground truth at all
    check("two identical readings agree",
          G.consistency(perfect, perfect)["verdict"]
          == "CONSISTENT")
    pert = G.consistency(perfect, liar)
    check("...and a reading that differs on the same record is "
          "named with the rows",
          pert["differ"] > 0 and pert["examples"]
          and "acute kidney injury" in
          pert["examples"][0]["only_second"])

    # --- source_followed, and absent is not zero
    def make_disagree(facts_, plan_):
        for f in facts_:
            if f.kind == "condition":
                f.placement = "both_disagree"
                return
    f3, fac3, pl3, led3 = built(seed=11, tweak=make_disagree)
    dis_lab = [f.label for f in fac3
               if f.placement == "both_disagree"][0]
    follow_text = [[{"label": e["label"], "present":
                     e["asserts_present"]}
                    for e in L if e["label"] == dis_lab]
                   for L in led3]
    sf = G.source_followed(led3, follow_text)
    check("a planted disagreement is counted, and a reader that "
          "believes the PROSE is recorded as following the text: "
          "{}".format(sf["followed"]),
          sf["disagreements"] > 20
          and sf["followed"]["text"] == sf["disagreements"])
    sf0 = G.source_followed(led, perfect)
    check("with no planted disagreement it says so rather than "
          "reporting a zero that reads as agreement",
          sf0["disagreements"] == 0 and sf0["note"])

    # --- a length mismatch is refused rather than zipped short
    try:
        G.audit(led, perfect[:5], frame=final, facts=facts,
                plan=plan)
        ok = False
    except ValueError:
        ok = True
    check("one ledger, one record row and one claim list per row "
          "- a mismatch raises rather than silently grading the "
          "first few", ok)

    cli_checks()

    print()
    if FAIL:
        print("{} of {} checks failed.".format(FAIL, PASS + FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
