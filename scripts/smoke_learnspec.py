"""Smoke: learned structure is legible, tunable, and survives all
the way into the free text.

The chain this defends: a relationship that exists in real data
should be MEASURED, described in clinical English, adjustable by a
dial, reproduced in generated records, and still recoverable by
reading the notes written from those records. A break anywhere in
that chain makes the text decorative rather than evidential.
"""
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit.condnet import CondNet          # noqa: E402
from synthkit import learnspec as L           # noqa: E402
from synthkit.transcribe import (             # noqa: E402
    TranscribeSpec, transcribe)

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def cohort(seed=3, patients=140, link=0.9):
    """CHF patients get a diuretic `link` of the time."""
    r = random.Random(seed)
    rows = []
    for pid in range(patients):
        chf = r.random() < 0.35
        bp = r.gauss(132, 16)
        for _ in range(r.randint(2, 10)):
            sbp = r.gauss(bp, 8)
            cs = [c for c in ("dm", "htn", "ckd")
                  if r.random() < 0.4]
            if chf:
                cs.append("chf")
            dr = [d for d in ("lisinopril", "metformin")
                  if r.random() < 0.5]
            if chf and r.random() < link:
                dr.append("furosemide")
            rows.append({
                "person_id": "P%03d" % pid,
                "systolic_blood_pressure": round(sbp, 1),
                "diastolic_blood_pressure":
                    round(0.55 * sbp + r.gauss(0, 5), 1),
                "conditions": "; ".join(sorted(cs)) or "none",
                "active_drugs": "; ".join(sorted(dr)) or "none"})
    return rows


def link_strength(rows, note_key=None):
    """How much more often does the diuretic appear WITH heart
    failure than without? Measured from records, or from the
    text alone when a note key is given."""
    def has(row, term):
        if note_key:
            txt = str(row.get(note_key, "")).lower()
            # only count plain assertions, not denials
            for clause in re.split(r"[.;\n]", txt):
                if term in clause:
                    if any(neg in clause for neg in
                           ("no ", "denies ", "not present",
                            "is not")):
                        return False
                    return True
            return False
        return term in str(row.get("conditions", "")).lower() + \
            str(row.get("active_drugs", "")).lower()
    with_chf = [r for r in rows if has(r, "chf")]
    without = [r for r in rows if not has(r, "chf")]
    if not with_chf or not without:
        return 0.0
    a = sum(1 for r in with_chf if has(r, "furosemide")) \
        / len(with_chf)
    b = sum(1 for r in without if has(r, "furosemide")) \
        / len(without)
    return a - b


def main():
    rows = cohort()
    src_link = link_strength(rows)
    check("the source data carries a planted clinical link "
          "(heart failure patients get a diuretic)",
          src_link > 0.5)

    net = CondNet(k=10, max_parents=3).learn(
        rows, group_by="person_id")

    # ---- legible ----
    nar = L.narrate(net)
    check("the model reports what it found in clinical English, "
          "not in symbol names",
          nar["findings"]
          and all(" moves with " in f["sentence"]
                  or " depends on " in f["sentence"]
                  for f in nar["findings"]))
    check("the planted link is among the findings",
          any("furosemide" in f["sentence"]
              or "chf" in f["sentence"]
              for f in nar["findings"]))
    check("findings and the pipeline's own arithmetic are "
          "reported as separate things",
          "findings" in nar and "bookkeeping" in nar)
    check("the headline counts PATIENTS and states how many "
          "relationships were found",
          "patients" in nar["headline"]
          and "relationship" in nar["headline"])
    check("the caveat says how much data each kind of pattern "
          "needs, so a thin result is not oversold",
          "800" in nar["caveat"] and "3,200" in nar["caveat"])

    # ---- tunable ----
    ds = L.dials(net)
    check("every finding is offered as a dial with a plain label",
          ds and all(d["label"] and "factor" in d for d in ds))
    target = [d["column"] for d in ds
              if "chf" in d["column"] or "furosemide"
              in d["column"]]
    check("the planted link is one of the dials",
          bool(target))

    strong = CondNet.from_json(net.to_json())
    flat = CondNet.from_json(net.to_json())
    L.apply_dials(flat, {c: 0.0 for c in target})
    gen_asfound = strong.sample(3000, seed=5)
    gen_flat = flat.sample(3000, seed=5)
    found_link = link_strength(gen_asfound)
    flat_link = link_strength(gen_flat)
    check("generated data reproduces the planted link when the "
          "dial is left as found", found_link > 0.3)
    check("turning that dial to zero REMOVES the link from the "
          "generated data — the pattern is genuinely under the "
          "user's control", flat_link < found_link / 2)

    # ---- survives into the text ----
    facts = L.facts_from_model(net)
    check("a note plan is derived from the learned columns",
          len(facts) >= 4)
    spec = TranscribeSpec(facts, rates=L.default_rates(), seed=4)
    noted, ledgers = transcribe(gen_asfound, spec)
    check("every generated record gets a note and a truth ledger",
          len(noted) == len(gen_asfound)
          and len(ledgers) == len(noted)
          and all("clinical_note" in r for r in noted))

    text_link = link_strength(noted, note_key="clinical_note")
    check("the planted relationship is STILL RECOVERABLE by "
          "reading the notes alone — the text carries the "
          "structure, it does not merely decorate it",
          text_link > 0.15)
    check("...and reading the text does not overstate it either",
          text_link <= src_link + 0.25)

    noted_flat, _ = transcribe(gen_flat, spec)
    flat_text = link_strength(noted_flat, note_key="clinical_note")
    check("when the dial removed the relationship, it is absent "
          "from the notes too",
          flat_text < text_link / 2 + 0.05)

    joined = " ".join(r["clinical_note"] for r in noted[:200])
    check("the notes are messy the way clinical prose is messy",
          any(m in joined for m in ("No ", "denies", "Possible",
                                    "History of", "per prior")))
    check("measurements are charted, not printed at full "
          "floating-point precision",
          not re.search(r"pressure \\d+\\.\\d{3,}", joined))

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
