"""Smoke: note facts derived from a cohort's own columns.

The load-bearing property is the UNDO. `transcribe` honors
`both_disagree` and `text_only` by overriding the fact's own
column - and for a set-derived fact that column is a derived
indicator, so an override there leaves the SET STRING still
asserting what the note denies. The set is what the recipient
reads, so a planted disagreement would be invisible and the
record would contradict its own note for a reason nobody could
see. Every check below that mentions "the set string" exists for
that.

And the k screen governs a note exactly as it governs a published
column, because a note is a release.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                           # noqa: E402
import pandas as pd                          # noqa: E402

from synthkit import sets as S               # noqa: E402
from synthkit.notefacts import (             # noqa: E402
    build_facts, reconcile, default_rates, label_for_token,
    label_for_column, NOTE_MEASURES)
from synthkit.transcribe import (            # noqa: E402
    CORRUPTIONS, TranscribeSpec, transcribe)

PASS = FAIL = 0
DESC = {"A": "type 2 diabetes mellitus",
        "B": "essential hypertension",
        "C": "gastro-oesophageal reflux",
        "D": "age-related osteoporosis"}


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


# The rare token is named so it sorts FIRST and is carried on more
# rows than any other, so neither alphabetical order nor frequency
# can be the thing that excludes it. Only the patient count can.
RARE = "ICD10-CM!AAA"


def cohort(seed=7):
    """A frame shaped like the extract: joined code sets, a few
    charted measurements, and a token held by TWO PATIENTS across
    MORE ROWS THAN ANY OTHER TOKEN - the shape the k rule exists
    for, and the only shape on which the patient screen is
    provably the thing doing the refusing.

    Two carriers get eighty visits each so the rare token outranks
    every common one by frequency; there are six common tokens at a
    low rate each so the mean set size still reads as a set.
    """
    rng = np.random.default_rng(seed)
    conds = ["ICD10-CM!" + c for c in "BCDEFG"]
    drugs = ["heparin", "ondansetron", "acetaminophen", "docusate"]
    rows = []
    for pid in range(100):
        visits = 80 if pid in (0, 1) else 4
        for v in range(visits):
            cs = [c for c in conds if rng.random() < 0.22]
            if pid in (0, 1):
                cs.append(RARE)
            rows.append(dict(
                person_id=pid, visit_number=v + 1,
                conditions=";".join(cs),
                active_drugs=";".join(
                    d for d in drugs if rng.random() < 0.4),
                spo2=round(float(rng.normal(96, 2)), 3),
                respiratory_rate=round(float(rng.normal(17, 3)), 3),
                glasgow_coma_score=int(rng.integers(11, 16)),
                span_days=int(rng.integers(1, 9))))
    return pd.DataFrame(rows)


def run_cli(*argv):
    """Through the process boundary, because a content check
    cannot see a dead path and the parser is part of what is
    being tested."""
    import subprocess
    r = subprocess.run(
        [sys.executable, "-m", "synthkit.cli", "notes"]
        + [str(a) for a in argv],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300)
    return r.returncode, r.stdout + r.stderr


def sparse_checks():
    """A VITAL RECORDED ON A FIFTH OF ENCOUNTERS IS STILL A VITAL.

    The first real run carried NO vitals line, and two faults were
    hiding behind one number: this list named the generic column
    forms where the extract uses specific ones, and the gate was a
    presence SHARE - on the inpatient slice every declared vital
    sits under 0.5. A fixture with fully populated measurements
    could reach neither.
    """
    from synthkit.notefacts import MIN_MEASURE_ROWS
    df = cohort()
    # spo2 recorded on about a fifth of rows, as in the extract
    keep = np.arange(len(df)) % 5 == 0
    # pandas 3 refuses a string into a float64 column, so the
    # column becomes object first - the same TypeError that once
    # passed here and crashed on the data machine
    df["spo2"] = df["spo2"].astype(object)
    df.loc[~keep, "spo2"] = ""
    share = float(keep.mean())
    rows_with = int(keep.sum())
    check("the fixture's vital is sparse like the real one - "
          "{} rows, a share of {:.3f}".format(rows_with, share),
          share < 0.5 and rows_with >= MIN_MEASURE_ROWS)

    frame, facts, plan = build_facts(
        df, group_by="person_id", k=10, tokens_per_set=4,
        descriptions=DESC)
    labels = [f.label for f in facts if f.kind == "measurement"]
    check("...and it STILL becomes a charted fact, because a "
          "measurement needs ROWS and not a share: {}".format(
              labels),
          "oxygen saturation" in labels)

    noted, led = transcribe(
        frame.to_dict("records"),
        TranscribeSpec(facts=facts, rates=default_rates(0.0),
                       seed=5))
    notes = pd.DataFrame(noted)["clinical_note"]
    absent = notes[~keep]
    check("a note on a row where nobody recorded it does NOT "
          "mention it - rendering an empty measurement produced "
          "'oxygen saturation:  %.'",
          not absent.str.contains("oxygen saturation").any())
    check("...while a row that HAS it does",
          notes[keep].str.contains("oxygen saturation").all())
    empties = [e for L in led for e in L
               if e.get("asserted_value") is not None
               and str(e["asserted_value"]).strip() == ""]
    check("and no ledger entry asserts a value of nothing, which "
          "would grade an extractor against text that says "
          "nothing", not empties)

    # the real extract's own column names, pinned so the finding
    # that cost a real run cannot regress
    for col, want in (("temperature_oral", "oral temperature"),
                      ("heart_rate_monitored", "heart rate")):
        check("the extract's own column {!r} is labelled - the "
              "first real run reported it as not-charted because "
              "this named the generic form".format(col),
              label_for_column(col) == want)


def cli_checks():
    import json
    import subprocess           # noqa: F401
    import tempfile
    from synthkit.transcribe import CORRUPTIONS

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        run = td / "run"
        run.mkdir()
        df = cohort()
        df["visit_type"] = ["INPATIENT" if i % 5 == 0
                            else "OUTPATIENT"
                            for i in range(len(df))]
        df.to_csv(run / "generated.csv", index=False)
        desc = td / "d.json"
        json.dump({c: "condition " + c for c in "BCDEFG"},
                  open(desc, "w"))

        # --- the refusals, asserted by EXIT CODE rather than by
        # any wording, so a reworded message cannot pass for a
        # working guard
        rc, _ = run_cli("-o", td / "x1")
        check("neither --run nor --csv is refused", rc != 0)
        empty = td / "empty"
        empty.mkdir()
        rc, _ = run_cli("--run", empty, "-o", td / "x2")
        check("a run directory with no generated.csv is refused",
              rc != 0)
        rc, _ = run_cli("--run", run, "-o", td / "x3",
                        "--visit-type", "NOSUCHTYPE")
        check("a scope that matches no row is refused rather than "
              "writing an empty notes file", rc != 0)
        check("...and nothing was written for it",
              not (td / "x3" / "notes.csv").exists())

        # --- the working path
        out = td / "out"
        rc, log = run_cli("--run", run, "-o", out,
                          "--descriptions", desc,
                          "--visit-type", "INPATIENT",
                          "--rate", "0.4")
        check("the command succeeds on a generated cohort", rc == 0)
        check("it writes notes.csv",
              (out / "notes.csv").exists())
        check("it writes ledger.json",
              (out / "ledger.json").exists())

        notes = pd.read_csv(out / "notes.csv")
        key = json.load(open(out / "ledger.json"))

        check("the scope filter reached the written file - {} "
              "rows, all INPATIENT".format(len(notes)),
              len(notes) > 10
              and set(notes["visit_type"]) == {"INPATIENT"})
        check("the note column carries prose",
              notes["clinical_note"].str.len().min() > 30)
        check("no derived indicator reached the written file",
              not any("__has__" in c for c in notes.columns))

        # --- THE ANSWER KEY DOES NOT TRAVEL. notes.csv is the
        # thing a recipient receives, so nothing in it may say
        # which corruption was applied or what the truth was.
        blob = (" ".join(map(str, notes.columns)) + " "
                + " ".join(notes["clinical_note"].astype(str)))
        leaked = [c for c in CORRUPTIONS if c in blob]
        check("notes.csv names no corruption, so the file that "
              "travels is not also the answer: {}".format(
                  leaked or "none"),
              not leaked)
        check("the ledger holds one entry list per row",
              isinstance(key.get("ledger"), list)
              and len(key["ledger"]) == len(notes))
        check("the ledger records the k it screened at",
              key.get("k") == 10)

        applied = key.get("corruptions_applied", {})
        check("every corruption type is applied at least once "
              "through the real CLI, so the grader has no failure "
              "type it cannot report on: {} of {}".format(
                  len(applied), len(CORRUPTIONS)),
              len(applied) == len(CORRUPTIONS))
        check("...and the run SAYS which were never applied, "
              "rather than leaving a zero to be noticed",
              "NEVER APPLIED" in log or not key.get(
                  "corruptions_never_applied"))

        # --- rate zero is clean prose, which is the control: it
        # proves the corruptions above come from the dial and not
        # from the renderer
        out0 = td / "out0"
        rc, _ = run_cli("--run", run, "-o", out0, "--rate", "0")
        key0 = json.load(open(out0 / "ledger.json"))
        check("at --rate 0 no documentation problem is planted, "
              "so the taxonomy above is the dial's doing",
              rc == 0 and not key0.get("corruptions_applied"))


def main():
    df = cohort()
    f2, facts, plan = build_facts(df, group_by="person_id", k=10,
                                  tokens_per_set=4,
                                  descriptions=DESC)

    # --- the fixture must contain the thing the checks are about
    sets_f = [f for f in facts if f.kind in ("condition", "med")]
    meas_f = [f for f in facts if f.kind == "measurement"]
    check("the fixture yields SET facts, without which every "
          "check below passes on an empty list: {}".format(
              len(sets_f)),
          len(sets_f) >= 6)
    check("...and measurement facts: {}".format(
        [f.label for f in meas_f]),
          len(meas_f) >= 2)

    # --- a note is a release, so the k screen governs it
    rare_rows = int(df["conditions"].str.contains(
        RARE, regex=False).sum())
    rare_pats = int(df.loc[df["conditions"].str.contains(
        RARE, regex=False), "person_id"].nunique())
    other_max = max(
        int(df["conditions"].str.contains(c, regex=False).sum())
        for c in ["ICD10-CM!" + x for x in "BCDEFG"])
    check("the fixture's rare token is the most FREQUENT of them "
          "all ({} rows against {}) while held by only {} "
          "patients, so neither order nor frequency can be what "
          "excludes it".format(rare_rows, other_max, rare_pats),
          rare_rows > other_max and rare_pats == 2)
    check("a token held by fewer than k PATIENTS never becomes a "
          "fact - a note that mentions what the file withholds "
          "has undone the screen",
          not any("AAA" in f.column for f in facts))

    # --- the vocabulary contract, asserted against a LIVE call.
    # Reading `sep` and `values` here returned no facts at all and
    # the whole module was inert; the field names that SOUND right
    # are not the contract.
    raw = df["conditions"].astype(str)
    vocab = S.vocabulary(raw, df["person_id"], k=10)
    check("sets.vocabulary names the separator `separator`, so a "
          "rename breaks this rather than a run on the machine "
          "that holds the data",
          isinstance(vocab, dict) and "separator" in vocab)
    check("...and the token list `tokens`, each entry carrying "
          "`value`",
          isinstance(vocab.get("tokens"), list)
          and bool(vocab["tokens"])
          and "value" in vocab["tokens"][0])

    # --- a charted measurement is a DECLARED thing
    check("only declared charted measures become measurement "
          "facts - taking every labelled numeric put 'span days "
          "recorded at 3.0' in a vitals line",
          all(f.column in NOTE_MEASURES for f in meas_f))
    check("and a numeric nobody charts is REPORTED rather than "
          "silently dropped: {}".format(plan["not_charted"]),
          "span_days" in plan["not_charted"]
          and "visit_number" in plan["not_charted"])

    # --- labels say what a column means, never what it looks like
    check("a token's description is used when supplied",
          label_for_token("ICD10-CM!A", DESC)
          == "type 2 diabetes mellitus")
    check("...and the bare code when it is not, rather than prose "
          "invented from a code",
          label_for_token("ICD10-CM!Z99.9", None) == "Z99.9")
    check("a code-shaped column name earns no invented phrasing",
          label_for_column("787_2") is None)
    check("...while a word-shaped one does",
          label_for_column("respiratory_rate") == "respiratory rate")

    # --- the taxonomy must be reachable
    assigned = set()
    for f in facts:
        assigned.update(f.corruptions)
    check("every corruption the renderer applies is assigned to "
          "some fact, or the grader has a failure type it can "
          "never report on: {} of {}".format(
              len(assigned), len(CORRUPTIONS)),
          len(assigned) >= 8)

    # --- run the chain
    spec = TranscribeSpec(facts=facts, rates=default_rates(0.0),
                          seed=5)
    noted, ledgers = transcribe(f2.to_dict("records"), spec)
    final = reconcile(pd.DataFrame(noted), plan)

    check("no derived indicator survives into the written frame - "
          "the recipient does not receive search scaffolding",
          not any(S.is_scaffolding(c) for c in final.columns)
          and not any("__has__" in c for c in final.columns))
    check("the note column is written",
          "clinical_note" in final.columns
          and final["clinical_note"].str.len().min() > 20)
    check("every column the cohort arrived with survives",
          all(c in final.columns for c in df.columns))

    # --- UNEXPANDED tokens must survive the rebuild
    kept = sum(1 for v in final["conditions"] if "AAA" in str(v))
    orig = sum(1 for v in df["conditions"] if "AAA" in str(v))
    check("a token that became no fact still survives reconcile - "
          "rebuilding the set from the indicators alone would "
          "delete every token nobody expanded ({} of {})".format(
              kept, orig),
          kept == orig and orig > 0)

    # --- THE LOAD-BEARING PAIR: an override reaches the SET
    df2 = cohort()
    g2, facts2, plan2 = build_facts(df2, group_by="person_id",
                                    k=10, tokens_per_set=4,
                                    descriptions=DESC)
    named = [f for f in facts2 if f.kind == "condition"]
    named[0].placement = "both_disagree"
    named[1].placement = "text_only"
    toks = plan2["sets"]["conditions"]["tokens"]
    inds = plan2["sets"]["conditions"]["indicators"]
    dis = [t for t, n in zip(toks, inds) if n == named[0].column][0]
    only = [t for t, n in zip(toks, inds) if n == named[1].column][0]

    before = df2["conditions"].tolist()
    noted2, _ = transcribe(g2.to_dict("records"),
                           TranscribeSpec(facts=facts2,
                                          rates=default_rates(0.0),
                                          seed=5))
    after = reconcile(pd.DataFrame(noted2), plan2)["conditions"]\
        .tolist()

    flipped = sum(1 for b, a in zip(before, after)
                  if (dis in b) != (dis in a))
    check("a both_disagree fact flips the SET STRING the recipient "
          "reads, not merely a derived indicator - {} rows".format(
              flipped),
          flipped > 20)
    still = sum(1 for a in after if only in a)
    check("a text_only fact is removed from the set entirely, so "
          "it can only be found by reading ({} left)".format(still),
          still == 0
          and sum(1 for b in before if only in b) > 20)

    # --- the ledger is the answer key, so it must match the prose
    df3 = cohort(seed=11)
    g3, facts3, plan3 = build_facts(df3, group_by="person_id",
                                    k=10, tokens_per_set=4,
                                    descriptions=DESC)
    rates = {c: 0.0 for c in CORRUPTIONS}
    rates["negation_simple"] = 1.0
    noted3, led3 = transcribe(
        g3.to_dict("records"),
        TranscribeSpec(facts=facts3, rates=rates, seed=5))
    neg = [(i, e) for i, L in enumerate(led3) for e in L
           if not e["asserts_present"]]
    check("a negation is ledgered as asserting ABSENCE, which is "
          "what makes a keyword matcher score badly on it: {} "
          "entries".format(len(neg)),
          len(neg) > 20)
    i0, e0 = neg[0]
    txt = pd.DataFrame(noted3)["clinical_note"].iloc[i0].lower()
    check("...and the prose for it really does deny the finding",
          e0["label"].split()[0].lower() in txt
          and any(w in txt for w in ("no ", "denies", "negative",
                                     "without", "ruled out",
                                     "not ")))

    sparse_checks()
    cli_checks()

    print()
    if FAIL:
        print("{} of {} checks failed.".format(FAIL, PASS + FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
