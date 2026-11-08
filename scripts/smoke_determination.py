"""Checks for the Expert Determination evidence package (ED-1).

THE LOAD-BEARING CHECK IS ON THE FIELD NAMES, ASSERTED AGAINST
THE BATTERIES' LIVE OUTPUT. The first cut of `assess` read
`arms`, `membership` and `attribute` out of one invented
`attack.json`. The real batteries write TWO files with different
shapes, so every element would have read OWED on a run that had
the evidence - and the package would have told an expert that
measurements this project has published for months did not
exist. Seventh instance of the wrong-key fault here, and this
suite applies the prescribed fix: call the batteries and assert
the names, never remember them.

AND ON `OWED` BEING A FIRST-CLASS ANSWER. A package that looks
complete and is not is worse than no package, because an expert
signs against evidence that does not exist. Three ways that goes
wrong are pinned: one adversary is not a threat model, one seed
is a draw rather than a measurement, and an excess with no
positive control is the check-that-cannot-fail shape.

The tool must also never call itself a determination. That is
the same discipline as refusing to describe an output as
differentially private when only the weights are accounted, and
a check asserts the refusal travels in the data rather than in
one renderer's prose.
"""

import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from synthkit import determination as DET            # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def full_artifacts(td):
    """A run directory plus battery outputs, all present."""
    json.dump({"summary": {"columns": 12}, "nn_ratio": 1.15},
              io.open(os.path.join(td, "fidelity.json"), "w"))
    json.dump({"marginals": {"x": {}},
               "bounds_are_k_anonymous": True},
              io.open(os.path.join(td, "blueprint.json"), "w"))
    json.dump({"rows": 100},
              io.open(os.path.join(td, "provenance.json"), "w"))
    mem = {"seeds": [11, 23, 37], "worst_auc_max": 0.551,
           "verdict": "PASS",
           "runs": [{"nearest_neighbor": {"auc": 0.514,
                                          "attacker_sees": "rows"},
                     "likelihood": {"auc": 0.479,
                                    "attacker_sees": "weights"},
                     "worst_auc": 0.514, "verdict": "PASS"}]}
    att = {"seeds": [{"excess_over_control": -0.009}],
           "positive_control": [{"excess_over_control": 0.145}]}
    # AND THE LINKAGE ARM. Adding an eleventh element correctly
    # broke the three checks that build a COMPLETE package -
    # they supplied no linkage evidence, so `ready` could never
    # be true again. That is the checks demanding evidence for a
    # new element rather than a regression.
    lkg = {"source": "fixture.csv", "control_caught": True,
           "verdict": "CLEAN",
           "honest": {"excess_in_sd": 0.006},
           "control": {"excess_in_sd": 0.071},
           "share_of_control": 0.08}
    scope = ("SCOPE. One cohort of 800 patients from one extract "
             "at k=10, output-only release. The determination does "
             "not extend to a different cohort, a different k, or "
             "a release of the trained network. Residual risk: the "
             "effect curves, interaction surfaces and dynamics "
             "have not been audited by the attack battery, and "
             "k-anonymity on what is published is not differential "
             "privacy.")
    return mem, att, scope, lkg


def main():
    # ---- 1. THE FIELD NAMES, AGAINST A LIVE CALL
    import membership_new_path as MN
    from synthkit.attack import attribute_disclosure
    _live = MN.run_once(60, 4, 11, 10, sets=0)
    check("the MEMBERSHIP field names come from a LIVE battery "
          "run, not from memory: run_once returns {} - and "
          "`assess` reads nearest_neighbor.auc, likelihood.auc "
          "and worst_auc out of exactly those"
          .format(sorted(k for k in _live
                         if k in ("nearest_neighbor",
                                  "likelihood", "worst_auc",
                                  "verdict"))),
          "nearest_neighbor" in _live and "likelihood" in _live
          and "auc" in _live["nearest_neighbor"]
          and "auc" in _live["likelihood"]
          and _live.get("worst_auc") is not None
          and _live.get("verdict"))

    def _rows(n, seed):
        r = np.random.default_rng(seed)
        return [{"sex": int(r.integers(0, 2)),
                 "site": int(r.integers(0, 3)),
                 "age": float(r.normal(60, 12)),
                 "hiv": int(r.integers(0, 2))} for _ in range(n)]
    _ad = attribute_disclosure("hiv", ["sex", "site", "age"],
                               _rows(400, 1), _rows(400, 2),
                               _rows(400, 3), seed=0)
    check("...and the ATTRIBUTE field names likewise - a live "
          "call returns excess_over_control and control_accuracy "
          "among {} keys, and the package's status turns on the "
          "CONTROL being present rather than on the excess "
          "alone".format(len(_ad)),
          "excess_over_control" in _ad
          and "control_accuracy" in _ad
          and "synthetic_accuracy" in _ad)

    # ---- 2. BOTH BRANCHES ARE REACHABLE
    with tempfile.TemporaryDirectory() as td:
        mem, att, scope, lkg = full_artifacts(td)
        _bare = DET.assess(
            fidelity=json.load(io.open(os.path.join(
                td, "fidelity.json"))),
            blueprint=json.load(io.open(os.path.join(
                td, "blueprint.json"))),
            provenance=json.load(io.open(os.path.join(
                td, "provenance.json"))))
        _full = DET.assess(
            fidelity=json.load(io.open(os.path.join(
                td, "fidelity.json"))),
            blueprint=json.load(io.open(os.path.join(
                td, "blueprint.json"))),
            provenance=json.load(io.open(os.path.join(
                td, "provenance.json"))),
            membership=mem, attribute=att, linkage=lkg,
            scrub_inventory=[{"category": "names"}],
            scope=scope)
    check("a run WITHOUT the batteries reads {} of {} with {} "
          "OWED, and one WITH them reads {} of {} and READY - "
          "both branches reachable, which the first cut was not: "
          "`residual` was unconditionally false, so `ready` "
          "could never be true and the every-element branch was "
          "dead code inside a tool built to catch exactly that"
          .format(_bare["elements_present"],
                  _bare["elements_total"],
                  len(_bare["elements_owed"]),
                  _full["elements_present"],
                  _full["elements_total"]),
          _bare["ready"] is False
          and len(_bare["elements_owed"]) >= 4
          and _full["ready"] is True
          and not _full["elements_owed"])

    # ---- 3. THE COUNTS ARE DERIVED FROM THE ROWS
    check("the totals are DERIVED from the element rows rather "
          "than written beside them - {} present and {} owed "
          "against {} rows, and the two add up"
          .format(_full["elements_present"],
                  len(_full["elements_owed"]),
                  len(_full["elements"])),
          _full["elements_present"]
          + len(_full["elements_owed"])
          == len(_full["elements"])
          == _full["elements_total"]
          and _bare["elements_present"]
          + len(_bare["elements_owed"])
          == _bare["elements_total"])

    # ---- 4. THE THREE WAYS `OWED` MUST STILL FIRE
    _one_arm = dict(mem)
    _one_arm["runs"] = [{"nearest_neighbor": {"auc": 0.51},
                         "worst_auc": 0.51, "verdict": "PASS"}]
    _r1 = DET.assess(membership=_one_arm)
    check("ONE ADVERSARY IS NOT A THREAT MODEL: a membership "
          "summary carrying only the output-surface arm reads "
          "threat_model OWED. The output surface and the weights "
          "surface are different recipients and this project's "
          "recorded numbers differ sharply between them, so one "
          "arm answers a question nobody asked",
          "threat_model" in _r1["elements_owed"])
    _one_seed = dict(mem, seeds=[11])
    _r2 = DET.assess(membership=_one_seed)
    check("...ONE SEED IS A DRAW, NOT A MEASUREMENT: a summary "
          "with a single seed reads membership OWED. The battery "
          "itself refuses fewer than two, so a summary with one "
          "did not come from it",
          "membership" in _r2["elements_owed"])
    _no_ctrl = {"seeds": [{"excess_over_control": -0.009}],
                "positive_control": []}
    _r3 = DET.assess(attribute=_no_ctrl)
    check("...and AN EXCESS WITH NO POSITIVE CONTROL reads "
          "attribute OWED - a clean excess whose leaky-generator "
          "arm never ran is the check-that-cannot-fail shape, "
          "and this project has recorded a reassuring number "
          "that was worth nothing for exactly that reason",
          "attribute" in _r3["elements_owed"])
    _r4 = DET.assess(scope="one cohort, one k")
    check("...and A ONE-LINE SCOPE IS NOT A SCOPE: residual risk "
          "stays OWED until a written statement exists, because "
          "it is the one element a RUN cannot supply - a scope "
          "nobody wrote is not a scope, and deriving one would "
          "be inventing the bound on somebody else's "
          "determination",
          "residual" in _r4["elements_owed"]
          and _r4["scope_chars"] < 200)

    # ---- 5. IT MUST NEVER CALL ITSELF A DETERMINATION
    _txt = "\n".join(DET.plain(_full))
    check("the package refuses to call itself a determination, "
          "in the DATA rather than one renderer's prose, so no "
          "renderer can drop it - and it says no source record "
          "travels with it",
          len(DET.NOT_A_DETERMINATION) >= 3
          and "EVIDENCE PACKAGE, not a determination"
          in " ".join(DET.NOT_A_DETERMINATION)
          and "No source record travels"
          in " ".join(DET.NOT_A_DETERMINATION)
          and all(line in _txt
                  for line in DET.NOT_A_DETERMINATION))
    check("...and a READY package says READINESS rather than "
          "anything stronger - the determination is still a "
          "qualified person's to make",
          "READINESS, not a determination" in _txt
          and "qualified person" in _txt)
    check("...and it is organised around the STATUTE's own two "
          "requirements, quoted, so a reader is not mapping our "
          "headings onto theirs",
          "164.514(b)(1)" in DET.STATUTE
          and "documents the methods and results" in DET.STATUTE
          and DET.STATUTE in _txt)

    # ---- 6. EVERY ELEMENT SAYS WHY IT IS ASKED
    check("every one of the {} elements names the EVIDENCE that "
          "answers it and WHY it is asked - a status a reader "
          "cannot go and check is a status taken on trust"
          .format(len(DET.ELEMENTS)),
          len(DET.ELEMENTS) >= 9
          and all(len(e["evidence"]) > 40 and len(e["why"]) > 60
                  and e["need"] for e in DET.ELEMENTS)
          and len(set(e["key"] for e in DET.ELEMENTS))
          == len(DET.ELEMENTS))

    # ---- 7. THE CLI, THROUGH THE PROCESS BOUNDARY
    with tempfile.TemporaryDirectory() as td:
        mem, att, scope, lkg = full_artifacts(td)
        mp = os.path.join(td, "mem.json")
        ap = os.path.join(td, "att.json")
        lp = os.path.join(td, "lkg.json")
        sp = os.path.join(td, "scope.txt")
        json.dump(mem, io.open(mp, "w"))
        json.dump(att, io.open(ap, "w"))
        json.dump(lkg, io.open(lp, "w"))
        io.open(sp, "w").write(scope)
        _bad = subprocess.run(
            [sys.executable, "-m", "synthkit.cli",
             "determination", td],
            capture_output=True, text=True, cwd=str(ROOT))
        _good = subprocess.run(
            [sys.executable, "-m", "synthkit.cli",
             "determination", td, "--membership", mp,
             "--attribute", ap, "--linkage", lp,
             "--scope", sp, "--scrubbed"],
            capture_output=True, text=True, cwd=str(ROOT))
        _missing = subprocess.run(
            [sys.executable, "-m", "synthkit.cli",
             "determination",
             os.path.join(td, "nope")],
            capture_output=True, text=True, cwd=str(ROOT))
        _wrote = os.path.exists(os.path.join(
            td, "determination.json"))
    check("the CLI exits 3 on an INCOMPLETE package and 0 on a "
          "complete one ({} and {}) - the same reasoning as "
          "`scrub --apply` refusing while a fresh detect still "
          "finds something: a document that says it is ready "
          "while half the work is outstanding is the worst "
          "outcome available"
          .format(_bad.returncode, _good.returncode),
          _bad.returncode == 3 and _good.returncode == 0)
    check("...and it REFUSES a missing run directory by name "
          "with exit 2 rather than a traceback",
          _missing.returncode == 2
          and "STOPPED" in _missing.stderr
          and "Traceback" not in _missing.stderr)
    check("...and it writes determination.json beside the run, "
          "so the package is an artifact rather than a terminal "
          "scroll", _wrote)
    check("...and the incomplete run NAMES what is owed in its "
          "output rather than only counting it, because a count "
          "a reader cannot act on is the fault this project "
          "records about every totals line",
          "OWED:" in _bad.stdout
          and "membership" in _bad.stdout
          and "signing against nothing" in _bad.stdout)

    # ---- EVIDENCE MEASURED ON ANOTHER DATASET (ED-2)
    #
    # The attack batteries default to their own cohort FIXTURE,
    # which this project's own posture calls "a floor, not a
    # certificate, and not a release decision for a different
    # cohort". A determination is scoped to ONE release, so
    # citing a fixture number for it is the strongest form of
    # the measured-at-one-shape fault: not a stale number, but a
    # number about somebody else.
    _fix_mem = {"source": "cohort fixture", "seeds": [1, 2],
                "worst_auc_max": 0.53, "verdict": "PASS",
                "runs": [{"nearest_neighbor": {"auc": 0.51},
                          "likelihood": {"auc": 0.49}}]}
    _r_no_src = DET.assess(membership=_fix_mem)
    _r_real = DET.assess(membership=_fix_mem,
                         source_name="tidy_visits.csv")
    _r_same = DET.assess(
        membership=dict(_fix_mem, source="tidy_visits.csv"),
        source_name="tidy_visits.csv")
    _st = dict((e["key"], e["status"]) for e in
               _r_real["elements"])
    check("evidence measured on ANOTHER DATASET reads FIXTURE "
          "rather than present: a membership summary whose "
          "source is the cohort fixture, against a run whose "
          "source is the extract, marks membership {} and "
          "threat_model {} - and both when the sources MATCH"
          .format(_st["membership"], _st["threat_model"]),
          _st["membership"] == "FIXTURE"
          and _st["threat_model"] == "FIXTURE"
          and all(e["status"] == "present"
                  for e in _r_same["elements"]
                  if e["key"] in ("membership", "threat_model")))
    check("...and it does not count toward readiness ({} "
          "present with the fixture evidence against {} when it "
          "matches the run) - a determination is scoped to one "
          "release"
          .format(_r_real["elements_present"],
                  _r_same["elements_present"]),
          _r_real["elements_present"]
          < _r_same["elements_present"]
          and _r_real["ready"] is False)
    check("...and FIXTURE is NOT also reported as OWED - the "
          "first cut listed such an element in both, so the "
          "output said FIXTURE on one line and OWED on another "
          "about the same thing, which is this project's own "
          "two-numbers-disagreeing alarm inside a governance "
          "document",
          "membership" not in _r_real["elements_owed"]
          and "membership" in _r_real["elements_elsewhere"]
          and not (set(_r_real["elements_owed"])
                   & set(_r_real["elements_elsewhere"])))
    check("...and with NO source on the run it does not guess: "
          "an unknown cohort cannot be compared against, so the "
          "evidence reads present rather than being flagged on "
          "a hunch",
          all(e["status"] == "present"
              for e in _r_no_src["elements"]
              if e["key"] in ("membership", "threat_model")))
    # TWO CAUSES, TWO LABELS. Different dataset and different
    # caps take DIFFERENT fixes - re-run on this cohort versus
    # re-run at these settings - so collapsing them into one
    # word would be the blocked-versus-unchanged fault.
    _same = {"source": "tidy_visits.csv", "seeds": [0, 1, 2],
             "worst_auc_max": 0.55, "verdict": "PASS",
             "token_cap": 60, "level_cap": 60,
             "runs": [{"nearest_neighbor": {"auc": 0.51},
                       "likelihood": {"auc": 0.49}}]}
    _ok = DET.assess(membership=_same,
                     source_name="tidy_visits.csv",
                     run_config={"token_cap": 60,
                                 "level_cap": 60})
    _cfg = DET.assess(membership=_same,
                      source_name="tidy_visits.csv",
                      run_config={"token_cap": 800,
                                  "level_cap": 60})
    _stc = dict((e["key"], e["status"]) for e in
                _cfg["elements"])
    check("evidence from the same data under DIFFERENT CAPS "
          "reads CONFIG, not FIXTURE and not present: a cap-60 "
          "attack against a cap-800 run marks membership {} "
          "where matching caps read {}. A larger token cap "
          "publishes more vocabulary about members, so the "
          "smaller-cap attack UNDERSTATES the risk - the wrong "
          "direction for a privacy number to be wrong in, and "
          "invisible without comparing"
          .format(_stc["membership"],
                  dict((e["key"], e["status"])
                       for e in _ok["elements"])["membership"]),
          _stc["membership"] == "CONFIG"
          and _ok["ready"] is False  # scope still owed
          and "membership" not in _ok["elements_elsewhere"]
          and "membership" in _cfg["elements_elsewhere"])
    check("...and the two causes keep SEPARATE words, because "
          "they take different fixes - re-run on this cohort "
          "versus re-run at these settings - and the text says "
          "which is which",
          _st["membership"] == "FIXTURE"
          and _stc["membership"] == "CONFIG"
          and "re-run it on this cohort" in "\n".join(
              DET.plain(_cfg))
          and "re-run it at these settings" in "\n".join(
              DET.plain(_cfg)))
    check("...and the attack RECORDS the caps it measured, or "
          "the comparison has nothing to read",
          '"token_cap": int(MAX_TOKENS),' in io.open(
              str(ROOT / "scripts" / "latent_challenger.py"),
              encoding="utf-8").read())

    _txt2 = "\n".join(DET.plain(_r_real))
    check("...and the text NAMES the elsewhere elements and says "
          "why that matters, rather than only counting them",
          "OUT OF SCOPE" in _txt2
          and "membership" in _txt2.split(
              "OUT OF SCOPE")[1][:200]
          and "scoped to one" in _txt2)

    # ---- THE ATTACK NOW WRITES WHAT THE PACKAGE READS
    _lcsrc2 = io.open(str(ROOT / "scripts"
                          / "latent_challenger.py"),
                      encoding="utf-8").read()
    check("the real-extract attack WRITES its summary where the "
          "package can read it - it printed only, so the one "
          "run with the strongest membership evidence would have "
          "read OWED",
          '"worst_auc_max"' in _lcsrc2
          and '"runs": runs,' in _lcsrc2
          and "--out" in _lcsrc2)
    check("...in the SHAPE membership_new_path already writes, "
          "deliberately - a third shape for one measurement is "
          "how a reader comes to need three parsers, and this "
          "module already lost a day to field names that were "
          "never written",
          all(f in _lcsrc2 for f in ('"worst_auc_mean"',
                                     '"verdict"', '"seeds"',
                                     '"reading"')))
    check("...and it records WHICH ARM the positive control "
          "controls, because feeding the members back as "
          "synthetic can only move the adversary that READS "
          "synthetic rows - the reconstruction adversary is "
          "blind to the output and cannot be controlled that way",
          '"arm": "nearest_neighbor (output surface)"' in _lcsrc2
          and "blind to the output" in _lcsrc2)

    # ---- THE SCOPE TEMPLATE
    #
    # Residual risk is the one element a RUN cannot supply, so
    # the operator has to write it - and a blank page is why it
    # stays OWED. The template fills in the MEASURED facts and
    # marks every JUDGMENT, so the person making the
    # determination starts from the measurements rather than
    # from nothing.
    #
    # THE LOAD-BEARING CHECK IS THAT IT CANNOT BE MISTAKEN FOR A
    # DETERMINATION, and that it does not pass `assess` by
    # length alone - the scope gate is a 200-character floor,
    # and a template full of unreplaced brackets would clear it
    # while saying nothing about this release.
    _tpl_path = ROOT / "docs" / "DETERMINATION_SCOPE.md"
    _tpl = io.open(str(_tpl_path), encoding="utf-8").read()
    check("the scope TEMPLATE exists and says in its own first "
          "lines that it is a template and NOT a determination "
          "- a scope statement that reads as authoritative is "
          "the one document here that could be misused",
          _tpl_path.exists()
          and "not a determination" in _tpl.split("\n\n")[0]
          + _tpl[:900]
          and "template" in _tpl[:400].lower())
    check("...and every JUDGMENT in it is marked for review ({} "
          "markers), because the measured facts and the "
          "conclusions drawn from them are different kinds of "
          "statement and a reader must be able to tell which is "
          "which".format(_tpl.count("[REVIEW]")),
          _tpl.count("[REVIEW]") >= 5
          and "[REVIEW] — every line here is a judgment" in _tpl)
    check("...and it carries the controls beside the numbers "
          "rather than the numbers alone, since a control that "
          "cannot catch a verbatim republish is a check that "
          "cannot fail",
          "control" in _tpl.lower()
          and "republish" in _tpl
          and "CAUGHT" in _tpl)
    check("...and it states the RETAINED information, because "
          "over-suppression is the trivial way to make risk "
          "very small and a determination without it has not "
          "described a trade",
          "Retained information" in _tpl
          and "Over-suppression" in _tpl)
    check("...and it names the FIXTURE and CONFIG scope rules, "
          "so a reader filling it in knows evidence from "
          "another dataset or other caps does not count",
          "FIXTURE" in _tpl and "CONFIG" in _tpl
          and "measure different things" in _tpl)
    # AND IT MUST NOT PASS THE SCOPE GATE WHILE STILL A
    # TEMPLATE. The gate is a 200-character floor, which an
    # unfilled template clears on length - so the point of this
    # check is that `assess` is NOT the thing stopping somebody
    # shipping a template as a scope. The brackets are, and a
    # reader has to be told that.
    _as_scope = DET.assess(scope=_tpl)
    check("the scope gate is a LENGTH floor, so the unfilled "
          "template clears it ({} chars) - which is stated here "
          "rather than pretended otherwise: `assess` cannot "
          "tell a filled scope from an unfilled one, the "
          "unreplaced [REVIEW] and [N] markers are what a "
          "reader checks, and the template says so"
          .format(_as_scope["scope_chars"]),
          "residual" not in _as_scope["elements_owed"]
          and "[N]" in _tpl and "[REVIEW]" in _tpl
          and "Replace the bracketed items" in _tpl)

    # ---- THE ATTRIBUTE CONTROL MUST HAVE FIRED (ED-3)
    #
    # The first cut required a non-empty `positive_control`
    # list, which a control that ran and caught NOTHING
    # satisfies. On a real extract the battery printed "THE
    # POSITIVE CONTROL DID NOT FIRE" at +0.019 excess and this
    # element would still have read present - the
    # check-that-cannot-fail shape, which the linkage element
    # already refuses via `control_caught`.
    _ab = {"seeds": [{"excess_over_control": 0.004}],
           "positive_control": [{"excess_over_control": 0.019}]}
    _old = DET.assess(attribute=dict(_ab))
    _fired = DET.assess(attribute=dict(_ab, control_fired=True))
    _dead = DET.assess(attribute=dict(_ab, control_fired=False))
    check("an attribute excess whose POSITIVE CONTROL DID NOT "
          "FIRE reads OWED, not present - a control that cannot "
          "catch a verbatim republish makes the clean number "
          "beside it worth nothing, which this project has "
          "recorded on this exact surface",
          "attribute" in _dead["elements_owed"]
          and "attribute" not in _fired["elements_owed"])
    check("...and a record PREDATING the flag still counts, "
          "because absent is not the same fact as did-not-fire "
          "- the same distinction drawn for `id_column` and for "
          "empty sets",
          "attribute" not in _old["elements_owed"])
    _adsrc = io.open(str(ROOT / "scripts"
                         / "attribute_disclosure.py"),
                     encoding="utf-8").read()
    check("...and the BAR lives at the battery's own decision "
          "site and travels as a flag, rather than being "
          "re-derived here - two halves deciding one threshold "
          "is how a privacy number came to be applied with one "
          "tail bound and reported with another",
          "control_bar = 0.03" in _adsrc
          and '"control_fired": bool(control_fired),' in _adsrc
          and "control_bar" not in io.open(
              str(ROOT / "synthkit" / "determination.py"),
              encoding="utf-8").read())

    # ---- AND THE BATTERY HAS A REAL-EXTRACT PATH AT ALL
    #
    # It only ever ran on its own cohort fixture, so
    # `determination` correctly marked its evidence FIXTURE and
    # a complete run of record was impossible.
    check("the attribute battery takes a REAL extract, so its "
          "evidence can be about the cohort being released "
          "rather than about somebody else's data",
          "--csv" in _adsrc and "--sensitive" in _adsrc
          and "--quasi" in _adsrc
          and 'src=src' in _adsrc)
    # AND THAT REFUSAL IS ASSERTED AS A DECISION, THROUGH THE
    # PROCESS BOUNDARY. A text check on the message passed while
    # a mutation turned the refusal into a note, because the
    # asserted words lived in the same string either way -
    # asserting the message instead of the decision, which this
    # project records repeatedly.
    with tempfile.TemporaryDirectory() as _td2:
        import numpy as _np2
        import pandas as _pd2
        _rr = _np2.random.default_rng(3)
        _n = 600
        _real = _pd2.DataFrame({
            "person_id": ["P{:03d}".format(i // 3)
                          for i in range(_n)],
            "age": _rr.normal(50, 12, _n).round(1),
            # a CATEGORY as a quasi-identifier: `_num` returns
            # None for it, so the adversary gets an empty column
            "site_name": _rr.choice(["north", "south", "east"],
                                    _n),
            "hiv_status": _rr.choice(["positive", "negative"],
                                     _n)})
        _csv = os.path.join(_td2, "real.csv")
        _real.to_csv(_csv, index=False)
        _cat = subprocess.run(
            [sys.executable,
             str(ROOT / "scripts" / "attribute_disclosure.py"),
             "--csv", _csv, "--sensitive", "hiv_status",
             "--quasi", "age,site_name", "--seeds", "1"],
            capture_output=True, text=True, cwd=str(ROOT))
        _nojudge = subprocess.run(
            [sys.executable,
             str(ROOT / "scripts" / "attribute_disclosure.py"),
             "--csv", _csv, "--seeds", "1"],
            capture_output=True, text=True, cwd=str(ROOT))
    check("it REFUSES a quasi-identifier that does not coerce "
          "to numbers, exit {} with no traceback - `_num` "
          "returns None for a category, so naming one hands the "
          "adversary an EMPTY column and the lower accuracy "
          "then reads as PRIVACY. This project records that "
          "fault; the fix is to refuse by name rather than wait "
          "for it".format(_cat.returncode),
          _cat.returncode == 2
          and "site_name" in _cat.stderr
          and "Traceback" not in _cat.stderr)
    check("...and it REFUSES --csv without the two judgments, "
          "exit {}, because which column is sensitive and what "
          "an adversary holds are the operator's calls - "
          "guessing them would measure an attack nobody is "
          "worried about".format(_nojudge.returncode),
          _nojudge.returncode == 2
          and "--sensitive" in _nojudge.stderr
          and "Traceback" not in _nojudge.stderr)
    check("...and it records the SOURCE it measured, so the "
          "package can tell evidence about this release from "
          "evidence about a fixture",
          '"source": (str(a.csv) if a.csv' in _adsrc)

    # ---- THE PATHS, AGAINST ARTIFACTS THE REAL WRITER MADE
    #
    # THIS IS THE CHECK THAT WOULD HAVE CAUGHT THE EIGHTH
    # WRONG-KEY FAULT. `assess` read `fidelity.nn_ratio`,
    # `fidelity.reconstruction_gap` and
    # `blueprint.bounds_are_k_anonymous`; a real run writes
    # `fidelity.reconstruction.ratio` and a whole
    # `blueprint.privacy` block, and no nn_ratio at all. So
    # `memorization` and `aggregates` read OWED on a run that
    # HAD the evidence - and every check above missed it,
    # because they built the artifacts BY HAND with the names I
    # had invented. The fixture agreed with the reader and both
    # disagreed with the only thing that matters.
    #
    # This project records that exact fault for the
    # relationships page, where twelve rows of em-dashes
    # shipped behind forty passing checks. The fix prescribed
    # there is the fix applied here: drive the REAL writer.
    sys.path.insert(0, str(ROOT / "scripts"))
    from latent_challenger import LatentGen, write_run_artifacts
    _rr3 = np.random.default_rng(5)
    _rows3 = []
    for _p in range(90):
        _lvl = float(_rr3.normal(0, 1))
        for _v in range(4):
            _rows3.append({
                "person_id": "P{:03d}".format(_p),
                "sys": round(126 + 12 * _lvl
                             + _rr3.normal(0, 4), 1),
                "dia": round(78 + 7 * _lvl
                             + _rr3.normal(0, 3), 1),
                "grp": "a" if _lvl > 0 else "b"})
    _rdf = pd.DataFrame(_rows3).astype(str)
    with tempfile.TemporaryDirectory() as _td3:
        # THE HOLDOUT SPLIT IS THE ENGINE'S OWN, and the
        # reconstruction measurement comes from the engine's own
        # method - NOT hand-built, which is the whole point of
        # this block. A dict assembled here would carry whatever
        # shape I guessed, which is how the fault being fixed
        # got in.
        _ids3 = sorted(_rdf["person_id"].unique())
        _hold3 = _rdf[_rdf["person_id"].isin(_ids3[-15:])]
        _train3 = _rdf[_rdf["person_id"].isin(_ids3[:-15])]
        _g3 = LatentGen(seed=0, k=5)
        _g3.fit(_train3, "person_id")
        _gen3 = _g3.generate(len(_rdf), seed=0)
        _recon3 = _g3.reconstruction_gap(_hold3)
        _run3 = os.path.join(_td3, "run")
        write_run_artifacts(_rdf, _gen3, "person_id", _run3,
                            source_name="fixture.csv",
                            reconstruction=_recon3,
                            gen_engine=_g3, k=5)
        _rfid = json.load(io.open(os.path.join(
            _run3, "fidelity.json")))
        _rbp = json.load(io.open(os.path.join(
            _run3, "blueprint.json")))
        _rprov = json.load(io.open(os.path.join(
            _run3, "provenance.json")))
    _rreal = DET.assess(fidelity=_rfid, blueprint=_rbp,
                        provenance=_rprov)
    _rst = dict((e["key"], e["status"])
                for e in _rreal["elements"])
    check("MEMORIZATION and AGGREGATES read present from "
          "artifacts the REAL WRITER produced - {} and {} - "
          "which is the check that would have caught the eighth "
          "wrong-key fault here: the paths were invented, the "
          "hand-built fixtures used the same invented names, and "
          "both elements read OWED on a real run that HAD the "
          "evidence"
          .format(_rst["memorization"], _rst["aggregates"]),
          _rst["memorization"] == "present"
          and _rst["aggregates"] == "present")
    check("...and the paths are the writer's own: "
          "fidelity.reconstruction.ratio is {} and "
          "blueprint.privacy carries k={} counted in {!r} - "
          "there is no `nn_ratio` in this artifact at all, that "
          "number belongs to the attack and the challenger"
          .format(round(float((_rfid.get("reconstruction")
                               or {}).get("ratio", 0)), 3),
                  (_rbp.get("privacy") or {}).get("k"),
                  str((_rbp.get("privacy") or {})
                      .get("counted_in"))[:22]),
          (_rfid.get("reconstruction") or {}).get("ratio")
          is not None
          and (_rbp.get("privacy") or {}).get("k")
          and "nn_ratio" not in _rfid)
    check("...and the k rule's record is checked for the PATIENT "
          "wording, not merely for a number - a k counted in "
          "ROWS would let one person seen two hundred times set "
          "a published bound alone, which is the whole reason "
          "the rule is written that way",
          "patient" in str((_rbp.get("privacy") or {})
                           .get("counted_in", "")).lower()
          and "aggregates" in DET.assess(blueprint={
              "columns": {}, "privacy": {
                  "k": 10, "counted_in": "rows",
                  "numeric_bounds": "x"}})["elements_owed"])
    check("...and an OLDER artifact carrying the flag instead "
          "of the block still counts, because absent is not the "
          "same fact as false",
          "aggregates" not in DET.assess(blueprint={
              "columns": {},
              "bounds_are_k_anonymous": True})["elements_owed"])

    print()
    if FAIL:
        print("{} of {} checks failed.".format(FAIL, PASS + FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
