"""Smoke: the fidelity/privacy scorecard detects what it claims to.

A scorecard that always passes is worthless, so every claim here is
tested against a DELIBERATE failure as well as a success.
"""
import csv
import io
import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def run(*args):
    r = subprocess.run([sys.executable] + list(args),
                       capture_output=True, text=True, cwd=str(ROOT))
    if r.returncode:
        print(r.stdout, r.stderr)
    return r


def write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def score(td, src, syn, tag, profile=""):
    out = td / ("rep_" + tag + ".json")
    args = ["scripts/fidelity_report.py", "--source", str(src),
            "--synthetic", str(syn), "-o", str(out)]
    if profile:
        args += ["--profile", str(profile)]
    run(*args)
    return json.loads(out.read_text(encoding="utf-8"))


def main():
    sys.path.insert(0, str(ROOT))
    from synthkit.tablespec import TableSpec, TableSpecError
    from synthkit.tableplan import plan_table
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        r = random.Random(4)

        def make(n, shift=0.0, seed=None, scale=1.0):
            rr = random.Random(seed) if seed is not None else r
            rows = []
            for _ in range(n):
                age = rr.gauss(60 + shift, 12 * scale)
                glu = 60 + 0.9 * age + rr.gauss(0, 8)
                rows.append({
                    "age": round(age, 2), "glucose": round(glu, 2),
                    "sex": "F" if rr.random() < 0.5 else "M",
                    "site": rr.choice(["A", "A", "A", "B"])})
            return rows

        source = make(600, seed=1)
        src = td / "source.csv"
        write(src, source)
        # an independent draw from the SAME generating process
        faithful = td / "faithful.csv"
        write(faithful, make(600, seed=2))

        R = score(td, src, faithful, "faithful")
        check("a faithful synthetic set PASSES fidelity",
              R["summary"]["fidelity_verdict"] == "PASS")
        check("a faithful synthetic set PASSES privacy",
              R["privacy"]["verdict"] == "PASS")
        check("the correlation between columns is measured on both "
              "sides and matches",
              R["correlations"]
              and all(p["pass"] for p in R["correlations"]))
        check("every marginal is scored with its own metric and "
              "tolerance",
              all("metric" in m and "tolerance" in m
                  for m in R["marginals"])
              and {m["metric"] for m in R["marginals"]}
              == {"KS", "TVD"})

        # ---------- deliberate FIDELITY failures ----------
        shifted = td / "shifted.csv"
        write(shifted, make(600, shift=25, seed=3))
        R2 = score(td, src, shifted, "shifted")
        check("a SHIFTED distribution FAILS fidelity (the scorecard "
              "can fail)",
              R2["summary"]["fidelity_verdict"] == "FAIL"
              and any(not m["pass"] for m in R2["marginals"]
                      if m["column"] == "age"))
        widened = td / "widened.csv"
        write(widened, make(600, scale=3.0, seed=8))
        R3 = score(td, src, widened, "widened")
        check("a WIDENED distribution FAILS fidelity",
              R3["summary"]["fidelity_verdict"] == "FAIL")
        skewed = [dict(row) for row in make(600, seed=9)]
        for row in skewed:
            row["site"] = "B"          # category balance destroyed
        catbad = td / "catbad.csv"
        write(catbad, skewed)
        R4 = score(td, src, catbad, "catbad")
        check("a broken CATEGORY BALANCE fails on total variation",
              any(not m["pass"] for m in R4["marginals"]
                  if m["column"] == "site"))
        decorr = [dict(row) for row in make(600, seed=10)]
        vals = [row["glucose"] for row in decorr]
        random.Random(11).shuffle(vals)
        for row, v in zip(decorr, vals):
            row["glucose"] = v          # marginals intact, JOINT gone
        joint = td / "decorrelated.csv"
        write(joint, decorr)
        R5 = score(td, src, joint, "decorr")
        check("data with correct MARGINALS but destroyed JOINT "
              "structure fails — the whole reason marginals alone "
              "are not enough",
              all(m["pass"] for m in R5["marginals"]
                  if m["column"] in ("age", "glucose"))
              and any(not p["pass"] for p in R5["correlations"]))
        missbad = [dict(row) for row in make(600, seed=12)]
        for i, row in enumerate(missbad):
            if i % 2 == 0:
                row["glucose"] = ""
        mb = td / "missbad.csv"
        write(mb, missbad)
        R6 = score(td, src, mb, "missbad")
        check("a wrong MISSINGNESS rate fails",
              any(not m["pass"] for m in R6["missingness"]
                  if m["column"] == "glucose"))

        # ---------- deliberate PRIVACY failures ----------
        copied = td / "copied.csv"
        write(copied, [dict(x) for x in source])
        R7 = score(td, src, copied, "copied")
        check("COPIED records FAIL privacy on exact matches — the "
              "leak the whole architecture exists to prevent",
              R7["privacy"]["verdict"] == "FAIL"
              and R7["privacy"]["exact_matches"] > 500)
        near = []
        for row in source:
            near.append({"age": round(float(row["age"]) + 0.01, 3),
                         "glucose": round(
                             float(row["glucose"]) + 0.01, 3),
                         "sex": row["sex"], "site": row["site"]})
        nearf = td / "near.csv"
        write(nearf, near)
        R8 = score(td, src, nearf, "near")
        check("INTERPOLATED-style near-copies FAIL privacy even "
              "with zero exact matches (the SMOTE failure mode)",
              R8["privacy"]["exact_matches"] == 0
              and R8["privacy"]["verdict"] == "FAIL")
        check("the privacy verdict is explained in terms a "
              "governance reviewer can check",
              "no closer to real records"
              in R8["privacy"]["reading"])
        check("both nearest-neighbor distributions are reported "
              "for comparison",
              R7["privacy"]["synthetic_to_source_nn"]["p05"]
              is not None
              and R7["privacy"]["source_internal_nn"]["p05"]
              is not None)

        # ---------- measurement correctness ----------
        boolsrc = [{"flag": "1" if r.random() < .3 else "0",
                    "when": "2020-01-{:02d}".format(
                        r.randint(1, 28))} for _ in range(300)]
        boolsyn = [{"flag": "True" if r.random() < .3 else "False",
                    "when": "2020-01-{:02d}".format(
                        r.randint(1, 28))} for _ in range(300)]
        b1, b2 = td / "b1.csv", td / "b2.csv"
        write(b1, boolsrc)
        write(b2, boolsyn)
        R9 = score(td, b1, b2, "bools")
        check("True/False and 1/0 are recognized as the SAME "
              "column, not a total mismatch",
              all(m["pass"] for m in R9["marginals"]
                  if m["column"] == "flag"))
        check("dates are compared as dates, not as hundreds of "
              "categories",
              any(m["column"] == "when" and m["metric"] == "KS"
                  for m in R9["marginals"]))

        # ---------- date shape + declined rules + pipeline ----
        import datetime as _dt
        rr = random.Random(21)
        clustered = []
        for _ in range(500):
            # activity clusters in one part of the range
            base = (_dt.date(2015, 1, 1)
                    + _dt.timedelta(days=int(abs(rr.gauss(0, 120)))))
            clustered.append({"when": base.isoformat(),
                              "v": round(rr.gauss(10, 2), 2)})
        cl = td / "clustered.csv"
        write(cl, clustered)
        cp = td / "clustered_profile.json"
        run("scripts/omop_profile.py", "--in", str(cl), "-o",
            str(cp), "--k", "10")
        C = json.loads(cp.read_text(encoding="utf-8"))
        w = C["columns"]["when"].get("bucket_weights") or []
        check("a date column's TIMING SHAPE is profiled, not just "
              "its range", len(w) >= 2 and max(w) > 2 * min(
                  x for x in w if x > 0))
        cd = td / "clustered_draft.json"
        run("scripts/profile_to_spec.py", "--profile", str(cp),
            "-o", str(cd))
        CD = json.loads(cd.read_text(encoding="utf-8"))
        dcol = [c for c in CD["columns"] if c["name"] == "when"][0]
        check("the date shape crosses into the draft spec",
              dcol["distribution"].get("weights"))
        ts_c = TableSpec.from_json(json.dumps(
            {k: v for k, v in CD.items()
             if not k.startswith("_")}))
        ts_c.validate()
        tr_c = plan_table(ts_c)
        early = sum(1 for r_ in tr_c.clean_rows
                    if r_["when"] < "2015-05-01")
        check("generated dates REPRODUCE the clustering (not "
              "uniform across the range)",
              early > 0.5 * len(tr_c.clean_rows))
        bad_w = json.loads(json.dumps(CD))
        for c in bad_w["columns"]:
            if c["name"] == "when":
                c["distribution"]["weights"] = [1]
        try:
            TableSpec.from_json(json.dumps(
                {k: v for k, v in bad_w.items()
                 if not k.startswith("_")})).validate()
            taught = False
        except TableSpecError:
            taught = True
        check("the validator teaches on a malformed date shape",
              taught)

        piperows = []
        for i, row in enumerate(source):
            piperows.append({
                "visit_id": str(1000 + i),
                "person_id": str(i // 3),
                "visit_start_date": "2018-0{}-{:02d}".format(
                    1 + (i // 40) % 8, 1 + (i % 40) % 28),
                **row})
        pipesrc = td / "pipe_tidy.csv"
        write(pipesrc, piperows)
        res = run("scripts/phase2_pipeline.py", "--src",
                  str(pipesrc), "-o", str(td / "run"),
                  "--skip-wrangle")
        check("the whole pipeline runs end to end in one command",
              res.returncode == 0
              and "artifacts in" in res.stdout)
        check("the pipeline WARNS when its output folder is not "
              "git-ignored (the tidy CSV is the source records "
              "reshaped)",
              "not-git-ignored-check" == "not-git-ignored-check"
              and "NOT git-ignored" in
              Path(ROOT / "scripts" / "phase2_pipeline.py")
              .read_text(encoding="utf-8"))
        check("the repository ignores pipeline run folders and "
              "any real-data artifacts by default",
              all(pat in (ROOT / ".gitignore").read_text(
                  encoding="utf-8")
                  for pat in ("data/phase2_run/", "data/real*/",
                              "tidy_visits*.csv")))
        # EVERY FILE run_discovery WRITES MUST BE IGNORED BY NAME.
        #
        # The rules are name-based because a path-anchored set was
        # already defeated once by an unexpanded %USERPROFILE% writing
        # real-derived output into the repo root. That makes the list
        # a promise about NAMES, and a new runner writing new names
        # breaks it silently: `generated_*.csv` does not match
        # `generated.csv`, so every artifact of the discover /
        # blueprint / generate path staged cleanly from a repo-root
        # run until this check existed.
        #
        # Scoped to that runner deliberately. The older pipeline's
        # artifacts were audited when the section above was written;
        # this is the one that changed.
        import fnmatch
        pats = [ln.strip() for ln in
                (ROOT / ".gitignore").read_text(
                    encoding="utf-8").splitlines()
                if ln.strip() and not ln.strip().startswith("#")
                and not ln.strip().startswith("!")]
        writes = ("catalogue.json", "blueprint.json", "findings.txt",
                  "generated.csv", "fidelity.json")
        uncovered = [n for n in writes
                     if not any(fnmatch.fnmatch(n, q) for q in pats)]
        check("every file run_discovery writes is ignored by NAME, so "
              "an artifact landing in the repo from a machine holding "
              "real data cannot be committed: {}".format(
                  ", ".join(writes)),
              not uncovered)
        check("the pipeline reports its seven stages",
              res.stdout.count("[") >= 7 and "/7]" in res.stdout)
        for f in ("profile.json", "draft_spec.json",
                  "generated.csv", "fidelity.json"):
            check("pipeline produced {}".format(f),
                  (td / "run" / f).exists())

        # ---------- structure beyond correlation ----------
        rr2 = random.Random(41)
        deep = []
        for _ in range(6000):
            age = rr2.gauss(62, 14)
            sod = rr2.gauss(139, 5)
            u = abs(sod - 139) / 5.0
            cre = max(0.4, rr2.gauss(1.1, 0.4))
            it = cre * (1.0 if age > 70 else 0.15)
            risk = 0.02 + 0.25 * min(u, 3) / 3 \
                + 0.35 * min(it, 2) / 2
            deep.append({"age": round(age, 1),
                         "sodium": round(sod, 1),
                         "creatinine": round(cre, 2),
                         "event": 1 if rr2.random() < risk else 0})
        dsrc = td / "deep_src.csv"
        write(dsrc, deep)
        # a generator that gets every MARGINAL right and every
        # relationship wrong: shuffle each column independently
        shuffled = [dict(x) for x in deep]
        # NOTE: each column needs its OWN permutation. Re-seeding
        # per column hands every column the SAME permutation, which
        # merely reorders the rows and leaves the joint structure
        # perfectly intact — a fixture that quietly tests nothing.
        shuf_rng = random.Random(42)
        for col in ("age", "sodium", "creatinine", "event"):
            vals = [x[col] for x in shuffled]
            shuf_rng.shuffle(vals)
            for x, v in zip(shuffled, vals):
                x[col] = v
        dshuf = td / "deep_shuffled.csv"
        write(dshuf, shuffled)
        Rd = score(td, dsrc, dshuf, "deep")
        ds = Rd["dependence_shape"]
        check("the scorecard PROFILES the shape of dependence, "
              "not just its correlation",
              ds["pairs_tested"] > 0)
        check("it identifies TURNING relationships in the source — "
              "the ones rank correlation cannot represent at all",
              ds["nonlinear_relationships_in_source"] >= 1)
        check("column-shuffled data (perfect marginals, no joint "
              "structure) FAILS the shape tests",
              any(not d["pass"] for d in ds["detail"]))
        check("...while its marginals all still pass — proving "
              "marginal fidelity alone proves nothing",
              all(m["pass"] for m in Rd["marginals"]))

        Rt = score(td, dsrc, dshuf, "deep_t")
        out_t = td / "rep_deept.json"
        run("scripts/fidelity_report.py", "--source", str(dsrc),
            "--synthetic", str(dshuf), "--target", "event",
            "-o", str(out_t))
        Rt = json.loads(out_t.read_text(encoding="utf-8"))
        check("with a target declared, INTERACTIONS are searched "
              "for and found in the source",
              len(Rt["interactions"]["detail"]) >= 1)
        check("an interaction present in the source but absent "
              "from the synthetic data FAILS",
              any(not d["pass"]
                  for d in Rt["interactions"]["detail"]))

        # a faithful generator on the same source must clear them
        sys.path.insert(0, str(ROOT))
        from synthkit.condnet import CondNet
        net = CondNet(k=10, max_parents=3).learn(
            deep, targets=["event"])
        dgood = td / "deep_condnet.csv"
        write(dgood, net.sample(6000, seed=77))
        out_g = td / "rep_good.json"
        run("scripts/fidelity_report.py", "--source", str(dsrc),
            "--synthetic", str(dgood), "--target", "event",
            "-o", str(out_g))
        Rg = json.loads(out_g.read_text(encoding="utf-8"))
        check("a generator that models the JOINT distribution "
              "passes the shape tests the shuffled one failed",
              all(d["pass"] for d in
                  Rg["dependence_shape"]["detail"]))
        check("...and reproduces the interactions too",
              Rg["interactions"]["detail"]
              and all(d["pass"] for d in
                      Rg["interactions"]["detail"]))

        # ---------- engine selection in the pipeline ----------
        res = run("scripts/phase2_pipeline.py", "--src",
                  str(pipesrc), "-o", str(td / "runc"),
                  "--skip-wrangle", "--engine", "condnet")
        check("the pipeline runs with --engine condnet",
              res.returncode == 0)
        check("it saves the learned model and its own generated "
              "data",
              (td / "runc" / "condnet_model.json").exists()
              and (td / "runc" / "generated_condnet.csv").exists())
        check("the conditional model is parameters, not records — "
              "it states its own privacy contract",
              "no record is stored" in
              (td / "runc" / "condnet_model.json").read_text(
                  encoding="utf-8"))
        check("the condnet run is scored too",
              (td / "runc" / "fidelity_condnet.json").exists())

        res = run("scripts/phase2_pipeline.py", "--src",
                  str(pipesrc), "-o", str(td / "runb"),
                  "--skip-wrangle", "--engine", "both")
        check("--engine both runs each and prints a head-to-head",
              res.returncode == 0
              and "HEAD TO HEAD" in res.stdout
              and (td / "runb" / "generated.csv").exists()
              and (td / "runb" / "generated_condnet.csv").exists())
        check("both engines are scored against the SAME question "
              "set (the profile describes the source, not the "
              "engine)",
              json.loads((td / "runb" / "fidelity.json").read_text(
                  encoding="utf-8"))["correlations"] is not None
              and json.loads(
                  (td / "runb" / "fidelity_condnet.json")
                  .read_text(encoding="utf-8"))[
                      "correlations"] is not None)
        Rn = json.loads((td / "runb" / "fidelity_condnet.json")
                        .read_text(encoding="utf-8"))
        check("relationships that are only sampling noise in the "
              "source are SKIPPED, not scored — reproducing noise "
              "is not fidelity",
              Rn["dependence_shape"]["pairs_skipped_as_noise"] > 0)

        # ---- a list column must not be failed for being private --
        rr3 = random.Random(61)
        lsrc, lsyn = [], []
        pool = ["dm", "htn", "ckd", "chf", "copd"]
        for i in range(1200):
            cs = [c for c in pool if rr3.random() < 0.4]
            # the source also carries rare one-off conditions that
            # suppression will legitimately drop
            if rr3.random() < 0.08:
                cs.append("rare_%d" % i)
            lsrc.append({"conditions": "; ".join(sorted(cs))
                         or "none",
                         "age": round(rr3.gauss(60, 12), 1)})
        for i in range(1200):
            cs = [c for c in pool if rr3.random() < 0.4]
            lsyn.append({"conditions": "; ".join(sorted(cs))
                         or "none",
                         "age": round(rr3.gauss(60, 12), 1)})
        ls, lg = td / "lsrc.csv", td / "lsyn.csv"
        write(ls, lsrc)
        write(lg, lsyn)
        outl = td / "lrep.json"
        run("scripts/fidelity_report.py", "--source", str(ls),
            "--synthetic", str(lg), "-o", str(outl))
        Rl = json.loads(outl.read_text(encoding="utf-8"))
        lm = [m for m in Rl["marginals"]
              if m["column"] == "conditions"]
        check("a list column is compared ITEM BY ITEM, not as "
              "whole strings",
              lm and lm[0]["type"] == "list")
        check("...so suppressing combinations unique to one "
              "person does not read as a fidelity failure — a "
              "scorecard that cries wolf teaches people to ignore "
              "it", lm and lm[0]["pass"])
        check("the row explains what was actually compared",
              lm and "privacy rule had failed" in lm[0]["note"])
        check("the item comparison still has teeth: a missing "
              "common item fails",
              True)
        drop = [{"conditions": "; ".join(
            sorted(c for c in
                   r["conditions"].split("; ") if c != "chf"))
            or "none", "age": r["age"]} for r in lsyn]
        ld = td / "ldrop.csv"
        write(ld, drop)
        outd = td / "ldrop.json"
        run("scripts/fidelity_report.py", "--source", str(ls),
            "--synthetic", str(ld), "-o", str(outd))
        Rd2 = json.loads(outd.read_text(encoding="utf-8"))
        lm2 = [m for m in Rd2["marginals"]
               if m["column"] == "conditions"]
        check("...a COMMON item going missing is caught, so the "
              "leniency is narrow rather than blanket",
              lm2 and not lm2[0]["pass"])

    # THE M0 GATE, AS A THING THAT CAN BE RUN.
    #
    # It was a pair of absolute counts carried in conversation -
    # "close at least 116, direction about 124" - set when a run
    # related 132 pairs. The denominator then moved to 115 and to
    # 114, and 124 became unreachable BY CONSTRUCTION: larger than
    # the number of pairs the run has. Two runs went by without
    # anyone noticing, because the target lived in a chat log rather
    # than beside the numbers.
    #
    # Restated as proportions of whatever the run related. The
    # inversion criterion stays ABSOLUTE - one inverted relationship
    # reads as a finding whatever the denominator is.
    def _gate(summary):
        with tempfile.TemporaryDirectory() as _t:
            _d = Path(_t)
            (_d / "fidelity.json").write_text(
                json.dumps({"summary": summary}), encoding="utf-8")
            _r = subprocess.run(
                [sys.executable, "scripts/m0_gate.py", str(_d)],
                capture_output=True, text=True, cwd=str(ROOT))
            return _r.returncode, (_r.stdout or "")

    _clean = {"pairs": 114, "pairs_sign_ok": 114, "pairs_close": 114,
              "pairs_inverted": 0, "coverage_ok": 42, "columns": 42,
              "set_tokens_ok": 62, "set_tokens_compared": 62,
              "set_empty_ok": 4, "set_empty_compared": 4}
    _rc, _out = _gate(_clean)
    check("a run that meets every criterion clears the M0 gate "
          "(exit {})".format(_rc), _rc == 0 and "M0 MET" in _out)

    _short = dict(_clean, pairs_sign_ok=104, pairs_close=87)
    _rc2, _out2 = _gate(_short)
    check("...and the last real run's numbers do NOT clear it - "
          "104/114 direction is 91.2% against 93.9%, 87/114 close is "
          "76.3% against 87.9% (exit {})".format(_rc2),
          _rc2 == 1 and "M0 NOT MET" in _out2
          and "direction kept" in _out2 and "close" in _out2)

    # A FAIL COMES WITH ITS OPTIONS, CLEARLY LABELED. The operator
    # hit NOT MET on the real extract and asked, correctly, "what
    # are my options?" - the gate said NOT MET and stopped talking.
    # The guidance lives in synthkit.gate (ONE module), so the
    # script and the bench print the same words; each failed
    # criterion gets what it MEANS and numbered next steps, cheapest
    # first, and "proceed with the failure stated" is on the list
    # because a gate is a floor, not a release decision.
    check("...and a NOT MET prints WHAT NOW for each failed "
          "criterion - meaning plus numbered options, from the "
          "shared gate module",
          "DIRECTION KEPT - " in _out2 and "CLOSE - " in _out2
          and "1." in _out2 and "2." in _out2
          and "NOT MET is a reading, not a wall" in _out2)
    check("...and a MET run prints NO guidance - advice under a "
          "green gate is noise",
          "What now" not in _out
          and "NOT MET is a reading" not in _out)
    _inv = dict(_clean, pairs_inverted=1)
    _rc3, _out3 = _gate(_inv)
    check("...and an INVERTED failure says STOP and 'do not send "
          "this file', because inverted reads as a finding",
          _rc3 == 1 and "STOP" in _out3
          and "Do not send this file" in _out3)

    # THE OLD ABSOLUTE FORM WOULD HAVE BEEN UNREACHABLE HERE. 124 of
    # 114 pairs cannot happen, so a gate written that way reports
    # failure for a reason that has nothing to do with the data.
    check("...and the gate is expressed against the run's OWN "
          "denominator, so it stays reachable when the pair count "
          "moves - 124 of 114 pairs is not a bar, it is arithmetic",
          "114" in _out2 and "93.9%" in _out2)

    _inv = dict(_clean, pairs_inverted=1)
    _rc3, _out3 = _gate(_inv)
    check("...while a single INVERTED relationship fails it outright, "
          "however large the denominator - it reads as a finding, "
          "which is worse than a missing one (exit {})".format(_rc3),
          _rc3 == 1 and "inverted" in _out3)

    # THE DECK: original vs synthetic, drawn, and itself k-screened.
    #
    # Every number in it existed already - in fidelity.json and the
    # blueprint - and the person who asked for it is a statistician
    # who said, correctly, that nobody reads JSON. The deck draws the
    # comparison the way the field draws it, in the house palette,
    # and adds the one thing no vendor page shows: source histogram
    # bins backed by fewer than k patients are SUPPRESSED from the
    # report itself, because a bin of three people is a group small
    # enough to gossip about.
    import numpy as _np2
    import pandas as _pd2
    with tempfile.TemporaryDirectory() as _t:
        _d = Path(_t)
        _r20 = _np2.random.RandomState(3)
        _n20, _npat20 = 800, 80
        _g20 = _np2.repeat(_np2.arange(_npat20), _n20 // _npat20)
        _df20 = _pd2.DataFrame({
            "person_id": ["P{:03d}".format(x) for x in _g20],
            "visit_start_date": ["2024-01-{:02d}".format(i % 28 + 1)
                                 for i in range(_n20)],
            "val": _np2.round(_r20.normal(40, 8, _n20), 2),
            "val2": 0.0, "val3": 0.0,
            "grp": _r20.choice(list("xyz"), _n20)})
        _df20["val2"] = _np2.round(
            0.8 * _df20["val"].astype(float)
            + _r20.normal(0, 2.5, _n20), 2)
        _df20["val3"] = _np2.round(_r20.normal(10, 3, _n20), 2)
        # a TWO-parent child, so the SHAP attribution section has
        # something to attribute - its own column, because a column
        # added to a fixture other checks depend on once broke
        # three of them
        _df20["val4"] = _np2.round(
            0.5 * _df20["val"].astype(float)
            + 0.9 * _df20["val3"].astype(float)
            + _r20.normal(0, 1.5, _n20), 2)
        _srcp = _d / "tiny.csv"
        _df20.to_csv(_srcp, index=False)
        _rundir = _d / "run"
        _r1 = subprocess.run(
            [sys.executable, "scripts/run_discovery.py",
             "--src", str(_srcp), "--out", str(_rundir),
             "--group-by", "person_id", "--generate"],
            capture_output=True, text=True, cwd=str(ROOT))
        _deck = _d / "deck.html"
        _r2 = subprocess.run(
            [sys.executable, "scripts/fidelity_deck.py",
             "--src", str(_srcp), "--run", str(_rundir),
             "-o", str(_deck), "--group-by", "person_id",
             "--compare", "again={}".format(_rundir)],
            capture_output=True, text=True, cwd=str(ROOT))
        _h = _deck.read_text(encoding="utf-8") if _deck.exists()             else ""
        # INSIDE THE TEMP BLOCK, because _rundir is deleted when it
        # closes - the first version of this check ran AFTER the
        # block and failed on the 1x run being gone, not on the
        # compare it meant to test.
        _r3 = subprocess.run(
            [sys.executable, "scripts/fidelity_deck.py",
             "--src", str(_srcp), "--run", str(_rundir),
             "-o", str(_deck), "--group-by", "person_id",
             "--compare", "5x=/no-such-run"],
            capture_output=True, text=True, cwd=str(ROOT))
        # the SHAP omission path, captured while _rundir exists:
        # block the import and rebuild in-process
        import builtins as _bi
        import fidelity_deck as _fd0
        _real_imp = _bi.__import__

        def _block(nm, *a, **k):
            if nm == "shap":
                raise ImportError("blocked by smoke check")
            return _real_imp(nm, *a, **k)
        _bi.__import__ = _block
        try:
            _doc_blocked, _ = _fd0.build_deck(
                str(_srcp), str(_rundir), "person_id")
        finally:
            _bi.__import__ = _real_imp
        # THE SCALE CAPTION SAYS WHICH COMPARISON IT IS. It
        # claimed "same blueprint" unconditionally and the
        # operator's first real compare paired two INDEPENDENT
        # fits - the denominators contradicted the caption on the
        # exhibit itself. Same-run compare must say same
        # blueprint; a compare against a run whose blueprint
        # differs by one byte must say independent fits, claimed
        # as the STRONGER form.
        import shutil as _sh
        _rd2 = Path(_d) / "run2"
        _sh.copytree(_rundir, _rd2)
        _bp2 = _rd2 / "blueprint.json"
        _bp2.write_text(_bp2.read_text(encoding="utf-8") + "\n",
                        encoding="utf-8")
        _doc_cross, _ = _fd0.build_deck(
            str(_srcp), str(_rundir), "person_id",
            ["x={}".format(_rd2)])
    check("the deck writes a self-contained page with both series, "
          "gate chips and paired heatmaps",
          _r2.returncode == 0 and "<svg" in _h
          and 'class="chip' in _h
          and "Original" in _h and "Synthetic" in _h)
    check("...in the house palette - near-black ink, gray original, "
          "cardinal synthetic, system font - not the bench's "
          "station colors",
          "-apple-system" in _h and "#8c1515" in _h
          and "#6e6e73" in _h)
    check("...and the report is ITSELF k-screened: source bins "
          "under k patients are suppressed and the footer says so, "
          "which no vendor page does",
          "suppressed" in _h
          and "not differential privacy" in _h)
    check("...and the scale table renders one row per compared run, "
          "answering 'does 5x degrade it' with measurements",
          "Does scale degrade it" in _h
          and _h.count("<td>again</td>") == 1)

    # THE PATTERNS, DRAWN - the section no vendor page has. Fabric
    # compares CORRELATIONS, one number per pair, which a threshold,
    # a saturation and a straight line can all share. The deck draws
    # each discovered relationship three times - published curve,
    # measured on the original, measured on the synthetic - so a
    # kept bend is visible and a lost one is flagged. The fixture
    # plants val2 = 0.8*val + noise, so at least one pattern card
    # must exist.
    check("the deck draws pattern cards - each relationship as "
          "published / original / synthetic curves with a "
          "plain-English verdict",
          "The patterns, drawn" in _h
          and _h.count("&larr;") >= 1
          and ("tracks the original within" in _h
               or "DEPARTS" in _h))
    # THE CHARTS ARE ALIVE, AND THE FILE CARRIES ITS OWN MOTION.
    # Press-and-hold pulls original and synthetic apart; release
    # lets them settle back into overlap - the fidelity claim,
    # performed. The script and styles are inline because this file
    # must animate identically as a shared artifact with no network.
    # Verified fail-first: the pre-animation output fails this exact
    # contract.
    check("each chart's two series are grouped so they can move as "
          "bodies, and the press-and-hold pull-apart is wired with "
          "its styles and script inline",
          'class="ser src"' in _h and 'class="ser syn"' in _h
          and 'svg.live.apart .ser.src' in _h
          and "pointerdown" in _h
          and 'path class="draw"' in _h)
    check("...the page says how to use it, and motion respects the "
          "reader who asked for none",
          "Press and hold any chart" in _h
          and "prefers-reduced-motion" in _h)
    # THE HEATMAPS CROSSFADE IN PLACE, AND THE TITLE RIDES INSIDE
    # EACH LAYER. Holding fades each side into the other so a
    # differing pair blinks into view - and during the fade the
    # half must be labeled by what it is SHOWING, or the
    # "Synthetic" panel would display original data under a false
    # heading. Fail-first: no hm layer exists in the prior build.
    check("the correlation heatmaps crossfade on hold - both cell "
          "layers present, opacity swap wired, and the titles "
          "travel with their data",
          'class="live hm"' in _h
          and _h.count("hm-self") >= 3 and _h.count("hm-other") >= 3
          and "svg.hm.apart .hm-self{opacity:0" in _h
          and "svg.hm.apart .hm-other{opacity:1" in _h
          and "crossfades each side" in _h)

    check("...with the drivers attributed when there is more than "
          "one, and shapes named in words (a 'threshold' is not a "
          "correlation)",
          "confirmed on held-out patients" in _h)

    # SHAPES AND SURFACES: WHAT RANK CORRELATION CANNOT SEE. A
    # U-shape has Spearman near zero on BOTH tables, so a U-shaped
    # relationship was excluded from the close criterion entirely -
    # generation could flatten it and the gate stayed green. And
    # the published interaction surfaces' survival had only ever
    # been proven on the instrumented fixture, never per run.
    # synthkit.curvecheck is the ONE implementation; the pipeline
    # records it, the gate reads it, the deck draws from it.
    import numpy as _np9
    import pandas as _pd9
    from synthkit import curvecheck as _cc
    from synthkit import gate as _g2
    import fidelity_deck as _fd
    _r9 = _np9.random.RandomState(7)
    _n9 = 3000
    _x9 = _r9.uniform(0, 1, _n9)

    def _mk9(y):
        return _pd9.DataFrame({
            "pid": _np9.repeat(_np9.arange(300), 10).astype(str),
            "x": _x9, "y": y})
    _srcU = _mk9((_x9 - 0.5) ** 2 * 8 + _r9.normal(0, .1, _n9))
    _flat = _mk9(_r9.normal(float(_srcU["y"].mean()),
                            float(_srcU["y"].std()), _n9))
    _bp9 = {"relationships": [{"child": "y", "parents": ["x"],
            "evidence": {"effect": {"x": {"grid_kind":
                                          "numeric"}}}}]}
    _kept9 = _cc.measure_shapes(_srcU, _srcU.copy(), _bp9,
                                _srcU["pid"], 10)
    _lost9 = _cc.measure_shapes(_srcU, _flat, _bp9,
                                _srcU["pid"], 10)
    _rho9 = float(_srcU["x"].corr(_srcU["y"], method="spearman"))
    check("a planted U-shape is INVISIBLE to rank correlation "
          "(|rho| {:.3f}) yet the shape criterion reads a kept U "
          "as tracked and a flattened U as departed by {:.1f} "
          "sd".format(abs(_rho9), _lost9["claims"][0]["gap_sd"]),
          abs(_rho9) < 0.1
          and _kept9["claims"][0]["tracks"]
          and not _lost9["claims"][0]["tracks"]
          and _lost9["claims"][0]["gap_sd"] > 1.0)
    _a9 = _r9.uniform(0, 1, _n9)
    _b9 = _r9.uniform(0, 1, _n9)
    _xor9 = ((_a9 > .5) ^ (_b9 > .5)).astype(float)

    def _mk29(y):
        return _pd9.DataFrame({
            "pid": _np9.repeat(_np9.arange(300), 10).astype(str),
            "a": _a9, "b": _b9, "y": y})
    _srcX = _mk29(_xor9 * 2 + _r9.normal(0, .15, _n9))
    _lostX = _mk29(_r9.normal(float(_srcX["y"].mean()),
                              float(_srcX["y"].std()), _n9))
    _bpX = {"relationships": [{"child": "y", "parents": ["a", "b"],
            "evidence": {"interaction": {"pair": ["a", "b"],
                         "grid_a": [0, 1], "grid_b": [0, 1],
                         "response": [[0, 1], [1, 0]]}}}]}
    _keptX = _cc.measure_surfaces(_srcX, _srcX.copy(), _bpX,
                                  _srcX["pid"], 10)
    _lostX2 = _cc.measure_surfaces(_srcX, _lostX, _bpX,
                                   _srcX["pid"], 10)
    check("a planted XOR interaction surface reads as tracked when "
          "kept and departed when generation loses it - the "
          "per-run metric fixtures could only imply",
          _keptX["surfaces"][0]["tracks"]
          and not _lostX2["surfaces"][0]["tracks"]
          and _lostX2["surfaces"][0]["gap_sd"] > 0.5)
    # THE SURFACE COUNT STATES ITS TOTAL. The real extract
    # publishes 72 surfaces of which only 6 touch columns in the
    # written files - the rest are token-indicator surfaces that
    # exist only inside the search. A criterion reading "2/6"
    # without naming the other 66 reads as coverage it does not
    # have, so measure_surfaces counts every skip by reason and
    # the note says so.
    _bpT = {"relationships": [
        {"child": "y", "parents": ["a", "b"],
         "evidence": {"interaction": {"pair": ["a", "b"],
                      "grid_a": [0, 1], "grid_b": [0, 1],
                      "response": [[0, 1], [1, 0]]}}},
        {"child": "meds__has__x", "parents": ["a"],
         "evidence": {"interaction": {"pair": ["a", "meds__n"]}}},
        {"child": "z", "parents": ["a"],
         "evidence": {"interaction": {"pair": ["a",
                                              "not_a_column"]}}}]}
    _resT = _cc.measure_surfaces(_srcX.rename(columns={
        "a": "a", "b": "b", "y": "y"}), _srcX.copy(),
        _bpT, _srcX["pid"], 10)
    check("measure_surfaces states its TOTAL - published, "
          "measured, and every skip counted by reason, with the "
          "note saying skipped is never silent",
          _resT["published"] == 3
          and _resT["compared"] == 1
          and _resT["skipped_token_columns"] == 1
          and _resT["skipped_absent_columns"] == 1
          and _resT["published"] == _resT["compared"]
          + _resT["skipped_token_columns"]
          + _resT["skipped_absent_columns"]
          + _resT["skipped_thin_data"]
          and "never silently" in _resT["note"])

    _vS = _g2.assess({"summary": {
        "pairs": 10, "pairs_sign_ok": 10, "pairs_close": 10,
        "pairs_inverted": 0, "shapes_compared": 5, "shapes_ok": 4,
        "surfaces_compared": 2, "surfaces_ok": 2}})
    check("the gate reads the new counters - a departed shape "
          "fails 'shapes tracked' while absent measurements fail "
          "nothing",
          [c["ok"] for c in _vS["criteria"]
           if c["name"] == "shapes tracked"] == [False]
          and _g2.assess({"summary": {
              "pairs": 5, "pairs_sign_ok": 5, "pairs_close": 5,
              "pairs_inverted": 0}})["met"])
    _fidS = {"shapes": {"claims": [
        {"child": "y", "parent": "x", "shape": "u_shape",
         "gap_sd": 1.69, "tracks": False}]},
        "surfaces": {"surfaces": []},
        "relationships": {"pairs": [], "inverted": []}}
    _vS2 = {"criteria": [{"name": "shapes tracked", "ok": False,
                          "detail": "4/5"}], "met": False}
    check("...and the deck enumerates a departed shape by name "
          "with its gap",
          "u_shape" in _fd.gate_issues_html(_fidS, _vS2)
          and "1.69" in _fd.gate_issues_html(_fidS, _vS2))

    # THE centre_miss READER. The diagnosis block has waited in
    # fidelity.json since it was built, unread on real data because
    # reading it needed a hand-typed one-liner on a terminal that
    # mangles pastes. peek.py centre is the reader; the fixture
    # CONTAINS misses, because a reader checked against an empty
    # list proves nothing.
    with tempfile.TemporaryDirectory() as _tc:
        (Path(_tc) / "fidelity.json").write_text(json.dumps({
            "summary": {"centre_ok": 1},
            "columns": [
                {"column": "lab_a", "mean_source": 1.0,
                 "centre_miss": {"by_sd": 0.71,
                                 "direction": "generated above source",
                                 "skew_source": 2.8,
                                 "tail_shape_published": False,
                                 "integral": False}},
                {"column": "lab_b", "mean_source": 2.0,
                 "centre_miss": {"by_sd": 0.15,
                                 "direction": "generated below source",
                                 "skew_source": 0.1,
                                 "tail_shape_published": True,
                                 "integral": True}},
                {"column": "ok_col", "mean_source": 3.0}]}),
            encoding="utf-8")
        _rp = subprocess.run(
            [sys.executable, "scripts/peek.py", _tc, "centre"],
            capture_output=True, text=True, cwd=str(ROOT))
    check("peek.py centre reads the diagnosis worst-first, flags "
          "the skewed/no-tail suspects, and echoes its settings",
          _rp.returncode == 0
          and "centre_view on" in _rp.stdout
          and _rp.stdout.index("lab_a") < _rp.stdout.index("lab_b")
          and "1 of 2 missed columns are skewed" in _rp.stdout
          and "1 had NO tail shape published" in _rp.stdout)

    # GUIDANCE GIVES COMMANDS, NOT ADVICE - and survives both
    # renderers. The seed step carries the exact CLI with
    # bracket-free placeholders, because <angle brackets> are HTML
    # tags to the bench and silently vanish - caught by eye in a
    # screenshot, with the command rendering as "--src --out".
    from synthkit import gate as _g3
    _seed_step = _g3.NEXT_STEPS["close"]["steps"][1]
    check("the seed guidance states the exact command - two seeds, "
          "new out directory, all other flags identical - with "
          "renderer-safe placeholders",
          "--seed 11" in _seed_step and "--seed 37" in _seed_step
          and "THE_SAME_CSV" in _seed_step
          and "20260731" in _seed_step
          and all("<" not in st for v in _g3.NEXT_STEPS.values()
                  for st in v["steps"]))
    # THE k-RULE INFORMATION SITS BESIDE THE DRIFT. A drifted pair
    # whose column loses magnitude to the published bound is MARKED
    # in the gate-issues table - the operator asked for the
    # information, not for directions to go find it.
    _fidK = {"relationships": {"pairs": [
        {"child": "lab", "parent": "age", "kind": "numeric",
         "source": 0.6, "generated": 0.25, "delta": -0.35}],
        "inverted": []}}
    _vK = {"criteria": [{"name": "close", "ok": False,
                         "detail": "x"}], "met": False}
    _bpK = {"columns": {"lab": {"marginal": {
        "share_of_magnitude_outside_bounds": 0.87}}}}
    check("the gate-issues table MARKS drift on bound-clipped "
          "columns as the k rule by design, and stays silent when "
          "no column is clipped",
          "loses 87% of its magnitude to the k bound"
          in _fd.gate_issues_html(_fidK, _vK, _bpK)
          and "k bound" not in _fd.gate_issues_html(_fidK, _vK))
    # MULTI-VARIABLE IS SAID OUT LOUD. The operator looked for the
    # multivariate/non-linear position and could not find it: each
    # multi-driver pattern now states how many variables act
    # jointly, whether a two-way surface is published, and that
    # higher-order is not modeled.
    check("each multi-driver pattern card states its arity, its "
          "surface (or the absence), and the two-way limit",
          "MULTI-VARIABLE pattern" in _h
          and "above two-way are not modeled" in _h)

    # peek surfaces - the remote diagnosis for the surface fix's
    # NON-TRANSFER. The fix cured the fixture (2/5-seeds-failing to
    # 0/5) and moved nothing on the extract (same 2/6, identical
    # gate line), so the next answer must come from the RUN
    # DIRECTORY, not a third guessed fixture. The view flags the
    # three suspect conditions by name; the fixture here contains
    # all three.
    with tempfile.TemporaryDirectory() as _tp:
        _pr = Path(_tp)
        (_pr / "blueprint.json").write_text(json.dumps({
            "relationships": [
                {"child": "a", "parents": ["x", "y"],
                 "evidence": {"interaction": {
                     "pair": ["x", "z"]}}},
                {"child": "b", "parents": ["p", "q"],
                 "evidence": {"interaction": {
                     "pair": ["p", "q"]}}}]}), encoding="utf-8")
        (_pr / "fidelity.json").write_text(json.dumps({
            "surfaces": {"surfaces": [
                {"child": "b", "pair": ["p", "q"],
                 "gap_sd": 1.2, "tracks": False}]},
            "generation": {"edges_dropped": [
                {"child": "b", "parents": ["p", "q", "r_lag1"],
                 "kept_parents": ["p"]}]}}), encoding="utf-8")
        _rv = subprocess.run(
            [sys.executable, "scripts/peek.py", str(_pr),
             "surfaces"],
            capture_output=True, text=True, cwd=str(ROOT))
    check("peek.py surfaces flags all three suspects by name - a "
          "pair member absent from the parents, a lost LAG "
          "parent, and a lost pair member - beside each "
          "surface's gap",
          _rv.returncode == 0
          and "PAIR MEMBER NOT IN PARENTS: z" in _rv.stdout
          and "LAG PARENT LOST: r_lag1" in _rv.stdout
          and "SURFACE PAIR MEMBER LOST: q" in _rv.stdout
          and "gap 1.2 sd, tracks False" in _rv.stdout)

    # THE SIGN-OFF PAGE - goal 6's last piece. One roll-up from
    # the run's own artifacts: the gate with the same criteria the
    # Verdict station shows, the privacy posture in the
    # non-overclaiming words with measured counts beside it, the
    # limitations restated, and a signature block that says what
    # signing accepts. Exercised on a NOT-MET fixture shaped like
    # the real extract, because a sign-off that only renders green
    # runs has never been read.
    with tempfile.TemporaryDirectory() as _ts:
        _sr = Path(_ts)
        (_sr / "fidelity.json").write_text(json.dumps({
            "summary": {"pairs": 115, "pairs_sign_ok": 108,
                        "pairs_close": 88, "pairs_inverted": 0,
                        "coverage_ok": 42, "columns": 42,
                        "set_tokens_ok": 62,
                        "set_tokens_compared": 62,
                        "set_empty_ok": 4, "set_empty_compared": 4,
                        "shapes_compared": 9, "shapes_ok": 8,
                        "surfaces_compared": 6, "surfaces_ok": 1},
            "contradictions": [], "disobedience": []}),
            encoding="utf-8")
        (_sr / "blueprint.json").write_text(json.dumps({
            "columns": {"conditions": {"marginal": {
                "type": "list", "bounds_are_k_anonymous": 10,
                "unpublishable_row_share": 0.027}}}}),
            encoding="utf-8")
        _so = Path(_ts) / "signoff.html"
        _rs = subprocess.run(
            [sys.executable, "scripts/signoff.py", str(_sr),
             "-o", str(_so)],
            capture_output=True, text=True, cwd=str(ROOT))
        _sh = (_so.read_text(encoding="utf-8")
               if _so.exists() else "")
        check("the sign-off page renders a NOT-MET gate honestly - "
              "banner naming the failed criteria, all eight chips, "
              "and the full criterion table",
              _rs.returncode == 0
              and "NOT MET" in _sh
              and "FAIL close" in _sh
              and "FAIL interaction surfaces" in _sh
              and _sh.count('class="chip') == 8)
        check("...with the privacy posture in the non-overclaiming "
              "words and MEASURED counts beside it, and a "
              "signature block that says what signing accepts",
              "NOT differential privacy" in _sh
              and "floor measured on one cohort" in _sh
              and "conditions 3%" in _sh
              and "What signing accepts" in _sh
              and "reviewed by" in _sh)
        _rs2 = subprocess.run(
            [sys.executable, "scripts/signoff.py",
             str(Path(_ts) / "nope"), "-o", str(_so)],
            capture_output=True, text=True, cwd=str(ROOT))
        check("...and a missing run refuses in a sentence",
              _rs2.returncode != 0
              and "STOPPED" in (_rs2.stdout + _rs2.stderr)
              and "Traceback" not in (_rs2.stdout + _rs2.stderr))

    # THE GATE, ENUMERATED. FAIL chips said which criterion and
    # stopped; the operator asked where the misses were and how
    # large. gate_issues_html names every drifted pair with its
    # magnitude - and renders NOTHING when the gate is met, because
    # highlighting under a green gate is noise. Verified against a
    # crafted report: the close table excludes close pairs and
    # weak-source pairs, flags direction-lost apart from faded, and
    # sorts largest drift first.
    import fidelity_deck as _fd
    _ffid = {"relationships": {"pairs": [
        {"child": "crp", "parent": "stress", "kind": "numeric",
         "source": 0.62, "generated": 0.31, "delta": -0.31},
        {"child": "hr", "parent": "age", "kind": "numeric",
         "source": 0.45, "generated": 0.44, "delta": -0.01},
        {"child": "sleep", "parent": "stress", "kind": "numeric",
         "source": -0.38, "generated": 0.02, "delta": 0.40}],
        "inverted": []}}
    _fv = {"criteria": [
        {"name": "close", "ok": False, "detail": "x"}],
        "met": False}
    _gh = _fd.gate_issues_html(_ffid, _fv)
    check("the deck ENUMERATES a failed gate criterion - every "
          "drifted pair by name with magnitude, direction-lost "
          "flagged apart, largest first, close pairs excluded",
          "The gate, enumerated" in _gh
          and _gh.count("<tr>") == 3
          and "direction lost" in _gh and "faded" in _gh
          and "hr" not in _gh.replace("relationship", "")
          and _gh.index("sleep") < _gh.index("crp"))
    check("...and a MET gate renders no issues section at all",
          _fd.gate_issues_html(_ffid, {"criteria": [
              {"name": "close", "ok": True}], "met": True}) == "")

    # STATION REFERENCES RENDER AS THE STATION. gate guidance
    # carries [[dashboard]] tokens; plain() names the station for
    # terminals, and m0_gate must never print a raw token.
    from synthkit import gate as _g2
    check("station tokens exist in the guidance and plain() names "
          "the station for terminals",
          "[[dashboard]]" in _g2.NEXT_STEPS["close"]["steps"][0]
          and "the VIEW Dashboard station" in _g2.plain(
              _g2.NEXT_STEPS["close"]["steps"][0])
          and all("[[" not in _g2.plain(st)
                  for v in _g2.NEXT_STEPS.values()
                  for st in v["steps"]))
    check("...and every criterion carries an explanation AND an "
          "inspect-it-yourself pointer into the dashboard",
          set(_g2.EXPLAIN) == {c["name"] for c in [
              {"name": n} for n in (
                  "direction kept", "close", "inverted",
                  "coverage", "set token shares",
                  "set EMPTY rate", "shapes tracked",
                  "interaction surfaces")]}
          and all("[[dashboard]]" in v["inspect"]
                  for v in _g2.EXPLAIN.values()))

    # THE CANDIDATE POOL IS EVERY CLAIM, AND THE EMPTY CASE IS
    # STATED. The first version screened only the top-8 drawn
    # cards; on the real extract those are token-indicator pairs
    # whose columns live only inside the search, so the section
    # rendered NOTHING - the exact silent absence the omission
    # note exists to prevent, shipped in the branch beside it.
    import pandas as _pd8
    _srcE = _pd8.DataFrame({"a": ["1", "2", "3"] * 40,
                            "b": ["2", "3", "4"] * 40,
                            "y": ["5", "6", "7"] * 40,
                            "d": ["2024-01-01"] * 120})
    _genE = _srcE.copy()
    check("shap candidates come from EVERY claim - a numeric "
          "two-driver child qualifies, a token child and a "
          "date-thinned parent list do not",
          _fd.shap_candidates(
              [{"child": "y", "skill": .8, "predictors": [
                  {"column": "a"}, {"column": "b"}]}],
              _srcE, _genE) == [(0.8, "y", ["a", "b"])]
          and _fd.shap_candidates(
              [{"child": "meds__has__x", "skill": .9,
                "predictors": [{"column": "a"},
                               {"column": "b"}]}],
              _srcE, _genE) == []
          and _fd.shap_candidates(
              [{"child": "y", "skill": .8, "predictors": [
                  {"column": "a"}, {"column": "d"}]}],
              _srcE, _genE) == [])
    check("...and when nothing qualifies the section STATES the "
          "limit rather than vanishing",
          "nothing to attribute" in io.open(
              "scripts/fidelity_deck.py",
              encoding="utf-8").read()
          and "This is a stated " in io.open(
              "scripts/fidelity_deck.py",
              encoding="utf-8").read())

    # THE DRIVERS, ATTRIBUTED - SHAP is OPTIONAL and its absence
    # is STATED. When installed, each drawn pattern child with two
    # or more numeric parents gets paired driver bars (original
    # beside synthetic) and the strongest jointly-acting pair from
    # SHAP interaction values, cross-referenced against whether the
    # contract publishes a surface for it. When absent, the deck
    # says so and names the install command - a silently missing
    # section reads as "nothing to show".
    try:
        import shap as _shap_probe  # noqa: F401
        _has_shap = True
    except Exception:
        _has_shap = False
    if _has_shap:
        check("with shap installed, the deck attributes drivers - "
              "paired bars per parent and the strongest "
              "jointly-acting pair, cross-referenced against the "
              "contract's surfaces",
              "The drivers, attributed" in _h
              and "who drives it" in _h
              and "Strongest jointly-acting pair" in _h
              and ("publishes a surface for this pair" in _h
                   or "does NOT publish a surface" in _h))
    else:
        check("without shap, the deck STATES the omission and "
              "names the install command",
              "is not installed on this machine" in _h
              and "pip install shap" in _h)
    # the omission path must hold on EVERY machine; _doc_blocked
    # was captured INSIDE the temp block - the run directory is
    # gone by now, the same trap the compare-refusal check hit
    check("...and with the import blocked, the same build emits "
          "the stated omission, never a silent gap",
          "is not installed on this machine" in _doc_blocked
          and "pip install shap" in _doc_blocked
          and "who drives it" not in _doc_blocked)

    check("the scale caption states WHICH comparison it is - "
          "same-run compare says same blueprint; a one-byte "
          "blueprint difference flips it to independent fits, "
          "claimed as the stronger form",
          "same blueprint" in _h
          and "independent fits" not in _h
          and "independent fits" in _doc_cross
          and "STRONGER form" in _doc_cross
          and "same blueprint" not in _doc_cross)

    # A MISSING COMPARE RUN IS A SENTENCE, NOT A STACK. The operator
    # hit a raw FileNotFoundError when --compare named a 5x run that
    # had not been generated yet - and the 1x deck they had ALREADY
    # written was fine, which the refusal now says. (_r3 was captured
    # above, inside the temp block, so _rundir still existed.)
    check("a --compare pointing at a run that does not exist gets a "
          "plain refusal naming the fix, not a traceback",
          _r3.returncode != 0
          and "STOPPED" in (_r3.stderr + _r3.stdout)
          and "does not exist yet" in (_r3.stderr + _r3.stdout)
          and "Traceback" not in (_r3.stderr + _r3.stdout))

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
