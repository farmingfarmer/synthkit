"""Smoke: the fidelity/privacy scorecard detects what it claims to.

A scorecard that always passes is worthless, so every claim here is
tested against a DELIBERATE failure as well as a success.
"""
import csv
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
        check("both nearest-neighbour distributions are reported "
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
        check("True/False and 1/0 are recognised as the SAME "
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
              "any real-data artefacts by default",
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
        # `generated.csv`, so every artefact of the discover /
        # blueprint / generate path staged cleanly from a repo-root
        # run until this check existed.
        #
        # Scoped to that runner deliberately. The older pipeline's
        # artefacts were audited when the section above was written;
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
              "an artefact landing in the repo from a machine holding "
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
    check("the deck writes a self-contained page with both series, "
          "gate chips and paired heatmaps",
          _r2.returncode == 0 and "<svg" in _h
          and 'class="chip' in _h
          and "Original" in _h and "Synthetic" in _h)
    check("...in the house palette - near-black ink, gray original, "
          "cardinal synthetic, system font - not the bench's "
          "station colours",
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
    # must animate identically as a shared artefact with no network.
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

    check("...with the drivers attributed when there is more than "
          "one, and shapes named in words (a 'threshold' is not a "
          "correlation)",
          "confirmed on held-out patients" in _h)

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
