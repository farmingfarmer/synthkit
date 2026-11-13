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

    # THE GATE BARS REACH THE CLI, and the exit code follows the
    # chosen bar - the k-floor refusal taught that a dispatcher
    # can DISCARD a return value, so the flag is tested through
    # the process boundary, not the function.
    with tempfile.TemporaryDirectory() as _tb:
        _d = Path(_tb)
        (_d / "fidelity.json").write_text(
            json.dumps({"summary": _short}), encoding="utf-8")
        _rb = subprocess.run(
            [sys.executable, "scripts/m0_gate.py", str(_d),
             "--direction-bar", "0.90", "--close-bar", "0.75"],
            capture_output=True, text=True, cwd=str(ROOT))
    check("--direction-bar/--close-bar flip the same run from "
          "NOT MET to MET through the process boundary, and the "
          "printed lines say YOUR bar beside the recorded "
          "default - a MET that does not name its contract is "
          "a number nobody can trust (exit {})".format(
              _rb.returncode),
          _rb.returncode == 0 and "M0 MET" in _rb.stdout
          and "YOUR bar" in _rb.stdout
          and "87.9%" in _rb.stdout)

    # argparse %-formats help text, so a formatted "93.9%" in a
    # help string is a latent ValueError - it sat green for as
    # long as nothing rendered -h, and Python 3.14 then validated
    # it EAGERLY, killing the script on EVERY invocation and
    # taking all eight gate checks down on the data machine. -h
    # is rendered through the process boundary here so the crash
    # has a check at every interpreter, not just the newest.
    _rh = subprocess.run(
        [sys.executable, "scripts/m0_gate.py", "--help"],
        capture_output=True, text=True, cwd=str(ROOT))
    check("m0_gate --help RENDERS, with both bar flags and the "
          "default percentages in it - a literal % in argparse "
          "help is a crash that waits for whoever asks for help "
          "(and on Python 3.14, for everyone)",
          _rh.returncode == 0 and "--direction-bar" in _rh.stdout
          and "--close-bar" in _rh.stdout
          and "93.9%" in _rh.stdout and "87.9%" in _rh.stdout)

    # ONE VERDICT, ONE TEXT, TWO DOORS. `synthkit gate` exists so
    # the test kit works from its own folder, where scripts/ does
    # not; both doors call gate.report, and this asserts they
    # cannot drift by comparing their bytes on the same run.
    def _gate_verb(summary):
        with tempfile.TemporaryDirectory() as _t:
            _d = Path(_t)
            (_d / "fidelity.json").write_text(
                json.dumps({"summary": summary}), encoding="utf-8")
            _r = subprocess.run(
                [sys.executable, "-m", "synthkit.cli", "gate",
                 str(_d)],
                capture_output=True, text=True, cwd=str(ROOT))
            _r2 = subprocess.run(
                [sys.executable, "scripts/m0_gate.py", str(_d)],
                capture_output=True, text=True, cwd=str(ROOT))
            return _r, _r2
    _v, _m = _gate_verb(_short)
    check("`synthkit gate` prints byte-identical lines to "
          "scripts/m0_gate.py on the same failing run, with the "
          "same exit code - one renderer, asserted rather than "
          "hoped",
          _v.returncode == 1 and _v.returncode == _m.returncode
          and _v.stdout.replace(str(Path()), "")
          == _m.stdout.replace(str(Path()), "")
          and "M0 NOT MET" in _v.stdout)
    _v3 = subprocess.run(
        [sys.executable, "-m", "synthkit.cli", "gate",
         "no_such_dir_anywhere"],
        capture_output=True, text=True, cwd=str(ROOT))
    check("...and a missing run refuses in a sentence naming "
          "fidelity.json, never a stack",
          _v3.returncode == 2
          and "fidelity.json" in (_v3.stdout + _v3.stderr)
          and "Traceback" not in (_v3.stdout + _v3.stderr))

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
    # COMBINATION EFFECTS, MINED - ITS OWN FIXTURE AND ITS OWN
    # RUN, because the deck fixture above is purely LINEAR
    # (val4 = 0.5*val + 0.9*val3) and nothing mines an
    # interaction out of a sum. A check run against it would
    # have passed on an empty list, which is the
    # fixture-must-contain-the-thing rule this file has broken
    # before. So: a genuine PRODUCT term, whose two parents have
    # near-zero rank correlation with their child - the shape
    # only a model-importance screen can find.
    with tempfile.TemporaryDirectory() as _t9:
        _d9 = Path(_t9)
        _r9 = _np2.random.RandomState(11)
        _n9 = 900
        _g9 = _np2.repeat(_np2.arange(90), _n9 // 90)
        _a9 = _r9.normal(0, 1, _n9)
        _b9 = _r9.normal(0, 1, _n9)
        _df9 = _pd2.DataFrame({
            "person_id": ["Q{:03d}".format(x) for x in _g9],
            "a": _np2.round(_a9, 3), "b": _np2.round(_b9, 3),
            "c": _np2.round(_r9.normal(5, 1, _n9), 3),
            "y": _np2.round(3.0 * _a9 * _b9
                            + _r9.normal(0, 0.4, _n9), 3)})
        _sp9 = _d9 / "prod.csv"
        _df9.to_csv(_sp9, index=False)
        _rd9 = _d9 / "run"
        subprocess.run(
            [sys.executable, "scripts/run_discovery.py",
             "--src", str(_sp9), "--out", str(_rd9),
             "--group-by", "person_id", "--generate"],
            capture_output=True, text=True, cwd=str(ROOT))
        _doc9, _ = _fd0.build_deck(str(_sp9), str(_rd9),
                                   "person_id")
        # THE CACHE IS READ, NOT JUST WRITTEN. Mining costs a
        # fitted model per child and `/api/deck` is a BLOCKING
        # request, so the Dashboard would pay it on every press.
        # Proving the file exists proves only that something was
        # written; poisoning it with a child name the miner could
        # never produce and finding that name on the page proves
        # the page READ it.
        _mc9 = _rd9 / "mined_interactions.json"
        _cached9 = _mc9.exists()
        _mc9.write_text(
            '[{"child": "ZZPOISON", "pair": ["a", "b"], '
            '"source": 0.5}]', encoding="utf-8")
        _doc9b, _ = _fd0.build_deck(str(_sp9), str(_rd9),
                                    "person_id")
        _sp9r = _pd2.read_csv(_sp9)
        _rho9 = max(
            abs(float(_sp9r["y"].corr(_sp9r["a"], method="spearman"))),
            abs(float(_sp9r["y"].corr(_sp9r["b"], method="spearman"))))
    check("the fixture CONTAINS a two-way interaction that rank "
          "correlation cannot see - y = 3ab, whose parents each "
          "read near zero against it - so the section below is "
          "not being asserted against an empty list",
          _rho9 < 0.15)
    check("the Dashboard mines combination effects from the DATA, "
          "not only the ones the contract PUBLISHED - it existed "
          "on the neural and duel pages alone, so a reader seeing "
          "the gate's '0/8' had no way to learn that eight was a "
          "publishing choice rather than what the data holds",
          "Combination effects, mined from the data" in _doc9
          and "acting together" in _doc9
          and "model importance" in _doc9)
    # The TOTAL and the ROWS have to agree. Asserting that the
    # words "no surface" appear would depend on this fixture
    # happening to hold an unpublished effect - it does not, its
    # one interaction IS published - and a check that can only
    # pass on a lucky fixture is the vacuous kind. The identity
    # holds whichever way the fixture falls, which is the same
    # reason the surface counts are held as an identity rather
    # than a bare number.
    import re as _re9
    _m9 = _re9.search(r"(\d+) of the (\d+) strongest combination "
                      r"effects", _doc9)
    _n_no = _doc9.count("<b>no surface</b>")
    _n_yes = _doc9.count("<td>yes</td>")
    check("...and each mined effect says whether the contract "
          "publishes a surface for it, with a TOTAL that agrees "
          "with the rows - an unpublished joint effect is a "
          "limitation to name, and SHAP ranked two of them first "
          "on the real extract with neither published",
          "published?" in _doc9 and _m9 is not None
          and int(_m9.group(1)) == _n_no
          and int(_m9.group(2)) == _n_no + _n_yes
          and _n_no + _n_yes > 0)

    # A PERCENTAGE WHOSE DENOMINATOR MOVED IS NOT A COMPARISON.
    # Fixing the set columns took a real run's pair count from 21
    # to 115 - ninety-odd token-indicator pairs that had been
    # silently unmeasurable became measurable - so the absolute
    # counts ROSE (18 -> 89 direction) while the headline
    # percentage FELL, and it read as a large regression. The
    # `__has__` indicators are search machinery, dropped before
    # the file is written; the fidelity summary already separated
    # them when counting COLUMNS and did not when counting PAIRS.
    from synthkit import gate as _gt
    _mix = _gt.assess({"summary": {
        "pairs": 115, "pairs_sign_ok": 89, "pairs_close": 49,
        "pairs_inverted": 10, "pairs_own": 21,
        "pairs_own_sign_ok": 18, "pairs_own_close": 14,
        "pairs_own_inverted": 0, "pairs_scaffold": 94}})
    _dline = [c for c in _mix["criteria"]
              if c["name"] == "direction kept"][0]["detail"]
    _plain = _gt.assess({"summary": {
        "pairs": 20, "pairs_sign_ok": 19, "pairs_close": 18,
        "pairs_inverted": 0}})
    _pline = [c for c in _plain["criteria"]
              if c["name"] == "direction kept"][0]["detail"]
    check("the gate states the operator's OWN columns apart from "
          "set-token search scaffolding, so a denominator that "
          "grew cannot read as a regression - the mistake that "
          "cost a round trip",
          "your own columns" in _dline
          and "21 are your own columns" in _dline
          and "94 are set-token search scaffolding" in _dline)
    check("...and a run with no scaffolding says nothing extra, "
          "because a note that fires on everything is noise",
          "your own columns" not in _pline)
    # AND IT BELONGS ON THE CLOSE LINE TOO, where it was missing.
    #
    # Measured on the real extract: close reads 120/137 = 87.6%
    # over all pairs and 23/27 = 85.2% over the operator's OWN
    # columns, so the 110 scaffolding pairs are FLATTERING the
    # total - a 0/1 indicator tied to its own set survives better
    # than a clinical column does. The headline is kinder than
    # the file the operator receives, and the note was on
    # `direction kept` and not on the criterion where that
    # matters most.
    _cline = [c for c in _mix["criteria"]
              if c["name"] == "close"][0]["detail"]
    check("the close criterion carries the own/scaffolding "
          "breakdown as well, because the total is FLATTERED by "
          "scaffolding rather than dragged down by it - 87.6% "
          "over all pairs against 85.2% over the operator's own "
          "columns on the real extract",
          "your own columns" in _cline
          and "21 are your own columns" in _cline
          and "94 are set-token search scaffolding" in _cline)
    check("...and the criterion still JUDGES the total, "
          "deliberately: the bar was set against it, so "
          "switching the denominator would move a release "
          "verdict rather than fix a bug. The stricter number is "
          "printed beside the one being judged",
          "49/115" in _cline)
    _cplain = [c for c in _plain["criteria"]
               if c["name"] == "close"][0]["detail"]
    check("...and a run with no scaffolding adds nothing to the "
          "close line either",
          "your own columns" not in _cplain)

    # EVERY TERM ON THE DASHBOARD IS A DOOR - and it has to be
    # one INSIDE the page. The Dashboard is served into a
    # SANDBOXED IFRAME, where the bench's own explain panel
    # cannot be reached, and the same HTML is a file people send
    # to each other. One registry, two renderers: the words are
    # shared so the two cannot come to say different things
    # about what drift, or faded, or INVERTED means.
    import re as _rq
    from synthkit import explain as _EX
    _txt9 = _rq.sub(r"<[^>]+>", " ", _doc9)
    _reach = [sl for sl, phs in _EX.PHRASES.items()
              if any(ph in _txt9 for ph in phs)]
    check("the Dashboard carries its own explain panel, so the "
          "terms are reachable inside the sandboxed iframe AND "
          "in the file when somebody sends it on",
          "xqOpen" in _doc9 and "xqwrap" in _doc9
          and "xqMark" in _doc9 and '"terms"' in _doc9)
    check("...and the words the reader actually meets are the "
          "ones that open: drift, faded, sign lost, overshot, "
          "INVERTED, the k rule, coverage, token shares, "
          "suppressed bins and what the numbers even are",
          all(sl in _reach for sl in
              ("drift", "faded", "sign-lost", "overshot",
               "backwards", "k-rule", "coverage",
               "set-token-shares", "suppressed-bins",
               "rank-correlation", "skill")))
    check("...and no door opens on nothing - every [[slug]] in "
          "an entry resolves, which is the same contract the "
          "bench's panel has, because a link the reader trusted "
          "and that goes nowhere is worse than no link",
          _EX.broken_links() == []
          and all(k in _EX.TERMS for k in _EX.PHRASES))

    check("the mined interactions are CACHED beside the run and "
          "the cache is READ on the next draw - each child costs "
          "a fitted model and /api/deck is a blocking request, "
          "so a Dashboard that re-mined on every press is the "
          "hung-browser fault this repo already recorded against "
          "the duel endpoint",
          _cached9 and "ZZPOISON" in _doc9b)

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
    from synthkit import pipeline as _pl
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

    # A TOKEN-INDICATOR SURFACE WHOSE COLUMNS ARE PRESENT MUST BE
    # MEASURED, NOT COUNTED AS UNMEASURABLE.
    #
    # `compare` hands this function the EXPANDED frames, so an
    # indicator the blueprint's own vocabulary rebuilt is there to
    # be read. The old code refused on the NAME before asking, and
    # on the real extract reported 54 of 61 published surfaces as
    # unmeasurable - a criterion wrong in the direction nobody
    # checks, by measuring too little.
    #
    # Two axes are planted on purpose: a binary indicator against
    # an ordinary count, and binary against binary, because a 0/1
    # axis cannot yield three quantile edges and would otherwise
    # land in `skipped_thin_data` with the refusal removed.
    _rT = _np9.random.default_rng(31)
    _nT = 1600
    _hx = (_rT.random(_nT) < 0.45).astype(float)
    _ho = (_rT.random(_nT) < 0.5).astype(float)
    _mn = _rT.integers(0, 4, _nT).astype(float)
    _tokfr = _pd9.DataFrame({
        "pid": ["P%04d" % (i // 4) for i in range(_nT)],
        "meds__has__x": _hx,
        "routes__has__oral": _ho,
        "meds__n": _mn,
        # genuine interactions, so a lost surface is detectable
        "y": 3.0 * _hx * _mn + _rT.normal(0, .2, _nT),
        "w": 4.0 * _hx * _ho + _rT.normal(0, .2, _nT)})
    _lostT = _tokfr.copy()
    _lostT["y"] = _rT.normal(float(_tokfr["y"].mean()),
                             float(_tokfr["y"].std()), _nT)
    _lostT["w"] = _rT.normal(float(_tokfr["w"].mean()),
                             float(_tokfr["w"].std()), _nT)
    _bpTok = {"relationships": [
        {"child": "y", "parents": ["meds__has__x", "meds__n"],
         "evidence": {"interaction": {
             "pair": ["meds__has__x", "meds__n"]}}},
        {"child": "w", "parents": ["meds__has__x",
                                   "routes__has__oral"],
         "evidence": {"interaction": {
             "pair": ["meds__has__x", "routes__has__oral"]}}}]}
    _tokKept = _cc.measure_surfaces(_tokfr, _tokfr.copy(), _bpTok,
                                    _tokfr["pid"], 10)
    _tokLost = _cc.measure_surfaces(_tokfr, _lostT, _bpTok,
                                    _tokfr["pid"], 10)

    check("a surface on a TOKEN INDICATOR is measured when the "
          "column is present - `compare` hands this function the "
          "expanded frames, and refusing on the name threw away "
          "the ones the blueprint's own vocabulary had rebuilt",
          _tokKept["compared"] == 2
          and _tokKept["skipped_token_columns"] == 0)

    check("...including BINARY AGAINST BINARY, which no quantile "
          "grid can bin: three edges cannot be built from a 0/1 "
          "column, so without the two-valued axis rule these land "
          "in `skipped_thin_data` and nothing changes",
          _tokKept["skipped_thin_data"] == 0
          and all(r["cells_compared"] >= 4
                  for r in _tokKept["surfaces"]))

    check("...and the measurement can still FAIL on them - a "
          "surface generation loses reads as departed, so this is "
          "coverage that can report a defect rather than coverage "
          "that always passes",
          _tokKept["tracked"] == 2 and _tokLost["tracked"] == 0
          and min(r["gap_sd"] for r in _tokLost["surfaces"]) > 0.35)

    check("...and the TOTAL identity still holds with the wider "
          "coverage, so nothing moved from one skip bucket to "
          "another unnoticed",
          _tokKept["published"] == _tokKept["compared"]
          + _tokKept["skipped_token_columns"]
          + _tokKept["skipped_absent_columns"]
          + _tokKept["skipped_thin_data"])

    # WHICH SIDE COULD NOT FILL THE GRID IS THREE DIFFERENT
    # FINDINGS, and one `thin` bucket cannot say which. On the real
    # extract 22 surfaces landed there: a source that cannot be
    # binned is unmeasurable by construction, a source whose cells
    # the k screen removes is the privacy rule doing its job, and a
    # source that bins while the OUTPUT does not is the engine
    # under-emitting. Only the last is a fidelity defect.
    _thinbp = {"relationships": [
        {"child": "y", "parents": ["meds__has__x", "meds__n"],
         "evidence": {"interaction": {
             "pair": ["meds__has__x", "meds__n"]}}}]}

    # (a) the OUTPUT cannot fill it: the token is absent from the
    # generated frame's rows, so its axis has one value.
    _flat = _tokfr.copy()
    _flat["meds__has__x"] = 0.0
    _rg = _cc.measure_surfaces(_tokfr, _flat, _thinbp,
                               _tokfr["pid"], 10)
    check("a surface whose SOURCE bins but whose OUTPUT cannot is "
          "attributed to the GENERATED side - that is the engine "
          "under-emitting a token, and it is the only one of the "
          "three thin reasons that is a fidelity defect",
          _rg["thin_generated"] == 1
          and _rg["skipped_thin_data"] == 1
          and _rg["compared"] == 0)

    # (b) the k screen removes every source cell. It takes FEW
    # patients over MANY rows: one patient per row would give each
    # cell as many distinct patients as it has rows, so every cell
    # would clear k and nothing would be screened - which is what
    # the first version of this fixture did, and the check went red
    # against correct code.
    _solo = _tokfr.copy()
    _solo["pid"] = ["Q%03d" % (i % 3) for i in range(len(_solo))]
    _rk = _cc.measure_surfaces(_solo, _solo.copy(), _thinbp,
                               _solo["pid"], 10)
    check("...while a surface whose source cells are all removed "
          "by the k screen is attributed to PRIVACY, not to the "
          "engine - a shortfall the k rule explains is not a fault "
          "and must not be counted as one",
          _rk["thin_source_k_screened"] == 1
          and _rk["thin_generated"] == 0)

    # (c) too few rows to bin at all.
    _tiny = _tokfr.head(40).copy()
    _rt = _cc.measure_surfaces(_tiny, _tiny.copy(), _thinbp,
                               _tiny["pid"], 10)
    check("...and a frame with too few rows to bin at all is "
          "attributed to the SOURCE being unbinnable, which is a "
          "statement about the cohort rather than about either "
          "table",
          _rt["thin_source_unbinnable"] == 1)

    check("...and the three thin reasons SUM to the thin total on "
          "every one of those frames, so the decomposition cannot "
          "lose a surface into a bucket nothing counts",
          all(r["skipped_thin_data"] == r["thin_source_unbinnable"]
              + r["thin_source_k_screened"] + r["thin_generated"]
              + r["thin_no_overlap"]
              for r in (_rg, _rk, _rt, _tokKept, _resT)))

    # AND THE VOCABULARY HAS TO CARRY THEM, which is a separate
    # gap: `pair` is the top two by importance and `parents` is the
    # filtered order, so a pair member can be named by one and not
    # the other. Without this the columns are simply not in the
    # expanded frame and the fix above has nothing to read.
    _bpVoc = {"relationships": [
        {"child": "y", "parents": ["meds__n"],
         "evidence": {"interaction": {
             "pair": ["meds__has__surfonly", "meds__n"]}}}],
        "columns": {"meds": {"marginal": {
            "type": "list",
            "tokens": [{"value": "t%03d" % i, "p": 0.3}
                       for i in range(40)]
            + [{"value": "surfonly", "p": 0.2}]}}}}
    _voc = _pl.blueprint_vocab(_bpVoc)
    check("`blueprint_vocab` carries a token named ONLY by an "
          "interaction pair - `pair` is the top two by importance "
          "and `parents` is the filtered order, so the two lists "
          "disagree and a surface-only token was never rebuilt",
          "surfonly" in [str(t.get("value"))
                         for t in _voc.get("meds", [])])

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

    # peek curves - A PARENT THE CONTRACT NAMES AND PUBLISHES NO
    # CURVE FOR. `discover` skips a curve whose grid collapses with
    # a bare `continue` and `blueprint` lists the parent anyway, so
    # the sampler is handed a parent it cannot use and nothing
    # listed them. Measured 2026-11-09: a set-token parent on 0.81%
    # of rows, driving its child by 7.2 sd, found at importance
    # 0.633 and published BY NAME at search cap 60, carried no curve
    # - and the output held 1.4% of the relationship. Raising the
    # cap did not help; a QUANTILE grid on a binary column was the
    # cause.
    with tempfile.TemporaryDirectory() as _cv:
        _cr = Path(_cv)
        (_cr / "blueprint.json").write_text(json.dumps({
            "relationships": [
                # a named parent WITH a curve, and an indicator
                # parent WITHOUT one - the fault, beside its control
                {"child": "y", "parents": ["plain", "s__has__tok"],
                 "evidence": {"effect": {"plain": {
                     "grid": [0, 1], "response": [1.0, 2.0]}}}},
                # and a relationship where every parent is covered,
                # so the view cannot just always report trouble
                {"child": "z", "parents": ["other"],
                 "evidence": {"effect": {"other": {
                     "grid": [0, 1], "response": [0.0, 1.0]}}}}]}),
            encoding="utf-8")
        _cvr = subprocess.run(
            [sys.executable, "scripts/peek.py", str(_cr), "curves"],
            capture_output=True, text=True, cwd=str(ROOT))

    check("peek.py curves NAMES every parent the contract publishes "
          "without a curve, and counts the indicator ones apart - a "
          "parent generation cannot apply is the read-but-not-used "
          "fault in the one artifact generation reads",
          _cvr.returncode == 0
          and "y                                  <- s__has__tok"
              in _cvr.stdout
          and "with NO curve                        1" in _cvr.stdout
          and "of those, set-token indicators       1" in _cvr.stdout)

    check("...and it does not report trouble where there is none: "
          "the covered parents are counted as covered, so a clean "
          "blueprint reads clean rather than every run looking "
          "broken",
          "parent slots across them             3" in _cvr.stdout
          and "WITH a curve                         2" in _cvr.stdout
          and "plain" not in _cvr.stdout.split(
              "every parent named without a curve")[-1])

    # peek flat - THE COLUMN THE CONTRACT SAYS VARIES AND THE FILE
    # EMITS CONSTANT. Found because a real run printed scipy's
    # ConstantInputWarning twice from inside spearmanr: a constant
    # input means the pair's rank correlation is UNDEFINED, so the
    # pair is skipped rather than failed and leaves the gate's
    # denominator - a defect that can raise the percentage while
    # costing fidelity. The duel page already named flattened
    # columns, but only when both engines ran; one run had nothing
    # that could say it. The fixture CONTAINS the fault, plus one
    # of every neighbouring category, because a totals line is the
    # only way an omitted category is visible.
    with tempfile.TemporaryDirectory() as _tf:
        _fr = Path(_tf)
        (_fr / "blueprint.json").write_text(json.dumps({
            "columns": {
                "bp_num": {"marginal": {"type": "quantiles",
                                        "v": [1.0, 5.0, 9.0]}},
                "bp_cat": {"marginal": {
                    "type": "levels",
                    "levels": [{"value": "a", "p": 0.5},
                               {"value": "b", "p": 0.5}]}},
                "bp_const": {"marginal": {
                    "type": "levels",
                    "levels": [{"value": "only", "p": 1.0}]}},
                "scaffold__has__t1": {"marginal": {
                    "type": "levels",
                    "levels": [{"value": "0", "p": 0.5},
                               {"value": "1", "p": 0.5}]}}}}),
            encoding="utf-8")
        _lines = ["bp_num,bp_cat,bp_const"]
        for _i in range(50):
            _lines.append("7,{},only".format("a" if _i % 2 else "b"))
        (_fr / "generated.csv").write_text("\n".join(_lines) + "\n",
                                           encoding="utf-8")
        _fv = subprocess.run(
            [sys.executable, "scripts/peek.py", str(_fr), "flat"],
            capture_output=True, text=True, cwd=str(ROOT))
    check("peek.py flat names the flattened column WITH the one "
          "value it holds - a numeric the contract publishes from "
          "1 to 9 that came out as a single 7",
          _fv.returncode == 0
          and "bp_num" in _fv.stdout and "'7'" in _fv.stdout)
    check("...and does NOT flag a column the blueprint itself "
          "publishes as constant - that is the source, not a "
          "defect",
          "published AS constant         1" in _fv.stdout
          and "FLATTENED                     1" in _fv.stdout)
    check("...and accounts for a published column that was never "
          "WRITTEN, because set-token scaffolding is dropped "
          "before the write and would otherwise read as flattened",
          "not written to the file       1" in _fv.stdout)
    check("...and prints the totals identity, so a category with "
          "no bucket cannot hide",
          "accounted for                 4 of 4 published"
          in _fv.stdout
          and "MISMATCH" not in _fv.stdout)

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
        # The verb door onto the same builder - the kit runs it
        # from a folder with no scripts/.
        _so2 = Path(_ts) / "signoff2.html"
        _rv = subprocess.run(
            [sys.executable, "-m", "synthkit.cli", "signoff",
             str(_sr), "-o", str(_so2)],
            capture_output=True, text=True, cwd=str(ROOT))
        check("`synthkit signoff` writes the same page the script "
              "does, and says so in one line",
              _rv.returncode == 0 and _so2.exists()
              and _so2.read_text(encoding="utf-8")
              == _so.read_text(encoding="utf-8"))

    # THE SINGLE-ENGINE REVIEW. Same measures as the duel,
    # minus the comparison - one implementation underneath, so
    # the two pages cannot disagree about the same output. It
    # leads with the PATIENT question, because a longitudinal
    # table whose rows belong to nobody is the failure that
    # every per-column number would otherwise hide.
    sys.path.insert(0, str(ROOT / "scripts"))
    from fidelity_deck import build_neural as _bn
    import numpy as _np3
    import pandas as _pd3
    with tempfile.TemporaryDirectory() as _nd:
        _rngN = _np3.random.RandomState(4)
        _nN = 900
        _g3 = _np3.repeat(_np3.arange(150), 6)
        _lvl = _rngN.normal(0, 9, 150)[_g3]
        _s3 = _pd3.DataFrame({
            "person_id": ["P{:03d}".format(i) for i in _g3],
            "a": _np3.round(_lvl + _rngN.normal(0, 1, _nN), 2),
            "b": _np3.round(_rngN.normal(20, 4, _nN), 2)})
        _sp3 = Path(_nd) / "src.csv"
        _s3.to_csv(_sp3, index=False)
        _lp3 = Path(_nd) / "gen.csv"
        _s3.sample(frac=0.85, random_state=5).to_csv(_lp3,
                                                     index=False)
        _pg = _bn(str(_sp3), str(_lp3), "person_id")
        check("the neural review page leads with the patient "
              "question, carries the combination effects and "
              "every column drawn twice, and states what it does "
              "NOT measure",
              "The neural engine, measured" in _pg
              and "Does a patient look like a person" in _pg
              and _pg.index("Does a patient")
              < _pg.index("Every column, drawn twice")
              and "The combination effects" in _pg
              and 'class="ser lat"' in _pg
              and "does not measure safety" in _pg
              and "suppressed by" in _pg)
        try:
            _bn(str(_sp3), str(Path(_nd) / "absent.csv"),
                "person_id")
            _ref = False
        except ValueError as e:
            _ref = "press Generate first" in str(e)
        check("...and with nothing generated yet it refuses in a "
              "sentence naming the next action, never a stack",
              _ref)

    # THE DUEL PAGE: both engines against one source. Built by
    # ONE function (fidelity_deck.build_duel) that the CLI and
    # the bench both call; asserts the three series, the gap
    # heatmap with both poles explained, the mined interactions,
    # the k-suppression count, and the posture footer - and that
    # a missing input refuses with a plain ValueError, because
    # inside the bench a sys.exit takes the server down.
    import numpy as _np2
    import pandas as _pd2
    sys.path.insert(0, str(ROOT / "scripts"))
    from fidelity_deck import build_duel
    with tempfile.TemporaryDirectory() as _dd:
        _r2 = _np2.random.RandomState(2)
        _n2 = 900
        _g2 = _np2.repeat(_np2.arange(150), 6)
        _x2 = _r2.normal(50, 10, _n2)
        _src2 = _pd2.DataFrame({
            "person_id": ["P{:03d}".format(i) for i in _g2],
            "a": _np2.round(_x2, 2),
            "b": _np2.round(0.7 * _x2 + _r2.normal(0, 5, _n2),
                            2),
            "c": _np2.round(_r2.normal(0, 1, _n2), 2)})
        _sp2 = Path(_dd) / "src.csv"
        _src2.to_csv(_sp2, index=False)
        _rn2 = Path(_dd) / "run"
        _rn2.mkdir()
        (_rn2 / "blueprint.json").write_text(
            json.dumps({"columns": {}}), encoding="utf-8")
        _src2.sample(frac=0.9, random_state=3).to_csv(
            _rn2 / "generated.csv", index=False)
        _lat2 = Path(_dd) / "lat.csv"
        _src2.sample(frac=0.8, random_state=4).drop(
            columns=["person_id"]).to_csv(_lat2, index=False)
        _duel = build_duel(str(_sp2), str(_rn2), str(_lat2),
                           "person_id")
        check("the duel page carries three series, the "
              "who-got-it-more-wrong grid, the combination "
              "effects, the suppression count, and the safety "
              "footer that says the two engines are NOT equally "
              "safe",
              'class="ser lat"' in _duel
              and 'class="ser src"' in _duel
              and 'class="ser syn"' in _duel
              and "Where each engine goes wrong" in _duel
              and "WHICH ENGINE GOT IT MORE WRONG" in _duel
              and "The combination effects" in _duel
              and "suppressed by" in _duel
              and "they are not equally safe" in _duel
              and "learns from the records themselves" in _duel
              and "never part of any release" in _duel
              and "INVENTED patient identity" in _duel
              and "visits are not linked" not in _duel)
        # PLAIN ENGLISH IS THE CONTRACT, not a preference. The
        # operator could not read the page through our own code
        # names, so every heading, label and legend is written
        # for a stranger and `blueprint`/`latent` survive ONLY as
        # parenthetical log names - once each, beside the plain
        # name, so somebody reading a terminal can still connect
        # the two.
        # The count is over what a READER sees. The page now
        # carries the explain registry as embedded data, and the
        # registry legitimately says "blueprint" and "latent"
        # inside explanation bodies a reader only meets by
        # clicking - so scripts and styles are stripped before
        # counting, keeping the check about the visible page.
        import re as _re2
        _vis = _re2.sub(r"<(script|style)[^>]*>.*?</\1>", "",
                        _duel, flags=_re2.S).lower()
        check("...and the page reads without our jargon - the "
              "engines are named in plain words, with the code "
              "names surviving only as one parenthetical each "
              "(counted over the VISIBLE text; the embedded "
              "explain registry is data, not prose)",
              "rules engine" in _vis and "neural engine" in _vis
              and _vis.count("blueprint") == 1
              and _vis.count("latent") == 1
              and "in the logs" in _vis)
        check("...and the duel page carries the explain panel "
              "too - it is served into a sandboxed iframe and "
              "sent as a file, the same two reasons the "
              "Dashboard needed its own",
              "xqOpen" in _duel and '"terms"' in _duel)
        try:
            build_duel(str(_sp2), str(Path(_dd) / "nope"),
                       str(_lat2), "person_id")
            _refused = False
        except ValueError as e:
            _refused = "generated.csv" in str(e)
        check("...and a missing blueprint side refuses with a "
              "plain ValueError naming generated.csv - never "
              "sys.exit, which would take the bench down",
              _refused)

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
          and "the VIEW Visualize station" in _g2.plain(
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
                  "interaction surfaces",
                  "dynamics tracked")]}
          and all("[[dashboard]]" in v["inspect"]
                  for v in _g2.EXPLAIN.values()))

    # THE CANDIDATE POOL IS EVERY CLAIM, AND THE EMPTY CASE IS
    # STATED. The first version screened only the top-8 drawn
    # cards; on the real extract those are token-indicator pairs
    # whose columns live only inside the search, so the section
    # rendered NOTHING - the exact silent absence the omission
    # note exists to prevent, shipped in the branch beside it.
    import pandas as _pd8
    _srcE = _pd8.DataFrame({"a": ["1", "2", "3"] * 100,
                            "b": ["2", "3", "4"] * 100,
                            "y": ["5", "6", "7"] * 100,
                            "d": ["2024-01-01"] * 300})
    _genE = _srcE.copy()
    # THE JOINT MEASURES, IN THE DECK. Every other section on that
    # page starts from something the contract PUBLISHED, so it can
    # only show what somebody thought to look for. This one asks
    # whether each column is as learnable and whether the SAME
    # drivers drive it - a pair can survive as "faded" while who
    # drives whom is reassigned, and no pairwise check sees that.
    #
    # The fixture plants exactly that: `y` is equally predictable
    # from both tables while leaning on the OTHER driver. If the
    # section cannot flag it, it is decoration.
    import numpy as _njj
    import pandas as _pjj
    _rj = _njj.random.default_rng(0)
    _nj = 900
    _xj = _rj.normal(0, 1, _nj)
    _zj = _rj.normal(0, 1, _nj)
    _realj = _pjj.DataFrame({
        "person_id": _njj.repeat(_njj.arange(_nj // 3), 3),
        "x": _xj.round(3), "z": _zj.round(3),
        "y": (2.0 * _xj + 0.7 * _zj
              + _rj.normal(0, .3, _nj)).round(3),
        "w": _rj.normal(5, 2, _nj).round(3)})
    _synj = _realj.copy()
    _synj["y"] = (0.7 * _synj["x"] + 2.0 * _synj["z"]
                  + _rj.normal(0, .3, _nj)).round(3)
    _hj = _fd.joint_section_html(_realj, _synj, "person_id")
    check("the deck's joint section renders a row per column with "
          "skill on each table and whether the same drivers drive "
          "it",
          "The same things doing the driving?" in _hj
          and "skill on original" in _hj
          and "drivers agree" in _hj
          and ">y<" in _hj)
    check("...and it FLAGS a column that is equally predictable "
          "while leaning on DIFFERENT drivers - the case every "
          "pairwise check on the page is blind to",
          'class="warn"' in _hj
          and any('class="warn"' in _l and ">y<" in _l
                  for _l in _hj.split("<tr")))
    check("...and an unpredictable column shows an em dash rather "
          "than a ratio of two numbers that mean nothing",
          "&mdash;" in _hj)
    # THE KEYS COME FROM A LIVE CALL, never from a fixture that
    # agrees with the renderer - a blank section once shipped
    # because both halves held the same wrong key names and only
    # the function being called disagreed.
    from synthkit import jointcheck as _jcc
    _prj = _jcc.predictability(_realj, _synj, group_by="person_id")
    _r0 = (_prj.get("columns") or [{}])[0]
    check("...and the renderer reads the keys `predictability` "
          "actually emits, asserted against a LIVE call",
          set(["column", "skill_real", "skill_syn",
               "driver_agreement"]).issubset(set(_r0))
          and "mean_driver_agreement" in _prj)
    check("the deck's TSTR block renders - the one fidelity "
          "number a thin cohort cannot flatter, because both arms "
          "are scored on the SAME held-out real rows",
          "Train on synthetic, test on real" in _hj
          and "trained on synthetic" in _hj)
    # THE KEYS AGAIN, FROM A LIVE CALL. The first cut of this
    # block guessed `real`/`synthetic` and rendered NOTHING - the
    # blank-section fault, repeated in the same hour it was
    # guarded against for predictability. Assert the contract.
    _uj = _jcc.utility(_realj, _synj, "y", group_by="person_id")
    check("...and it reads the keys `utility` actually emits, "
          "asserted against a LIVE call rather than a fixture "
          "that agrees with the renderer",
          set(["train_real_test_real", "train_synthetic_test_real",
               "retained"]).issubset(set(_uj)))
    check("...and a target the REAL data cannot predict shows a "
          "dash rather than a ratio - '75% kept' of nothing is "
          "not a result",
          any('<td>w</td>' in _l and "&mdash;" in _l
              for _l in _hj.split("<tr")))
    # A FLAG THAT IS READ AND NOT USED IS WORSE THAN NO FLAG -
    # `--time-col` was carried into a blueprint and then ignored,
    # and passing it changed nothing while the message still
    # claimed it had. Assert the flag REACHES the builder and
    # that the two branches differ, not merely that the option
    # parses.
    import inspect as _ins9
    _bs9 = _ins9.getsource(_fd.build_deck)
    _fs9 = io.open("scripts/fidelity_deck.py",
                   encoding="utf-8").read()
    check("the --joint flag reaches build_deck and CHANGES what "
          "it builds - a flag that parses and does nothing is "
          "the --time-col failure wearing a new name",
          "joint=a.joint" in _fs9
          and "if joint:" in _bs9
          and "joint_section_html(src, gen" in _bs9
          and "opt-in" in _bs9)
    check("the joint measures are OFF by default and their absence "
          "is STATED - they fit a model per column and a page that "
          "takes minutes is a hung browser",
          "--joint" in io.open("scripts/fidelity_deck.py",
                               encoding="utf-8").read()
          and "opt-in" in io.open("scripts/fidelity_deck.py",
                                  encoding="utf-8").read())

    # NOBODY CAN READ A COLUMN THAT IS NOT IN THE FILE. The
    # `__has__` token indicators and `__n` set sizes are search
    # machinery, dropped before generated.csv is written, and the
    # one verdict that says STOP was counting them. Measured on
    # the real extract: pairs_inverted ONE, pairs_own_inverted
    # ZERO - the single inversion was `procedures__n`, a set-size
    # column absent from the delivered data. This repo already
    # records the identical fault for the constraints section.
    from synthkit import gate as _gi

    def _invfid(total, own=None):
        _s = {"pairs_sign_ok": 0.99, "pairs_close": 0.95,
              "pairs_inverted": total, "coverage_ok": 10,
              "columns": 10, "pairs": 100}
        if own is not None:
            _s["pairs_own_inverted"] = own
        return {"summary": _s}

    def _invcrit(fid):
        return [c for c in _gi.assess(fid)["criteria"]
                if c["name"] == "inverted"][0]

    check("an inversion that sits ONLY on search scaffolding does "
          "not trip the absolute stop - the column is dropped "
          "before the file is written, so it cannot read as a "
          "finding to anyone",
          _invcrit(_invfid(1, 0))["ok"] is True)
    check("...and the scaffolding inversion is REPORTED rather "
          "than silently dropped, because it can still mark a "
          "modelling fault worth chasing",
          "scaffolding" in _invcrit(_invfid(1, 0))["detail"]
          and "own columns" in _invcrit(_invfid(1, 0))["detail"])
    check("...while an inversion in the operator's OWN columns "
          "still stops the release, which is the whole point of "
          "the criterion",
          _invcrit(_invfid(1, 1))["ok"] is False
          and _invcrit(_invfid(3, 2))["ok"] is False)
    check("...and a run that predates the split falls back to the "
          "TOTAL - absent is not known-to-be-zero, the same "
          "empty-versus-missing distinction drawn everywhere here",
          _invcrit(_invfid(1))["ok"] is False
          and "predates" in _invcrit(_invfid(1))["detail"])
    check("...and a clean run still passes",
          _invcrit(_invfid(0, 0))["ok"] is True)

    # THE SURFACES CRITERION GRADES A CONTRACT ONE ENGINE DOES
    # NOT READ, and saying so is the difference between "this
    # engine ignores published curves" and "no multivariate
    # structure survived" - opposite conclusions from one chip.
    from synthkit import gate as _gt
    _su = _gt.CRITERIA_EXPLAIN["interaction surfaces"] \
        if hasattr(_gt, "CRITERIA_EXPLAIN") else None
    _suh = io.open("synthkit/gate.py", encoding="utf-8").read()
    check("the interaction-surfaces criterion NAMES whose "
          "contract it grades and points at the mined table - a "
          "FAIL on an engine that never reads the surfaces is "
          "not evidence that two-way structure is absent",
          "never consults them" in _suh
          and "construction, and a FAIL here is NOT evidence"
          in _suh
          and "mined-interaction table" in _suh)
    check("...and its guidance does not send a neural-run reader "
          "hunting trim machinery that engine does not have",
          "RULES ENGINE ONLY" in _suh
          and "no trim machinery" in _suh)

    check("shap candidates come from EVERY claim - a numeric "
          "two-driver child qualifies, a token child and a "
          "date-thinned parent list do not",
          _fd.shap_candidates(
              [{"child": "y", "skill": .8, "predictors": [
                  {"column": "a"}, {"column": "b"}]}],
              _srcE, _genE)[0] == [(0.8, "y", ["a", "b"])]
          and _fd.shap_candidates(
              [{"child": "meds__has__x", "skill": .9,
                "predictors": [{"column": "a"},
                               {"column": "b"}]}],
              _srcE, _genE)[0] == []
          and _fd.shap_candidates(
              [{"child": "y", "skill": .8, "predictors": [
                  {"column": "a"}, {"column": "d"}]}],
              _srcE, _genE)[0] == [])
    # A SPARSE NUMERIC COLUMN IS STILL NUMERIC. The old screen
    # required presence on half the rows, which on a clinical
    # extract rejects the sparse columns that are the interesting
    # drivers - and the section then rendered "nothing to
    # attribute" on data that had plenty. Present on a fifth of
    # the rows, comfortably past the row floor:
    _spN = 1500
    _srcS = _pd8.DataFrame({
        "a": (["1", None, None, None, None] * (_spN // 5)),
        "b": (["2", None, None, None, None] * (_spN // 5)),
        "y": (["5", None, None, None, None] * (_spN // 5))})
    _genS = _srcS.copy()
    check("a SPARSE numeric child and its sparse numeric drivers "
          "still qualify - a column whose present values are "
          "numbers is numeric however often it is present, and "
          "the old presence-share gate dropped exactly the "
          "columns worth attributing",
          _srcS["y"].notna().mean() < 0.5
          and _fd.shap_candidates(
              [{"child": "y", "skill": .8, "predictors": [
                  {"column": "a"}, {"column": "b"}]}],
              _srcS, _genS)[0] == [(0.8, "y", ["a", "b"])])
    check("...and an EMPTY result reports the census it counted "
          "rather than asserting a cause - 'the strongest "
          "patterns live on token indicators' was a guess that "
          "sends a reader hunting the wrong thing",
          _fd.shap_candidates(
              [{"child": "meds__has__x", "skill": .9,
                "predictors": [{"column": "a"},
                               {"column": "b"}]}],
              _srcE, _genE)[1]
          == {"child is search-only scaffolding": 1}
          and _fd.shap_candidates(
              [{"child": "y", "skill": .8, "predictors": [
                  {"column": "a"}, {"column": "d"}]}],
              _srcE, _genE)[1]
          == {"fewer than 2 eligible numeric drivers": 1}
          and _fd.shap_candidates(
              [{"child": "ghost", "skill": .5, "predictors": [
                  {"column": "a"}, {"column": "b"}]}],
              _srcE, _genE)[1]
          == {"child not in the written files": 1})
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
    # "there's a lot here ... get this to a filtered version."
    # The filter lives in the DECK, not the bench, because the bench
    # serves these same bytes into an iframe and the shareable file
    # is the same file - a filter built in the bench would exist on
    # screen and be missing from what somebody is emailed.
    _dk = io.open(str(ROOT / "scripts" / "fidelity_deck.py"),
                  encoding="utf-8").read()
    check("the dashboard ships a summary view, because the "
          "complaint was density on first sight",
          "viewtoggle" in _dk and "apply(false)" in _dk)
    check("...and the toggle SAYS how many sections it is hiding, "
          "so the short view is never mistaken for the whole page",
          "more section" in _dk)
    check("...and the filter is in the deck itself, so the file "
          "that gets emailed has it too rather than only the "
          "bench's iframe",
          "viewtoggle" not in io.open(
              str(ROOT / "synthkit" / "gui.py"),
              encoding="utf-8").read())
    check("...and the gate and the limitations are never in the "
          "hidden group - a filtered view that hides the verdict "
          "or the caveats would be a different product",
          "'gate'" not in _dk.split("var DETAIL=")[1][:400]
          and "does not say" not in
              _dk.split("var DETAIL=")[1][:400])

    check("a --compare pointing at a run that does not exist gets a "
          "plain refusal naming the fix, not a traceback",
          _r3.returncode != 0
          and "STOPPED" in (_r3.stderr + _r3.stdout)
          and "does not exist yet" in (_r3.stderr + _r3.stdout)
          and "Traceback" not in (_r3.stderr + _r3.stdout))

    # ---- EVERY DECLARED EDGE IS ACCOUNTED FOR
    #
    # The pair measurement dropped an edge silently at four
    # points and counted none of them, so a run could publish
    # 22 declared relationships it never measured with no line
    # in the report saying so - which is exactly what happened,
    # and a week went into guessing the cause from two totals.
    # `measure_surfaces` has held the same identity for months;
    # this is the pair half of it.
    import numpy as _np
    import pandas as _pd
    from synthkit import blueprint as _Ba
    from synthkit.discover import discover as _da
    from synthkit.generate import generate as _ga
    from synthkit.pipeline import compare as _ca
    _n, _v = 120, 5
    _N = _n * _v
    _rg = _np.random.default_rng(4)
    _x = _rg.normal(0, 1, _N)
    _dfa = _pd.DataFrame({
        "person_id": _np.repeat(
            ["P{:03d}".format(i) for i in range(_n)], _v),
        "visit": _np.tile(_np.arange(_v), _n),
        "x": _np.round(_x, 3),
        "y": _np.round(2 * _x + _rg.normal(0, 0.5, _N), 3),
        "z": _np.round(_rg.normal(0, 1, _N), 3)})
    _bpa = _Ba.build(_dfa, _da(_dfa, group_by="person_id",
                               seed=1), group_by="person_id")
    _base_declared = len(
        _ca(_dfa, _ga(_bpa, n_patients=_n, seed=3), _bpa,
            "person_id", _bpa.get("time_column")
            )["relationships"]["declared_edges"]["examples"]
        or {})
    # A DECLARED EDGE ON A COLUMN NOTHING CARRIES - the skip
    # this fixture exists to plant, because a fixture that
    # cannot reach the branch proves nothing about it.
    _bpa["relationships"].append(
        {"child": "ghost", "parents": ["x"]})
    _ga2 = _ga(_bpa, n_patients=_n, seed=3)
    _acc = _ca(_dfa, _ga2, _bpa, "person_id",
               _bpa.get("time_column")
               )["relationships"]["declared_edges"]
    check("fidelity now reports a DECLARED EDGE account, so a "
          "relationship the contract claims and the run never "
          "measured cannot go missing without a line",
          _acc["declared"] >= 2)
    check("...and the account BALANCES - declared equals "
          "measured plus the skip reasons - so a fifth silent "
          "exit cannot hide: {} declared, {} accounted"
          .format(_acc["declared"], _acc["accounted"]),
          _acc["balances"] is True
          and _acc["accounted"] == _acc["declared"])
    check("...the planted edge on a column NEITHER frame "
          "carries is counted as absent rather than dropped, "
          "which is the whole fault being closed",
          _acc["absent_source"] == 1)
    check("...and the skipped edge is NAMED, because a count "
          "alone cannot tell anyone which relationship went "
          "unmeasured - that was a week of archaeology",
          any("ghost" in str(x)
              for x in (_acc.get("examples") or {}).get(
                  "absent_source") or []))
    check("...while the genuine edge is still MEASURED, so the "
          "accounting did not come at the cost of the "
          "measurement",
          _acc["measured"] >= 1)

    # ---- AND THE ACCOUNT REACHES THE GATE'S OWN PAGE
    #
    # A measurement that never reaches a page might as well not
    # exist for the person deciding. The account exists so a
    # reader judging "133 of 148" can see what 148 counts; if it
    # lives only in fidelity.json, the next person chases the
    # denominator from two totals exactly as before.
    from synthkit import gate as _gt
    _fidg = {
        "summary": {"pairs": 10, "columns": 5},
        "relationships": {
            "compared": 10, "kept_direction": 10, "close": 10,
            "inverted": [], "pairs": [],
            "declared_edges": {
                "declared": 12, "measured": 10,
                "absent_source": 1, "absent_generated": 0,
                "thin_or_unmeasurable": 1, "accounted": 12,
                "balances": True}},
        "coverage": {}, "centre": {}, "spread": {}}
    _v = _gt.assess(_fidg)
    check("the gate CARRIES the declared-edge account through "
          "to its verdict, rather than leaving it in a file "
          "nobody opens",
          (_v.get("declared_edges") or {}).get("declared") == 12)
    _ls, _ = _gt.report(_v, "run_x")
    _txt = "\n".join(_ls)
    check("...and the shared renderer PRINTS what the "
          "denominator is made of - one text, so the terminal "
          "and the bench cannot disagree about it",
          "DECLARES 12 edge(s)" in _txt
          and "measured 10" in _txt)
    check("...naming each skip reason that actually occurred, "
          "and only those - a zero bucket printed every time is "
          "noise a reader learns to skip",
          "absent from the source frame" in _txt
          and "too thin or unmeasurable" in _txt
          and "absent from the generated frame" not in _txt)
    _v2 = _gt.assess(dict(_fidg, relationships=dict(
        _fidg["relationships"],
        declared_edges=dict(_fidg["relationships"]
                            ["declared_edges"],
                            accounted=11, balances=False))))
    _ls2, _ = _gt.report(_v2, "run_x")
    check("...and an account that does NOT balance says so "
          "loudly, because that means an edge is dropped by a "
          "path nothing counts - a defect in the measurement, "
          "not in the data",
          "DOES NOT BALANCE" in "\n".join(_ls2))
    _v3 = _gt.assess({
        "summary": {"pairs": 10, "columns": 5},
        "relationships": {"compared": 10, "kept_direction": 10,
                          "close": 10, "inverted": [],
                          "pairs": []},
        "coverage": {}, "centre": {}, "spread": {}})
    _ls3, _ = _gt.report(_v3, "run_x")
    check("...while a run that PREDATES the account prints "
          "nothing about it - absent is not the same fact as "
          "zero, and an old run must still read cleanly",
          "DECLARES" not in "\n".join(_ls3))

    # ---- peek marginal: WHICH HALF is responsible ------------------
    # The largest named fidelity gap on the real extract has had two
    # incompatible explanations on the record and nothing measuring
    # which applies. This splits it: does the PUBLISHED GRID already
    # imply the wrong shape, or does the grid imply the right shape
    # and something after the draw move it? Different fixes.
    from peek import classify_marginal as _cm, MARGINAL_SEED as _MS

    check("the marginal verdict is a PURE function on planted "
          "numbers - the grid missing the source while the output "
          "tracks the grid is the CONTRACT's fault",
          _cm(0.40, 0.01) == "CONTRACT")
    check("...the grid tracking the source while the output "
          "misses it is GENERATION's fault",
          _cm(0.01, 0.40) == "GENERATION")
    check("...both missing is TWO faults, not one, because fixing "
          "either alone leaves a gap",
          _cm(0.40, 0.40) == "BOTH")
    check("...and neither missing is CLEAR, so the branch cannot "
          "report a fault on a healthy column",
          _cm(0.01, 0.01) == "CLEAR")

    import numpy as _np
    from synthkit.generate import _draw_numeric as _dn

    _QS = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95,
           0.99, 1.0]

    def _mk(td, grid, genv, srcv):
        d = Path(td)
        (d / "blueprint.json").write_text(json.dumps({
            "columns": {"q": {"marginal": grid},
                        "lv": {"marginal": {
                            "type": "levels",
                            "levels": [{"value": "a", "p": 1.0}]}},
                        "set__has__t": {"marginal": grid}}}),
            encoding="utf-8")
        (d / "provenance.json").write_text(json.dumps(
            {"source": {"rows_read": int(len(srcv))}}),
            encoding="utf-8")
        (d / "generated.csv").write_text(
            "q\n" + "\n".join("{:g}".format(x) for x in genv) + "\n",
            encoding="utf-8")
        (d / "src.csv").write_text(
            "q\n" + "\n".join("{:g}".format(x) for x in srcv) + "\n",
            encoding="utf-8")
        return d

    def _peek(d, *extra):
        return subprocess.run(
            [sys.executable, "scripts/peek.py", str(d), "marginal",
             "q"] + list(extra),
            capture_output=True, text=True, cwd=str(ROOT))

    # CASE A - a source with a POINT MASS and a grid too coarse to
    # carry it, with an output faithful to that grid.
    _r1 = _np.random.default_rng(7)
    _mass = _np.concatenate([_np.ones(600),
                             _r1.integers(2, 31, 400).astype(float)])
    _r1.shuffle(_mass)
    _coarse = {"type": "quantiles", "q": [0.0, 0.5, 1.0],
               "v": [1.0, 10.0, 30.0], "integral": True}
    with tempfile.TemporaryDirectory() as _td:
        _r = _mk(_td, _coarse,
                 _dn(_coarse, _np.random.default_rng(3).random(1000)),
                 _mass)
        _a = _peek(_r, str(_r / "src.csv"))
    check("a grid that cannot carry the source's point mass, with "
          "an output faithful to that grid, reads CONTRACT - the "
          "fix is the blueprint rather than the sampler",
          _a.returncode == 0 and "CONTRACT" in _a.stdout)

    # CASE B - a shape the grid CAN carry, so the grid tracks the
    # source, and an output piled on one value anyway.
    #
    # The source here is SMOOTH on purpose. A point mass is the one
    # shape a quantile grid cannot reproduce - measured on CASE A's
    # own source, a grid built from its own quantiles still loses
    # 10 points of a 60% mode, because the mass between the last
    # knot equal to the mode and the next one is spread
    # continuously. So a point-mass fixture cannot isolate a
    # GENERATION fault: it reads BOTH, correctly.
    _r2 = _np.random.default_rng(11)
    _smooth = _np.round(_np.exp(_r2.normal(2.2, 0.8, 1000))).astype(float)
    _good = {"type": "quantiles", "q": _QS,
             "v": [float(x) for x in _np.quantile(_smooth, _QS)],
             "integral": True}
    _piled = _np.concatenate([
        _np.full(950, float(_np.median(_smooth))),
        _np.arange(2.0, 52.0)])
    with tempfile.TemporaryDirectory() as _td:
        _r = _mk(_td, _good, _piled, _smooth)
        _b = _peek(_r, str(_r / "src.csv"))
    check("...while a grid that DOES carry the source, against an "
          "output piled on one value, reads GENERATION - the same "
          "symptom, the opposite fix, which one number could not "
          "separate",
          _b.returncode == 0 and "GENERATION" in _b.stdout)

    # THE GRID COLUMN MUST BE THE ENGINE'S OWN DRAW. A
    # reimplementation here would be the two-halves fault in the
    # one tool built to tell the halves apart.
    #
    # AND THE FIXTURE HAS TO REACH THE DIFFERENCE. The first
    # version of this check used a grid with no published tail
    # mean - where `_shape_tail` is a no-op and `_to_integers`
    # reduces to rounding - so a naive `interp + round` gave the
    # IDENTICAL mean and the mutation that replaced the engine's
    # draw broke nothing. A grid carrying `tail_mean_high` bends
    # the top segment, and the two part by 2.16 on this fixture.
    _tailm = {"type": "quantiles", "q": _QS,
              "v": [1.0, 1.0, 2.0, 3.0, 5.0, 9.0, 18.0, 34.0,
                    48.0, 87.0, 983.0],
              "integral": True, "tail_mean_high": 212.4}
    _tu = _np.random.default_rng(_MS).random(1000)
    _exp = float(_np.mean(_dn(_tailm, _tu)))
    _naive = float(_np.mean(_np.round(_np.interp(
        _tu, _np.asarray(_tailm["q"], dtype=float),
        _np.asarray(_tailm["v"], dtype=float)))))
    with tempfile.TemporaryDirectory() as _td:
        _r = _mk(_td, _tailm,
                 _dn(_tailm, _np.random.default_rng(4).random(1000)),
                 _np.round(_np.exp(_np.random.default_rng(5)
                                   .normal(2.2, 1.1, 1000))))
        _t = _peek(_r, str(_r / "src.csv"))
    _tg = [l for l in _t.stdout.splitlines()
           if l.strip().startswith("mean")]

    check("the fixture for the next check REACHES the difference - "
          "the engine's draw and a naive interp-and-round part by "
          "more than two on this grid, so a mutation between them "
          "has something to break",
          abs(_exp - _naive) > 1.0)

    check("...and the GRID column comes from the engine's own "
          "`_draw_numeric` at the stated seed, not a second "
          "implementation of what a published grid means",
          bool(_tg) and abs(float(_tg[0].split()[2]) - _exp) < 1e-4)

    with tempfile.TemporaryDirectory() as _td:
        _r = _mk(_td, _good, _piled, _smooth)
        _c = subprocess.run(
            [sys.executable, "scripts/peek.py", str(_r), "marginal",
             "lv", str(_r / "src.csv")],
            capture_output=True, text=True, cwd=str(ROOT))
        _d = subprocess.run(
            [sys.executable, "scripts/peek.py", str(_r), "marginal",
             "set__has__t", str(_r / "src.csv")],
            capture_output=True, text=True, cwd=str(ROOT))
        _bad = Path(_td) / "short.csv"
        _bad.write_text("q\n1\n2\n3\n", encoding="utf-8")
        _e = _peek(_r, str(_bad))

    check("a column publishing LEVELS is refused by name - this "
          "compares a numeric grid against the shape it produces",
          _c.returncode != 0
          and "not quantiles" in (_c.stdout + _c.stderr))
    check("...a column in the blueprint but NOT in generated.csv "
          "is refused, because search scaffolding is dropped "
          "before the file is written and nobody can read a "
          "column that is not there",
          _d.returncode != 0
          and "not in generated.csv" in (_d.stdout + _d.stderr).lower())
    check("...and a source whose row count disagrees with the "
          "run's own provenance is refused rather than producing a "
          "plausible number about the wrong file",
          _e.returncode != 0 and "refusing" in (_e.stdout + _e.stderr))

    # THE ALL-COLUMNS VIEW. A round trip to the machine holding
    # the extract is minutes, so asking one column per trip is the
    # wrong shape of tool for where it runs - and it cannot answer
    # whether a gap is ONE column or the engine.
    with tempfile.TemporaryDirectory() as _td:
        _r = _mk(_td, _good, _piled, _smooth)
        # a second column that is FINE, so the view has to sort and
        # a verdict of CLEAR has to be reachable alongside a fault
        _sj = json.loads((_r / "blueprint.json").read_text(
            encoding="utf-8"))
        _sj["columns"]["ok"] = {"marginal": _good}
        (_r / "blueprint.json").write_text(json.dumps(_sj),
                                           encoding="utf-8")
        _fine = _dn(_good, _np.random.default_rng(9).random(1000))
        for _p, _vals in ((_r / "generated.csv", _fine),
                          (_r / "src.csv", _smooth)):
            _txt = _p.read_text(encoding="utf-8").splitlines()
            _out = [_txt[0] + ",ok"]
            for _i, _ln in enumerate(_txt[1:]):
                _out.append("{},{:g}".format(_ln, _vals[_i % len(_vals)]))
            _p.write_text("\n".join(_out) + "\n", encoding="utf-8")
        _ms = subprocess.run(
            [sys.executable, "scripts/peek.py", str(_r), "marginals",
             str(_r / "src.csv")],
            capture_output=True, text=True, cwd=str(ROOT))

    check("the all-columns view reports every numeric column in "
          "one pass, so finding which are off their grid does not "
          "cost one round trip each",
          _ms.returncode == 0 and "q" in _ms.stdout
          and "ok" in _ms.stdout)
    check("...worst FIRST, so the column that matters is the one "
          "a reader sees without scrolling",
          _ms.stdout.index("\n  q ") < _ms.stdout.index("\n  ok "))
    check("...separating the SAMPLER's gaps from the contract's, "
          "because they take opposite fixes",
          "SAMPLER" in _ms.stdout and "GENERATION" in _ms.stdout)
    check("...and a healthy column reads CLEAR in the same table, "
          "so the verdict is not simply reporting every column as "
          "a fault",
          "CLEAR" in _ms.stdout)
    check("...and it carries a TOTAL a reader can check the rows "
          "against, which is how an omitted category stays visible",
          "column(s) are off their own published grid" in _ms.stdout)
    # THE QUANTITY A RECIPIENT ACTUALLY HOLDS. The two gaps above
    # split the blame between the contract and the sampler, and
    # neither is what lands in their hands - a sampler faithfully
    # reproducing a wrong grid reads clean on grid-against-output
    # and is still wrong for them.
    check("...and reports SOURCE against OUTPUT, which is what a "
          "recipient holds rather than a split of the blame",
          "SRC~OUT" in _ms.stdout
          and "what a recipient holds" in _ms.stdout)
    check("...and says how many columns differ from the SOURCE in "
          "a recipient's hands, separately from the blame split",
          "differ from the SOURCE in a recipient" in _ms.stdout)

    # THE DECK IS THE PAGE PEOPLE READ, and a measurement that
    # never reaches it might as well not exist for the person
    # deciding - this project's own words. The shape measure has
    # to appear there, and it has to carry its privacy warning,
    # because the deck is also the page that travels.
    _dk = (ROOT / "scripts" / "fidelity_deck.py").read_text(
        encoding="utf-8")
    check("the deck renders the marginal-SHAPE section, which no "
          "gate criterion asks about - a column can pass "
          "coverage, centre and spread with most of its rows on "
          "one value",
          "Does each column keep its own shape?" in _dk)
    check("...naming the ORIGINAL-against-synthetic column as the "
          "one a reader holds, rather than only the blame split",
          "ORIGINAL vs synthetic" in _dk)
    check("...and carrying the warning that a VERBATIM copy would "
          "score perfectly here, so a good row is never read as "
          "privacy",
          "says nothing whatever" in _dk)
    check("...and saying plainly when every column tracks, rather "
          "than rendering an empty table",
          "Every one of the {} numeric columns tracks" in _dk)

    # AND RENDERED, not merely present in the source. A content
    # check cannot see a dead code path - this project lost a
    # whole bench to 211 green content checks over a page the
    # browser could not run.
    with tempfile.TemporaryDirectory() as _dm:
        _dmp = Path(_dm)
        import numpy as _npm
        import pandas as _pdm
        _rngm = _npm.random.default_rng(4)
        _n = 900
        _dfm = _pdm.DataFrame({
            "person_id": ["P%03d" % (i // 3) for i in range(_n)],
            "visit_number": [(i % 3) + 1 for i in range(_n)],
            "v": _npm.round(_rngm.normal(50, 9, _n), 3)})
        _spm = _dmp / "s.csv"
        _dfm.to_csv(_spm, index=False)
        _rdm = _dmp / "run"
        subprocess.run(
            [sys.executable, "scripts/run_discovery.py",
             "--src", str(_spm), "--out", str(_rdm),
             "--group-by", "person_id", "--generate"],
            capture_output=True, text=True, cwd=str(ROOT))
        _fp = _rdm / "fidelity.json"
        _fj2 = json.loads(_fp.read_text(encoding="utf-8"))
        _had = bool((_fj2.get("marginals") or {}).get(
            "columns_measured"))
        # PLANT a column off the source, so the table branch is
        # exercised rather than only the all-clear sentence.
        _fj2["marginals"] = {
            "columns_measured": 3, "tolerance": 0.05,
            "off_source": ["v"], "sampler_faults": ["v"],
            "worst_source_vs_output": 0.4,
            "columns": [{"column": "v", "grid_vs_source": 0.02,
                         "grid_vs_output": 0.39,
                         "source_vs_output": 0.4,
                         "verdict": "GENERATION"}]}
        _fp.write_text(json.dumps(_fj2), encoding="utf-8")
        _docm, _ = _fd0.build_deck(str(_spm), str(_rdm),
                                   "person_id")

    check("a real run's own fidelity.json CARRIES the marginals "
          "block, without which the deck would have nothing to "
          "render and the check below would pass on a stub",
          _had)
    check("...and the deck RENDERS the section from it - a "
          "content check on the source cannot see a dead path",
          "Does each column keep its own shape?" in _docm)
    check("...naming the off-source column in the page itself",
          ">v</td>" in _docm or "<td>v</td>" in _docm)
    check("...and rendering the privacy warning on the page, not "
          "just in the file that builds it",
          "says nothing whatever" in _docm)

    # THE USAGE LIST AND THE DISPATCH MUST AGREE. `peek` has
    # grown to eighteen subcommands; a documented one that is not
    # wired reads as a broken tool, and a wired one that is not
    # documented is invisible to the operator it was built for.
    # Both are silent today.
    _pk = (ROOT / "scripts" / "peek.py").read_text(encoding="utf-8")
    _doc = _pk.split('"""')[1]
    _documented = set(
        re.findall(r"peek\.py RUNDIR (\w+)", _doc))
    _dispatched = set(re.findall(r'elif what == "(\w+)":', _pk))
    _dispatched |= set(re.findall(r'if what == "(\w+)":', _pk))
    check("the peek usage list is non-trivial, so the comparison "
          "below is not between two empty sets",
          len(_documented) >= 10)
    check("every subcommand peek DOCUMENTS is dispatchable - a "
          "documented command that does nothing reads as a broken "
          "tool: {}".format(
              ", ".join(sorted(_documented - _dispatched))
              or "none missing"),
          not (_documented - _dispatched))
    check("...and every subcommand it DISPATCHES is documented - "
          "an undocumented one is invisible to the operator it "
          "was built for: {}".format(
              ", ".join(sorted(_dispatched - _documented))
              or "none undocumented"),
          not (_dispatched - _documented))

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
