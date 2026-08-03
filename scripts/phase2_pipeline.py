"""Run the whole real-data-to-synthetic loop in one command.

    python scripts/phase2_pipeline.py --src DIR [-o OUTDIR]
                                      [--window 30] [--k 10]
                                      [--rows N] [--skip-wrangle]

Stages:
  1 wrangle   OMOP CSVs        -> tidy one-row-per-visit
  2 label     tidy             -> outcome column (derived timing)
  3 diagnose  labeled          -> is there learnable structure?
  4 profile   labeled          -> parameters only, k-anonymous
  5 compile   profile          -> draft spec (every value a dial)
  6 generate  spec             -> synthetic data
  7 score     source + synth   -> fidelity + privacy scorecards

Everything lands in OUTDIR. The draft spec is a DRAFT: review it,
change any dial, author the outcome you want planted, and re-run
generation. Nothing here decides clinical truth for you.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:                       # console-safe on Windows terminals
    from _console import console_safe
    console_safe()
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent


def _isnum(v):
    try:
        float(str(v).replace(",", ""))
        return True
    except ValueError:
        return False


def step(n, title):
    print("\n[{}/7] {}".format(n, title))
    print("-" * 58)


def run(*args, quiet=False):
    r = subprocess.run([sys.executable] + [str(a) for a in args],
                       cwd=str(ROOT), capture_output=True,
                       text=True)
    if r.returncode:
        print(r.stdout)
        print(r.stderr)
        sys.exit("pipeline stopped at: {}".format(args[0]))
    if not quiet:
        print(r.stdout.rstrip())
    return r.stdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="folder of OMOP CSVs (or a tidy CSV with "
                         "--skip-wrangle)")
    ap.add_argument("-o", "--out", default="data/phase2_run")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--rows", type=int, default=0)
    ap.add_argument("--skip-wrangle", action="store_true",
                    help="--src is already a tidy CSV")
    ap.add_argument("--engine", default="spec",
                    choices=["spec", "condnet", "both"],
                    help="spec: fitted marginals + confirmed "
                         "correlations, compiled into an editable "
                         "recipe. condnet: a conditional "
                         "dependency network that models the JOINT "
                         "distribution, capturing nonlinear and "
                         "interaction structure a recipe cannot "
                         "express. both: run each and compare.")
    ap.add_argument("--max-parents", type=int, default=3)
    ap.add_argument("--transcribe", action="store_true",
                    help="after generating, render some of the "
                         "structured facts into a messy clinical "
                         "note and emit the truth ledger, so "
                         "extraction can be graded by corruption "
                         "type")
    ap.add_argument("--group-by", default="person_id",
                    help="the PRIVACY UNIT column (default "
                         "person_id). k-anonymity counts these, "
                         "not rows.")
    ap.add_argument("--seed-out", type=int, default=20260731)
    ap.add_argument("--hierarchical", action="store_true",
                    help="generate PATIENTS with a course of "
                         "visits rather than loose rows: fixed "
                         "traits stay fixed and a drifting value "
                         "drifts from one visit to the next")
    ap.add_argument("--epsilon", type=float, default=0.0,
                    help="differential privacy budget. Covers "
                         "every published table. Lower is "
                         "stronger; the fidelity cost is real and "
                         "is reported")
    ap.add_argument("--audit", action="store_true",
                    help="attack the result: split the cohort, "
                         "fit on half, and measure whether an "
                         "adversary can tell who was in it")
    ap.add_argument("--plant", default="",
                    help="author an outcome and grade models "
                         "against a known ceiling. Format: "
                         "col=weight,col=weight")
    ap.add_argument("--prevalence", type=float, default=0.15)
    ap.add_argument("--vendor", default="",
                    help="a vendor model to grade alongside ours")
    a = ap.parse_args()
    out = (ROOT / a.out) if not Path(a.out).is_absolute() \
        else Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- data-safety gate -------------------------------------
    # The tidy and labeled CSVs are the SOURCE RECORDS reshaped.
    # If the output folder is not ignored by git, a reflexive
    # `git add -A` would publish them. Refuse to be that trap.
    at_risk = False
    try:
        chk = subprocess.run(
            ["git", "check-ignore", "-q", str(out)],
            cwd=str(ROOT), capture_output=True)
        # 0 = ignored (safe), 1 = tracked path (warn),
        # 128 = not a git repo (nothing to publish to)
        at_risk = chk.returncode == 1
    except Exception:
        at_risk = False
    if at_risk:
        print("\n!! WARNING: {} is NOT git-ignored.".format(out))
        print("   This folder will hold the source records "
              "reshaped (tidy_visits.csv) — if the source is real "
              "data, committing it would publish patient records.")
        print("   Add it to .gitignore, or pass -o with a path "
              "outside the repository, before running on anything "
              "real.\n")

    tidy = out / "tidy_visits.csv"
    if a.skip_wrangle:
        tidy = Path(a.src)
        step(1, "wrangle — SKIPPED (source is already tidy)")
    else:
        step(1, "wrangle: OMOP tables -> one row per visit")
        run("scripts/omop_wrangle.py", "--src", a.src,
            "-o", tidy, "--report")

    step(2, "label: derive the outcome from visit timing")
    labeled = out / "tidy_visits_labeled.csv"
    run("scripts/derive_label.py", "--in", tidy, "-o", labeled,
        "--window", a.window, "--report")

    step(3, "diagnose: is there learnable structure to begin with?")
    run("scripts/null_diagnostic.py", "--in", labeled,
        "--label", "returned_within_{}d".format(a.window))

    step(4, "profile: measure patterns as parameters, never records")
    profile = out / "profile.json"
    with labeled.open(encoding="utf-8-sig", newline="") as f:
        head = csv.DictReader(f)
        first = next(iter(head), {})
    gb = a.group_by if a.group_by in first else ""
    if gb:
        with labeled.open(encoding="utf-8-sig", newline="") as f:
            ids = {r.get(gb) for r in csv.DictReader(f)}
        with labeled.open(encoding="utf-8-sig") as f:
            nrows = sum(1 for _ in f) - 1
        print("  PRIVACY UNIT: {} — {} rows from {} people "
              "(mean {:.1f} each)".format(gb, nrows, len(ids),
                                          nrows / max(len(ids), 1)))
        print("  k counts PEOPLE, not rows: ten visits from one "
              "patient is a cell of one.")
    else:
        print("  no privacy-unit column found ({}): every row is "
              "treated as an independent person".format(a.group_by))
    args4 = ["scripts/omop_profile.py", "--in", labeled,
             "-o", profile, "--k", a.k, "--report"]
    if gb:
        args4 += ["--group-by", gb]
    run(*args4)

    sys.path.insert(0, str(ROOT))
    with labeled.open(encoding="utf-8-sig") as f:
        n_src = sum(1 for _ in f) - 1
    want_rows = a.rows or n_src
    label = "returned_within_{}d".format(a.window)
    outputs = []

    do_spec = a.engine in ("spec", "both")
    do_net = a.engine in ("condnet", "both")

    if do_spec:
        step(5, "compile (spec engine): profile -> a draft spec "
                "you can edit")
        draft = out / "draft_spec.json"
        run("scripts/profile_to_spec.py", "--profile", profile,
            "-o", draft, "--report")
        step(6, "generate (spec engine)")
        from synthkit.tablespec import TableSpec
        from synthkit.tableplan import plan_table
        D = {k: v for k, v in
             json.loads(draft.read_text(encoding="utf-8")).items()
             if not k.startswith("_")}
        D["rows"] = want_rows
        ts = TableSpec.from_json(json.dumps(D))
        ts.validate()
        tr = plan_table(ts)
        gen = out / "generated.csv"
        with gen.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f,
                               fieldnames=list(tr.dirty_rows[0]))
            w.writeheader()
            w.writerows(tr.dirty_rows)
        print("  generated {} rows x {} columns -> {}".format(
            len(tr.dirty_rows), len(tr.dirty_rows[0]), gen.name))
        outputs.append(("spec", gen, profile,
                        out / "fidelity.json"))

    if do_net:
        step(5 if not do_spec else 6,
             "learn (condnet engine): the JOINT distribution")
        from synthkit.condnet import CondNet
        with labeled.open(encoding="utf-8-sig", newline="") as f:
            rows_in = list(csv.DictReader(f))
        tgts = [label] if label in (rows_in[0] if rows_in else {}) \
            else []
        net = CondNet(k=a.k, max_parents=a.max_parents).learn(
            rows_in, targets=tgts, group_by=(gb or None),
            multilevel=bool(a.hierarchical and gb),
            epsilon=a.epsilon)
        if a.epsilon:
            dpr = net.report["differential_privacy"]
            print("  privacy budget {} split over {} published "
                  "tables".format(dpr["epsilon"],
                                  dpr["budget_split_over"]))
            print("  covers: {}".format(", ".join(dpr["covers"])))
        model = out / "condnet_model.json"
        model.write_text(net.to_json(), encoding="utf-8")
        rep = net.report
        print("  {} columns modelled at {} bins ({})".format(
            rep["columns_modelled"], rep["bins"],
            rep["bins_chosen"]))
        print("  effective sample size: {} (from {} rows)".format(
            rep["effective_n"], rep["rows"]))
        print("  {} dependencies learned; {} configurations too "
              "thin to publish (they back off)".format(
                  rep["edge_count"],
                  rep["suppressed_configurations"]))
        for e in rep["edges"][:8]:
            print("     {} <- {}".format(
                e["child"], ", ".join(e["parents"])))
        if len(rep["edges"]) > 8:
            print("     ... and {} more".format(
                len(rep["edges"]) - 8))
        if not rep["edges"]:
            print("     (no dependence survived correction — the "
                  "source columns are mutually independent)")
        step(6 if not do_spec else 7, "generate (condnet engine)")
        if a.hierarchical and getattr(net, "visit_counts", None):
            per = max(1.0, sum(kk * vv for kk, vv in
                               net.visit_counts.items())
                      / max(sum(net.visit_counts.values()), 1))
            syn = net.sample_patients(
                max(1, int(want_rows / per)), seed=a.seed_out)
            print("  generated {} patients averaging {:.1f} "
                  "visits each".format(
                      len({r.get(gb) for r in syn}), per))
        else:
            syn = net.sample(want_rows, seed=a.seed_out)
        gen_n = out / "generated_condnet.csv"
        with gen_n.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(syn[0]))
            w.writeheader()
            w.writerows(syn)
        print("  generated {} rows x {} columns -> {}".format(
            len(syn), len(syn[0]), gen_n.name))
        # The profile describes the SOURCE, so it governs which
        # relationships are worth checking regardless of which
        # engine produced the synthetic data. Passing it to one
        # engine and not the other would score them against
        # different question sets and make the comparison
        # meaningless.
        outputs.append(("condnet", gen_n, profile,
                        out / "fidelity_condnet.json"))

    if a.transcribe:
        step(7, "transcribe: structured facts -> messy prose")
        from synthkit.transcribe import (
            TranscribeSpec, transcribe)
        from synthkit import learnspec as _ls
        for name, path, _, _ in outputs:
            with Path(path).open(encoding="utf-8-sig",
                                 newline="") as f:
                gen_rows = list(csv.DictReader(f))
            if not gen_rows:
                continue
            # One implementation, shared with the bench. Sniffing
            # columns separately here was a guarantee the two
            # paths would drift apart.
            facts = _ls.facts_from_columns(gen_rows)
            if not facts:
                print("  no transcribable facts found in {}"
                      .format(Path(path).name))
                continue
            tspec = TranscribeSpec(facts,
                                   rates=_ls.default_rates())
            noted, ledgers = transcribe(gen_rows, tspec)
            npath = Path(path).with_name(
                Path(path).stem + "_noted.csv")
            with npath.open("w", newline="",
                            encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(noted[0]))
                w.writeheader()
                w.writerows(noted)
            lpath = Path(path).with_name(
                Path(path).stem + "_note_ledger.json")
            lpath.write_text(json.dumps(ledgers, indent=1),
                             encoding="utf-8")
            mentions = sum(len(L) for L in ledgers)
            print("  {}: {} notes, {} ledgered mentions across {} "
                  "corruption types -> {}".format(
                      name, len(noted), mentions,
                      len({c for L in ledgers for e in L
                           for c in e["corruptions"]}),
                      npath.name))
        print("  the ledger is the answer key: what each note "
              "asserts, and which corruption obscured it")

    if a.audit and do_net:
        step(7, "audit: can an adversary tell who was in it?")
        from synthkit.attack import membership_audit
        import random as _rnd
        by_p = {}
        for r in rows_in:
            by_p.setdefault(r.get(gb, id(r)), []).append(r)
        keys = sorted(by_p, key=str)
        _rnd.Random(7).shuffle(keys)
        cut2 = len(keys) // 2
        mem = [r for kk in keys[:cut2] for r in by_p[kk]]
        non = [r for kk in keys[cut2:] for r in by_p[kk]]
        half_net = CondNet(
            k=a.k, max_parents=a.max_parents).learn(
                mem, group_by=(gb or None),
                multilevel=bool(a.hierarchical and gb),
                epsilon=a.epsilon)
        aud = membership_audit(half_net, mem, non,
                               half_net.sample(1500, seed=3),
                               nn_sample=120)
        print("  strongest adversary scored {:.3f} ({})".format(
            aud["worst_auc"], aud["verdict"]))
        print("  {}".format(aud["reading"]))
        (out / "membership_audit.json").write_text(
            json.dumps(aud, indent=1), encoding="utf-8")

    step(7, "score: fidelity and privacy")
    scores = {}
    for name, path, prof_arg, rep_path in outputs:
        args = ["scripts/fidelity_report.py", "--source", labeled,
                "--synthetic", path, "-o", rep_path, "--report"]
        if prof_arg:
            args += ["--profile", prof_arg]
        if label:
            args += ["--target", label]
        txt = run(*args, quiet=True)
        R = json.loads(rep_path.read_text(encoding="utf-8"))
        scores[name] = R
        s = R["summary"]
        print("\n  --- {} engine ---".format(name))
        print("  fidelity {}/{} {}   privacy {}".format(
            s["passed"], s["checks"], s["fidelity_verdict"],
            s["privacy_verdict"]))
        ds = R.get("dependence_shape", {})
        print("  {} real relationships scored ({} skipped as "
              "noise); {} of them turn; {} not reproduced".format(
                  ds.get("pairs_tested", 0),
                  ds.get("pairs_skipped_as_noise", 0),
                  ds.get("nonlinear_relationships_in_source", 0),
                  len([d for d in ds.get("detail", [])
                       if not d["pass"]])))
        it = R.get("interactions", {}).get("detail", [])
        if it:
            print("  {} interactions found; {} not reproduced"
                  .format(len(it),
                          len([d for d in it if not d["pass"]])))

    if len(scores) == 2:
        print("\n  " + "=" * 54)
        print("  HEAD TO HEAD")
        a_s, b_s = scores["spec"], scores["condnet"]
        for lab, R in (("spec   ", a_s), ("condnet", b_s)):
            ds = R.get("dependence_shape", {})
            it = R.get("interactions", {}).get("detail", [])
            print("   {}  fidelity {:>3}/{:<3} | shape fails {:>2} "
                  "| interaction fails {:>2} | privacy {}".format(
                      lab, R["summary"]["passed"],
                      R["summary"]["checks"],
                      len([d for d in ds.get("detail", [])
                           if not d["pass"]]),
                      len([d for d in it if not d["pass"]]),
                      R["summary"]["privacy_verdict"]))
        print("   the spec engine yields an editable recipe; the "
              "conditional network yields joint structure a recipe "
              "cannot express. Compare the shape and interaction "
              "columns — that is where they differ.")

    print("\n" + "=" * 58)
    print("artifacts in {}".format(out))
    for f in ("tidy_visits.csv", "tidy_visits_labeled.csv",
              "profile.json", "draft_spec.json", "generated.csv",
              "fidelity.json", "condnet_model.json",
              "generated_condnet.csv", "fidelity_condnet.json",
              "generated_noted.csv",
              "generated_note_ledger.json",
              "generated_condnet_noted.csv",
              "generated_condnet_note_ledger.json"):
        p = out / f
        if p.exists():
            print("  {:26s} {:>9,} bytes".format(
                f, p.stat().st_size))
    if a.plant and do_net:
        step(7, "exam: plant an outcome and grade against it")
        from synthkit import learnspec as _ls2
        from synthkit.transcribe import (
            TranscribeSpec as _TS, transcribe as _tx)
        weights = {}
        for piece in a.plant.split(","):
            if "=" not in piece:
                continue
            col, val = piece.split("=", 1)
            weights[col.strip()] = float(val)
        known = {c["column"] for c in _ls2.outcome_candidates(net)}
        unknown = [c for c in weights if c not in known]
        if unknown:
            print("  not columns this model has: {}".format(
                ", ".join(unknown)))
            print("  available: {}".format(
                ", ".join(sorted(known))[:400]))
        weights = {c: w for c, w in weights.items() if c in known}
        if not weights:
            print("  nothing to plant")
        else:
            data = net.sample(max(2000, want_rows),
                              seed=a.seed_out + 5)
            planted = _ls2.plant_outcome(
                data, weights, name="planted_outcome",
                prevalence=a.prevalence)
            hide = [c for c in weights if "::" in c]
            spec2 = _TS(_ls2.facts_from_model(net,
                                              text_only=hide),
                        rates=_ls2.default_rates())
            planted["rows"], _ = _tx(planted["rows"], spec2)
            hidden = _ls2.columns_to_hide(net, hide)
            _ls2.blank_columns(planted["rows"], hidden)
            res = _ls2.showdown(planted["rows"], planted["probs"],
                                "planted_outcome",
                                vendor=a.vendor)
            print("  planted at {:.1%} prevalence (intercept "
                  "solved to {})".format(
                      planted["realized_prevalence"],
                      planted["intercept"]))
            if hidden:
                print("  causes removed from the columns and "
                      "written only into the prose: {}".format(
                          ", ".join(hidden)))
            print("  ceiling {:.3f} (known exactly) | ours "
                  "{:.3f}".format(res["ceiling"], res["reading"]))
            if res.get("blind") is not None:
                print("  notes withheld {:.3f} | reading worth "
                      "{:+.3f}".format(res["blind"],
                                       res["value_of_reading"]))
            if "vendor" in res:
                print("  vendor {:.3f} — {} ours".format(
                    res["vendor"],
                    "beats" if res["vendor_beats_ours"]
                    else "short of"))
            (out / "exam.json").write_text(
                json.dumps({"weights": planted["weights"],
                            "hidden": hidden, "showdown": res},
                           indent=1), encoding="utf-8")

    if do_spec:
        print("\nNEXT (spec engine): open draft_spec.json — every "
              "number is a dial. Author an outcome with known "
              "coefficients, then re-run generation to get planted "
              "truth with an exact ceiling.")
    if do_net:
        print("\nNEXT (condnet engine): condnet_model.json holds "
              "the learned dependencies. Every conditional table "
              "is a dial — CondNet.from_json(...).amplify(column, "
              "factor) strengthens a discovered relationship "
              "(factor > 1), weakens it (< 1), or removes it "
              "entirely (0), so a vendor can be tested against "
              "the pattern as found AND against deliberately "
              "harder versions of it.")


if __name__ == "__main__":
    main()
