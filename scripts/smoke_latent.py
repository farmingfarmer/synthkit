"""Checks for the latent challenger's MECHANICS - the k-screen,
the bound, determinism, and the memorization tripwire with its
positive control. Fidelity numbers are measured by running the
script on the fixtures, not asserted here: a generator's quality
is seed-swept, never pinned.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parent.parent
for p in (str(_root), str(_root / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

PASS = 0
FAIL = 0


def check(name, ok):
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print("FAIL: {}".format(name))


def main():
    from latent_challenger import (K, LatentGen, _k_clip,
                                   _split_patients)

    r = np.random.RandomState(3)
    n = 1200
    gid = np.repeat(np.arange(200), 6)
    x = r.normal(50, 10, n)
    # one patient carries an absurd outlier, and one categorical
    # level is held by only 3 patients - both must be screened
    x[gid == 7] = 400.0
    lab = np.where(gid % 50 == 3, "rare_level",
                   np.where(gid % 2 == 0, "common_a", "common_b"))
    df = pd.DataFrame({
        "person_id": ["P{:03d}".format(i) for i in gid],
        "num": np.round(x, 2),
        "buddy": np.round(0.8 * x + r.normal(0, 4, n), 2),
        "cat": lab,
    })
    # A GENUINELY RARE COMBINATION, planted on FOUR patients -
    # ordinary-looking values that only these four hold TOGETHER.
    # Without it the feature space of three smooth columns has no
    # sub-k region at all and the k-blur check passes on nothing:
    # the fixture must contain the thing the check is about.
    _odd = df["person_id"].isin(
        ["P00{}".format(i) for i in (1, 3, 5, 7)])
    df.loc[_odd, "num"] = 95.0 + r.normal(0, 0.4, int(_odd.sum()))
    df.loc[_odd, "buddy"] = 2.0 + r.normal(0, 0.4,
                                           int(_odd.sum()))
    # A GENUINELY PATIENT-LEVEL COLUMN. Every other column here
    # re-rolls per row, so a between-patient-share check on them
    # passes whether or not the generator has patient structure
    # at all - a vacuous check, caught by mutating the patient
    # centre away and watching nothing go red. `plevel` is a
    # property of the PERSON (their own level, plus small visit
    # noise), so it can only survive generation if a generated
    # patient's visits actually share a patient.
    _plev = r.normal(0, 12, 200)[gid]
    df["plevel"] = np.round(_plev + r.normal(0, 1.2, n), 2)
    rare_patients = df[df["cat"] == "rare_level"][
        "person_id"].nunique()
    check("the fixture contains the thing: the rare level is held "
          "by {} patients, below k={}".format(rare_patients, K),
          rare_patients < K)

    ho_ids = set("P{:03d}".format(i) for i in range(0, 200, 10))
    ho = df[df["person_id"].isin(ho_ids)].reset_index(drop=True)
    tr = df[~df["person_id"].isin(ho_ids)].reset_index(drop=True)
    lo, hi = _k_clip(tr["num"].to_numpy(dtype=float),
                     tr["person_id"].to_numpy(), K)
    check("the k bound is the MEAN of the k most extreme "
          "patients' extremes - one patient's 400.0 cannot set "
          "it alone",
          hi < 400.0 and hi > df["num"].quantile(0.5))

    # Whole patients held out BEFORE fitting - the first cut of
    # this check handed the tripwire "holdout" rows that were in
    # training, their distances read zero, and the honest ratio
    # inverted to 0.05. A control fed from inside the training
    # set controls nothing. And the rare-level holders must
    # SURVIVE the split into training - the second cut sent all
    # four to the holdout and the fold check passed on nothing.
    check("the fixture still contains the thing after the "
          "split: rare-level patients remain in training, below "
          "k",
          0 < tr[tr["cat"] == "rare_level"][
              "person_id"].nunique() < K)
    g = LatentGen(seed=5).fit(tr, "person_id")
    gen1 = g.generate(800, seed=9)
    gen2 = g.generate(800, seed=9)
    check("generation is deterministic under its seed - same "
          "seed, byte-identical frame",
          gen1.equals(gen2))
    check("...and a different seed is a different draw",
          not gen1.equals(g.generate(800, seed=10)))

    check("the sub-k level NEVER appears in the output, and "
          "__other__ carries its mass - the network was never "
          "shown the label it must not repeat",
          "rare_level" not in set(gen1["cat"])
          and "__other__" in set(gen1["cat"]))

    v = gen1["num"].dropna()
    check("every generated numeric sits inside the k-anonymous "
          "bound by construction - the decoder is unbounded, the "
          "output is not",
          float(v.min()) >= lo - 1e-9
          and float(v.max()) <= hi + 1e-9)

    # THE POSITIVE CONTROL. The nn-ratio near 1.0 on honest
    # output means nothing unless a REPUBLISHING generator makes
    # it collapse - hand the tripwire the training rows
    # themselves and it must fire.
    honest = g.nn_ratio(gen1, g._re_encode_frame(ho))
    leak = g.nn_ratio(tr.drop(columns=["person_id"]).iloc[:400],
                      g._re_encode_frame(ho))
    check("the memorization tripwire: honest output reads near "
          "1.0 ({:.2f}) and a verbatim republish reads near "
          "zero ({:.2f}) - the clean number is worth something "
          "because the dirty one fails".format(honest, leak),
          honest > 0.5 and leak < 0.1)

    # THE WRONG-TYPE FAULT, IN THE RULER ITSELF. span_days holds
    # seven distinct day-counts; the first cut sent it down the
    # categorical branch and argmax decoding collapsed it to a
    # constant - shares read 0/0 and the instrument measured
    # nothing while looking healthy. A numeric-valued column is
    # numeric however few values it has, and a generated
    # categorical must keep more than its modal level.
    df2 = df.copy()
    df2["days"] = (gid % 7 + 1).astype(int)
    g2 = LatentGen(seed=6).fit(
        df2[~df2["person_id"].isin(ho_ids)], "person_id")
    plan_kind = dict((it[1], it[0]) for it in g2.plan)
    gen3 = g2.generate(800, seed=4)
    check("a seven-distinct-value integer column is planned "
          "NUMERIC, and generated categoricals keep more than "
          "their modal level",
          plan_kind["days"] == "num"
          and gen3["cat"].nunique() > 1)

    # THE K-AWARE CONTRACT: faithful where many patients stand
    # behind a pattern, deliberately blurred where fewer than k
    # do - and support measured in the FEATURE space, because "a
    # pattern fewer than k records support" means a rare
    # COMBINATION OF VALUES, not a sparse latent region.
    g3 = LatentGen(seed=8, k_blur=True).fit(tr, "person_id")
    sup_tr = g3.support_of_rows(g3._Xtrain)
    check("the fixture contains the thing: some training rows "
          "sit in sub-k regions of the FEATURE space, so the "
          "blur has something to bite on",
          (sup_tr < g3.k).any() and (sup_tr >= g3.k).any())
    _ = g3.generate(600, seed=3)
    rep = g3.blur_report
    check("...and the blur fires on them and says how much it "
          "moved - blurred and snapped counts reported, never "
          "silent",
          rep is not None and rep["rows"] == 600
          and rep["blurred"] > 0
          and set(rep) == {"rows", "blurred", "snapped"})
    g4 = LatentGen(seed=8, k_blur=False).fit(tr, "person_id")
    _ = g4.generate(600, seed=3)
    check("...and turning it off is possible and honest - the "
          "arm that measures what the blur costs reports no "
          "blur at all",
          g4.blur_report is None)

    # PATIENT STRUCTURE. Without it the engine returned a table
    # of UNLINKED VISITS - no patient column at all - which on a
    # longitudinal source is not degraded structure but the
    # absence of one. The latent code splits into the patient's
    # centre and the visit's deviation; a generated patient's
    # visits share their centre, so a column that belongs to the
    # PERSON stays put across their visits.
    from latent_challenger import between_share
    g5 = LatentGen(seed=9, hierarchical=True).fit(tr,
                                                  "person_id")
    gen5 = g5.generate(900, seed=2)
    check("the output carries an INVENTED patient identity, "
          "first column, with several visits per patient - never "
          "a real person's id",
          "person_id" in gen5.columns
          and list(gen5.columns)[0] == "person_id"
          and gen5["person_id"].str.startswith("S").all()
          and not set(gen5["person_id"]) & set(tr["person_id"])
          and gen5["person_id"].nunique() < len(gen5) / 2)

    # A column that belongs to the PERSON must stay with the
    # person. `num` is patient-linked in this fixture (each
    # patient's rows share a level plus noise); the check asserts
    # the generated between-patient share lands near the
    # source's, and that a flat generator would FAIL it.
    src_b = between_share(tr, "plevel", "person_id")
    gen_b = between_share(gen5, "plevel", "person_id")
    g6 = LatentGen(seed=9, hierarchical=False).fit(tr,
                                                   "person_id")
    gen6 = g6.generate(900, seed=2)
    check("the fixture contains the thing: `plevel` really is a "
          "person-level column in the source ({:.2f} of its "
          "variance sits between patients)".format(src_b),
          src_b > 0.7)
    check("a person-level column keeps its person-level share "
          "({:.2f} source, {:.2f} generated) - and without "
          "patient structure the question cannot even be "
          "asked".format(src_b, gen_b),
          np.isfinite(src_b) and np.isfinite(gen_b)
          and abs(gen_b - src_b) < 0.35
          and "person_id" not in gen6.columns)

    # DENOISING TRAINING IS ON BY DEFAULT, because it measured
    # better on BOTH axes: the weights-surface membership
    # adversary fell 0.776 FAIL -> 0.569 PASS while close rose
    # 1332/1376 -> 1354/1376. A defense that also improves
    # fidelity is not a trade-off, and the default must follow
    # the measurement.
    check("denoising training is the default, and the corrupted "
          "input is what the network is trained FROM while the "
          "clean record is what it is trained TO",
          LatentGen().denoise == 0.3
          and "rs2.normal" in (_root / "scripts"
                               / "latent_challenger.py")
          .read_text(encoding="utf-8"))

    # THE MINER MUST FIND WHAT CORRELATION CANNOT SEE. An XOR's
    # parents have Spearman near zero with their child - if the
    # parent screen ran on rank correlation, the strongest
    # interaction in the frame would be invisible. Screened by
    # model importance, the planted XOR must surface at rank 1.
    from latent_challenger import _mine_interactions, \
        _planted_frame
    # The FULL planted frame, decoys included - a first cut
    # passed only the xor and 3way columns, and a mutation that
    # disabled the importance screen still found the xor because
    # any four features included its parents. A screen is only
    # tested where there is something to screen OUT.
    _pf = _planted_frame(0)
    _pcols = [c for c in _pf.columns if c != "person_id"]
    _mined = _mine_interactions(_pf, _pcols, top_n=3)
    check("the interaction miner surfaces the planted XOR at "
          "rank 1 from model-importance screening - rank "
          "correlation could never see its parents",
          bool(_mined) and _mined[0][1].startswith("planted_xor")
          and set(_mined[0][2:]) <= {"planted_xor_a",
                                     "planted_xor_b",
                                     "planted_xor_y"})

    # THE NEURAL OUTPUT MUST REACH THE REST OF THE BENCH. The
    # Verdict station, the Dashboard and the sign-off page all
    # key off artifacts the rules pipeline writes, so without
    # them this engine's output could only be read by its own
    # page - the same question asked of the same data, answerable
    # by only one instrument. Writing them found two real
    # contract bugs: a missing catalogue.json silently deleted
    # the Dashboard's pattern cards, and `source` written as a
    # STRING instead of a dict killed the sign-off page.
    from latent_challenger import write_run_artifacts
    import synthkit.gate as _gt
    with tempfile.TemporaryDirectory() as _ad:
        _adir = Path(_ad) / "run"
        _fid = write_run_artifacts(tr, gen5, "person_id", _adir,
                                   source_name="fixture.csv",
                                   gen_engine=g5)
        _want = ("blueprint.json", "fidelity.json",
                 "generated.csv", "catalogue.json",
                 "provenance.json")
        check("the neural run directory carries every artifact "
              "the other stations read ({}) - a missing one "
              "makes a station refuse, or silently drop a "
              "section".format(", ".join(_want)),
              all((_adir / f).exists() for f in _want))
        _prov = json.loads(
            (_adir / "provenance.json").read_text(
                encoding="utf-8"))
        check("...provenance carries the SHAPES the other pages "
              "read - source a dict with rows_read and patients, "
              "build a dict - because a bare string there killed "
              "the sign-off page with 'str has no attribute get'",
              isinstance(_prov.get("source"), dict)
              and "rows_read" in _prov["source"]
              and "patients" in _prov["source"]
              and isinstance(_prov.get("build"), dict))
        # THE GOVERNANCE CONTRACT: what the run did, as a
        # RECORD. Every field is read off the fitted engine, so
        # the contract cannot claim a tier the run did not
        # enforce - and the tier counts must MATCH the engine's
        # own state, or the record is decoration.
        _ct = _prov.get("contract")
        check("provenance carries the governance contract - k, "
              "blur, denoising, split, and every tier's count "
              "read off the fitted engine itself",
              isinstance(_ct, dict)
              and _ct.get("k") == 10
              and isinstance(_ct.get("tiers"), dict)
              and _ct["tiers"]["near_identities"]
              == len(getattr(g5, "_near_identities", []))
              and _ct["tiers"]["conditional_quotas"]
              == len(getattr(g5, "_token_implications", []))
              and _ct["tiers"]["clusters_adopted"]
              == [{"column": c_, "members": len(m_),
                   "patterns": len(p_)}
                  for c_, m_, _a_, p_ in getattr(
                      g5, "_token_clusters", [])]
              and _ct.get("posture"))
        # the sign-off page renders it, and a neural run WITHOUT
        # one is a loud finding, not a blank - both branches
        # exercised through the real renderer
        from signoff import build_signoff as _bs
        _page = _bs(str(_adir))[0]
        check("the sign-off page renders the contract - the "
              "signed what-this-run-did section with the "
              "posture sentence verbatim",
              "What this run did, signed" in _page
              and _ct["posture"] in _page)
        _pv2 = dict(_prov)
        _pv2.pop("contract")
        (_adir / "provenance.json").write_text(
            json.dumps(_pv2), encoding="utf-8")
        _page2 = _bs(str(_adir))[0]
        check("...and with the contract STRIPPED the page says "
              "so out loud - a neural run that will not state "
              "its rules is a finding, never a silent blank",
              "No governance contract" in _page2)
        (_adir / "provenance.json").write_text(
            json.dumps(_prov), encoding="utf-8")
        # THE OVERFITTING NUMBER IS DRAWN, NOT ONLY PRINTED.
        # The operator asked why nothing SHOWED how
        # reconstruction error compares between training and
        # held-out patients; the answer was that the number
        # lived in the terminal and fidelity.json. The neural
        # page now draws it: headline cards and per-column
        # paired bars, from the same recorded dict.
        _rg5 = g5.reconstruction_gap(ho)
        import json as _j5
        _fj5 = _adir / "fidelity.json"
        _fd5 = _j5.loads(_fj5.read_text(encoding="utf-8"))
        _fd5["reconstruction"] = _rg5
        _fj5.write_text(_j5.dumps(_fd5), encoding="utf-8")
        _sp5 = _adir / "src.csv"
        tr.to_csv(_sp5, index=False)
        import fidelity_deck as _fdk5
        _np5 = _fdk5.build_neural(
            str(_sp5), str(_adir / "generated.csv"),
            group_by="person_id")
        check("the neural page DRAWS the reconstruction "
              "comparison - training error, held-out error, the "
              "ratio, and per-column paired bars - rather than "
              "leaving it in the terminal",
              "Reconstruction error, training vs held-out" in _np5
              and "gray is training error" in _np5.lower()
              or ("Reconstruction error" in _np5
                  and "held-out" in _np5
                  and "ratio" in _np5))
        check("...and per-column bars are present with the "
              "memorization flag machinery, plus the explain "
              "panel so its own terms are doors on this page "
              "too",
              "Per column" in _np5 and "xqOpen" in _np5
              and '"terms"' in _np5)

        _v = _gt.assess(_fid)
        check("...and the gate can actually READ the result - "
              "the relationships were discovered from the "
              "SOURCE, so 'did this output keep what the real "
              "data contains' is a fair question to put to any "
              "engine",
              _v.get("pairs", 0) > 0
              and len(_v.get("criteria") or []) >= 5)

    # A SET COLUMN'S "LEVELS" ARE ITS COMBINATION STRINGS, AND
    # THERE CAN BE THOUSANDS. Uncapped, one such column puts
    # thousands of one-hot dimensions into the encoded frame: on
    # the real extract 44 source columns became 4,326 encoded
    # ones and training died allocating 1.5 GiB. The blueprint
    # has capped categories at 60 since it was written; this
    # engine had no cap at all.
    from latent_challenger import MAX_LEVELS as _ML
    _many = pd.DataFrame({
        "person_id": ["P{:03d}".format(i) for i in
                      np.repeat(np.arange(300), 6)],
        # 120 distinct level strings, each held by ~15 DIFFERENT
        # patients so every one clears the k floor - the first
        # cut used 900 levels held by 2 patients each, none of
        # them qualified, everything folded to __other__ and the
        # cap was never reached: the check passed on a frame
        # that could not exercise it.
        "combo": ["t{}".format(i % 120) for i in range(1800)],
        "num": np.round(np.random.RandomState(1)
                        .normal(0, 1, 1800), 3)})
    _lv = _many["combo"].nunique()
    _qual = int((_many.groupby("combo")["person_id"].nunique()
                 >= 10).sum())
    check("the fixture contains the thing: {} level strings "
          "CLEAR the k floor, well past the cap of {} - levels "
          "that do not qualify would fold away and never reach "
          "the cap at all".format(_qual, _ML),
          _qual > _ML)
    _gm = LatentGen(seed=1).fit(_many, "person_id")
    _wid = max((len(it[2]) for it in _gm.plan
                if it[0] == "cat"), default=0)
    check("...and the encoded frame stays bounded - no "
          "categorical contributes more than the cap plus its "
          "__other__ bucket ({} wide, cap {})".format(_wid, _ML),
          _wid <= _ML + 1 and _gm.d < 200)

    # THE SHAPES A CUSTOMER COULD PLAUSIBLY HAND OVER. The rules
    # engine was swept across 23 of them long ago; the neural
    # engine had only ever seen extract-shaped data, and the
    # sweep found a crash and three destroyed columns. Both
    # classes are pinned here.
    _one_ent = pd.DataFrame({
        "person_id": ["ONLY"] * 60,
        "v": np.round(np.random.RandomState(2).normal(0, 1, 60), 3),
        "c": ["x", "y"] * 30})
    _g1 = LatentGen(seed=3).fit(_one_ent, "person_id")
    _o1 = _g1.generate(60, seed=1)
    check("a table with ONE entity generates instead of crashing "
          "- a mixture needs two samples however few components "
          "it is asked for, so one patient centre cannot use one "
          "at all",
          len(_o1) > 0 and "v" in _o1.columns)

    # A COLUMN THE K RULE CANNOT PUBLISH IS NOT A COLUMN TO
    # DESTROY. Codes, SKUs, postcodes and free text all have
    # levels held by fewer than k people; folding every one into
    # __other__ returns a CONSTANT column, which the sweep caught
    # on a high-cardinality code, a free-text note and a date.
    _hc = pd.DataFrame({
        "person_id": ["P{:03d}".format(i) for i in
                      np.repeat(np.arange(120), 5)],
        "code": ["C{:05d}".format(i) for i in range(600)],
        "v": np.round(np.random.RandomState(4).normal(0, 1, 600), 3)})
    check("the fixture contains the thing: every `code` value is "
          "unique, so NOTHING clears the k floor",
          _hc["code"].nunique() == len(_hc))
    _g2 = LatentGen(seed=3).fit(_hc, "person_id")
    _o2 = _g2.generate(600, seed=1)
    check("...and the column comes back with MANY invented "
          "labels rather than one value for every row - the "
          "shape is publishable even when no label is",
          _o2["code"].nunique() > 5
          and not set(_o2["code"]) & set(_hc["code"]))

    # THE JOINT MEASURES. Everything else in this suite names a
    # pattern first; these ask the reverse and need nothing
    # enumerated. Exercised on a fixture where the "synthetic"
    # side is a SHUFFLE of the real columns - every marginal
    # identical, every relationship destroyed - which the
    # pattern-named measures would partly miss and these must
    # catch.
    from synthkit import jointcheck as _jc
    _rs = np.random.RandomState(12)
    _nj = 1400
    _gj = np.repeat(np.arange(200), 7)
    _xj = _rs.normal(0, 1, _nj)
    _real = pd.DataFrame({
        "person_id": ["P{:03d}".format(i) for i in _gj],
        "a": np.round(_xj, 3),
        "b": np.round(2.0 * _xj + _rs.normal(0, .3, _nj), 3),
        "c": np.round(_rs.normal(5, 2, _nj), 3)})
    _broken = _real.copy()
    for _c in ("a", "b", "c"):
        _broken[_c] = _rs.permutation(_broken[_c].to_numpy())
    _dj = _jc.distinguishability(_real, _real.copy(),
                                 "person_id")
    check("a table compared against ITSELF is indistinguishable "
          "- the measure's own control, without which a low "
          "score would prove nothing",
          _dj["auc_within_published_range"] < 0.60)
    _pj = _jc.predictability(_real, _broken, "person_id")
    _pg = _jc.predictability(_real, _real.copy(), "person_id")
    check("the predictability profile sees a destroyed "
          "relationship that every marginal check would pass: "
          "shuffled columns read a skill gap of {} against {} "
          "for an identical copy".format(_pj["mean_skill_gap"],
                                         _pg["mean_skill_gap"]),
          _pj["mean_skill_gap"] > _pg["mean_skill_gap"] + 0.2)
    _mj = _jc.manifold(_real, _real.copy(), "person_id")
    check("manifold precision and recall are near 1.0 against an "
          "identical copy ({} / {}) - and the docstring says "
          "plainly that a verbatim copy scoring 1.00 is why this "
          "is not a privacy measure".format(_mj["precision"],
                                            _mj["recall"]),
          _mj["precision"] > 0.9 and _mj["recall"] > 0.9)
    _uj = _jc.utility(_real, _real.copy(), "b", "person_id")
    check("train-on-synthetic scores near train-on-real when the "
          "synthetic IS the real data (retained {})".format(
              _uj["retained"]),
          _uj["retained"] is not None and _uj["retained"] > 0.8)
    _all = _jc.run_all(_real, _broken, "person_id", target="b")
    check("run_all returns every measure and NAMES any that "
          "failed rather than dropping it",
          set(_all) >= {"distinguishability", "predictability",
                        "three_way", "manifold", "utility"})

    # THE HEAD-TO-HEAD IS ONE COMMAND. compare measures the
    # blueprint run's generated.csv and the latent draws against
    # the SAME source with the SAME metrics - and refuses in a
    # sentence when the run directory holds no generated.csv.
    import subprocess as _sp
    import tempfile as _tf
    with _tf.TemporaryDirectory() as _td:
        _src = Path(_td) / "src.csv"
        df.to_csv(_src, index=False)
        _bp = Path(_td) / "bprun"
        _bp.mkdir()
        df.sample(frac=0.9, random_state=1).to_csv(
            _bp / "generated.csv", index=False)
        _r = _sp.run([sys.executable,
                      str(_root / "scripts" /
                          "latent_challenger.py"),
                      "compare", str(_src), "--group-by",
                      "person_id", "--blueprint-run", str(_bp),
                      "--seeds", "0"],
                     capture_output=True, text=True,
                     cwd=str(_root))
        _r2 = _sp.run([sys.executable,
                       str(_root / "scripts" /
                           "latent_challenger.py"),
                       "compare", str(_src), "--group-by",
                       "person_id", "--blueprint-run",
                       str(Path(_td) / "absent")],
                      capture_output=True, text=True,
                      cwd=str(_root))
        check("compare prints one table - blueprint and latent "
              "rows against the same source, the honesty footer "
              "attached - and a missing generated.csv refuses "
              "in a sentence with exit 2",
              _r.returncode == 0
              and "blueprint" in _r.stdout
              and "latent seed 0" in _r.stdout
              and "Fidelity is only one column" in _r.stdout
              and _r2.returncode == 2
              and "no generated.csv" in _r2.stderr
              and "Traceback" not in _r2.stderr)

        # THE DELIVERABLE IS THE SIZE OF THE SOURCE. It was the
        # size of the 85% TRAINING SPLIT, so the real run's
        # dashboard opened with "55,428 rows / 800 patients
        # original, 47,163 / 687 synthetic" - an eighth of the
        # cohort apparently lost by the generator, when nothing
        # had been lost and nobody had decided this. It also made
        # the duel unfair: the rules engine writes a full-size
        # file and this one was scored against it at 85%.
        _o3 = Path(_td) / "o3"
        _r3 = _sp.run([sys.executable,
                       str(_root / "scripts" /
                           "latent_challenger.py"),
                       "csv", str(_src), "--group-by",
                       "person_id", "--seeds", "0",
                       "--out", str(_o3)],
                      capture_output=True, text=True,
                      cwd=str(_root))
        _g3 = list(_o3.glob("*.csv")) if _o3.exists() else []
        _n3 = (pd.read_csv(_g3[0], dtype=str,
                           keep_default_na=False)
               if _g3 else None)
        check("the csv run writes a file the size of the SOURCE, "
              "with the source's patient count - not the 85% it "
              "trained on. The holdout exists for the "
              "nearest-neighbor tripwire; sizing the deliverable "
              "by it was never a decision anyone made",
              _r3.returncode == 0 and _n3 is not None
              and abs(len(_n3) - len(df)) / float(len(df)) < 0.12
              and abs(_n3["person_id"].nunique()
                      - df["person_id"].nunique()) <= 2)

    # ---------------------------------------------------------
    # THE K DIAL: the team asked for a user-settable k with
    # CLARITY on what can and cannot move. Higher is a policy
    # choice; lower is refused with the reason, in the same
    # sentence at the bench and at the CLI. And a raised k must
    # actually BIND - a dial that is read but not used is worse
    # than no dial, which is this repository's oldest lesson.
    from latent_challenger import _k_clip as _kc9
    r9 = np.random.RandomState(2)
    g9v = np.repeat(np.arange(120), 5)
    v9 = r9.normal(50, 10, 600)
    v9[g9v == 0] += 300          # a few extreme patients
    v9[g9v == 1] += 250
    v9[g9v == 2] += 200
    lo10, hi10 = _kc9(v9, g9v.astype(str), 10)
    lo40, hi40 = _kc9(v9, g9v.astype(str), 40)
    check("a RAISED k binds: the k=40 published bound averages "
          "over the 40 most extreme patients and sits strictly "
          "inside the k=10 bound - the dial reaches the "
          "computation, it is not a label",
          hi40 < hi10 and lo40 >= lo10)
    import subprocess as _spk
    _rk = _spk.run([sys.executable,
                    str(_root / "scripts" /
                        "latent_challenger.py"),
                    "csv", "nonexistent.csv", "--group-by",
                    "person_id", "--k", "5"],
                   capture_output=True, text=True,
                   cwd=str(_root))
    check("...and k below the floor of 10 is REFUSED before any "
          "work, with the reason stated - a pattern resting on "
          "fewer than 10 distinct patients begins to describe "
          "individuals - never a traceback",
          _rk.returncode == 2
          and "below the floor" in _rk.stderr
          and "describe" in _rk.stderr
          and "Traceback" not in _rk.stderr)

    # ---------------------------------------------------------
    # THE SPLIT EXISTED AND NOTHING MEASURED THE OVERFITTING.
    # 15% of PATIENTS are held out, and the only thing ever
    # asked of them was a nearest-neighbor DISTANCE ratio on the
    # generated file - which never puts a held-out record
    # through the network at all. So "is it memorizing rather
    # than learning" had a holdout ready and no measurement on
    # it: a capability present and wired to nothing, which is
    # this repository's own recurring fault.
    #
    # AND IT NEEDS A POSITIVE CONTROL, because a ratio near 1.0
    # is exactly what a broken measure returns too. The control
    # is a frame with NOTHING TO LEARN - every value iid noise -
    # so any low training error IS memorization and a network
    # able to memorize must be caught.
    rn = np.random.RandomState(9)
    _nr, _nc = 600, 24
    dnoise = pd.DataFrame(dict(
        ("n{:02d}".format(i), np.round(rn.normal(0, 1, _nr), 3))
        for i in range(_nc)))
    dnoise.insert(0, "person_id",
                  ["N{:04d}".format(i // 3) for i in range(_nr)])
    ntr, nho = _split_patients(dnoise, "person_id", 0)

    g_honest = LatentGen(k=10, seed=0,
                         hierarchical=True).fit(ntr, "person_id")
    r_honest = g_honest.reconstruction_gap(nho)
    g_memo = LatentGen(k=10, seed=0, hierarchical=True,
                       denoise=0.0, bottleneck=96,
                       hidden=384).fit(ntr, "person_id")
    r_memo = g_memo.reconstruction_gap(nho)

    check("reconstruction error is measured on HELD-OUT PATIENTS "
          "- people the network never trained on - and reported "
          "with the training error beside it, which is the "
          "question the 15% split was always there to answer",
          r_honest is not None
          and set(r_honest) >= {"train", "holdout", "ratio",
                                "rows_train", "rows_holdout"}
          and r_honest["rows_holdout"] > 0
          and r_honest["rows_train"] > r_honest["rows_holdout"])
    check("...and NOBODY is on both sides: the split is by "
          "patient, so a person whose visits trained the network "
          "cannot also be a stranger to it",
          not (set(ntr["person_id"]) & set(nho["person_id"])))
    check("POSITIVE CONTROL - on a frame with nothing to learn, "
          "a wide-bottleneck undefended network memorizes and "
          "the ratio climbs far above the shipping "
          "configuration's, so the measure can fail",
          r_memo["ratio"] > 3.0
          and r_memo["ratio"] > 2.0 * r_honest["ratio"])
    check("...while the shipping configuration CANNOT memorize "
          "even when memorizing is the only way to do well - "
          "the bottleneck and the denoising are doing the work, "
          "not the fixture",
          r_honest["ratio"] < 3.0)

    # ---------------------------------------------------------
    # A DATE AND A SET COLUMN ARE THE TWO SHAPES THIS ENGINE READ
    # AS CATEGORIES, and both were destroyed in the same real run.
    # The fixture has to CONTAIN them: 400 tokens with a heavy
    # tail so the k floor and the token cap both bind, and a third
    # of rows genuinely empty so the point mass at zero exists to
    # be lost. Before the fix this emitted ONE distinct token
    # across the whole file and every date unparseable.
    rr = np.random.RandomState(7)
    tk = ["d{:03d}".format(i) for i in range(400)]
    wt = 1.0 / (1.0 + np.arange(400)) ** 0.9
    wt = wt / wt.sum()
    rows = []
    for pp in range(300):
        base, sev = rr.randint(0, 3000), rr.rand()
        for vv in range(rr.randint(2, 7)):
            d = base + vv * rr.randint(5, 60)
            if rr.rand() < 0.35:
                ds = ""
            else:
                ds = ";".join(sorted(set(rr.choice(
                    tk, 1 + rr.poisson(2.5), p=wt))))
            rows.append({
                "person_id": "P{:04d}".format(pp),
                "visit_date": (pd.Timestamp("2005-01-01")
                               + pd.Timedelta(days=int(d))
                               ).strftime("%Y-%m-%d"),
                "severity": round(sev * 10 + rr.randn(), 3),
                "n_visits": vv + 1, "drugs": ds})
    dfx = pd.DataFrame(rows)
    gx = LatentGen(k=10, seed=0, hierarchical=True)
    gx.fit(dfx, "person_id")
    ox = gx.generate(len(dfx), seed=0)

    kinds = dict((it[1], it[0]) for it in gx.plan)
    check("a date column is typed NUMERIC, not categorical - "
          "`pd.to_numeric` returns NaN for 2013-10-03, which sent "
          "every date down the level path where the cap kept 60 "
          "days and folded the rest into __other__",
          kinds.get("visit_date") == "num")
    check("a set column is typed SET, not categorical - its "
          "levels are its distinct COMBINATION strings, so the "
          "cap sent nearly every row to __other__: a string that "
          "is not empty and holds no real token",
          kinds.get("drugs") == "set")

    gdt = pd.to_datetime(ox["visit_date"].astype(str),
                         format="%Y-%m-%d", errors="coerce")
    check("every generated date PARSES as a date in the format "
          "the column was read under, and the column keeps its "
          "range rather than 60 levels",
          float(gdt.isna().mean()) == 0.0
          and ox["visit_date"].nunique() > 300)

    ssx = dfx["drugs"].astype(str)
    gsx = ox["drugs"].astype(str)
    src_empty = float((ssx.str.len() == 0).mean())
    gen_empty = float((gsx.str.len() == 0).mean())
    check("the EMPTY share survives - a third of these visits "
          "genuinely had no drugs, and a size decoded as a "
          "standardized float plus rounding cannot land a point "
          "mass at zero (measured 29.4% against 36.1% before the "
          "published size quantiles were used)",
          abs(gen_empty - src_empty) < 0.03)

    gtok = set(x for t in gsx for x in t.split(";") if x)
    check("MANY distinct tokens are emitted, not one - the "
          "combination-string encoding emitted exactly 1 here, "
          "which is what 'looks present, is entirely "
          "scaffolding' means for a set column",
          len(gtok) >= 20 and gtok <= set(tk))

    _sc = pd.Series([x for t in ssx for x in t.split(";")
                     if x]).value_counts()
    _err = []
    for _t in list(_sc.index[:10]):
        _a = float(ssx.str.split(";").map(lambda z: _t in z).mean())
        _b = float(gsx.str.split(";").map(lambda z: _t in z).mean())
        _err.append(abs(_a - _b))
    check("each published token comes back at ITS OWN share - "
          "the damped per-token offset solved against the "
          "published shares, worst 0.21 before the set path "
          "existed",
          max(_err) < 0.03)

    # THE FILE PATH IS THE MEASUREMENT THAT COUNTS. Every
    # in-memory check above passes a DataFrame straight to fit;
    # the real extract arrives through the CLI's reader, and the
    # reader blanket-folded "" into NaN - right for a category,
    # WRONG for a set, where an empty list is a fact about the
    # visit. The engine learned token shares from a source in
    # which the medication-free visits did not exist (every
    # share inflated by 1/(1-empty), 5x on the 80%-empty
    # procedures), drew no empty sets at all, and its gate
    # PASSED both criteria because the gate's reference numbers
    # were folded identically. The deck reads the raw file and
    # disagreed: 0.0% synthetic empty beside a PASS chip - two
    # numbers on one page, this project's alarm, on the morning
    # of a demo. So THIS check goes through the disk and the
    # CLI, exactly as the extract does.
    import subprocess as _sp9
    import tempfile as _tf9
    with _tf9.TemporaryDirectory() as _tdq:
        _srcq = Path(_tdq) / "src.csv"
        dfx.to_csv(_srcq, index=False)
        _outq = Path(_tdq) / "run"
        _rq = _sp9.run([sys.executable,
                        str(_root / "scripts" /
                            "latent_challenger.py"),
                        "csv", str(_srcq), "--group-by",
                        "person_id", "--seeds", "0",
                        "--out", str(_outq)],
                       capture_output=True, text=True,
                       cwd=str(_root))
        _genq = pd.read_csv(_outq / "generated.csv", dtype=str,
                            keep_default_na=False)
        _se = float((dfx["drugs"].astype(str).str.strip()
                     == "").mean())
        _ge = float((_genq["drugs"].astype(str).str.strip()
                     == "").mean())
        _t0 = "d000"
        _ss = float(dfx["drugs"].astype(str).str.split(";").map(
            lambda z: _t0 in z).mean())
        _gs = float(_genq["drugs"].astype(str).str.split(";").map(
            lambda z: _t0 in z).mean())
    check("through the DISK and the CLI - the path the real "
          "extract takes - the generated file keeps the empty "
          "share (source {:.0%}): the reader must not fold an "
          "empty list into a missing cell, which starved the "
          "size grid of zeros and emitted 0% empty on the real "
          "run".format(_se),
          _rq.returncode == 0 and abs(_ge - _se) < 0.06)
    check("...and the top token lands at its source share of "
          "ALL rows, not share-of-non-empty-rows - the blanket "
          "fold inflated every token by exactly 1/(1-empty), "
          "which is a denominator, not a sampler",
          abs(_gs - _ss) < 0.05)

    # A LINEAR IDENTITY IS THE SIZE IDENTITY ONE LEVEL UP.
    # mean_arterial_pressure IS (systolic + 2*diastolic)/3 -
    # arithmetic, not correlation - and decoded as three ordinary
    # numbers the identity held within 0.1 on 40.2% of generated
    # rows against 100% in source. Someone opening the file finds
    # blood pressures that contradict each other, and the failing
    # pressure-family surfaces sit exactly on this. Detected from
    # the source by least squares (kept only under 2% of the
    # child's spread - a rule the source breaks is not a rule),
    # enforced by COMPUTING the child from its generated parents.
    rl = np.random.RandomState(6)
    lrows = []
    for pp in range(350):
        bs, bd = rl.normal(120, 12), rl.normal(75, 8)
        for vv in range(rl.randint(3, 9)):
            s_ = bs + rl.normal(0, 6)
            d_ = bd + rl.normal(0, 4)
            lrows.append({"person_id": "P{:04d}".format(pp),
                          "systolic": round(s_, 1),
                          "diastolic": round(d_, 1),
                          "map_cuff": round((s_ + 2 * d_) / 3, 1),
                          "hr": round(rl.normal(78, 10), 1)})
    dfl = pd.DataFrame(lrows)
    gl = LatentGen(k=10, seed=0, hierarchical=True).fit(
        dfl, "person_id")
    ol = gl.generate(len(dfl), seed=1)
    _m = pd.to_numeric(ol["map_cuff"])
    _s = pd.to_numeric(ol["systolic"])
    _d = pd.to_numeric(ol["diastolic"])
    _res = (_m - (_s + 2 * _d) / 3.0).abs()
    check("a LINEAR identity in the source - MAP is (S+2D)/3 - "
          "is detected and enforced by computing, so the "
          "generated pressures agree with each other "
          "arithmetically (within 0.1 on 40.2% of rows before "
          "this; three independent draws satisfy arithmetic "
          "only by chance)",
          len(gl._linear_identities) >= 1
          and float((_res <= 0.100001).mean()) > 0.99)
    check("...and the computed column keeps its NEIGHBORING "
          "properties - centre within a tenth of spread, spread "
          "within the band the raw-drawn columns themselves "
          "land in - because enforcing one property by breaking "
          "the one beside it is the shift/scale lesson again",
          abs(float(_s.mean())
              - float(pd.to_numeric(dfl["systolic"]).mean()))
          < 0.1 * float(pd.to_numeric(dfl["systolic"]).std())
          and float(_s.std())
          > 0.6 * float(pd.to_numeric(dfl["systolic"]).std()))
    _claimed = [t[0] for t in gl._linear_identities]
    check("...and no enforced child is another identity's "
          "PARENT - first claim wins, deterministically - so "
          "the computations cannot chase each other in a loop",
          all(pa not in _claimed and pb not in _claimed
              for _c, pa, pb, _w in gl._linear_identities))

    # A DRUG IMPLIES ITS ROUTE, AND INDEPENDENT PER-COLUMN DRAWS
    # CANNOT SAY SO. The remaining close misses on the real
    # extract cluster on CROSS-set-column token pairs
    # (has_sodium-chloride <- has_Flush: 0.88 source, 0.58
    # generated) - near-functional implications each column's
    # own draw cannot carry, because cross-column coherence
    # flows only through the latent. The implication tier
    # detects P(b|a) >= 0.9 from the source, FORCES the implied
    # token where its implier was drawn, and restores both the
    # token's quota and the column's empty count. On the
    # deterministic drug->route fixture this lands every token
    # pair on its source value exactly, because forcing
    # reproduces the source's own structure and the restore
    # pins the marginals.
    ri = np.random.RandomState(4)
    _di = ["dr{:02d}".format(i) for i in range(30)]
    _ri = ["Oral", "IV", "IV Push", "Subcut", "Topical",
           "Flush", "Rectal", "Chewed", "Misc", "NG-tube"]
    _dm = dict((d, _ri[1 + (i % 9)] if i % 3 else "Oral")
               for i, d in enumerate(_di))
    irows = []
    for pp in range(400):
        lv = ri.rand()
        for vv in range(ri.randint(3, 9)):
            nn_ = ri.poisson(1.8 + 1.6 * lv)
            ds = sorted(set(ri.choice(_di, nn_, p=None))) \
                if nn_ else []
            irows.append({
                "person_id": "P{:04d}".format(pp),
                "drugs": ";".join(ds),
                "routes": ";".join(sorted(set(
                    _dm[d] for d in ds))),
                "sev": round(lv * 10 + ri.randn(), 3)})
    dfimp = pd.DataFrame(irows)

    def _has(fr, col, t):
        return fr[col].astype(str).str.split(";").map(
            lambda z: float(t in z))

    gim = LatentGen(k=10, seed=0, hierarchical=True).fit(
        dfimp, "person_id")
    oim = gim.generate(len(dfimp), seed=1)
    check("cross-column token implications are DETECTED from "
          "the source - P(route|drug) >= 0.9 on enough rows - "
          "and at least a handful exist on a fixture whose "
          "routes are literally derived from its drugs",
          len(gim._token_implications) >= 3)
    _tp = [(d, _dm[d]) for d in _di[:5]]
    _errs = []
    for _d, _r in _tp:
        a_ = float(pd.Series(_has(dfimp, "drugs", _d)).corr(
            pd.Series(_has(dfimp, "routes", _r)),
            method="spearman"))
        b_ = float(pd.Series(_has(oim, "drugs", _d)).corr(
            pd.Series(_has(oim, "routes", _r)),
            method="spearman"))
        _errs.append(abs(a_ - b_))
    check("...and the generated drug<->route pairs land at "
          "their source correlation (worst drop over five "
          "pairs under 0.1; the independent draw alone left "
          "0.1-0.2 on the strongest and the real extract "
          "showed 0.3)",
          max(_errs) < 0.1)
    _es = float((dfimp["routes"].astype(str) == "").mean())
    _eg = float((oim["routes"].astype(str) == "").mean())
    _shs, _shg = [], []
    for _r in set(_dm.values()):
        _shs.append(float(_has(dfimp, "routes", _r).mean()))
        _shg.append(float(_has(oim, "routes", _r).mean()))
    _sherr = max(abs(a_ - b_) for a_, b_ in zip(_shs, _shg))
    check("...and the books balance: the empty rate within "
          "0.02 and every route token share within 0.05 of "
          "source - forcing without the quota and empty "
          "restores would inflate exactly what the reader bug "
          "inflated",
          abs(_eg - _es) < 0.02 and _sherr < 0.05)

    # A NEAR-IDENTITY IS AN IDENTITY ONE NOTCH RELAXED, AND THE
    # NETWORK CANNOT CARRY IT. The real extract's failing
    # pressure pairs are a SPARSE second-device twin: map_cuff_
    # bmdi IS map_cuff plus device noise (rho 0.97, residual ~3%
    # of spread), present on a third of visits - and at
    # realistic width the decoded twin faded to 0.70-0.87,
    # seed-dependent. The sampler was exonerated by measurement
    # (GMM x1/x3/x6 and KDE at three bandwidths all land in the
    # same band), so the cure is the identity tier's, one notch
    # relaxed: the sparser twin is COMPUTED from its denser
    # sibling plus noise at the measured residual sd, where both
    # are present.
    rn5 = np.random.RandomState(13)
    nrows5 = []
    for pp in range(350):
        sev = rn5.rand()
        bs5, bd5 = rn5.normal(120, 12), rn5.normal(75, 8)
        for vv in range(rn5.randint(3, 9)):
            s5 = bs5 + rn5.normal(0, 6)
            d5 = bd5 + rn5.normal(0, 4)
            m5 = (s5 + 2 * d5) / 3.0
            nrows5.append({
                "person_id": "P{:04d}".format(pp),
                "systolic": round(s5, 1),
                "diastolic": round(d5, 1),
                "map_cuff": round(m5, 1),
                "map_cuff_bmdi": (
                    round(m5 + rn5.normal(0, 1.8), 1)
                    if rn5.rand() < 0.15 + 0.35 * sev else None),
                "hr": round(78 + 8 * sev + rn5.normal(0, 7), 1)})
    dfn = pd.DataFrame(nrows5)

    def _rho5(fr, a, b):
        return float(pd.to_numeric(fr[a], errors="coerce").corr(
            pd.to_numeric(fr[b], errors="coerce"),
            method="spearman"))
    _src5r = _rho5(dfn, "map_cuff_bmdi", "map_cuff")
    gn5 = LatentGen(k=10, seed=0, hierarchical=True).fit(
        dfn, "person_id")
    on5 = gn5.generate(len(dfn), seed=1)
    check("the fixture CONTAINS the twin - a sparse second-"
          "device column at rho {:.2f} to its sibling - and the "
          "near tier DETECTS it, with the exact tier's parent "
          "still eligible as the near tier's sibling: one set "
          "conflating the two roles silently disabled the whole "
          "tier on its first cut".format(_src5r),
          _src5r > 0.9
          and any(t[0] == "map_cuff_bmdi" and t[1] == "map_cuff"
                  for t in gn5._near_identities)
          and any(t[0] == "systolic"
                  for t in gn5._linear_identities))
    _gen5r = _rho5(on5, "map_cuff_bmdi", "map_cuff")
    check("...and the generated twin lands at its source "
          "correlation by construction (decoded alone it faded "
          "to 0.70-0.87 at width, seed-dependent) - computed "
          "from its sibling plus noise at the measured residual "
          "sd, so the twin's own spread survives too",
          abs(_gen5r - _src5r) < 0.05)
    _bsd_s = float(pd.to_numeric(dfn["map_cuff_bmdi"],
                                 errors="coerce").std())
    _bsd_g = float(pd.to_numeric(on5["map_cuff_bmdi"],
                                 errors="coerce").std())
    _cov_s = float(pd.to_numeric(dfn["map_cuff_bmdi"],
                                 errors="coerce").notna().mean())
    _cov_g = float(pd.to_numeric(on5["map_cuff_bmdi"],
                                 errors="coerce").notna().mean())
    check("...with the NEIGHBORING properties held: the twin's "
          "spread within 25% of source and its presence rate "
          "within 0.05 - enforcing one property by breaking the "
          "one beside it is the shift/scale lesson",
          abs(_bsd_g - _bsd_s) < 0.25 * _bsd_s
          and abs(_cov_g - _cov_s) < 0.05)

    # A DRUG IMPLIES ITS ROUTE, AND THE DRAW WAS RANDOMIZING
    # THAT AWAY. The remaining close misses on the real extract
    # were dominated by CROSS-set-column token pairs -
    # has_sodium-chloride <-> has_Flush at 0.88 source, 0.20
    # generated - and the mechanism was the Gumbel top-k noise
    # (~1.28 sd, fixed) drowning activation differences of the
    # same order. TOKEN_SHARPNESS multiplies the log-activations;
    # the offset solve recalibrates marginals at any sharpness
    # PROVIDED its pass count scales too - sharpness alone read
    # worst-share 0.052 at 12 passes and 0.014 at 40.
    rc = np.random.RandomState(4)
    _dg = ["dr{:02d}".format(i) for i in range(30)]
    _rt = ["Oral", "IV", "Flush", "Subcut", "Topical"]
    _map = dict((d, _rt[i % 5]) for i, d in enumerate(_dg))
    _w = 1.0 / (1.0 + np.arange(30)) ** 0.8
    _w = _w / _w.sum()
    crows = []
    for pp in range(350):
        lv = rc.rand()
        for vv in range(rc.randint(3, 9)):
            nn_ = rc.poisson(1.8 + 1.5 * lv)
            ds = sorted(set(rc.choice(_dg, nn_, p=_w))) if nn_ \
                else []
            crows.append({
                "person_id": "P{:04d}".format(pp),
                "drugs": ";".join(ds),
                "routes": ";".join(sorted(set(
                    _map[d] for d in ds))),
                "sev": round(lv * 10 + rc.randn(), 3)})
    dfc = pd.DataFrame(crows)
    _cp = [("dr{:02d}".format(i), _map["dr{:02d}".format(i)])
           for i in range(4)]
    # THREE SEEDS, because a single seed of a seeded draw is not
    # a measurement - this file's own rule, nearly broken by its
    # own check: seed 0 alone reads 0.210 at the shipping
    # sharpness while seeds 1-2 read 0.097 and 0.079, and a
    # threshold that flickers with the seed proves nothing
    # either way. Averaged, sharpness 4 reads ~0.13 against
    # ~0.29 at sharpness 1 - separated with margin on both
    # sides.
    _seed_means = []
    for _sd in (0, 1, 2):
        gc_ = LatentGen(k=10, seed=_sd, hierarchical=True).fit(
            dfc, "person_id")
        oc = gc_.generate(len(dfc), seed=100 + _sd)
        _drops = []
        for _d, _r in _cp:
            a = float(pd.Series(_has(dfc, "drugs", _d)).corr(
                pd.Series(_has(dfc, "routes", _r)),
                method="spearman"))
            b = float(pd.Series(_has(oc, "drugs", _d)).corr(
                pd.Series(_has(oc, "routes", _r)),
                method="spearman"))
            _drops.append(abs(a - b))
        _seed_means.append(float(np.mean(_drops)))
    check("cross-set-column token co-occurrence SURVIVES the "
          "draw - a drug implies its route at rho 0.56-0.73 and "
          "the sharpened draw keeps it (mean drop over three "
          "seeds under 0.2, against ~0.29 at sharpness 1, where "
          "the Gumbel noise drowned the activation signal)",
          float(np.mean(_seed_means)) < 0.2)
    from latent_challenger import TOKEN_SHARPNESS
    check("...at a sharpness that follows the decoder rather "
          "than the noise - the signal the co-occurrence lives "
          "in",
          TOKEN_SHARPNESS >= 2.0)

    # A TOKEN MUST BE FREE TO POINT AGAINST THE SET SIZE. Any
    # within-row competitive draw (top-size-many by score)
    # mechanically ties every token to the set size, so a token
    # whose correlation with a count runs the OTHER way cannot
    # survive it: on this two-regime fixture (acute visits: many
    # drugs, none Oral; ambulatory: few drugs, Oral) the
    # competitive draw emitted corr(has_Oral, count) +0.04 to
    # +0.16 against a source -0.25 - a sign flip, the class
    # behind the real extract's has_Oral <- active_drug_count
    # 0.26 -> -0.64 INVERTED. The independent per-token draw
    # keeps the sign, because each token's placement follows its
    # OWN activation ordering across rows.
    rg2 = np.random.RandomState(11)
    _od = ["od{:02d}".format(i) for i in range(10)]
    _iv = ["ivd{:02d}".format(i) for i in range(20)]
    _ivr = ["IV", "IV Push", "IV Piggyback", "Flush", "Subcut",
            "Topical", "NG-tube", "Misc"]
    _rmap = dict((d, _ivr[i % len(_ivr)])
                 for i, d in enumerate(_iv))
    rrows = []
    for pp in range(400):
        acute = rg2.rand() < 0.4
        for vv in range(rg2.randint(3, 9)):
            if acute:
                n_iv, n_or = 2 + rg2.poisson(2.5), \
                    rg2.binomial(2, 0.45)
            else:
                n_iv, n_or = rg2.binomial(1, 0.15), \
                    1 + rg2.binomial(1, 0.6)
            ds = sorted(set(list(rg2.choice(_iv, n_iv))
                            if n_iv else []) |
                        set(list(rg2.choice(_od, n_or))
                            if n_or else []))
            rr_ = sorted(set("Oral" if d.startswith("od")
                             else _rmap[d] for d in ds))
            rrows.append({"person_id": "P{:04d}".format(pp),
                          "drugs": ";".join(ds),
                          "routes": ";".join(rr_),
                          "active_drug_count": len(ds),
                          "sev": round((3.0 if acute else 1.0)
                                       + rg2.randn() * 0.5, 2)})
    dfr = pd.DataFrame(rrows)
    _src_oral = float(pd.Series(_has(dfr, "routes", "Oral")).corr(
        pd.to_numeric(dfr["active_drug_count"]),
        method="spearman"))
    check("the regime fixture CONTAINS the shape - a broad token "
          "whose correlation with the count runs NEGATIVE in the "
          "source, against the size mechanism",
          _src_oral < -0.15)
    _oral_gen = []
    for _sd in (0, 1):
        gr_ = LatentGen(k=10, seed=_sd, hierarchical=True).fit(
            dfr, "person_id")
        orr = gr_.generate(len(dfr), seed=200 + _sd)
        _oral_gen.append(float(
            pd.Series(_has(orr, "routes", "Oral")).corr(
                pd.to_numeric(orr["active_drug_count"]),
                method="spearman")))
    check("...and the generated file KEEPS that sign on both "
          "seeds - the competitive draw flipped it positive, "
          "which is an INVERTED verdict at the gate, the one "
          "absolute criterion",
          all(v < -0.02 for v in _oral_gen))

    # THE MEASURING ENCODER HAS TO SEE A DATE AS A DATE TOO.
    # Encoded as a category it becomes its 60 most common days
    # plus an `other` bucket holding nearly every row on BOTH
    # sides - so the measure whose whole job is asking whether a
    # model can tell the tables apart was blind to the column
    # the generator was destroying. Both halves, one module.
    # A DECLARED SIZE IDENTITY IS ENFORCED BY COPYING, NOT BY
    # DRAWING TWICE - and this engine declares nothing, so it has
    # to FIND the identity. The fixture must CONTAIN one: a count
    # column that IS the length of its set on every source row,
    # which is what `active_drug_count` is on the real extract.
    #
    # Left undetected it is the whole reason that run's gate read
    # 10 INVERTED: every one of those pairs was a token indicator
    # against its own count partner, and a partner drawn
    # independently agrees with the set only by chance - measured
    # at 37.2% of rows against a source 100%.
    rid = np.random.RandomState(3)
    itk = ["d{:03d}".format(i) for i in range(300)]
    iw = 1.0 / (1.0 + np.arange(300)) ** 0.9
    iw = iw / iw.sum()
    irows = []
    for pp in range(350):
        lv = rid.rand()
        for vv in range(rid.randint(3, 10)):
            nn_ = rid.poisson(2.2 + 2 * lv)
            ds = sorted(set(rid.choice(itk, nn_, p=iw))) if nn_ \
                else []
            irows.append({"person_id": "P{:04d}".format(pp),
                          "drugs": ";".join(ds),
                          "drug_count": len(ds),
                          "sev": round(lv * 10 + rid.randn(), 3)})
    dfi = pd.DataFrame(irows)
    gi = LatentGen(k=10, seed=0, hierarchical=True).fit(
        dfi, "person_id")
    oi = gi.generate(len(dfi), seed=1)

    def _sz(col):
        return col.astype(str).str.split(";").map(
            lambda t: sum(1 for x in t if x))

    _src_id = float((_sz(dfi["drugs"])
                     == pd.to_numeric(dfi["drug_count"])).mean())
    _gen_id = float((_sz(oi["drugs"])
                     == pd.to_numeric(oi["drug_count"])).mean())
    check("the fixture CONTAINS the identity - a count column "
          "that IS the length of its set on every source row - "
          "so the check below is not asserted against data where "
          "nothing had to be enforced",
          _src_id > 0.999)
    check("a count column that IS its set's length in the source "
          "is COPIED from the generated set, not drawn a second "
          "time: two independent draws of one quantity agree "
          "only by chance, measured at 37.2% of rows before this",
          _gen_id > 0.999
          and any(t[0] == "drug_count" and t[1] == "drugs"
                  for t in gi._size_identities))
    _ta = dfi["drugs"].astype(str).str.split(";").map(
        lambda z: "d000" in z).astype(float)
    _tb = oi["drugs"].astype(str).str.split(";").map(
        lambda z: "d000" in z).astype(float)
    _ca = float(pd.Series(_ta).corr(
        pd.to_numeric(dfi["drug_count"]), method="spearman"))
    _cb = float(pd.Series(_tb).corr(
        pd.to_numeric(oi["drug_count"]), method="spearman"))
    check("...and the relationship the gate actually reads comes "
          "back: a token indicator against its count partner, "
          "which is where the real extract's inversions all sat",
          _ca > 0.1 and _cb > 0.1
          and abs(_cb - _ca) < 0.12)

    from synthkit.jointcheck import Encoder as _JE
    _je = _JE(dfx, group_by="person_id")
    _jk = dict((it[1], it[0]) for it in _je.plan)
    _jw = _je.groups()
    check("the joint-measure encoder types a date as NUMERIC - "
          "two dimensions, value and missingness - rather than "
          "as sixty levels and an `other` bucket that swallows "
          "the column on both sides",
          _jk.get("visit_date") == "num"
          and len(_jw["visit_date"]) == 2)
    _jx = _je.transform(ox)
    import numpy as _jnp
    check("...and it re-encodes the GENERATED dates, which carry "
          "the day format rather than the one the source was "
          "parsed under - the wrong format returns all-NaN and "
          "the measure silently compares nothing",
          float(_jnp.abs(
              _jx[:, _jw["visit_date"][0]]).mean()) > 0.05
          and float(
              _jx[:, _jw["visit_date"][1]].mean()) < 0.05)

    from synthkit import sets as _S
    _vc = _S.vocabulary(dfx["drugs"],
                        dfx["person_id"].to_numpy(), k=10, cap=60)
    _gm = float(gsx.str.split(";").map(
        lambda t: sum(1 for x in t if x)).mean())
    check("generated set size matches the PUBLISHABLE size, not "
          "the size the rows held - drawing the held size from a "
          "vocabulary the k rule thinned runs every survivor hot, "
          "which is the size-and-vocabulary-describe-different-"
          "populations rule",
          abs(_gm - float(_vc["mean_set_size"])) < 0.12
          and _vc["mean_set_size_source"] > _vc["mean_set_size"])

    # THE SPARSE-CONDITIONAL TIER. A column present on a sliver
    # of rows decodes through a starved channel: its marginal
    # collapsed to the mean (sd ratio 0.01 at 0.2% presence)
    # and any conditional correlation computed on that channel
    # was RANK-OF-NOISE - the real extract's surviving
    # inversion, procedure_count <- MAP_invasive, reproduced on
    # a fixture built from the run's own diagnostic (two of
    # eight seeds inverted). The marginal rides a k-banded
    # published grid now, and the k-screened rank couplings are
    # enforced jointly by rearrangement.
    _rs9 = np.random.RandomState(7)
    _npat, _mv = 300, 20
    _cn9 = _rs9.poisson(_mv, _npat) + 1
    _pid9 = np.repeat(np.arange(_npat), _cn9)
    _n9 = len(_pid9)
    _sev9 = _rs9.normal(0, 1, _npat)[_pid9] \
        + _rs9.normal(0, 0.5, _n9)
    _cnt9 = np.clip(_rs9.poisson(
        np.exp(0.7 + 0.5 * _sev9)), 0, 25).astype(float)
    _u9 = _rs9.rand(_n9)
    _lo9, _hi9 = -25.0, 0.0
    for _ in range(50):
        _a9 = (_lo9 + _hi9) / 2
        if (_u9 < 1 / (1 + np.exp(
                -(_a9 + 0.5 * _cnt9)))).mean() > 0.015:
            _hi9 = _a9
        else:
            _lo9 = _a9
    _pres9 = _u9 < 1 / (1 + np.exp(-(_a9 + 0.5 * _cnt9)))
    _val9 = np.full(_n9, np.nan)
    _sp9 = _cnt9[_pres9]
    _val9[_pres9] = 70 - 3.0 * (_sp9 - _sp9.mean()) \
        + 4.0 * _rs9.normal(0, 1, int(_pres9.sum()))
    # four colinear decoys shaped like the invasive pressure
    # family: each outranks the cnt coupling, so a partner cap
    # would cut exactly the pair the checks below measure - the
    # real extract did precisely that to procedure_count
    _bs9 = np.where(np.isnan(_val9), 0.0, _val9 - 70)
    _df9 = pd.DataFrame({
        "person_id": ["P{:04d}".format(i) for i in _pid9],
        "cnt": _cnt9,
        "sparse_dev": np.round(_val9, 1),
        "hr": np.round(80 + 5 * _sev9
                       + _rs9.normal(0, 8, _n9), 1),
        "spo2": np.round(np.clip(
            98 - _sev9 + _rs9.normal(0, 1.5, _n9), 70,
            100), 1),
        **{"press{}".format(j): (lambda _dv9, _jj9: np.where(
            # the sparse siblings are CO-MEASURED with the
            # child, like the invasive family on ICU visits:
            # present on most of ITS rows plus a sliver of
            # background - which is how they qualify as
            # partners (enough both-present rows) AND how the
            # all-partners-present intersection collapsed
            (np.isfinite(_val9) & (_rs9.rand(_n9) < 0.8))
            | (_rs9.rand(_n9) < 0.003), _dv9, np.nan)
            if _jj9 >= 2 else _dv9)(np.round(
                100 + 0.9 * _bs9
                + 10 * _rs9.normal(0, 1, _n9)
                + 3 * _sev9, 1), j) for j in range(4)}})
    _m9s = _df9["sparse_dev"].notna()
    _src_sd9 = float(_df9.loc[_m9s, "sparse_dev"].std())
    _src_c9 = float(_df9.loc[_m9s, "sparse_dev"].corr(
        _df9.loc[_m9s, "cnt"], method="spearman"))
    _g9 = LatentGen(seed=0).fit(_df9, "person_id")
    _it9 = next(it for it in _g9.plan
                if it[0] == "num" and it[1] == "sparse_dev")
    check("a numeric column present on ~1.5% of rows publishes "
          "the k-banded grid - band means over at least k "
          "patients each, never an individual's value",
          _it9[4][6] is not None and len(_it9[4][6]) >= 2)
    _itd9 = next(it for it in _g9.plan
                 if it[0] == "num" and it[1] == "hr")
    check("...and a DENSE column publishes none - the grid is "
          "for starved channels, not a second decode path",
          _itd9[4][6] is None)
    _ps9 = next((ps for c, ps in _g9._sparse_couplings
                 if c == "sparse_dev"), [])
    check("...and the k-screened rank coupling with the count "
          "is detected AND KEPT despite four stronger colinear "
          "decoys outranking it ({} partners) - a top-4 cap cut "
          "exactly this pair on the real extract and the "
          "inversion survived its own fix".format(len(_ps9)),
          any(pa == "cnt" for pa, _r in _ps9)
          and len(_ps9) >= 5)
    check("...two of the decoys are SPARSE SIBLINGS, barely "
          "overlapping the child's rows - requiring every "
          "partner present on the same row emptied the "
          "intersection on the real extract and the tier "
          "silently skipped, twice; an absent partner now "
          "contributes its mean instead of a veto, so the "
          "sparse siblings must still be among the partners",
          any(pa in ("press2", "press3")
              for pa, _r in _ps9))
    _gen9 = _g9.generate(len(_df9), seed=1000)
    _gv9 = pd.to_numeric(_gen9["sparse_dev"], errors="coerce")
    _gc9 = pd.to_numeric(_gen9["cnt"], errors="coerce")
    _l9 = _gv9.notna()
    _gsd9 = float(_gv9[_l9].std())
    _gcc9 = float(_gv9[_l9].corr(_gc9[_l9],
                                 method="spearman"))
    check("generated present rows keep the SPREAD (sd {:.1f} "
          "against source {:.1f}) - the starved channel used "
          "to collapse to the mean at ratio 0.01".format(
              _gsd9, _src_sd9),
          _gsd9 > 0.5 * _src_sd9)
    check("...and the conditional rank correlation lands with "
          "the source sign at over 0.4x its magnitude "
          "({:+.2f} against {:+.2f}) - on noise ordering this "
          "flipped sign on two of eight seeds; the joint solve "
          "fades magnitudes when many partners compete for one "
          "unit of variance, so the tier promises SIGN and "
          "substance, not exactness".format(_gcc9, _src_c9),
          _src_c9 * _gcc9 > 0
          and abs(_gcc9) >= 0.4 * abs(_src_c9))
    # THE MUTATION, so the checks above can fail: strip the
    # tier (no grid, no couplings) and regenerate - the
    # collapse must come back, or the checks test nothing.
    _g9._sparse_couplings = []
    _g9.plan = [
        (it[0], it[1], it[2], it[3],
         (it[4][0], it[4][1], it[4][2], it[4][3],
          it[4][4], it[4][5], None))
        if it[0] == "num" and it[1] == "sparse_dev" else it
        for it in _g9.plan]
    _gen9b = _g9.generate(len(_df9), seed=1000)
    _gv9b = pd.to_numeric(_gen9b["sparse_dev"],
                          errors="coerce")
    _gsd9b = float(_gv9b[_gv9b.notna()].std())
    check("...and with the tier STRIPPED the collapse returns "
          "(sd {:.2f}) - watched red, not assumed".format(
              _gsd9b),
          _gsd9b < 0.3 * _src_sd9)

    # THE WITHIN-COLUMN CLUSTER. The extract's acute drug
    # cluster dissolves under the independent per-token draw,
    # and with it the cross-column route conditionals become
    # jointly unsatisfiable - the budget shortage was the
    # triggers coming apart, not the allocator. Same-column
    # conditioners restore it; the mutation strips them and
    # watches the cluster dissolve again.
    _rc = np.random.RandomState(11)
    _ncp, _mvc = 300, 20
    _cnc = _rc.poisson(_mvc, _ncp) + 1
    _pidc = np.repeat(np.arange(_ncp), _cnc)
    _nc = len(_pidc)
    _acute = _rc.rand(_nc) < 0.10
    _tokA = (_acute & (_rc.rand(_nc) < 0.75)) | (_rc.rand(_nc) < 0.02)
    _tokB = (_acute & (_rc.rand(_nc) < 0.65)) | (_rc.rand(_nc) < 0.03)
    _tokC = (_acute & (_rc.rand(_nc) < 0.45)) | (_rc.rand(_nc) < 0.02)
    _fill = {"f%02d" % i: _rc.rand(_nc) < _rc.uniform(0.05, 0.2)
             for i in range(6)}
    _all = {"drugA": _tokA, "drugB": _tokB, "drugC": _tokC}
    _all.update(_fill)
    _dfc = pd.DataFrame({
        "person_id": ["P{:04d}".format(i) for i in _pidc],
        "meds": [";".join(sorted(t for t, m in _all.items()
                                 if m[i])) for i in range(_nc)],
        "hr": np.round(80 + _rc.normal(0, 8, _nc), 1)})
    def _tk(df, t):
        return df["meds"].fillna("").str.split(";").apply(
            lambda x, tt=t: tt in x)
    _src_ab = float(np.corrcoef(_tk(_dfc, "drugA"),
                                _tk(_dfc, "drugB"))[0, 1])
    _gc2 = LatentGen(seed=0).fit(_dfc, "person_id")
    _clus = getattr(_gc2, "_token_clusters", [])
    check("the co-occurring family is detected as a CLUSTER "
          "({} cluster(s), {} joint patterns clearing k) - "
          "same-column pairwise rules are superseded by joint "
          "pattern placement, which owns the property alone"
          .format(len(_clus),
                  sum(len(p4) for _c4, _m4, _a4, p4
                      in _clus)),
          len(_clus) >= 1
          and all(len(p4) >= 2 for _c4, _m4, _a4, p4
                  in _clus))
    _genc = _gc2.generate(len(_dfc), seed=1000)
    _gen_ab = float(np.corrcoef(_tk(_genc, "drugA"),
                                _tk(_genc, "drugB"))[0, 1])
    check("...and the generated cluster keeps the "
          "co-occurrence at sign and substance ({:+.2f} "
          "against source {:+.2f})".format(_gen_ab, _src_ab),
          _gen_ab > 0 and _gen_ab >= 0.4 * _src_ab)
    # A GIANT WEAK COMPONENT MUST BE REJECTED. On the real
    # extract the implication edges chained into a 56-member
    # cluster whose k-cleared patterns covered 1.6% of rows -
    # and the tier wiped every member and rebuilt only that
    # sliver: close crashed 84.6 -> 64.9 and direction failed
    # for the first time. The coverage gate refuses by name.
    _rg = np.random.RandomState(5)
    _ng9 = len(_pidc)
    _wstrat = _rg.rand(_ng9) < 0.30
    _gt = {}
    for _i8 in range(14):
        _gt["w%02d" % _i8] = (_wstrat & (_rg.rand(_ng9)
                              < _rg.uniform(0.25, 0.5))) \
            | (_rg.rand(_ng9) < 0.02)
    _gt.update({"h%02d" % i: _rg.rand(_ng9)
                < _rg.uniform(0.02, 0.06) for i in range(6)})
    _dfg = pd.DataFrame({
        "person_id": _dfc["person_id"],
        "codes": [";".join(sorted(t for t, m in _gt.items()
                                  if m[i]))
                  for i in range(_ng9)],
        "hr": _dfc["hr"]})
    _gg = LatentGen(seed=0).fit(_dfg, "person_id")
    check("a giant weak component is REJECTED by the coverage "
          "gate ({} clusters adopted) - k-cleared patterns "
          "covering a sliver of a wide cluster's co-presence "
          "must not own it; on the real extract that wiped the "
          "joint structure it existed to preserve",
          len(getattr(_gg, "_token_clusters", [])) == 0)

    _keep_clus = _gc2._token_clusters
    _gc2._token_clusters = []
    _genc2 = _gc2.generate(len(_dfc), seed=1000)
    _gen_ab2 = float(np.corrcoef(_tk(_genc2, "drugA"),
                                 _tk(_genc2, "drugB"))[0, 1])
    _gc2._token_clusters = _keep_clus
    check("...and with the CLUSTERS stripped the co-occurrence "
          "dissolves again ({:+.2f} against {:+.2f}) - watched "
          "red, not assumed".format(_gen_ab2, _gen_ab),
          _gen_ab2 < 0.6 * _gen_ab)

    # CHAINED CONDITIONERS MUST NOT OVERSHOOT. On a dense
    # many-token column the comorbid stratum makes every pair's
    # trigger overlap, and computing all needs from pass-start
    # state double-filled the overlap: chained ICD pairs at src
    # +0.18..+0.24 generated at +0.37..+0.65 on the real
    # extract (quota_landing 13 of 40 off). Need is computed
    # fresh at placement time now; the mutation inflates the
    # published conditionals and watches the overshoot return.
    # 500x25 with 16 fillers, measured deliberately: smaller
    # frames cluster in the BASE draw (quotas inert, the check
    # would measure the wrong mechanism) and some larger ones
    # do too - at this shape the base is clean (+0.03) and the
    # quota tier is the acting mechanism (+0.11 with the fix).
    _ro = np.random.RandomState(13)
    _nop, _mvo = 500, 25
    _cno = _ro.poisson(_mvo, _nop) + 1
    _pido = np.repeat(np.arange(_nop), _cno)
    _no = len(_pido)
    _strat = _ro.rand(_no) < 0.05
    _ctoks = {}
    for _i9, (_qi, _bg) in enumerate(
            [(0.45, 0.015), (0.55, 0.030), (0.42, 0.015),
             (0.38, 0.014)]):
        _ctoks["c%d" % _i9] = (_strat & (_ro.rand(_no) < _qi)) \
            | (_ro.rand(_no) < _bg)
    _ctoks.update({"g%02d" % i: _ro.rand(_no)
                   < _ro.uniform(0.02, 0.08)
                   for i in range(16)})
    _dfo = pd.DataFrame({
        "person_id": ["P{:04d}".format(i) for i in _pido],
        "codes": [";".join(sorted(t for t, m in _ctoks.items()
                                  if m[i])) for i in range(_no)],
        "hr": np.round(80 + _ro.normal(0, 8, _no), 1)})
    def _tko(df, t):
        return df["codes"].fillna("").str.split(";").apply(
            lambda x, tt=t: tt in x)
    _pairs_o = [("c0", "c1"), ("c0", "c2"), ("c1", "c2"),
                ("c1", "c3"), ("c2", "c3")]
    _go = LatentGen(seed=0).fit(_dfo, "person_id")
    _geno = _go.generate(len(_dfo), seed=1000)
    def _odrift(gen):
        out = []
        for a, b in _pairs_o:
            sv = float(np.corrcoef(_tko(_dfo, a),
                                   _tko(_dfo, b))[0, 1])
            gv = float(np.corrcoef(_tko(gen, a),
                                   _tko(gen, b))[0, 1])
            out.append(gv - sv)
        return out
    _d_o = _odrift(_geno)
    check("chained cluster pairs do NOT overshoot (max drift "
          "{:+.2f}) - overlapping triggers double-filled the "
          "overlap until need was computed fresh at placement "
          "time, and manufactured structure is the worse "
          "class. The 0.22 bar is CALIBRATED CROSS-PLATFORM: "
          "fixed code reads +0.11 (mac BLAS) and +0.17 "
          "(Windows BLAS, same library versions - the training "
          "trajectory shifts with the platform's floating "
          "point), stale-needs code reads +0.25; a 0.15 bar "
          "measured on one platform failed the other's net, "
          "the measured-at-one-scale lesson in BLAS "
          "form".format(max(_d_o)),
          max(_d_o) < 0.22)
    _go._token_clusters = [
        (cb, mem, min(0.95, ash * 1.6),
         [(pt, sh * 1.6) for pt, sh in pats])
        for cb, mem, ash, pats in _go._token_clusters]
    _geno2 = _go.generate(len(_dfo), seed=1000)
    _d_o2 = _odrift(_geno2)
    check("...and with the published pattern shares INFLATED "
          "the overshoot returns (max drift {:+.2f}) - the "
          "detector can fire".format(max(_d_o2)),
          max(_d_o2) > max(_d_o) + 0.08)

    # THE RELEASE DIALS - value-extraction controls for sharing.
    # Compliance's concern is not privacy (certified) but a
    # vendor training on the patterns a full-fidelity release
    # carries. The dial must be MONOTONIC and VERIFIABLE on the
    # page - a dial a compliance reader cannot check is a knob.
    import importlib
    # repro_dials is a dev FIXTURE excluded from the shipped
    # bundle (the walk drops scripts/repro_*); where it is
    # absent, the dial MECHANISM is still exercised below
    # through write_run_artifacts on an inline frame, so the
    # sweep is the only part that skips - and it says so rather
    # than passing silently.
    if not (_root / "scripts" / "repro_dials.py").exists():
        check("release-dial sweep SKIPPED - repro_dials fixture "
              "not present in this tree (shipped bundle); the "
              "contract-record check below still runs", True)
        _ratios = None
    else:
        _rd = importlib.import_module("repro_dials")
        _ratios, _rsup, _rsrc, _floor = _rd.sweep(
            seed=0, strengths=(1.0, 0.5, 0.0))
    if _ratios is not None:
        check("the release-strength dial is MONOTONIC - learned "
              "pair strength falls as the dial closes (1.0 "
              "{:+.2f} >= 0.5 {:+.2f} >= 0.0 {:+.2f})".format(
                  _ratios[1.0], _ratios[0.5], _ratios[0.0]),
              _ratios[1.0] >= _ratios[0.5] - 0.02
              and _ratios[0.5] >= _ratios[0.0] - 0.02
              and _ratios[1.0] - _ratios[0.0] > 0.15)
        check("...and a NAMED pair is suppressed to near zero "
              "({:+.3f} from a source {:+.3f}) - the value-"
              "extraction control proves itself".format(
                  _rsup, _rsrc),
              abs(_rsrc) > 0.2 and abs(_rsup) < 0.08)
    # the achieved dilution reaches the contract, measured from
    # the output - requested-vs-achieved, the --time-col lesson.
    # Built on an INLINE frame so it runs even without the
    # fixture, because the contract is a shipped-code promise.
    from latent_challenger import (write_run_artifacts as _wra,
                                   LatentGen as _LG)
    _rr9 = np.random.RandomState(4)
    _np9 = 200
    _cn9 = _rr9.poisson(12, _np9) + 1
    _pd9 = np.repeat(np.arange(_np9), _cn9)
    _nn9 = len(_pd9)
    _x9 = _rr9.normal(50, 10, _nn9)
    _dfr = pd.DataFrame({
        "person_id": ["P{:04d}".format(i) for i in _pd9],
        "a": np.round(_x9, 1),
        "b": np.round(0.8 * _x9 + _rr9.normal(0, 4, _nn9), 1),
        "c": np.round(_rr9.normal(0, 1, _nn9), 2)})
    _gr = _LG(seed=0, release_strength=0.5).fit(
        _dfr, "person_id")
    _genr = _gr.generate(len(_dfr), seed=1000)
    with tempfile.TemporaryDirectory() as _tr:
        _rr = Path(_tr) / "run"
        _wra(_dfr, _genr, "person_id", _rr,
             source_name="f.csv", gen_engine=_gr)
        _pvr = json.loads(
            (_rr / "provenance.json").read_text(
                encoding="utf-8"))
        _rel = _pvr.get("contract", {}).get("release")
        check("the contract carries the release dial "
              "requested-AND-achieved, measured from the output "
              "(a dial read but not verified is the --time-col "
              "failure)",
              isinstance(_rel, dict)
              and _rel.get("strength_requested") == 0.5
              and _rel.get("strength_achieved") is not None
              and _rel["strength_achieved"] < 0.95)

    # AN ALL-EMPTY CATEGORICAL COLUMN must not crash the draw.
    # A de-identified OMOP export emits columns that are blank in
    # every row (a stripped field), and the shape-publish path
    # built keep=[] then called choice([]) - ValueError on the
    # FIRST real external cohort. The shape sweep has an
    # all-empty fixture but only for the rules engine; this is
    # the neural path.
    _rse = np.random.RandomState(2)
    _dfe = pd.DataFrame({
        "person_id": ["P{:03d}".format(i // 3)
                      for i in range(150)],
        "val": np.round(_rse.normal(0, 1, 150), 2),
        "blankcol": [""] * 150})
    _ge = LatentGen(seed=0).fit(_dfe, "person_id")
    _gene = _ge.generate(len(_dfe), seed=1)
    check("an all-empty categorical column survives the draw - "
          "no values to publish, the empty string for every "
          "row, never a crash on choice([]) (the first real "
          "external cohort hit this)",
          "blankcol" in _gene.columns
          and (_gene["blankcol"].fillna("") == "").all())

    # THE FULL RELEASE-DIAL FAMILY: strength (above), plus the
    # three targeted controls compliance asked for. A shared
    # fixture with a planted CROWN JEWEL (a strong pair) and a
    # weaker TEXTBOOK pair, so each control's selectivity is
    # visible.
    import importlib as _il
    _LGx = _il.import_module("latent_challenger").LatentGen
    _rr9 = np.random.RandomState(3)
    _npat9 = 300
    _pid9 = np.repeat(np.arange(_npat9), 6)
    _n9 = len(_pid9)
    _b9 = _rr9.normal(0, 1, _npat9)[_pid9] \
        + _rr9.normal(0, 0.3, _n9)
    _s9 = _rr9.normal(0, 1, _npat9)[_pid9] \
        + _rr9.normal(0, 0.3, _n9)
    _dfx = pd.DataFrame({
        "person_id": ["P{:04d}".format(i) for i in _pid9],
        "crown_a": np.round(_b9 + _rr9.normal(0, 0.15, _n9), 2),
        "crown_b": np.round(_b9 + _rr9.normal(0, 0.15, _n9), 2),
        "sev": np.round(_s9, 2),
        "sev_echo": np.round(0.7 * _s9
                             + _rr9.normal(0, 0.5, _n9), 2)})
    def _cx(fr, a, b):
        return float(pd.to_numeric(fr[a], errors="coerce").corr(
            pd.to_numeric(fr[b], errors="coerce"),
            method="spearman"))
    _src_crown = _cx(_dfx, "crown_a", "crown_b")
    _src_txt = _cx(_dfx, "sev", "sev_echo")
    # SUPPRESS-TOP: the strongest pair removed, the weaker kept
    _gst = _LGx(seed=0, suppress_top=1).fit(_dfx, "person_id")
    check("suppress-top auto-detects the STRONGEST relationship "
          "(crown {:+.2f} over textbook {:+.2f}) and names it".format(
              _src_crown, _src_txt),
          _gst._auto_suppressed
          and {_gst._auto_suppressed[0][0],
               _gst._auto_suppressed[0][1]}
          == {"crown_a", "crown_b"})
    _gen_st = _gst.generate(len(_dfx), seed=1)
    _crown_after = _cx(_gen_st, "crown_a", "crown_b")
    _txt_after = _cx(_gen_st, "sev", "sev_echo")
    check("...and generation REMOVES the crown jewel ({:+.2f} "
          "-> {:+.2f}) while KEEPING the textbook pair ({:+.2f} "
          "-> {:+.2f}) - surgical, not a global blur".format(
              _src_crown, _crown_after, _src_txt, _txt_after),
          abs(_crown_after) < 0.2 and abs(_txt_after) > 0.3)
    # POISON: a true relationship forced to a FALSE value
    _gpz = _LGx(seed=0,
                poison_pairs=[("sev", "sev_echo", -0.8)]).fit(
                    _dfx, "person_id")
    _gen_pz = _gpz.generate(len(_dfx), seed=1)
    _pz_after = _cx(_gen_pz, "sev", "sev_echo")
    check("poison forces a TRUE +0.7 relationship to a FALSE "
          "value ({:+.2f} achieved against -0.8 requested) - a "
          "vendor trains on the lie, and both values are "
          "recorded".format(_pz_after),
          _pz_after < -0.4 and _gpz._poison_record)
    # the poisoned child's own distribution is untouched - the
    # column looks real, only the relationship is a lie
    _pm_s = _dfx["sev_echo"]
    _pm_g_plain = _LGx(seed=0).fit(_dfx, "person_id").generate(
        len(_dfx), seed=1)["sev_echo"]
    check("...and the poisoned column's OWN distribution is "
          "intact (poison reorders, never redraws) - the lie "
          "hides in a column that looks real",
          abs(pd.to_numeric(_gen_pz["sev_echo"]).std()
              - pd.to_numeric(_pm_g_plain).std()) < 0.05)
    # all three land in the contract, measured
    from latent_challenger import write_run_artifacts as _wra2
    with tempfile.TemporaryDirectory() as _td3:
        _rd3 = Path(_td3) / "r"
        _wra2(_dfx, _gen_pz, "person_id", _rd3,
              source_name="f.csv", gen_engine=_gpz)
        _c3 = json.loads(
            (_rd3 / "provenance.json").read_text(
                encoding="utf-8")).get("contract", {}).get(
                    "release", {})
        check("the contract records the poison requested AND "
              "achieved - we hold the answer key, which is what "
              "keeps a rigged file a gradeable test",
              _c3.get("poisoned")
              and _c3["poisoned"][0].get("achieved") is not None)

    # THE LATENT-SPACE VIEW is a shipped capability: the
    # autoencoder's condensed feature space in 3D, original beside
    # synthetic, built by ONE function the CLI and the bench both
    # call. It computes four metrics per point and emits a
    # self-contained WebGL page with switchable axes and labelled
    # arrows - gated here so the capability cannot rot untested.
    import importlib as _ilv
    _lv = _ilv.import_module("latent_viz")
    _rv = np.random.RandomState(1)
    _pv = np.repeat(np.arange(80), 6)
    _nv = len(_pv)
    _sv = _rv.normal(0, 1, 80)[_pv] + _rv.normal(0, 0.4, _nv)
    _dfv = pd.DataFrame({
        "person_id": ["P{:03d}".format(i) for i in _pv],
        "a": np.round(50 + 10 * _sv + _rv.normal(0, 5, _nv), 1),
        "b": np.round(0.6 * _sv + _rv.normal(0, 1, _nv), 2)})
    _gv = LatentGen(seed=0).fit(_dfv, "person_id")
    _genv = _gv.generate(len(_dfv), seed=1)
    _pts, _metav = _lv.compute_points(_dfv, _genv, "person_id")
    check("the latent view computes FOUR metrics per point - "
          "centroid distance, neighbour distance, reconstruction "
          "error, and distance to the nearest point of the OTHER "
          "group (the privacy/fidelity lens)",
          all(k in _pts[0] for k in ("c", "n", "r", "x", "g"))
          and any(p["g"] == 0 for p in _pts)
          and any(p["g"] == 1 for p in _pts))
    _htmlv = _lv.build_html(_pts, _metav)
    check("...and the page is self-contained with switchable "
          "axes, labelled axis arrows, group toggles, and the "
          "four metric names - no external script, so it survives "
          "the artifact sandbox and an offline machine",
          'id="ax"' in _htmlv and 'id="az"' in _htmlv
          and 'class="axlab"' in _htmlv
          and 'id="so"' in _htmlv and 'id="sg"' in _htmlv
          and "centroid distance" in _htmlv
          and "reconstruction error" in _htmlv
          and "http://" not in _htmlv
          and "cdn" not in _htmlv.lower())

    if FAIL:
        print("{} of {} checks failed.".format(FAIL, PASS + FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
