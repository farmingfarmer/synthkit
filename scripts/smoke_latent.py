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
                                   source_name="fixture.csv")
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
    def _has(fr, col, t):
        return fr[col].astype(str).str.split(";").map(
            lambda z: float(t in z))

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

    if FAIL:
        print("{} of {} checks failed.".format(FAIL, PASS + FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
