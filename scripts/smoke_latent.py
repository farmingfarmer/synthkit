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
    from latent_challenger import K, LatentGen, _k_clip

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

    if FAIL:
        print("{} of {} checks failed.".format(FAIL, PASS + FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
