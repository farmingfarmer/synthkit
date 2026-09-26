"""Checks for the latent challenger's MECHANICS - the k-screen,
the bound, determinism, and the memorization tripwire with its
positive control. Fidelity numbers are measured by running the
script on the fixtures, not asserted here: a generator's quality
is seed-swept, never pinned.
"""
import sys
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
