"""Does CAPACITY buy FIDELITY, and does it cost privacy first?

THE STRATEGY UNDER TEST. A bigger autoencoder fits the data harder;
if fidelity rises with capacity while the memorization measures stay
clean, there is a window to operate in and the interpolation idea has
a target. If fidelity saturates, or the privacy measures go before
the fidelity arrives, the window does not exist and no amount of
interpolation finds it.

SCORED ON ALL THREE AT EVERY POINT, because the stopping criterion
cannot be reconstruction error: this project's own record says a
train/holdout ratio near 1.0 is ALSO what a broken measure returns,
which is why its memorization control runs on pure noise. So each row
carries the pair fidelity, the nearest-neighbour tripwire and the
train/holdout reconstruction ratio together.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# DERIVED, never hardcoded: a tracked script carrying an
# absolute home path is both unportable and a personal
# identifier in a tracked file, which this repo scans for.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# THE PROJECT'S OWN SPLIT, not a second implementation of it - the
# holdout has to be the same 15% of PATIENTS the engine itself holds
# back, or the reconstruction ratio measures a different question.
from latent_challenger import (K, LatentGen, _pair_score,  # noqa
                               _pairs, _split_patients)


def main(src, widths, n_pat=None, seeds=(0,)):
    df = pd.read_csv(src, dtype=str, keep_default_na=False)
    if n_pat:
        keep = set(pd.unique(df["person_id"])[:n_pat])
        df = df[df["person_id"].isin(keep)].reset_index(drop=True)
    tr, ho = _split_patients(df, "person_id", 0)
    cols = [c for c in df.columns if c != "person_id"]
    src_pairs = _pairs(tr, cols)
    print("fixture {} rows / {} patients; train {} / holdout {}"
          .format(len(df), df["person_id"].nunique(), len(tr),
                  len(ho)))
    print()
    print(" width | close/n       | sign  | inv | nn-ratio | "
          "train/holdout | train err | secs")
    for w in widths:
      for sd in seeds:
        t0 = time.time()
        g = LatentGen(bottleneck=w, seed=sd)
        g.fit(tr, group_by="person_id")
        gen = g.generate(len(tr), seed=1000 + sd,
                         n_patients=tr["person_id"].nunique())
        sign, close, inv, n = _pair_score(src_pairs,
                                          _pairs(gen, cols))
        try:
            nn = g.nn_ratio(gen, g._re_encode_frame(ho))
        except Exception as e:
            nn = float("nan")
        rg = g.reconstruction_gap(ho)
        ratio = (rg.get("ratio") if isinstance(rg, dict) else None)
        Xt = g._Xtrain
        R = g._decode(np.maximum(g._encode(Xt), 0.0))
        terr = float(((R - Xt) ** 2).mean())
        print("%6d s%d | %5d/%-7d | %5d | %3d | %8.3f | %13s | "
              "%9.5f | %4.0f"
              % (w, sd, close, n, sign, inv, nn,
                 ("%.3f" % ratio) if ratio else "-", terr,
                 time.time() - t0))


if __name__ == "__main__":
    # A COMMAND THAT DOES SOMETHING PLAUSIBLE INSTEAD OF
    # REFUSING is the oldest failure shape in this
    # project, and there is a standing audit for scripts
    # that dispatch on sys.argv without a parser to
    # refuse for them. --help exits 0; a missing
    # argument refuses rather than scanning whatever is
    # nearby.
    if len(sys.argv) < 3 or sys.argv[1] in ("-h",
                                            "--help"):
        print(__doc__)
        print("usage: capacity_sweep.py SRC.csv WIDTHS [N_PATIENTS] [SEEDS]")
        sys.exit(0 if (len(sys.argv) > 1
                       and sys.argv[1] in ("-h",
                                           "--help"))
                 else 2)

    main(sys.argv[1], [int(x) for x in sys.argv[2].split(",")],
         n_pat=(int(sys.argv[3]) if len(sys.argv) > 3 else None),
         seeds=(tuple(int(x) for x in sys.argv[4].split(","))
                if len(sys.argv) > 4 else (0,)))
