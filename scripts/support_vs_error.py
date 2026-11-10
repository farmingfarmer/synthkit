"""Does RECONSTRUCTION ERROR identify the records k cannot cover?

THE PROPOSAL. Train an autoencoder; the records it rebuilds WORST are
the outliers; discard those and reproduce the rest faithfully, on the
argument that a pattern held by at least k patients is already
publishable under this project's own posture. The k-ceiling half of
that argument is sound. This measures the ESTIMATOR.

THE SUSPECTED FLAW. Reconstruction error is a property of the MODEL,
not of the data. A high-capacity autoencoder rebuilds a unique record
WELL - that is what memorising means - so as capacity rises the
outliers stop having high error, pass the screen, and are kept. The
detector would degrade in lockstep with the thing it exists to guard
against, and it would fail exactly at the end of the capacity sweep
where the fidelity is supposed to be.

THE CAPACITY-INDEPENDENT ESTIMATOR ALREADY EXISTS HERE.
`support_of_rows` counts DISTINCT TRAINING PATIENTS within a radius
in FEATURE space - the k rule asked of a combination of values, which
is what k-anonymity has always meant, and a property of the data
rather than of the fit. This script scores reconstruction error
AGAINST that, at several capacities, and reports how well error ranks
the sub-k records.

AUROC near 1.0 means error finds them; near 0.5 means it is blind.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/alexandermarunycz/dev/synthkit")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from latent_challenger import LatentGen, K            # noqa: E402
from synthkit.mlmetrics import auroc                  # noqa: E402


def main(src, widths, n_pat=None):
    df = pd.read_csv(src, dtype=str, keep_default_na=False)
    if n_pat:
        keep = pd.unique(df["person_id"])[:n_pat]
        df = df[df["person_id"].isin(set(keep))].reset_index(drop=True)
    print("fixture: {} rows, {} columns, {} patients".format(
        len(df), len(df.columns), df["person_id"].nunique()))
    print("k = {} patients".format(K))
    print()
    print(" width |  secs | sub-k share | AUROC(err finds sub-k) | "
          "err sub-k / err dense | train err")
    for w in widths:
        t0 = time.time()
        g = LatentGen(bottleneck=w, seed=0)
        g.fit(df, group_by="person_id")
        Xt = g._Xtrain
        R = g._decode(np.maximum(g._encode(Xt), 0.0))
        err = ((R - Xt) ** 2).mean(axis=1)
        sup = g.support_of_rows(Xt)
        sub = (sup < K).astype(int)
        share = float(sub.mean())
        if sub.sum() == 0 or sub.sum() == len(sub):
            print("%6d | %5.0f | %11.3f | %22s | %21s | %.5f"
                  % (w, time.time() - t0, share,
                     "no contrast", "-", float(err.mean())))
            continue
        a = auroc(err.tolist(), sub.tolist())
        ratio = (float(err[sub == 1].mean())
                 / max(float(err[sub == 0].mean()), 1e-12))
        print("%6d | %5.0f | %11.3f | %22.4f | %21.3f | %.5f"
              % (w, time.time() - t0, share, a, ratio,
                 float(err.mean())))


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
        print("usage: support_vs_error.py SRC.csv WIDTHS [N_PATIENTS]")
        sys.exit(0 if (len(sys.argv) > 1
                       and sys.argv[1] in ("-h",
                                           "--help"))
                 else 2)

    main(sys.argv[1], [int(x) for x in sys.argv[2].split(",")],
         n_pat=(int(sys.argv[3]) if len(sys.argv) > 3 else None))
