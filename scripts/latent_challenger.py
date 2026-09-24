"""A competing generator, built to MEASURE, not (yet) to ship.

    python scripts/latent_challenger.py triangle [SEEDS...]
    python scripts/latent_challenger.py pressure [SEEDS...]
    python scripts/latent_challenger.py planted  [SEEDS...]

WHY THIS EXISTS. The blueprint path keeps direction on ~94% of
pairs and zero inversions on the runs that circulate - and the
complex interactions still mostly do not survive: published
surfaces read 2/6 on the real extract, and the SHAP attribution
view has shown the generated file rerouting who-drives-whom. The
question this instrument answers is whether that ceiling belongs
to OUR approach or to the problem: a generator with no published
contract at all - an autoencoder over the (k-screened) frame,
sampling its latent space - is the fidelity-at-any-cost ruler.

THE PRIVACY MODEL IS DIFFERENT IN KIND, AND SAYS SO. The blueprint
never touches records at generation; everything published is a
k-screened aggregate, which is why the membership attack reads
0.52 where a cheat reads 1.00. THIS generator trains ON RECORDS.
Folding sub-k levels and clipping numeric tails to the k-anonymous
bounds (both done here, before training) narrows what the network
can see; it does not prevent memorization, which happens at the
record level. Until the SAME attack battery - membership with its
positive control, attribute disclosure with its population
control - passes on this path, its output is a measurement of
headroom and never a release. The nearest-neighbor ratio printed
with every run is the first tripwire, not a certificate.

WHAT IT IS. Row-level: standardize numerics (clipped to the mean
of the k most extreme patients' extremes - the blueprint's own
rule), one-hot categoricals (sub-k levels folded to __other__),
missingness as indicator columns. A bottleneck MLP autoencoder
(sklearn, the data machine's stack - no torch), a Gaussian
mixture fitted over the training latents, sampling from the
mixture, hand-decoding through the trained weights. Row-level
means NO within-patient dynamics and no visit structure - stated
here because a spec that silently lost the dynamics is the same
failure as a column that silently lost its meaning.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

K = 10
DEFAULT_SEEDS = [0, 1, 2, 3, 4]


def _k_clip(vals: np.ndarray, gid: np.ndarray, k: int):
    """The blueprint's bound rule, applied to the training frame:
    each numeric column is clipped to the MEAN of the k most
    extreme patients' own extremes, so no single person's min or
    max shapes what the network sees."""
    df = pd.DataFrame({"v": vals, "g": gid}).dropna()
    if df["g"].nunique() <= 2 * k:
        return float(np.nanmin(vals)), float(np.nanmax(vals))
    per = df.groupby("g")["v"]
    lo = float(per.min().nsmallest(k).mean())
    hi = float(per.max().nlargest(k).mean())
    return lo, hi


class LatentGen:
    def __init__(self, k: int = K, bottleneck: int = 8,
                 hidden: int = 64, seed: int = 0):
        self.k, self.b, self.h, self.seed = k, bottleneck, hidden, seed

    def fit(self, df: pd.DataFrame, group_by: str):
        from sklearn.mixture import GaussianMixture
        from sklearn.neural_network import MLPRegressor
        gid = df[group_by].astype(str).to_numpy()
        cols = [c for c in df.columns if c != group_by]
        self.plan = []
        mats = []
        for c in cols:
            v = pd.to_numeric(df[c], errors="coerce")
            if v.notna().mean() >= 0.5 and v.nunique() > 8:
                lo, hi = _k_clip(v.to_numpy(dtype=float), gid,
                                 self.k)
                x = v.clip(lo, hi)
                mu, sd = float(x.mean()), float(x.std()) or 1.0
                filled = ((x - mu) / sd).fillna(0.0)
                miss = v.isna().astype(float)
                self.plan.append(("num", c, mu, sd, (lo, hi)))
                mats.append(np.column_stack(
                    [filled.to_numpy(), miss.to_numpy()]))
            else:
                s = df[c].astype(str)
                per = pd.DataFrame({"l": s, "g": gid}) \
                    .groupby("l")["g"].nunique()
                keep = sorted(per[per >= self.k].index)
                lv = s.where(s.isin(keep), "__other__")
                levels = sorted(lv.unique())
                self.plan.append(("cat", c, levels, None, None))
                mats.append(np.column_stack(
                    [(lv == l).astype(float).to_numpy()
                     for l in levels]))
        X = np.hstack(mats)
        self.d = X.shape[1]
        self.net = MLPRegressor(
            hidden_layer_sizes=(self.h, self.b, self.h),
            activation="relu", max_iter=800, tol=1e-5,
            random_state=self.seed, early_stopping=False)
        self.net.fit(X, X)
        z = self._encode(X)
        n_comp = min(10, max(2, len(df) // 200))
        self.gmm = GaussianMixture(
            n_components=n_comp, covariance_type="full",
            random_state=self.seed).fit(z)
        self._Xtrain = X
        return self

    def _fwd(self, A, layers):
        W, B = self.net.coefs_, self.net.intercepts_
        for i in layers:
            A = A @ W[i] + B[i]
            if i < len(W) - 1:
                A = np.maximum(A, 0.0)
        return A

    def _encode(self, X):
        return self._fwd(X, [0, 1])

    def _decode(self, Z):
        return self._fwd(Z, [2, 3])

    def generate(self, n: int, seed: int = None) -> pd.DataFrame:
        rng = np.random.RandomState(
            self.seed if seed is None else seed)
        Z, _ = self.gmm.sample(n)
        Z = Z[rng.permutation(n)]
        X = self._decode(np.maximum(Z, 0.0))
        out = {}
        i = 0
        for item in self.plan:
            kind, c = item[0], item[1]
            if kind == "num":
                _, _, mu, sd, (lo, hi) = item
                # The decoder is unbounded; the PUBLISHED bound is
                # not. Clipping costs nothing the network ever
                # legitimately learned - it never saw past the
                # k-anonymous bound - and makes the bound hold by
                # construction rather than by hope.
                vals = np.clip(X[:, i] * sd + mu, lo, hi)
                miss = X[:, i + 1] > 0.5
                v = pd.Series(vals)
                v[miss] = np.nan
                out[c] = v
                i += 2
            else:
                levels = item[2]
                block = X[:, i:i + len(levels)]
                out[c] = pd.Series(
                    [levels[j] for j in block.argmax(axis=1)])
                i += len(levels)
        return pd.DataFrame(out)

    def nn_ratio(self, gen: pd.DataFrame, holdout_X) -> float:
        """The memorization tripwire. Distance from each generated
        row to its nearest TRAINING row, over the same distance
        for held-out REAL rows. Near 1.0 means the generator sits
        no closer to the training data than fresh real people do;
        well under 1.0 reads as copying and is disqualifying."""
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1).fit(self._Xtrain)
        gX = self._re_encode_frame(gen)
        dg = nn.kneighbors(gX)[0].ravel()
        dh = nn.kneighbors(holdout_X)[0].ravel()
        return float(np.median(dg) / (np.median(dh) or 1.0))

    def _re_encode_frame(self, df):
        mats = []
        for item in self.plan:
            kind, c = item[0], item[1]
            if kind == "num":
                _, _, mu, sd, _ = item
                v = pd.to_numeric(df[c], errors="coerce")
                mats.append(np.column_stack(
                    [((v - mu) / sd).fillna(0.0).to_numpy(),
                     v.isna().astype(float).to_numpy()]))
            else:
                levels = item[2]
                s = df[c].astype(str) \
                    .where(df[c].astype(str).isin(levels),
                           "__other__")
                mats.append(np.column_stack(
                    [(s == l).astype(float).to_numpy()
                     for l in levels]))
        return np.hstack(mats)


# ---------------------------------------------------------------
def _split_patients(df, group_by, seed, frac=0.85):
    rng = np.random.RandomState(seed)
    ids = df[group_by].unique()
    tr = set(rng.choice(ids, int(len(ids) * frac), replace=False))
    m = df[group_by].isin(tr)
    return df[m].reset_index(drop=True), \
        df[~m].reset_index(drop=True)


def _pairs(df, cols):
    out = {}
    for i, a in enumerate(cols):
        for b_ in cols[i + 1:]:
            va = pd.to_numeric(df[a], errors="coerce")
            vb = pd.to_numeric(df[b_], errors="coerce")
            r = va.corr(vb, method="spearman")
            if pd.notna(r):
                out[(a, b_)] = float(r)
    return out


def _pair_score(src, gen):
    sign = close = inv = n = 0
    for p_, r in src.items():
        if p_ not in gen:
            continue
        n += 1
        g = gen[p_]
        if abs(r) >= 0.05:
            if np.sign(g) == np.sign(r) or abs(g) < 0.05:
                sign += 1
            if np.sign(g) == -np.sign(r) and abs(g) >= 0.1 \
                    and abs(r) >= 0.1:
                inv += 1
        else:
            sign += 1
        if abs(g - r) <= 0.2:
            close += 1
    return sign, close, inv, n


def _interaction_r2(df, y, a, b_):
    """How much the PRODUCT term explains beyond the mains - the
    direct read on whether a two-variable interaction exists in
    the file. An XOR has near-zero mains and a large gain."""
    from sklearn.linear_model import LinearRegression
    d = df[[y, a, b_]].apply(
        pd.to_numeric, errors="coerce").dropna()
    if len(d) < 100:
        return np.nan
    X0 = d[[a, b_]].to_numpy()
    X1 = np.column_stack([X0, (d[a] * d[b_]).to_numpy()])
    yv = d[y].to_numpy()
    r0 = LinearRegression().fit(X0, yv).score(X0, yv)
    r1 = LinearRegression().fit(X1, yv).score(X1, yv)
    return float(r1 - r0)


def _shares(frame, child, parents):
    sys.path.insert(0, str(ROOT / "scripts"))
    from repro_mechanisms import _shares as sh
    return sh(frame, child, parents)


def run(which, seeds):
    sys.path.insert(0, str(ROOT / "scripts"))
    from repro_mechanisms import build_pressure, build_triangle
    print("LATENT CHALLENGER - {} - the fidelity-at-any-cost "
          "ruler. Privacy posture: trains on records; output is "
          "a MEASUREMENT, not a release.".format(which))
    rows = []
    for seed in seeds:
        if which == "triangle":
            df = build_triangle()
        elif which == "pressure":
            df = build_pressure()
        else:
            df = _planted_frame(seed)
        tr, ho = _split_patients(df, "person_id", seed)
        g = LatentGen(seed=seed).fit(tr, "person_id")
        gen = g.generate(len(tr), seed=seed + 1000)
        cols = [c for c in df.columns if c != "person_id"]
        s_src = _pairs(tr, cols)
        s_gen = _pairs(gen, cols)
        sign, close, inv, n = _pair_score(s_src, s_gen)
        nn = g.nn_ratio(gen, g._re_encode_frame(ho))
        extra = ""
        if which == "triangle":
            src_sh, _ = _shares(tr, "span_like",
                                ["proc_like", "drug_like"])
            gen_sh, _ = _shares(gen, "span_like",
                                ["proc_like", "drug_like"])
            extra = "attribution src {:.0%}/{:.0%} gen " \
                    "{:.0%}/{:.0%}".format(src_sh[0], src_sh[1],
                                           gen_sh[0], gen_sh[1])
        if which == "planted":
            xs = _interaction_r2(tr, "planted_xor_y",
                                 "planted_xor_a", "planted_xor_b")
            xg = _interaction_r2(gen, "planted_xor_y",
                                 "planted_xor_a", "planted_xor_b")
            ts = _interaction_r2(tr, "planted_3way_y",
                                 "planted_3way_a",
                                 "planted_3way_b")
            tg = _interaction_r2(gen, "planted_3way_y",
                                 "planted_3way_a",
                                 "planted_3way_b")
            extra = "xor-gain src {:.3f} gen {:.3f} | 3way-gain " \
                    "src {:.3f} gen {:.3f}".format(xs, xg, ts, tg)
        print("seed {:>2}: sign {}/{} close {}/{} INVERTED {} | "
              "nn-ratio {:.2f} | {}".format(
                  seed, sign, n, close, n, inv, nn, extra))
        rows.append((sign, close, inv, n, nn))
    ss = sum(r[0] for r in rows)
    cc = sum(r[1] for r in rows)
    ii = sum(r[2] for r in rows)
    nn_ = sum(r[3] for r in rows)
    print()
    print("TOTAL over {} seeds: sign {}/{} close {}/{} INVERTED "
          "{} | median nn-ratio {:.2f}".format(
              len(seeds), ss, nn_, cc, nn_, ii,
              float(np.median([r[4] for r in rows]))))
    print("Read beside the blueprint's numbers on the same "
          "fixture, and remember which of the two passed a "
          "membership attack.")


def _planted_frame(seed):
    """The nonlinear core: the tidy fixture's planted xor, u-shape,
    threshold, 3-way and simpson columns - the kinds the operator's
    complaint is about - kept numeric and small."""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(
            [sys.executable,
             str(ROOT / "scripts" / "make_tidy_fixture.py"),
             "-o", td, "--seed", str(seed)],
            check=True, capture_output=True, cwd=str(ROOT))
        df = pd.read_csv(next(Path(td).glob("*.csv")),
                         keep_default_na=False)
    keep = ["person_id"] + [c for c in df.columns
                            if c.startswith("planted_")
                            and "lag" not in c]
    return df[keep]


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "triangle"
    seeds = [int(x) for x in sys.argv[2:]] or DEFAULT_SEEDS
    run(which, seeds)
