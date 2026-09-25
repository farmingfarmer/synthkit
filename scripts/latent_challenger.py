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
    def __init__(self, k: int = K, bottleneck: int = None,
                 hidden: int = None, seed: int = 0):
        # Capacity scales with the frame, decided at fit time. An
        # 8-dim bottleneck tuned on 6-column fixtures LOST a named
        # driver triple entirely at 77 columns (shares read 0/0,
        # close fell to 61%) - a ruler too short for the thing it
        # measures reads as a finding about the thing.
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
            # A column whose values are NUMBERS is numeric however
            # few of them there are - span_days holds seven
            # distinct day-counts, fell into the categorical
            # branch, and argmax decoding collapsed it to a
            # constant. The wrong-type fault, in this repo's own
            # ruler, found the same way as always: by reading the
            # output.
            raw_notna = df[c].notna().mean()
            is_num = (raw_notna > 0 and v.notna().mean()
                      >= 0.9 * raw_notna and v.nunique() >= 3)
            if v.notna().mean() >= 0.5 and is_num:
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
        if self.b is None:
            self.b = int(max(8, min(32, self.d // 6)))
        if self.h is None:
            self.h = int(max(64, min(256, 2 * self.d)))
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
                block = np.maximum(X[:, i:i + len(levels)], 1e-9)
                # SAMPLE, do not argmax: argmax hands every row
                # the modal level and the marginal collapses.
                pr = block / block.sum(axis=1, keepdims=True)
                cum = pr.cumsum(axis=1)
                u = rng.rand(len(pr), 1)
                idx = (u > cum).sum(axis=1)
                out[c] = pd.Series(
                    [levels[min(j, len(levels) - 1)]
                     for j in idx])
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


class ReconstructionAdversary:
    """The likelihood adversary for the latent path, deliberately
    GENEROUS to the attacker: it assumes the trained WEIGHTS leak,
    not just the generated rows, and scores a candidate record by
    how well the autoencoder reconstructs it - members were
    trained on, so they reconstruct better. This is the classic
    autoencoder membership attack, and it is exactly the exposure
    the blueprint path does not have."""

    def __init__(self, gen: LatentGen):
        self.g = gen

    def log_likelihood(self, row) -> float:
        df = pd.DataFrame([row])
        if "person_id" in df.columns:
            df = df.drop(columns=["person_id"])
        X = self.g._re_encode_frame(df)
        R = self.g._decode(self.g._encode(X))
        return -float(np.sum((X - R) ** 2))


def run_attack(seeds, csv=None, group_by="person_id"):
    """The SAME membership question the blueprint path answered at
    worst 0.52 - asked of the latent path, with the same bands,
    plus the republish control that makes any PASS worth reading.
    Default: the shared cohort fixture, for comparability. With
    --csv: the REAL frame at its real width, which is where an
    autoencoder's memorization risk actually lives - members are
    half the patients, non-members the other half of the SAME
    population."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from membership_new_path import cohort
    from synthkit.attack import membership_audit
    print("MEMBERSHIP ATTACK ON THE LATENT PATH - two "
          "adversaries, worse one reported; the reconstruction "
          "adversary assumes the WEIGHTS leak.")
    if csv:
        print("frame: {} (real width - rows are subsampled to "
              "1,500 per side for the row-scored adversary)"
              .format(csv))
    base = None
    if csv:
        base = pd.read_csv(csv, dtype=str,
                           keep_default_na=False).replace(
                               "", np.nan)
    worst = []
    for seed in seeds:
        df = base if base is not None else cohort(600, 6, seed)
        ids = sorted(df[group_by].astype(str).unique())
        rng = np.random.RandomState(seed + 1)
        rng.shuffle(ids)
        half = len(ids) // 2
        mem = df[df[group_by].astype(str).isin(set(ids[:half]))]
        non = df[df[group_by].astype(str).isin(set(ids[half:]))]
        g = LatentGen(seed=seed).fit(mem, group_by)
        synth = g.generate(min(len(mem), 4000), seed=seed + 2)
        audit = membership_audit(
            ReconstructionAdversary(g),
            mem.to_dict("records")[:1500],
            non.to_dict("records")[:1500],
            synthetic=synth.to_dict("records"))
        print("seed {:>2}: nn {:.3f}  reconstruction {:.3f}  "
              "worst {:.3f}  {}".format(
                  seed,
                  audit.get("nearest_neighbor", {}).get(
                      "auc", 0.5),
                  audit["likelihood"]["auc"],
                  audit["worst_auc"], audit["verdict"]))
        worst.append(audit["worst_auc"])
    # THE POSITIVE CONTROL - a "generator" that republishes its
    # members must FAIL, or the numbers above are decoration.
    df = base if base is not None else cohort(600, 6, seeds[0])
    ids = sorted(df[group_by].astype(str).unique())
    rng = np.random.RandomState(seeds[0] + 1)
    rng.shuffle(ids)
    half = len(ids) // 2
    mem = df[df[group_by].astype(str).isin(set(ids[:half]))]
    non = df[df[group_by].astype(str).isin(set(ids[half:]))]
    g = LatentGen(seed=seeds[0]).fit(mem, group_by)
    leak = membership_audit(
        ReconstructionAdversary(g),
        mem.to_dict("records")[:1500],
        non.to_dict("records")[:1500],
        synthetic=mem.to_dict("records")[:2000])
    print()
    print("republish control: worst {:.3f} {} (must FAIL, or "
          "the attack cannot see)".format(
              leak["worst_auc"], leak["verdict"]))
    print("latent path worst over {} seeds: mean {:.3f}, max "
          "{:.3f} - read beside the blueprint's recorded 0.52, "
          "same cohort fixture, same bands.".format(
              len(seeds), float(np.mean(worst)),
              float(np.max(worst))))
    return worst, leak


def run_csv(argv):
    """The real-extract read, for the data machine. Echoes every
    setting it received - a missing flag must be visible."""
    import argparse
    ap = argparse.ArgumentParser(prog="latent_challenger.py csv")
    ap.add_argument("csv")
    ap.add_argument("--group-by", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default=None,
                    help="directory for generated.csv per seed - "
                         "REAL-DERIVED output stays on this "
                         "machine")
    ap.add_argument("--shares", default=None,
                    help="CHILD=PARENT1,PARENT2 - SHAP driver "
                         "shares on a named triple, source vs "
                         "generated")
    a = ap.parse_args(argv)
    seeds = [int(x) for x in a.seeds.split(",")]
    print("csv: --group-by {} --seeds {} --out {} --shares {}"
          .format(a.group_by, a.seeds, a.out or "(none)",
                  a.shares or "(none)"))
    df = pd.read_csv(a.csv, dtype=str, keep_default_na=False)
    df = df.replace("", np.nan)
    print("read {}: {} rows x {} columns, {} patients".format(
        a.csv, len(df), df.shape[1],
        df[a.group_by].nunique()))
    for seed in seeds:
        tr, ho = _split_patients(df, a.group_by, seed)
        g = LatentGen(seed=seed).fit(tr, a.group_by)
        gen = g.generate(len(tr), seed=seed + 1000)
        cols = [c for c in df.columns if c != a.group_by]
        sign, close, inv, n = _pair_score(
            _pairs(tr, cols), _pairs(gen, cols))
        nn = g.nn_ratio(gen, g._re_encode_frame(ho))
        extra = ""
        if a.shares:
            child, ps = a.shares.split("=")
            parents = ps.split(",")
            # BOTH frames get the same numeric coercion - the
            # first cut coerced the source and handed the
            # generated frame over raw, where a folded __other__
            # string poisons the column and the shares read 0/0.
            def _numframe(fr):
                return fr.assign(**{
                    c: pd.to_numeric(fr[c], errors="coerce")
                    for c in [child] + parents}).dropna(
                        subset=[child] + parents)
            ss, _ = _shares(_numframe(tr), child, parents)
            gs, _ = _shares(_numframe(gen), child, parents)
            extra = " | shares src " + "/".join(
                "{:.0%}".format(x) for x in ss) + " gen " +                 "/".join("{:.0%}".format(x) for x in gs)
        print("seed {:>2}: sign {}/{} close {}/{} INVERTED {} | "
              "nn-ratio {:.2f}{}".format(seed, sign, n, close,
                                         n, inv, nn, extra))
        if a.out:
            outd = Path(a.out)
            outd.mkdir(parents=True, exist_ok=True)
            gen.to_csv(outd / "latent_gen_seed{}.csv".format(
                seed), index=False)
            print("  wrote {}".format(
                outd / "latent_gen_seed{}.csv".format(seed)))
    print("Row-level: no within-patient dynamics. Trains on "
          "records: a MEASUREMENT, not a release.")


def _mine_interactions(df, cols, group_by=None, top_n=8,
                       seed=0, max_children=40):
    """MINE the strongest two-variable interactions from the
    SOURCE, so both engines are judged on what the real data
    actually contains rather than on a pair somebody named. Parent
    screening goes through model importance, NOT rank correlation
    - an XOR's parents have Spearman near zero with their child,
    which is precisely why they matter. The gain measured is the
    product term's R2 beyond the two mains: a screen for TWO-way
    interactions, said plainly; higher orders are not mined."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    num = {}
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce")
        if v.notna().mean() >= 0.5 and v.nunique() >= 8:
            num[c] = v
    found = []
    children = list(num)[:max_children]
    for child in children:
        feats = [c for c in num if c != child]
        if len(feats) < 2:
            continue
        d = pd.DataFrame({c: num[c] for c in feats + [child]}
                         ).dropna(subset=[child])
        if len(d) < 200:
            continue
        X = d[feats].to_numpy()
        y = d[child].to_numpy()
        m = HistGradientBoostingRegressor(
            max_iter=60, random_state=seed,
            early_stopping=False).fit(X, y)
        sub = min(len(d), 2000)
        imp = permutation_importance(
            m, X[:sub], y[:sub], n_repeats=3,
            random_state=seed).importances_mean
        top = [feats[i] for i in np.argsort(-imp)[:4]]
        for i, a in enumerate(top):
            for b_ in top[i + 1:]:
                g = _interaction_r2(d, child, a, b_)
                if pd.notna(g) and g > 0.005:
                    found.append((float(g), child, a, b_))
    found.sort(reverse=True)
    return found[:top_n]


def run_compare(argv):
    """THE HEAD-TO-HEAD, one command, one table. The blueprint
    run's generated.csv and the latent path's draws are measured
    against the SAME source frame with the SAME metrics - a
    comparison assembled by eye from two terminals is not a
    comparison. Frames are restricted to the columns all three
    share, and what was dropped is SAID."""
    import argparse
    ap = argparse.ArgumentParser(
        prog="latent_challenger.py compare")
    ap.add_argument("csv")
    ap.add_argument("--group-by", required=True)
    ap.add_argument("--blueprint-run", required=True,
                    help="a finished fit run directory holding "
                         "generated.csv")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--shares", default=None, action="append",
                    help="CHILD=P1,P2 - repeatable")
    ap.add_argument("--interactions", type=int, default=8,
                    help="mine the top-N two-way interactions "
                         "from the SOURCE and measure both "
                         "engines on them; 0 disables")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    seeds = [int(x) for x in a.seeds.split(",")]
    print("compare: --group-by {} --blueprint-run {} --seeds {} "
          "--shares {} --out {}".format(
              a.group_by, a.blueprint_run, a.seeds,
              a.shares or "(none)", a.out or "(none)"))
    bp_csv = Path(a.blueprint_run) / "generated.csv"
    if not bp_csv.exists():
        print("STOPPED: no generated.csv in {} - point "
              "--blueprint-run at a finished `synthkit fit "
              "--generate` run".format(a.blueprint_run),
              file=sys.stderr)
        return 2
    src = pd.read_csv(a.csv, dtype=str,
                      keep_default_na=False).replace("", np.nan)
    bpg = pd.read_csv(bp_csv, dtype=str,
                      keep_default_na=False).replace("", np.nan)
    common = [c for c in src.columns
              if c in bpg.columns or c == a.group_by]
    dropped = [c for c in src.columns if c not in common]
    if dropped:
        print("columns not in the blueprint output, dropped "
              "from ALL sides so the metrics match: {}".format(
                  ", ".join(dropped[:8])
                  + (", ..." if len(dropped) > 8 else "")))
    src = src[common]
    cols = [c for c in common if c != a.group_by]

    def score(gen, label, g_for_nn=None, ho=None):
        sign, close, inv, n = _pair_score(
            _pairs(src, cols), _pairs(gen, cols))
        extra = ""
        for spec in (a.shares or []):
            child, ps = spec.split("=")
            parents = ps.split(",")

            def nf(fr):
                return fr.assign(**{
                    c: pd.to_numeric(fr[c], errors="coerce")
                    for c in [child] + parents}).dropna(
                        subset=[child] + parents)
            try:
                gs, _ = _shares(nf(gen), child, parents)
                extra += " | {} shares ".format(child) + "/".join(
                    "{:.0%}".format(x) for x in gs)
            except Exception as e:
                extra += " | {} shares unmeasurable ({})".format(
                    child, e)
        nnr = ""
        if g_for_nn is not None:
            nnr = " | nn-ratio {:.2f}".format(
                g_for_nn.nn_ratio(gen,
                                  g_for_nn._re_encode_frame(ho)))
        print("  {:<18} sign {}/{} close {}/{} INVERTED {}{}{}"
              .format(label, sign, n, close, n, inv, nnr, extra))

    for spec in (a.shares or []):
        child, ps = spec.split("=")
        parents = ps.split(",")

        def nf0(fr):
            return fr.assign(**{
                c: pd.to_numeric(fr[c], errors="coerce")
                for c in [child] + parents}).dropna(
                    subset=[child] + parents)
        ss, _ = _shares(nf0(src), child, parents)
        print("  {:<18} {} shares {}".format(
            "source", child, "/".join("{:.0%}".format(x)
                                      for x in ss)))
    score(bpg, "blueprint")
    latent_gens = []
    for seed in seeds:
        tr, ho = _split_patients(src, a.group_by, seed)
        g = LatentGen(seed=seed).fit(tr, a.group_by)
        gen = g.generate(len(tr), seed=seed + 1000)
        latent_gens.append((seed, gen))
        score(gen, "latent seed {}".format(seed),
              g_for_nn=g, ho=ho)
        if a.out:
            outd = Path(a.out)
            outd.mkdir(parents=True, exist_ok=True)
            gen.to_csv(outd / "latent_gen_seed{}.csv".format(
                seed), index=False)
            print("    wrote {}".format(
                outd / "latent_gen_seed{}.csv".format(seed)))
    if a.interactions:
        print()
        print("THE MINED INTERACTIONS - top {} two-way product "
              "gains found in the SOURCE (parents screened by "
              "model importance, so an XOR's parents are "
              "findable; higher orders not mined). gain = R2 of "
              "the product term beyond the two mains.".format(
                  a.interactions))
        mined = _mine_interactions(src, cols,
                                   top_n=a.interactions)
        if not mined:
            print("  none cleared the 0.005 gain floor - "
                  "stated, not silent.")
        for gv, child, pa, pb in mined:
            gb = _interaction_r2(bpg, child, pa, pb)
            row = "  {} <- {} x {}: src {:.3f} | blueprint "                   "{}".format(child, pa, pb, gv,
                              "{:.3f}".format(gb)
                              if pd.notna(gb) else "n/a")
            for seed, gen in latent_gens:
                gl = _interaction_r2(gen, child, pa, pb)
                row += " | latent s{} {}".format(
                    seed, "{:.3f}".format(gl)
                    if pd.notna(gl) else "n/a")
            print(row)
    print("Same source, same metrics, both engines. The latent "
          "path trains on records and has no dynamics; the "
          "blueprint publishes k-screened aggregates and passed "
          "its attack battery. Fidelity is only one column of "
          "that table.")
    return 0


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
    if which == "csv":
        run_csv(sys.argv[2:])
    elif which == "compare":
        sys.exit(run_compare(sys.argv[2:]))
    elif which == "attack":
        import argparse
        ap = argparse.ArgumentParser(
            prog="latent_challenger.py attack")
        ap.add_argument("seeds", nargs="*", type=int,
                        default=[0, 1, 2])
        ap.add_argument("--csv", default=None)
        ap.add_argument("--group-by", default="person_id")
        aa = ap.parse_args(sys.argv[2:])
        run_attack(aa.seeds or [0, 1, 2], csv=aa.csv,
                   group_by=aa.group_by)
    else:
        seeds = [int(x) for x in sys.argv[2:]] or DEFAULT_SEEDS
        run(which, seeds)
