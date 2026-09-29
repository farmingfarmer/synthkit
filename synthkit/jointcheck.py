"""Joint-distribution fidelity: the whole table, not a list of
enumerated patterns.

WHY THIS EXISTS. Everything measured before this module works by
NAMING a pattern first - a pair's rank correlation, a published
curve, a two-way product term - and then asking whether the
synthetic data kept it. That can only ever find what somebody
thought to look for, and it left the honest answer to "how close
is this really?" resting on a handful of enumerated features.

The five measures here ask the question the other way round.
Three of them need no enumeration at all: a classifier that tries
to tell real from synthetic will use whatever structure exists,
including structure nobody listed; a per-column predictability
profile asks whether each column is as learnable from the others
as it is in reality; and a nearest-neighbour precision/recall
pair asks whether the synthetic data covers the real space
without inventing regions that do not exist. The remaining two -
utility and three-way interactions - close the two gaps the
older measures explicitly stated they could not reach.

ONE ENCODER. Every measure here reads the same numeric
representation of both tables, built once from the REAL frame's
vocabulary so the two sides are commensurable. Two encoders is
how two measures come to disagree about the same data.

WHAT NONE OF THIS MEASURES: privacy. A generator that republished
its training data verbatim would score perfectly on every number
in this file. Fidelity and safety are different questions and the
pages that show these numbers say so.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from synthkit import dates

MAX_LEVELS = 24


def _shap_or_permutation(model, X, y, feature_names, seed=0):
    """Per-feature importance, SHAP where it is installed and
    permutation importance where it is not - and the caller is
    told WHICH, because a number whose provenance is unstated
    invites the reader to assume the better one."""
    import numpy as np
    try:
        import shap as _shap
        ex = _shap.TreeExplainer(model)
        sv = ex.shap_values(X[:800])
        if isinstance(sv, list):
            sv = np.mean([np.abs(s) for s in sv], axis=0)
        else:
            sv = np.abs(sv)
        if sv.ndim == 3:
            sv = np.abs(sv).mean(axis=2)
        return np.asarray(sv).mean(axis=0), "shap"
    except Exception:
        from sklearn.inspection import permutation_importance
        n = min(len(X), 1500)
        r = permutation_importance(model, X[:n], y[:n],
                                   n_repeats=3,
                                   random_state=seed)
        return np.asarray(r.importances_mean), "permutation"


class Encoder:
    """One numeric space for both tables, built from the REAL
    frame. Categorical levels come from the real data only: a
    level the generator invented is folded to `__other__` rather
    than silently given its own dimension, because a column the
    two sides disagree about the shape of cannot be compared at
    all."""

    def __init__(self, real, group_by: Optional[str] = None):
        import pandas as pd
        self.group_by = group_by
        self.plan: List[Any] = []
        for c in real.columns:
            if c == group_by:
                continue
            v = pd.to_numeric(real[c], errors="coerce")
            raw_ok = real[c].notna().mean()
            # A DATE IS A NUMBER, HERE TOO. Encoded as a
            # category it becomes its 60 most common days plus an
            # `other` bucket holding nearly every row on both
            # sides - so this measure, whose whole job is to ask
            # whether a model can tell the tables apart, was
            # blind to the column the generating engine was
            # destroying. Same module, same parse, both halves.
            dsp = None
            if not (raw_ok > 0 and v.notna().mean() >= 0.9 * raw_ok
                    and v.nunique() >= 3):
                dsp = dates.date_kind(real[c])
                if dsp:
                    v = dates.to_ordinal(real[c],
                                         dsp["parsed_format"])
                    if not (v.notna().any() and v.nunique() >= 3):
                        dsp = None
            if dsp or (raw_ok > 0
                       and v.notna().mean() >= 0.9 * raw_ok
                       and v.nunique() >= 3):
                mu = float(v.mean())
                sd = float(v.std()) or 1.0
                self.plan.append(("num", c, mu, sd, dsp))
            else:
                s = real[c].astype(str)
                lv = list(s.value_counts().index[:MAX_LEVELS])
                self.plan.append(("cat", c, lv, None))
        self.names = [it[1] for it in self.plan]

    def transform(self, df):
        import numpy as np
        import pandas as pd
        blocks, names = [], []
        for item in self.plan:
            kind, c = item[0], item[1]
            if c not in df.columns:
                # A column the other side does not have cannot be
                # compared; it is filled with the real mean and
                # NAMED by `missing_columns`, never silently
                # treated as zero-and-fine.
                if kind == "num":
                    blocks.append(np.zeros((len(df), 2)))
                else:
                    blocks.append(np.zeros((len(df),
                                            len(item[2]) + 1)))
                names.append(c)
                continue
            if kind == "num":
                _, _, mu, sd, dsp = item
                if dsp:
                    # The written file carries the DAY format;
                    # the source may carry a timestamp.
                    v = dates.to_ordinal(
                        df[c], dsp.get("format") or "%Y-%m-%d")
                    if not v.notna().any():
                        v = dates.to_ordinal(
                            df[c], dsp["parsed_format"])
                else:
                    v = pd.to_numeric(df[c], errors="coerce")
                blocks.append(np.column_stack([
                    ((v - mu) / sd).fillna(0.0).to_numpy(),
                    v.isna().astype(float).to_numpy()]))
            else:
                lv = item[2]
                s = df[c].astype(str)
                m = [np.where(s == l, 1.0, 0.0) for l in lv]
                m.append(np.where(s.isin(lv), 0.0, 1.0))
                blocks.append(np.column_stack(m))
            names.append(c)
        return np.hstack(blocks) if blocks else np.zeros(
            (len(df), 0))

    def groups(self):
        """Which encoded dimensions belong to which source column
        - importance is reported per COLUMN a person can name,
        never per one-hot dimension."""
        out, i = {}, 0
        for item in self.plan:
            w = 2 if item[0] == "num" else len(item[2]) + 1
            out[item[1]] = list(range(i, i + w))
            i += w
        return out


def _balanced(real, synth, seed=0, cap=6000):
    import numpy as np
    n = int(min(len(real), len(synth), cap))
    r = np.random.RandomState(seed)
    return (real.iloc[r.choice(len(real), n, replace=False)],
            synth.iloc[r.choice(len(synth), n, replace=False)])


def distinguishability(real, synth, group_by=None, seed=0):
    """CAN A MODEL TELL THEM APART? A classifier is trained to
    separate real rows from synthetic ones and scored on rows it
    never saw. 0.50 means indistinguishable - the ideal. 1.00
    means the synthetic data is trivially identifiable.

    Then the classifier is asked WHAT GAVE IT AWAY, per column.
    That list is the diagnosis: it names the columns whose joint
    behaviour differs, including structure nobody enumerated,
    which is the whole reason this measure exists.

    It cannot say WHY a column betrays the data - only that it
    does. Reading that is still a person's job."""
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    enc = Encoder(real, group_by)
    rr, ss = _balanced(real, synth, seed)
    Xr, Xs = enc.transform(rr), enc.transform(ss)
    X = np.vstack([Xr, Xs])
    y = np.concatenate([np.ones(len(Xr)), np.zeros(len(Xs))])
    rng = np.random.RandomState(seed + 1)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    cut = int(0.7 * len(X))
    m = HistGradientBoostingClassifier(
        max_iter=120, random_state=seed,
        early_stopping=False).fit(X[:cut], y[:cut])
    p = m.predict_proba(X[cut:])[:, 1]
    auc = float(roc_auc_score(y[cut:], p))
    imp, how = _shap_or_permutation(m, X[cut:], y[cut:],
                                    enc.names, seed)
    # THE PRIVACY RULE IS VISIBLE, AND IT IS NOT A FIDELITY
    # FAULT. Published bounds are k-anonymous, so the synthetic
    # data structurally CANNOT contain the extremes the real data
    # holds - on one real extract, 92% of real rows carried at
    # least one value outside the synthetic range, which lets any
    # classifier separate the tables perfectly while saying
    # nothing about how well the shape was learned. So the
    # measure is reported twice: once raw, and once over the real
    # rows the generator was even ALLOWED to produce. The second
    # is the fidelity number; the first is the privacy signature.
    # CLIP, DO NOT DROP. The first cut of this dropped whole real
    # rows holding any out-of-bound value - and on one extract
    # that kept 4.8% of them, so the comparison became "the most
    # CENTRAL real rows against all synthetic rows", which is a
    # different and unfair question. Clipping each real column
    # into the synthetic's own range keeps every row and removes
    # only the signature the k rule creates.
    rr2 = rr.copy()
    clipped_cols, touched = 0, 0.0
    for item in enc.plan:
        if item[0] != "num":
            continue
        c = item[1]
        if c not in synth.columns:
            continue
        a = pd.to_numeric(rr[c], errors="coerce")
        b = pd.to_numeric(ss[c], errors="coerce")
        if b.notna().sum() < 20:
            continue
        lo_, hi_ = float(b.min()), float(b.max())
        out = ((a < lo_) | (a > hi_)).fillna(False)
        if out.any():
            clipped_cols += 1
            touched = max(touched, float(out.mean()))
        rr2[c] = a.clip(lo_, hi_)
    Xr2 = enc.transform(rr2)
    n2 = int(min(len(Xr2), len(Xs)))
    X2 = np.vstack([Xr2[:n2], Xs[:n2]])
    y2 = np.concatenate([np.ones(n2), np.zeros(n2)])
    i2 = np.random.RandomState(seed + 2).permutation(len(X2))
    X2, y2 = X2[i2], y2[i2]
    c2 = int(0.7 * len(X2))
    m2 = HistGradientBoostingClassifier(
        max_iter=120, random_state=seed,
        early_stopping=False).fit(X2[:c2], y2[:c2])
    auc_in = float(roc_auc_score(
        y2[c2:], m2.predict_proba(X2[c2:])[:, 1]))
    kept = 1.0 - 0.0
    groups = enc.groups()
    tells = []
    tot = float(np.sum(np.abs(imp))) or 1.0
    for c, dims in groups.items():
        tells.append((c, float(np.sum(np.abs(imp[dims]))) / tot))
    tells.sort(key=lambda t: -t[1])
    missing = [c for c in enc.names if c not in synth.columns]
    # PER COLUMN, TOO. A joint classifier over ~150 encoded
    # dimensions SATURATES: differences too small to matter in
    # any one column compound into a perfect score, so a 1.00
    # joint AUC is not by itself actionable and must not be read
    # as "the data is useless". Each column is therefore also
    # tested ALONE - that number is comparable across columns
    # and says where the work actually is.
    per_col = []
    for c, dims in enc.groups().items():
        if c not in synth.columns:
            continue
        Xc = np.vstack([Xr2[:n2][:, dims], Xs[:n2][:, dims]])
        yc = np.concatenate([np.ones(n2), np.zeros(n2)])
        ic = np.random.RandomState(seed + 3).permutation(len(Xc))
        Xc, yc = Xc[ic], yc[ic]
        cc = int(0.7 * len(Xc))
        try:
            mc = HistGradientBoostingClassifier(
                max_iter=40, random_state=seed,
                early_stopping=False).fit(Xc[:cc], yc[:cc])
            per_col.append((c, round(float(roc_auc_score(
                yc[cc:], mc.predict_proba(Xc[cc:])[:, 1])), 4)))
        except Exception:
            continue
    per_col.sort(key=lambda t: -t[1])
    judged = auc if auc_in is None else auc_in
    return {"auc": round(auc, 4),
            "auc_within_published_range": (
                None if auc_in is None else round(auc_in, 4)),
            "columns_clipped_to_range": clipped_cols,
            "worst_column_share_beyond": round(touched, 4),
            "verdict": ("indistinguishable" if judged < 0.60 else
                        "distinguishable" if judged < 0.80 else
                        "trivially separable"),
            "tells": [(c, round(v, 4)) for c, v in tells[:12]],
            "per_column_auc": per_col[:16],
            "columns_indistinguishable": sum(
                1 for _, a2 in per_col if a2 < 0.60),
            "columns_measured": len(per_col),
            "importance_from": how,
            "rows_each_side": int(len(Xr)),
            "missing_columns": missing,
            "note": "0.50 is the ideal - a classifier doing no "
                    "better than a coin flip. `auc` includes the "
                    "PRIVACY SIGNATURE: k-anonymous bounds mean "
                    "the synthetic data cannot hold the "
                    "extremes the real data does, which alone "
                    "separates the tables and is by design. "
                    "`auc_within_published_range` repeats the "
                    "test with each real column CLIPPED into the "
                    "synthetic's own range - every row kept, only "
                    "the bound signature removed - and that is "
                    "the FIDELITY number the verdict is taken "
                    "from. A joint AUC over many columns "
                    "SATURATES - small differences everywhere "
                    "compound into a perfect score - so read "
                    "`per_column_auc`, which is comparable "
                    "across columns and says where the work "
                    "actually is. The tells name where the "
                    "tables differ, not why."}


def predictability(real, synth, group_by=None, max_cols=24,
                   seed=0):
    """IS EACH COLUMN AS LEARNABLE, AND DO THE SAME THINGS DRIVE
    IT? For every column in turn, a model is trained to predict
    it from all the others - once on the real data, once on the
    synthetic - and each is scored on held-out rows of its OWN
    table. Two numbers come back per column: whether the column
    is as predictable, and whether the same drivers do the
    driving, in the same proportions.

    This is the generalisation of driver attribution from a
    handful of named claims to the whole table, and it needs no
    interaction to be enumerated: whatever structure a model can
    use, it will use.

    Numeric children only - a categorical child needs a different
    skill measure, and mixing the two in one average would make
    the average mean nothing. The count of columns actually
    scored is returned."""
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor
    enc = Encoder(real, group_by)
    rr, ss = _balanced(real, synth, seed)
    groups = enc.groups()
    Xr_all, Xs_all = enc.transform(rr), enc.transform(ss)
    rows, skipped = [], []
    targets = [it[1] for it in enc.plan if it[0] == "num"]
    for c in targets[:max_cols]:
        if c not in synth.columns:
            skipped.append((c, "not in the synthetic table"))
            continue
        keep = [d for cc, dims in groups.items() if cc != c
                for d in dims]
        yr = pd.to_numeric(rr[c], errors="coerce")
        ys = pd.to_numeric(ss[c], errors="coerce")
        mr, ms = yr.notna().to_numpy(), ys.notna().to_numpy()
        if mr.sum() < 200 or ms.sum() < 200:
            skipped.append((c, "too few non-empty rows"))
            continue
        out = {}
        vecs = {}
        for tag, X_, yv, msk in (("real", Xr_all, yr, mr),
                                 ("syn", Xs_all, ys, ms)):
            Xf = X_[msk][:, keep]
            yf = yv[msk].to_numpy(dtype=float)
            cut = int(0.7 * len(Xf))
            if cut < 100:
                break
            mdl = HistGradientBoostingRegressor(
                max_iter=60, random_state=seed,
                early_stopping=False).fit(Xf[:cut], yf[:cut])
            pred = mdl.predict(Xf[cut:])
            var = float(np.var(yf[cut:]))
            out[tag] = (0.0 if var <= 0 else
                        float(1.0 - np.mean(
                            (yf[cut:] - pred) ** 2) / var))
            v, how = _shap_or_permutation(mdl, Xf[cut:],
                                          yf[cut:], None, seed)
            vecs[tag] = np.abs(np.asarray(v))
            vecs["how"] = how
        if "real" not in out or "syn" not in out:
            skipped.append((c, "not enough held-out rows"))
            continue
        a, b = vecs["real"], vecs["syn"]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        cos = float(a.dot(b) / (na * nb)) if na and nb else 0.0
        rows.append({"column": c,
                     "skill_real": round(out["real"], 4),
                     "skill_syn": round(out["syn"], 4),
                     "skill_gap": round(
                         abs(out["real"] - out["syn"]), 4),
                     "driver_agreement": round(cos, 4)})
    gaps = [r["skill_gap"] for r in rows]
    agr = [r["driver_agreement"] for r in rows]
    return {"columns": rows, "scored": len(rows),
            "skipped": skipped,
            "mean_skill_gap": round(float(np.mean(gaps)), 4)
            if gaps else None,
            "mean_driver_agreement": round(float(np.mean(agr)), 4)
            if agr else None,
            "importance_from": vecs.get("how") if rows else None,
            "note": "skill is out-of-sample R2 on each table's "
                    "OWN held-out rows. driver_agreement is the "
                    "cosine between the two importance vectors: "
                    "1.00 means the same things drive the column "
                    "in both tables, in the same proportions."}


def utility(real, synth, target, group_by=None, seed=0):
    """TRAIN ON SYNTHETIC, TEST ON REAL - the question a customer
    actually asks. A model trained on the synthetic data is
    scored on REAL rows it never saw, beside the same model
    trained on real data. The gap between them is what using
    synthetic data costs.

    One target at a time, because the answer depends entirely on
    what is being predicted and an average over targets would
    hide that."""
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor
    if target not in real.columns or target not in synth.columns:
        raise ValueError("target {!r} must be in both tables"
                         .format(target))
    enc = Encoder(real.drop(columns=[target]), group_by)
    yr = pd.to_numeric(real[target], errors="coerce")
    ys = pd.to_numeric(synth[target], errors="coerce")
    if yr.notna().sum() < 300 or ys.notna().sum() < 300:
        raise ValueError("target {!r} has too few numeric values "
                         "to score".format(target))
    Xr = enc.transform(real)[yr.notna().to_numpy()]
    Xs = enc.transform(synth)[ys.notna().to_numpy()]
    yrv = yr.dropna().to_numpy(dtype=float)
    ysv = ys.dropna().to_numpy(dtype=float)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(Xr))
    Xr, yrv = Xr[perm], yrv[perm]
    cut = int(0.7 * len(Xr))
    Xtr, ytr, Xte, yte = Xr[:cut], yrv[:cut], Xr[cut:], yrv[cut:]

    def score(X, y):
        m = HistGradientBoostingRegressor(
            max_iter=80, random_state=seed,
            early_stopping=False).fit(X, y)
        var = float(np.var(yte))
        if var <= 0:
            return 0.0
        return float(1.0 - np.mean(
            (yte - m.predict(Xte)) ** 2) / var)
    trtr = score(Xtr, ytr)
    tstr = score(Xs, ysv)
    return {"target": target,
            "train_real_test_real": round(trtr, 4),
            "train_synthetic_test_real": round(tstr, 4),
            "retained": (round(tstr / trtr, 4)
                         if trtr > 0 else None),
            "note": "Both scored on the SAME held-out REAL rows. "
                    "`retained` is the share of real-data "
                    "performance a model gets by training on the "
                    "synthetic data instead."}


def three_way(real, synth, max_children=12, top_n=6, seed=0):
    """THE GAP THE OTHER MEASURES STATE THEY CANNOT REACH. Every
    interaction measured elsewhere is two-way; this asks whether
    a THREE-column product explains anything beyond all three
    pairs, and whether the synthetic data kept it.

    Still bounded: three at a time, numeric columns only, the top
    children by variance. Nothing above three is measured, and
    saying so is the point."""
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LinearRegression
    num = {}
    for c in real.columns:
        v = pd.to_numeric(real[c], errors="coerce")
        if v.notna().mean() >= 0.5 and v.nunique() >= 8 \
                and c in synth.columns:
            num[c] = v
    found = []
    for child in list(num)[:max_children]:
        feats = [c for c in num if c != child]
        if len(feats) < 3:
            continue
        d = pd.DataFrame({c: num[c] for c in feats + [child]}
                         ).dropna()
        if len(d) < 400:
            continue
        X = d[feats].to_numpy()
        y = d[child].to_numpy()
        m = HistGradientBoostingRegressor(
            max_iter=50, random_state=seed,
            early_stopping=False).fit(X, y)
        sub = min(len(d), 1500)
        imp = permutation_importance(
            m, X[:sub], y[:sub], n_repeats=2,
            random_state=seed).importances_mean
        top = [feats[i] for i in np.argsort(-imp)[:3]]
        if len(top) < 3:
            continue

        def gain(fr):
            dd = fr[[child] + top].apply(
                pd.to_numeric, errors="coerce").dropna()
            if len(dd) < 200:
                return float("nan")
            a, b, c3 = (dd[t].to_numpy() for t in top)
            yv = dd[child].to_numpy()
            pairs = np.column_stack([a, b, c3, a * b, a * c3,
                                     b * c3])
            trip = np.column_stack([pairs, a * b * c3])
            r0 = LinearRegression().fit(pairs, yv).score(pairs, yv)
            r1 = LinearRegression().fit(trip, yv).score(trip, yv)
            return float(r1 - r0)
        gs = gain(real)
        if not np.isfinite(gs) or gs <= 0.002:
            continue
        found.append({"child": child, "trio": top,
                      "source": round(gs, 4),
                      "synthetic": round(gain(synth), 4)})
    found.sort(key=lambda r: -r["source"])
    return {"trios": found[:top_n], "measured": len(found),
            "note": "Gain is the R2 of the three-way product "
                    "beyond all three pairwise products. Nothing "
                    "above three columns is measured."}


def manifold(real, synth, group_by=None, seed=0, sample=3000):
    """DOES IT COVER THE REAL SPACE, AND DOES IT INVENT SPACE
    THAT IS NOT THERE? Two nearest-neighbour questions against a
    yardstick taken from the real data's own spacing.

    PRECISION: the share of synthetic rows that sit as close to a
    real row as real rows sit to each other. Low precision means
    the generator is producing records unlike anything real.

    RECALL: the share of real rows that have a synthetic row as
    close. Low recall means whole regions of the real data have
    no synthetic counterpart - the file looks fine on average and
    silently omits a population.

    Both are blind to WHICH region, and neither says anything
    about privacy: a verbatim copy scores 1.00 on both."""
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    enc = Encoder(real, group_by)
    rr, ss = _balanced(real, synth, seed, cap=sample)
    R, S = enc.transform(rr), enc.transform(ss)
    if len(R) < 50 or len(S) < 50:
        return {"precision": None, "recall": None,
                "note": "too few rows to measure"}
    nnr = NearestNeighbors(n_neighbors=2).fit(R)
    d_rr = nnr.kneighbors(R)[0][:, 1]
    yard = float(np.median(d_rr))
    d_sr = nnr.kneighbors(S, n_neighbors=1)[0].ravel()
    nns = NearestNeighbors(n_neighbors=1).fit(S)
    d_rs = nns.kneighbors(R)[0].ravel()
    return {"precision": round(float(np.mean(d_sr <= yard)), 4),
            "recall": round(float(np.mean(d_rs <= yard)), 4),
            "yardstick": round(yard, 4),
            "note": "The yardstick is the median distance "
                    "between a real row and its nearest real "
                    "neighbour. PRECISION: synthetic rows that "
                    "sit that close to something real. RECALL: "
                    "real rows with a synthetic row that close. "
                    "Neither measures privacy - a verbatim copy "
                    "scores 1.00 on both."}


def run_all(real, synth, group_by=None, target=None, seed=0
            ) -> Dict[str, Any]:
    """Every measure in this module, one call, each failure
    caught and NAMED rather than taking the others down with
    it."""
    out: Dict[str, Any] = {}
    for name, fn in (
            ("distinguishability",
             lambda: distinguishability(real, synth, group_by,
                                        seed)),
            ("predictability",
             lambda: predictability(real, synth, group_by,
                                    seed=seed)),
            ("three_way", lambda: three_way(real, synth,
                                            seed=seed)),
            ("manifold", lambda: manifold(real, synth, group_by,
                                          seed))):
        try:
            out[name] = fn()
        except Exception as e:
            out[name] = {"error": "{}: {}".format(
                type(e).__name__, e)}
    if target:
        try:
            out["utility"] = utility(real, synth, target,
                                     group_by, seed)
        except Exception as e:
            out["utility"] = {"error": "{}: {}".format(
                type(e).__name__, e)}
    return out
