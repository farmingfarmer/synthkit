"""Reference generators, so a fidelity number has something to beat.

WHY THIS EXISTS. Every fidelity number this project has produced is
ABSOLUTE. "center within 10% of spread on 31/33 columns" - against
what? A structural generator that learns marginals, a relationship
graph, effect curves, dynamics and interactions is a large thing to
build and maintain, and nothing here has ever established that it
beats the naive alternative on the numbers it reports. Without a
reference, a good score and a bad score look the same.

So there are two references, chosen to bracket the question:

  INDEPENDENT   every column drawn from its own marginal, nothing
                else. The NULL: correct one column at a time and
                structurally empty. Any generator that fails to beat
                this on the pair measures is not modeling anything.

  COPULA        a Gaussian copula over the numeric columns - rank to
                normal scores, one correlation matrix, sample, map
                back through the empirical quantile function. This is
                the off-the-shelf answer a reviewer will ask why you
                did not use. It gets the marginals exactly right by
                construction and the linear rank structure with them.

THESE ARE NOT RELEASE CANDIDATES AND MUST NEVER BE TREATED AS ONE.
Both fit the empirical quantile function of the real column and
resample from it, so published values are real patients' values with
no k-screen anywhere. That is precisely what `blueprint.py` refuses to
do. They exist to put a number beside a number, they are never
written to a blueprint, and `bench_baselines.py` says so on every run.

WHAT THEY DELIBERATELY DO NOT MODEL, so a loss against them is read
correctly: neither carries within-patient dynamics, so both will lose
badly on steadiness and ICC, and that is not evidence about the
copula - it is what a row-wise model is. The copula also draws
categoricals from their marginals independently, because a Gaussian
copula has no vocabulary for them.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _visit_counts(df: pd.DataFrame, group_by: str) -> np.ndarray:
    """The source's own visits-per-patient, to be resampled."""
    return df.groupby(group_by).size().to_numpy()


def _numeric_cols(df: pd.DataFrame, group_by: str) -> List[str]:
    return [c for c in df.columns
            if c != group_by and pd.api.types.is_numeric_dtype(df[c])]


def _frame(counts: np.ndarray, group_by: str) -> pd.DataFrame:
    pids = np.array(["SYN{:06d}".format(i) for i in range(len(counts))])
    return pd.DataFrame({
        group_by: np.repeat(pids, counts),
        "visit_number": np.concatenate(
            [np.arange(1, c + 1) for c in counts]) if len(counts)
        else np.array([], dtype=int)})


def _draw_counts(src_counts: np.ndarray, n_patients: int, rng):
    return rng.choice(src_counts, size=n_patients, replace=True)


def _empirical_quantile(values: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Map uniforms through the column's own sorted values."""
    v = np.sort(values[np.isfinite(values)])
    if not len(v):
        return np.full(len(u), np.nan)
    idx = np.clip((u * len(v)).astype(int), 0, len(v) - 1)
    return v[idx]


class Independent(object):
    """Each column from its own marginal. Nothing between columns."""

    name = "independent"

    def __init__(self, df, group_by, time_col=None):
        self.group_by = group_by
        self.counts = _visit_counts(df, group_by)
        self.cols = [c for c in df.columns if c != group_by]
        self.values = dict((c, df[c].to_numpy()) for c in self.cols)

    def sample(self, n_patients: int, seed: int) -> pd.DataFrame:
        rng = np.random.RandomState(seed)
        counts = _draw_counts(self.counts, n_patients, rng)
        out = _frame(counts, self.group_by)
        n = len(out)
        for c in self.cols:
            v = self.values[c]
            out[c] = v[rng.randint(0, len(v), size=n)]
        return out


class GaussianCopula(object):
    """One correlation matrix over normal scores of the numerics.

    Missingness is reproduced as an independent per-column rate: the
    copula has nothing to say about WHICH rows are measured, and
    pretending otherwise would flatter it."""

    name = "copula"

    def __init__(self, df, group_by, time_col=None):
        from scipy.special import ndtri
        self.group_by = group_by
        self.counts = _visit_counts(df, group_by)
        self.num = _numeric_cols(df, group_by)
        self.other = [c for c in df.columns
                      if c != group_by and c not in self.num]
        self.values = dict((c, df[c].to_numpy(dtype=float))
                           for c in self.num)
        self.other_values = dict((c, df[c].to_numpy())
                                 for c in self.other)
        self.coverage = dict(
            (c, float(np.isfinite(self.values[c]).mean()))
            for c in self.num)

        # Rank -> normal score, per column, ignoring the missing.
        scores = {}
        for c in self.num:
            v = self.values[c]
            ok = np.isfinite(v)
            z = np.full(len(v), np.nan)
            if ok.sum() > 2:
                r = pd.Series(v[ok]).rank(method="average").to_numpy()
                z[ok] = ndtri(r / (ok.sum() + 1.0))
            scores[c] = z
        S = np.column_stack([scores[c] for c in self.num]) \
            if self.num else np.zeros((len(df), 0))
        self.corr = self._corr(S)

    @staticmethod
    def _corr(S: np.ndarray) -> np.ndarray:
        """Pairwise-complete correlation, repaired to be sampleable.

        Pairwise completion can produce a matrix that is not positive
        semi-definite, which `multivariate_normal` will either refuse
        or quietly fudge. Clipping the eigenvalues is the standard
        repair and is done here explicitly rather than left to the
        library's discretion."""
        k = S.shape[1]
        if k == 0:
            return np.zeros((0, 0))
        C = np.eye(k)
        for i in range(k):
            for j in range(i + 1, k):
                m = np.isfinite(S[:, i]) & np.isfinite(S[:, j])
                if m.sum() > 10:
                    a, b = S[m, i], S[m, j]
                    if a.std() > 0 and b.std() > 0:
                        C[i, j] = C[j, i] = float(
                            np.corrcoef(a, b)[0, 1])
        w, V = np.linalg.eigh(C)
        w = np.clip(w, 1e-6, None)
        C = V @ np.diag(w) @ V.T
        d = np.sqrt(np.diag(C))
        return C / np.outer(d, d)

    def sample(self, n_patients: int, seed: int) -> pd.DataFrame:
        from scipy.special import ndtr
        rng = np.random.RandomState(seed)
        counts = _draw_counts(self.counts, n_patients, rng)
        out = _frame(counts, self.group_by)
        n = len(out)
        if self.num:
            Z = rng.multivariate_normal(
                np.zeros(len(self.num)), self.corr, size=n)
            U = ndtr(Z)
            for i, c in enumerate(self.num):
                col = _empirical_quantile(self.values[c], U[:, i])
                cov = self.coverage[c]
                if cov < 1.0:
                    col = np.where(rng.random_sample(n) < cov,
                                   col, np.nan)
                out[c] = col
        for c in self.other:
            v = self.other_values[c]
            out[c] = v[rng.randint(0, len(v), size=n)]
        return out


REFERENCES = (Independent, GaussianCopula)
