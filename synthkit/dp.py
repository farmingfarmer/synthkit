"""Differentially private training for the neural path.

WHAT THIS IS, AND WHAT IT IS NOT. Read this before quoting an
epsilon from it anywhere.

It is a correct implementation of DP-SGD for the autoencoder the
neural engine trains: per-example gradient clipping to an L2
norm, Gaussian noise on the clipped sum, Poisson subsampling,
and an RDP accountant for the subsampled Gaussian mechanism. The
epsilon it reports covers THE AUTOENCODER WEIGHTS and nothing
else.

IT IS NOT END-TO-END DIFFERENTIAL PRIVACY FOR THE RELEASE. The
neural path touches the data in several places and only one of
them is accounted here. `UNACCOUNTED_STAGES` names every other
one, `scope_note()` prints them, and the CLI refuses to describe
its output as differentially private - because a half-accounted
epsilon is worse than none. That is the same argument this
project makes about redacting free text: a mechanism that
misses one path hands back false confidence, and false
confidence about privacy is the most expensive kind there is.

WHY SUBSAMPLING IS NOT OPTIONAL. Measured with the accountant
below: full-batch gradient descent gives eps ~ 4.8*sqrt(T)/sigma,
so the 800 steps this architecture trains for would need a noise
multiplier near 136 to reach eps 1 - which destroys training.
With Poisson subsampling at q=0.01 and sigma=2.0, the same 800
steps cost eps 0.771. Practical DP training IS privacy
amplification by subsampling; without it there is no usable
setting.

WHY THE TRAINER IS HAND-ROLLED. `MLPRegressor` exposes no
per-example gradients, so DP-SGD cannot be applied to it. That
makes the hand-rolled trainer a CONFOUND: any utility cost
measured against the shipping engine would mix "the cost of DP"
with "the cost of my trainer". So the control arm is the same
hand-rolled network with the noise and clipping turned OFF, and
the DP cost is measured against THAT. The comparison against
`MLPRegressor` is reported separately, as the trainer's own
cost.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

# Orders tried when converting RDP to (eps, delta). Integers,
# because the subsampled-Gaussian bound below is a closed form
# only for integer alpha - a fractional order would need the
# numerical integration this module deliberately avoids.
ALPHAS = tuple(range(2, 257))

# Every stage of the neural path that touches the data and is
# NOT covered by the epsilon this module reports. Named rather
# than counted, because a reader has to be able to act on it -
# and because an epsilon whose scope is a number is an epsilon
# nobody can audit.
UNACCOUNTED_STAGES = (
    ("k-screen and vocabulary selection",
     "which levels and set tokens get modelled is chosen from "
     "PATIENT COUNTS in the real data. Those counts are "
     "k-screened but not noised, so the selection itself leaks"),
    ("published numeric bounds",
     "columns are clipped to the blueprint's k-anonymous "
     "minimum and maximum, which are means of the k most "
     "extreme patients' own extremes - a k-anonymity "
     "guarantee, not a DP one"),
    ("the latent mixture",
     "the GaussianMixture is fitted on latent codes of the real "
     "rows. Even with DP weights, the codes are a function of "
     "real records and the mixture parameters are not noised"),
    ("visit-count and set-size grids",
     "how many visits a patient has, and how large a set is, "
     "come from k-screened empirical grids rather than from a "
     "DP mechanism"),
    ("identity and arithmetic tiers",
     "size identities and arithmetic identities are DETECTED "
     "from the real rows and enforced at generation"),
)


def _log_add(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    hi, lo = max(a, b), min(a, b)
    return hi + math.log1p(math.exp(lo - hi))


def rdp_gaussian(sigma: float, alpha: float) -> float:
    """RDP of the plain Gaussian mechanism at order alpha.

    Sensitivity 1, noise sd sigma. This is the full-batch case
    and the limit the subsampled bound must reduce to at q=1 -
    which is the first self-consistency check on the whole
    accountant.
    """
    return alpha / (2.0 * sigma * sigma)


def rdp_subsampled_gaussian(q: float, sigma: float,
                            alpha: int) -> float:
    """RDP of the Poisson-subsampled Gaussian, integer alpha.

    The standard bound:

        RDP(a) <= 1/(a-1) * log sum_{k=0..a} C(a,k)
                  (1-q)^(a-k) q^k exp(k(k-1)/(2 sigma^2))

    Computed in log space so the binomial term cannot overflow
    at the orders this needs (alpha up to 256). A closed form,
    auditable by hand, which is why it is used here rather than
    a tighter bound requiring numerical integration: a wrong
    epsilon is worse than a loose one.
    """
    if q <= 0.0:
        return 0.0
    if q >= 1.0:
        return rdp_gaussian(sigma, alpha)
    acc = -math.inf
    for k in range(int(alpha) + 1):
        term = (math.lgamma(alpha + 1) - math.lgamma(k + 1)
                - math.lgamma(alpha - k + 1)
                + (alpha - k) * math.log1p(-q)
                + (k * math.log(q) if k else 0.0)
                + k * (k - 1) / (2.0 * sigma * sigma))
        acc = _log_add(acc, term)
    return acc / (alpha - 1.0)


def dp_epsilon(q: float, sigma: float, steps: int,
               delta: float = 1e-5) -> Tuple[float, int]:
    """(eps, alpha*) for `steps` subsampled Gaussian steps."""
    best, best_a = float("inf"), ALPHAS[0]
    for a in ALPHAS:
        e = (steps * rdp_subsampled_gaussian(q, sigma, a)
             + math.log(1.0 / delta) / (a - 1.0))
        if e < best:
            best, best_a = e, a
    return best, best_a


def sigma_for_epsilon(q: float, steps: int, target_eps: float,
                      delta: float = 1e-5,
                      lo: float = 0.3, hi: float = 400.0
                      ) -> float:
    """The smallest noise multiplier reaching `target_eps`.

    Bisection on a monotone function - epsilon falls as sigma
    rises, which the accountant's own self-consistency check
    pins. Returns `hi` if the target is unreachable in range,
    and the caller is expected to notice rather than be handed
    a silent cap.
    """
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if dp_epsilon(q, mid, steps, delta)[0] > target_eps:
            lo = mid
        else:
            hi = mid
    return hi


def scope_note(eps: Optional[float] = None,
               delta: float = 1e-5) -> List[str]:
    """What the epsilon covers, and what it does not.

    Returned as lines so a caller prints them next to the
    number. Every stage below touches the real data and is not
    in the accounting; an epsilon quoted without them is a
    claim about one component dressed as a claim about a
    release.
    """
    out = []
    if eps is None:
        out.append("DP TRAINING IS OFF. No epsilon applies, and "
                   "the neural path's posture is the one it "
                   "always had: it trains on records, and its "
                   "privacy evidence is the attack battery "
                   "rather than a guarantee.")
        return out
    out.append(
        "DP TRAINING: the autoencoder weights are "
        "({:.3f}, {:.0e})-differentially private with respect "
        "to one training ROW.".format(eps, delta))
    out.append(
        "THIS IS NOT AN END-TO-END GUARANTEE FOR THE RELEASE, "
        "and the output must not be described as differentially "
        "private. {} other stages touch the real data and are "
        "NOT accounted:".format(len(UNACCOUNTED_STAGES)))
    for name, why in UNACCOUNTED_STAGES:
        out.append("  - {}: {}".format(name, why))
    out.append(
        "Composing those would need a mechanism per stage and "
        "an accounting across them. Until that exists, this "
        "epsilon is evidence about the WEIGHTS - which is the "
        "surface the weights-leak adversary reads - and not a "
        "release certificate.")
    return out


def _relu(a):
    return np.maximum(a, 0.0)


class DPAutoencoder(object):
    """A small autoencoder trainable with or without DP-SGD.

    Shape matches what the neural engine already builds:
    d -> h -> b -> h -> d, ReLU hidden, linear output, MSE loss.

    PER-EXAMPLE GRADIENT NORMS WITHOUT PER-EXAMPLE GRADIENTS.
    For a layer with input activations `a` and output deltas
    `g`, example i's weight gradient is the outer product
    a_i g_i^T, whose squared Frobenius norm is
    ||a_i||^2 ||g_i||^2. So the per-example total norm is a sum
    of products of row norms - no (batch x in x out) tensor is
    ever materialised, which is what makes clipping affordable
    at this width.
    """

    def __init__(self, d, h, b, seed=0, lr=0.01,
                 clip=1.0, sigma=None, q=0.01, steps=800,
                 beta1=0.9, beta2=0.999, eps_adam=1e-8):
        self.d, self.h, self.b = int(d), int(h), int(b)
        self.lr = float(lr)
        self.clip = float(clip)
        self.sigma = sigma          # None => no DP noise
        self.q = float(q)
        self.steps = int(steps)
        rs = np.random.RandomState(seed)
        sizes = [self.d, self.h, self.b, self.h, self.d]
        self.W, self.B = [], []
        for i in range(4):
            # He initialisation, which the ReLU stack wants.
            self.W.append((rs.randn(sizes[i], sizes[i + 1])
                           * math.sqrt(2.0 / sizes[i])
                           ).astype(np.float64))
            self.B.append(np.zeros(sizes[i + 1]))
        self.rs = rs
        self.history = []
        # ADAM, because the trainer is a CONFOUND and this is
        # what closes it. Plain SGD at a fixed rate reached
        # holdout MSE 0.065 where `MLPRegressor` - which uses
        # adam - reached 0.011 at equal wall clock, so a DP cost
        # measured against the SGD arm would have been five
        # parts optimiser and one part privacy. The noise is
        # added to the gradient BEFORE the optimiser sees it,
        # which is what DP-Adam does and what keeps the
        # accounting valid: the privatised gradient is the only
        # thing that touches the data.
        self.beta1, self.beta2 = float(beta1), float(beta2)
        self.eps_adam = float(eps_adam)
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mB = [np.zeros_like(b_) for b_ in self.B]
        self.vB = [np.zeros_like(b_) for b_ in self.B]
        self._t = 0

    def _forward(self, X):
        acts = [X]
        a = X
        for i in range(4):
            z = a @ self.W[i] + self.B[i]
            a = _relu(z) if i < 3 else z
            acts.append(a)
        return acts

    def _grads(self, acts, Y):
        """Deltas per layer, and the per-example squared norms."""
        n = acts[0].shape[0]
        # dL/dout for MSE averaged over FEATURES only; the sum
        # over examples happens after clipping, which is what
        # makes the sensitivity per example bounded.
        g = 2.0 * (acts[4] - Y) / float(self.d)
        deltas = [None] * 4
        deltas[3] = g
        for i in (2, 1, 0):
            g = (g @ self.W[i + 1].T) * (acts[i + 1] > 0)
            deltas[i] = g
        sq = np.zeros(n)
        for i in range(4):
            an = np.einsum("ij,ij->i", acts[i], acts[i])
            dn = np.einsum("ij,ij->i", deltas[i], deltas[i])
            sq += an * dn + dn          # weights, then bias
        return deltas, sq

    def fit(self, X, Y=None, report=None):
        """Train. `sigma=None` is the control arm: no noise, no
        clipping, same network and same schedule."""
        X = np.asarray(X, dtype=np.float64)
        Y = X if Y is None else np.asarray(Y, dtype=np.float64)
        n = X.shape[0]
        private = self.sigma is not None
        for t in range(self.steps):
            if private:
                # POISSON sampling, which is what the accountant
                # assumes. A fixed-size shuffle would need a
                # different (and weaker) amplification bound, and
                # using one while quoting this accountant would
                # make the epsilon wrong.
                take = self.rs.random_sample(n) < self.q
                if not take.any():
                    continue
                xb, yb = X[take], Y[take]
            else:
                # The CONTROL ARM draws the same expected
                # batch size, so the two arms differ in the
                # noise and the clipping and not in how much
                # data a step sees.
                m = max(1, int(round(self.q * n)))
                idx = self.rs.randint(0, n, m)
                xb, yb = X[idx], Y[idx]
            acts = self._forward(xb)
            deltas, sq = self._grads(acts, yb)
            nb = xb.shape[0]
            if private:
                norms = np.sqrt(np.maximum(sq, 1e-24))
                scale = np.minimum(1.0, self.clip / norms)
            else:
                scale = np.ones(nb)
            self._t += 1
            b1, b2 = self.beta1, self.beta2
            c1 = 1.0 - b1 ** self._t
            c2 = 1.0 - b2 ** self._t
            for i in range(4):
                gw = acts[i].T @ (deltas[i] * scale[:, None])
                gb = (deltas[i] * scale[:, None]).sum(axis=0)
                if private:
                    s = self.sigma * self.clip
                    gw = gw + self.rs.normal(0.0, s, gw.shape)
                    gb = gb + self.rs.normal(0.0, s, gb.shape)
                # The EXPECTED batch size, not the realised one.
                # Dividing by the realised count would make the
                # update depend on the sample size, which is
                # itself data - a small leak the accountant does
                # not cover.
                denom = max(1.0, self.q * n) if private else nb
                gw = gw / denom
                gb = gb / denom
                self.mW[i] = b1 * self.mW[i] + (1 - b1) * gw
                self.vW[i] = b2 * self.vW[i] + (1 - b2) * gw * gw
                self.mB[i] = b1 * self.mB[i] + (1 - b1) * gb
                self.vB[i] = b2 * self.vB[i] + (1 - b2) * gb * gb
                self.W[i] -= self.lr * (self.mW[i] / c1) / (
                    np.sqrt(self.vW[i] / c2) + self.eps_adam)
                self.B[i] -= self.lr * (self.mB[i] / c1) / (
                    np.sqrt(self.vB[i] / c2) + self.eps_adam)
            if report is not None and (t % 100 == 0):
                self.history.append(float(np.mean(
                    (self._forward(X)[4] - Y) ** 2)))
        if report is not None:
            report["train_mse"] = float(np.mean(
                (self._forward(X)[4] - Y) ** 2))
            if private:
                eps, alpha = dp_epsilon(self.q, self.sigma,
                                        self.steps)
                report["dp"] = {
                    "epsilon": eps, "delta": 1e-5,
                    "alpha": alpha, "sigma": self.sigma,
                    "clip": self.clip, "q": self.q,
                    "steps": self.steps,
                }
        return self

    # SKLEARN-SHAPED ON PURPOSE. The engine's own `_fwd` reads
    # `net.coefs_` and `net.intercepts_` and applies ReLU to
    # every layer but the last - which is exactly this network's
    # layout and activation rule. Exposing the two names makes
    # this a DROP-IN for `MLPRegressor` with no change to
    # `_encode`, `_decode`, `generate` or anything downstream,
    # so the DP arm and the shipping arm differ in the trainer
    # and in nothing else.
    @property
    def coefs_(self):
        return self.W

    @property
    def intercepts_(self):
        return self.B

    def encode(self, X):
        return self._forward(np.asarray(X, dtype=np.float64))[2]

    def decode(self, Z):
        a = np.asarray(Z, dtype=np.float64)
        for i in (2, 3):
            z = a @ self.W[i] + self.B[i]
            a = _relu(z) if i < 3 else z
        return a

    def predict(self, X):
        return self._forward(np.asarray(X, dtype=np.float64))[4]
