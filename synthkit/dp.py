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


# ---------------------------------------------------------------
# THE AGGREGATE MECHANISMS (DP-2)
#
# The epsilon above covers the autoencoder WEIGHTS. Five other
# stages touch the real data, and four of them are COUNTS or
# QUANTILES - which have standard mechanisms, so the gap was
# never a research problem. It was unbuilt.
#
# SENSITIVITY IS THE WHOLE DESIGN, AND IT IS PATIENT-LEVEL. The k
# rule in this project has always counted PATIENTS, never rows -
# one person seen 200 times could otherwise set a published bound
# alone. So the unit of privacy here is the PATIENT, and every
# mechanism below states what one patient can move:
#
#   a single count (patients holding a token)   ->  1
#   a histogram over many tokens                ->  L, the cap on
#                                                   tokens per
#                                                   patient
#   a quantile over per-patient values          ->  1 rank
#
# The histogram case is the one that is easy to get wrong: a
# patient holds MANY tokens, so their removal moves many counts
# at once and the L1 sensitivity is the number of tokens they
# hold - unbounded unless it is CLIPPED. `dp_histogram` clips and
# says it clipped, because a mechanism whose sensitivity claim is
# wrong is not a weaker guarantee, it is no guarantee.

# Laplace for counts. Pure epsilon-DP, no delta, and the tail is
# heavier than Gaussian at the same epsilon - which is the right
# trade for integer counts where a rare large error is survivable
# and a delta that has to be explained is not.
def laplace_count(value: float, sensitivity: float,
                  epsilon: float, rng=None) -> float:
    """One count, released under pure epsilon-DP."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if sensitivity <= 0:
        raise ValueError("sensitivity must be positive")
    rng = rng or np.random.default_rng()
    return float(value + rng.laplace(0.0,
                                     sensitivity / float(epsilon)))


def dp_histogram(counts: Dict[str, int], epsilon: float,
                 max_per_patient: int, rng=None
                 ) -> Tuple[Dict[str, float], Dict[str, object]]:
    """Noised patient counts for many items at once.

    `max_per_patient` is the CAP on how many of these items one
    patient can contribute to, and it is the L1 sensitivity. It
    is a parameter rather than a measurement because measuring
    it from the data would itself be a query on the data - the
    caller must impose it (by truncating each patient's token
    list) and pass what they imposed.
    """
    if max_per_patient < 1:
        raise ValueError("max_per_patient must be at least 1")
    rng = rng or np.random.default_rng()
    out = {}
    for k, v in counts.items():
        out[k] = laplace_count(float(v), float(max_per_patient),
                               epsilon, rng)
    return out, {
        "mechanism": "laplace histogram",
        "epsilon": float(epsilon),
        "l1_sensitivity": float(max_per_patient),
        "items": len(counts),
        "scale": float(max_per_patient) / float(epsilon),
        "note": "one patient can move at most {} of these counts "
                "by 1 each, so the L1 sensitivity is {} - the "
                "caller must have TRUNCATED each patient to that "
                "many items or the claim is void".format(
                    max_per_patient, max_per_patient)}


# The threshold has to RISE, and that is the whole correctness
# argument for selection.
#
# Releasing which items cleared a floor is not the same query as
# releasing their counts: an item held by exactly k patients
# appears or vanishes on one person's presence, so the SET itself
# leaks. Noising the counts and keeping the same floor does not
# fix it - an item at the floor still flips with probability near
# one half. The floor is raised by enough noise quantiles that a
# single patient cannot change the outcome except with
# probability under `tau_delta`.
def select_above(counts: Dict[str, int], floor: int,
                 epsilon: float, max_per_patient: int,
                 tau_delta: float = 1e-6, rng=None
                 ) -> Tuple[List[str], Dict[str, object]]:
    """Which items clear a floor, released privately."""
    noisy, rep = dp_histogram(counts, epsilon, max_per_patient,
                              rng)
    scale = rep["scale"]
    # THE TAIL IS ONE-SIDED, AND WHICH SIDE IS THE WHOLE POINT.
    #
    # For Laplace with scale b, P(X > t) = 0.5 exp(-t/b), so
    # t = b ln(0.5 / tau) is the tight bound. The failure that
    # matters is an item being INCLUDED when it should not be -
    # noise pushing a sub-floor count up - which is one-sided.
    # The other direction, an item dropped that should have been
    # kept, costs UTILITY and leaks nothing.
    #
    # The first version used ln(1/tau), which is conservative by
    # b ln 2 and had no reason stated. Conservative in the safe
    # direction is fine; conservative by accident, in a number
    # that sets a privacy threshold, is how a bound comes to be
    # trusted for the wrong reason. Caught by a check and the
    # implementation disagreeing about the arithmetic.
    bump = scale * math.log(0.5 / max(tau_delta, 1e-300))
    raised = float(floor) + bump + float(max_per_patient)
    kept = sorted(k for k, v in noisy.items() if v >= raised)
    rep = dict(rep)
    rep.update({
        "mechanism": "laplace histogram + raised threshold",
        "floor_requested": int(floor),
        "floor_used": raised,
        "tau_delta": float(tau_delta),
        "kept": len(kept), "offered": len(counts),
        "note": "the floor is RAISED from {} to {:.1f}. Keeping "
                "the requested floor would leak membership: an "
                "item held by exactly {} patients flips in or "
                "out on one person, so the SELECTED SET is "
                "itself a release and noising the counts alone "
                "does not fix it".format(
                    floor, raised, floor)})
    return kept, rep


# AND FOR A WIDE HISTOGRAM, GAUSSIAN BEATS LAPLACE BY sqrt(L).
#
# A patient holding L tokens moves the count vector by L in L1
# and only sqrt(L) in L2. Laplace's scale is set by L1, the
# Gaussian's by L2 - so at the vocabulary widths this project
# publishes (hundreds of tokens, a patient holding dozens) the
# Gaussian mechanism is not a refinement, it is the difference
# between a usable floor and an unusable one. It costs a delta,
# which the training accounting already carries.
def gaussian_scale(epsilon: float, delta: float,
                   max_per_patient: int) -> Tuple[float, float]:
    """The Gaussian noise scale for a count histogram.

    ONE function, because `gaussian_histogram` APPLIES this and
    `required_floor` REPORTS it, and computing it twice is how
    the Laplace floor came to be applied with one tail bound and
    reported with another. Returns (l2_sensitivity, sigma).

    A cap below one is refused here rather than at each call
    site: sqrt(0) is 0, which would make sigma ZERO and release
    the exact counts under a function named for a privacy
    mechanism. That is a silent total failure, not a weak
    guarantee.
    """
    if max_per_patient < 1:
        raise ValueError("max_per_patient must be at least 1")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1)")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    l2 = math.sqrt(float(max_per_patient))
    # the classical analytic Gaussian bound
    return l2, l2 * math.sqrt(2.0 * math.log(1.25 / delta)) \
        / float(epsilon)


def gaussian_histogram(counts: Dict[str, int], epsilon: float,
                       delta: float, max_per_patient: int,
                       rng=None
                       ) -> Tuple[Dict[str, float], Dict]:
    """Noised patient counts, Gaussian, L2 sensitivity."""
    l2, sigma = gaussian_scale(epsilon, delta, max_per_patient)
    rng = rng or np.random.default_rng()
    out = dict((k, float(v + rng.normal(0.0, sigma)))
               for k, v in counts.items())
    return out, {
        "mechanism": "gaussian histogram",
        "epsilon": float(epsilon), "delta": float(delta),
        "l2_sensitivity": l2, "scale": sigma,
        "items": len(counts),
        "note": "a patient holding {} items moves the count "
                "vector by {} in L1 and {:.2f} in L2, and the "
                "Gaussian scale is set by L2 - which is why "
                "this is the mechanism for a wide vocabulary "
                "and Laplace is the one for a single "
                "count".format(max_per_patient,
                               max_per_patient, l2)}


def required_floor(epsilon: float, max_per_patient: int,
                   tau_delta: float = 1e-6,
                   delta: Optional[float] = None,
                   floor: int = 10) -> Dict[str, float]:
    """How high the k floor must RISE to publish a selection.

    Returns both mechanisms so the choice is made on numbers.
    This is the quantity that decides whether DP vocabulary
    selection is usable at all: on the real extract a token held
    by fewer than the returned floor cannot be published, and
    the project's own k floor is 10.
    """
    lap_scale = float(max_per_patient) / float(epsilon)
    lap = float(floor) + lap_scale * math.log(
        0.5 / max(tau_delta, 1e-300)) + float(max_per_patient)
    out = {"floor": float(floor), "laplace": lap,
           "laplace_scale": lap_scale}
    if delta:
        l2, sig = gaussian_scale(epsilon, delta,
                                 max_per_patient)
        # a one-sided Gaussian tail at tau_delta
        z = math.sqrt(2.0) * _erfinv(1.0 - 2.0 * tau_delta)
        out["gaussian"] = float(floor) + z * sig \
            + float(max_per_patient)
        out["gaussian_scale"] = sig
        out["z"] = z
    return out


def _erfinv(y: float) -> float:
    """Inverse error function, bisection.

    Hand-rolled because this module deliberately depends on
    nothing but numpy - and a wrong value here would set a
    privacy threshold, so it is bisected to 1e-12 rather than
    approximated by a series whose error nobody checked.
    """
    lo, hi = -6.0, 6.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if math.erf(mid) < y:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def dp_quantile(values, q: float, lo: float, hi: float,
                epsilon: float, rng=None) -> Tuple[float, Dict]:
    """A quantile by the exponential mechanism.

    ONE VALUE PER PATIENT is assumed, and it is the caller's job
    to have aggregated to that - a patient contributing many
    rows moves the rank by many, and the sensitivity claim here
    is 1.

    Utility is -|rank(x) - target|, whose sensitivity is 1
    because adding or removing one value moves every rank by at
    most 1. The mechanism samples an INTERVAL between sorted
    values with probability proportional to its width times
    exp(epsilon * utility / 2), then samples uniformly inside -
    the standard construction, which is why the output can be a
    value nobody held. That property is the point: the published
    bound must not be an individual's value, which is a rule
    this project already enforces by averaging the k extremes.
    """
    v = np.asarray([x for x in values
                    if x is not None and np.isfinite(x)],
                   dtype=float)
    if v.size == 0:
        raise ValueError("no finite values")
    if not (hi > lo):
        raise ValueError("hi must exceed lo")
    rng = rng or np.random.default_rng()
    v = np.clip(v, lo, hi)
    v.sort()
    edges = np.concatenate(([lo], v, [hi]))
    widths = np.diff(edges)
    target = q * v.size
    # interval i sits between the i-th and (i+1)-th order
    # statistic, so every point inside it has rank i
    ranks = np.arange(v.size + 1, dtype=float)
    util = -np.abs(ranks - target)
    logw = np.log(np.maximum(widths, 1e-300)) \
        + 0.5 * float(epsilon) * util
    logw -= logw.max()
    w = np.exp(logw)
    w /= w.sum()
    i = int(rng.choice(len(w), p=w))
    out = float(rng.uniform(edges[i], edges[i + 1]))
    return out, {
        "mechanism": "exponential mechanism (quantile)",
        "epsilon": float(epsilon), "q": float(q),
        "rank_sensitivity": 1.0, "n": int(v.size),
        "range": [float(lo), float(hi)],
        "note": "one value per PATIENT is assumed; the output "
                "may be a value nobody held, which is the "
                "property that makes it publishable"}


def dp_bounds(values, epsilon: float, lo: float, hi: float,
              q_lo: float = 0.0, q_hi: float = 1.0, rng=None
              ) -> Tuple[float, float, Dict]:
    """A publishable minimum and maximum, split-budget.

    The budget SPLITS across the two queries rather than being
    spent twice, because sequential composition of two
    epsilon/2 mechanisms is epsilon - stating it here so nobody
    later passes the same epsilon to both and reports the sum as
    epsilon.
    """
    half = float(epsilon) / 2.0
    a, ra = dp_quantile(values, q_lo, lo, hi, half, rng)
    b, rb = dp_quantile(values, q_hi, lo, hi, half, rng)
    if a > b:
        a, b = b, a
    return a, b, {
        "mechanism": "two exponential mechanisms, split budget",
        "epsilon": float(epsilon), "each": half,
        "parts": [ra, rb],
        "note": "epsilon SPLITS across the two queries - "
                "sequential composition of two eps/2 mechanisms "
                "is eps, so passing the full epsilon to both "
                "and reporting eps would understate the cost by "
                "a factor of two"}


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
