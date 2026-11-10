"""The token solve is ITERATION-STARVED at a real vocabulary.

WHY THIS EXISTS. The real extract's remaining `close` misses are
dominated by cross-set token pairs, and reading the artifacts
acquitted both obvious suspects: the pair IS detected and IS
published - `sodium-chloride` against `Flush` carries phi 0.88483
with a conditional of 0.954, matching the source to three decimals.
So the contract holds the structure and the output does not, which
puts the fault in the DRAW.

MEASURED. On a feasible 1,827-token vocabulary with 400 published
conditional pairs, the solve lands 27 of 400 at the shipped 24
passes and 93 of 400 at 224 - monotone, with the token marginals
unharmed in every arm. The engine's own comment claimed the solve
"plateaus" by 30 passes; that was measured on ONE planted pair and
does not hold at 400.

AND THE FIRST FIXTURE FOR IT WAS INFEASIBLE IN AGGREGATE. It gave
1,827 tokens an average share of 0.15, so the shares summed to about
274 tokens per row against a set size of at most 12 - every token
ran cold and 1,638 of 1,827 marginals missed by more than 0.05 on
BOTH arms. A frame whose targets cannot all be met at once is not
measuring the solve, so the shares here are SCALED to sum to the
mean set size, which is what the extract's own vocabulary does.

WHAT IS NOT CLAIMED: that passes are the whole cause. Even 224 land
only 23% of the published conditionals, and the remaining candidate
is named in the engine - the set SIZE is drawn independently of
context.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import synthkit.generate as G                           # noqa: E402
from synthkit import sets as S                          # noqa: E402
from synthkit.generate import (TOKEN_SOLVE_MAX_PASSES,   # noqa: E402
                               TOKEN_SOLVE_PASSES,
                               _solve_passes, generate)

PASS = FAIL = 0
SIZE_V = [1, 3, 6, 12]
SIZE_P = [0.25, 0.35, 0.25, 0.15]
MEAN_SIZE = sum(v * p for v, p in zip(SIZE_V, SIZE_P))


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def vocab_bp(n_tok, n_pairs, frac=0.90, with_cooc=True):
    raw = np.array([1.0 / (1.0 + (i % 97)) ** 0.7
                    for i in range(n_tok)])
    p = np.clip(raw / raw.sum() * MEAN_SIZE, 1e-4, 0.95)
    toks = [{"value": "t%04d" % i, "p": float(p[i])}
            for i in range(n_tok)]
    cooc = []
    if with_cooc:
        for i in range(0, min(n_pairs * 2, n_tok - 1), 2):
            pa, pb = p[i], p[i + 1]
            feasible = min(1.0, pb / pa)
            cooc.append({
                "a": "s__has__t%04d" % i,
                "b": "s__has__t%04d" % (i + 1),
                "lift": float(min(9.0, feasible / max(pb, 1e-9))),
                "cond": float(round(frac * feasible, 5)),
                "base": float(pb), "phi": 0.6, "patients": 60})
    bp = {
        "columns": {"s": {
            "kind": "set", "level": "visit", "coverage": 1.0,
            "dials": {},
            "marginal": {"type": "list", "separator": ";",
                         "tokens": toks,
                         "set_size": {"v": SIZE_V, "p": SIZE_P}}}},
        "relationships": [],
        "patients": {"count": 900,
                     "visits": {"q": [0.0, 0.5, 1.0],
                                "v": [3.0, 4.0, 5.0], "mean": 4.0},
                     "dials": {}}}
    if with_cooc:
        bp["token_cooccurrence"] = cooc
    return bp, p


def score(g, bp):
    s = g["s"].fillna("")
    cache = {}

    def ind(tok):
        if tok not in cache:
            cache[tok] = np.nan_to_num(
                S.has_token(s, ";", tok).to_numpy(dtype=float)
            ).astype(bool)
        return cache[tok]

    ok = sc = 0
    for c in (bp.get("token_cooccurrence") or []):
        A = ind(c["a"].split(S.HAS, 1)[1])
        B = ind(c["b"].split(S.HAS, 1)[1])
        na = int(A.sum())
        if na == 0:
            continue
        sc += 1
        if abs(float((A & B).sum()) / float(na) - c["cond"]) <= 0.15:
            ok += 1
    worst, over = 0.0, 0
    for t in bp["columns"]["s"]["marginal"]["tokens"]:
        d = abs(float(ind(t["value"]).mean()) - float(t["p"]))
        worst = max(worst, d)
        if d > 0.05:
            over += 1
    return ok, sc, worst, over


def main():
    # ---- THE SCALING RULE, AS A PURE FUNCTION ------------------
    _default_rate = G.TOKEN_SOLVE_PER_PAIR
    check("a column with NO published conditionals earns exactly "
          "the base pass count, which is what makes this change "
          "inert where its mechanism is absent",
          _solve_passes(None, None) == TOKEN_SOLVE_PASSES
          and _solve_passes({}, {}) == TOKEN_SOLVE_PASSES)

    check("...and the per-pair rate is ZERO by default, so the "
          "engine is byte-identical to the build every existing run "
          "was made with - the evidence here is from FIXTURES, and "
          "the token-target tier was reversed by a real-extract "
          "control arm the same night",
          G.TOKEN_SOLVE_PER_PAIR == 0.0)

    # The scaling checks set the rate EXPLICITLY, because it is a
    # module constant read at draw time and the shipped default
    # turns the scaling off.
    G.TOKEN_SOLVE_PER_PAIR = 0.5
    few = _solve_passes(None, dict((i, 1) for i in range(10)))
    many = _solve_passes(dict((str(i), 1) for i in range(400)), None)
    check("...and the passes RISE with the number of published "
          "pairs ({} at ten, {} at four hundred) - this project's "
          "own rule from the sharpness work: passes proportional to "
          "the push".format(few, many),
          TOKEN_SOLVE_PASSES <= few < many)

    check("...and they are CAPPED, so a pathological vocabulary "
          "cannot price a run nobody asked for",
          _solve_passes(dict((str(i), 1) for i in range(100000)),
                        None) == TOKEN_SOLVE_MAX_PASSES)

    # ---- INERT WHERE THE MECHANISM IS ABSENT -------------------
    bp0, _ = vocab_bp(40, 0, with_cooc=False)
    h = []
    for per in (0.0, 0.5):
        G.TOKEN_SOLVE_PER_PAIR = per
        g = generate(bp0, n_patients=300, seed=9)
        h.append(hashlib.md5(
            g.to_csv(index=False).encode("utf-8")).hexdigest())
    G.TOKEN_SOLVE_PER_PAIR = 0.5
    check("a blueprint publishing NO conditionals generates "
          "BYTE-IDENTICALLY however the per-pair rate is set, so "
          "every run this project has already made is unaffected",
          h[0] == h[1])

    # ---- THE FIXTURE MUST BE FEASIBLE --------------------------
    bp1, p1 = vocab_bp(300, 80)
    check("the fixture's token shares SUM to the mean set size "
          "({:.2f} against {:.2f}), so every marginal can be met at "
          "once - the first version summed to about 274 tokens per "
          "row against a size of at most twelve, and measured "
          "nothing but its own infeasibility"
          .format(float(p1.sum()), MEAN_SIZE),
          abs(float(p1.sum()) - MEAN_SIZE) < 0.5)

    check("...and it publishes enough pairs to STARVE the base "
          "pass count - a fixture with ten pairs lands all of them "
          "at 24 passes and could not show this at all",
          len(bp1["token_cooccurrence"]) >= 60)

    # ---- THE MEASUREMENT --------------------------------------
    G.TOKEN_SOLVE_PER_PAIR = 0.0
    g_base = generate(bp1, n_patients=900, seed=3,
                      use_cooccurrence=True)
    ok_b, sc_b, w_b, ov_b = score(g_base, bp1)
    G.TOKEN_SOLVE_PER_PAIR = 0.5
    g_more = generate(bp1, n_patients=900, seed=3,
                      use_cooccurrence=True)
    ok_m, sc_m, w_m, ov_m = score(g_more, bp1)

    check("scaled passes LAND MORE published conditionals at a real "
          "vocabulary: {} of {} at the base against {} of {} scaled"
          .format(ok_b, sc_b, ok_m, sc_m),
          sc_b > 0 and sc_m > 0 and ok_m > ok_b + 5)

    # THE NEIGHBOURING PROPERTY, which this project requires in the
    # SAME test: the sharpness work traded the marginals away for
    # exactly this kind of gain, and only a check on both caught it.
    check("...and the token MARGINALS are no worse for it (worst "
          "miss {:.4f} against {:.4f}; {} against {} over the 0.05 "
          "bar) - the sharpness work bought a co-occurrence gain by "
          "trading the shares away, and only measuring both saw it"
          .format(w_b, w_m, ov_b, ov_m),
          ov_m <= ov_b and w_m <= w_b + 0.02)

    check("...and the base arm is genuinely STARVED rather than "
          "merely imperfect - it lands under three quarters of what "
          "the scaled arm does, which is what makes this a "
          "measurement of the iteration count",
          ok_b < 0.75 * ok_m)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
