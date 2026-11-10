"""A near-deterministic implication has to be FORCED, not nudged.

WHY THIS EXISTS. `cooc_landing` on the real extract's shipping run
reads 4,471 published conditional pairs with 2,729 landing within
0.15 - so the solve is not broadly broken - and the WORST misses read
**published 1.0, achieved 0.0**: three ophthalmic drugs against one
procedure code, a perfect implication landing at nothing. A source
rank correlation of 0.885 IS a near-perfect implication, so the band
that a multiplicative weight cannot reach is exactly the band `close`
is made of. The engine's own detector comment says so in as many
words. The neural engine has had a forcing tier since 2026-09-30;
the rules engine never did.

WHAT IS LOAD-BEARING HERE IS THE BUDGET. Forcing every pair above the
bar lands 69 of 69 conditionals and pushes SEVENTEEN token shares
past the 0.05 bar, worst by 0.31 - measured, not imagined. The cause
is the UNION: a token implied by many others is forced wherever ANY
of them lands, and that union can exceed the token's own published
share even when each implier fits inside it. No weight can take it
back, because the solve can drive the UNFORCED picks to zero and the
forced ones remain. So forcing has a per-token budget of its own
published share, and the conditional lands on as many rows as the
marginal can afford and no more.

Three seeds with the budget: 4 -> 20, 9 -> 19, 13 -> 28 landed, with
ZERO shares over the bar in every arm.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import synthkit.generate as G                           # noqa: E402
from synthkit import sets as S                          # noqa: E402
from synthkit.generate import (COOC_FORCE_MAX_SHARE,     # noqa: E402
                               COOC_FORCE_MIN,
                               _force_budget, generate)

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


def make_bp(n_tok=300, n_pairs=80, broad=False, with_cooc=True):
    raw = np.array([1.0 / (1.0 + (i % 97)) ** 0.7
                    for i in range(n_tok)])
    p = np.clip(raw / raw.sum() * MEAN_SIZE, 1e-4, 0.95)
    if broad:
        # A BROAD implied token - the `Oral` shape. Forcing it
        # wherever any implier lands replaces its own placement with
        # the union of theirs, and this project measured that
        # flipping a guarded negative correlation to positive.
        #
        # AND THE REST ARE RESCALED SO THE TOTAL STILL EQUALS THE
        # MEAN SET SIZE. Setting one share to 0.80 and leaving the
        # others pushes the sum past what any row can hold, and then
        # the CONTROL arm misses the broad token too - measured at
        # 0.6886 against a published 0.80, which I first read as the
        # forcing dragging it. Third infeasible fixture of the
        # night: a frame whose targets cannot all be met at once
        # measures nothing but its own infeasibility.
        p[1] = 0.80
        rest = np.array([i for i in range(n_tok) if i != 1])
        room = max(MEAN_SIZE - 0.80, 1e-6)
        p[rest] = p[rest] / p[rest].sum() * room
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
                "cond": float(round(feasible, 5)),
                "base": float(pb), "phi": 0.6, "patients": 60})
        cooc = [c for c in cooc if c["cond"] >= 0.90]
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


def score(g, bp, tol=0.15):
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
        if abs(float((A & B).sum()) / float(na) - c["cond"]) <= tol:
            ok += 1
    worst, over = 0.0, 0
    for t in bp["columns"]["s"]["marginal"]["tokens"]:
        d = abs(float(ind(t["value"]).mean()) - float(t["p"]))
        worst = max(worst, d)
        if d > 0.05:
            over += 1
    sizes = s.apply(lambda x: 0 if not x else len(x.split(";")))
    return ok, sc, worst, over, float(sizes.mean())


def arm(force, seed, **kw):
    G.COOC_FORCE_MIN = force
    bp, p = make_bp(**kw)
    g = generate(bp, n_patients=900, seed=seed,
                 use_cooccurrence=True)
    return score(g, bp), bp, p, g


def main():
    try:
        # ---- OFF BY DEFAULT ---------------------------------
        check("the forcing tier is OFF by default - the evidence "
              "below is from FIXTURES, and this project reversed a "
              "fixture-measured token tier on the real extract the "
              "same night", COOC_FORCE_MIN is None)

        # ---- INERT WHERE IT IS OFF --------------------------
        bp0, _ = make_bp(40, 0, with_cooc=False)
        hs = []
        for fm in (None, 0.90):
            G.COOC_FORCE_MIN = fm
            g = generate(bp0, n_patients=300, seed=9)
            hs.append(hashlib.md5(
                g.to_csv(index=False).encode("utf-8")).hexdigest())
        check("a blueprint publishing NO conditionals generates "
              "BYTE-IDENTICALLY with the tier on or off, so it "
              "cannot act where its mechanism is absent",
              hs[0] == hs[1])

        # ---- IT LANDS MORE, ON EVERY SEED -------------------
        rows = []
        for sd in (3, 4, 5):
            o = arm(None, sd)[0]
            n = arm(0.90, sd)[0]
            rows.append((o, n))
        check("forcing lands MORE near-deterministic conditionals on "
              "EVERY seed ({} -> {}) - a range, never a point"
              .format(" ".join(str(o[0]) for o, _ in rows),
                      " ".join(str(n[0]) for _, n in rows)),
              all(n[0] > o[0] for o, n in rows))

        # ---- AND THE MARGINALS SURVIVE ----------------------
        # THE NEIGHBOURING PROPERTY, in the same test. Without the
        # budget this is exactly what breaks: 17 shares over the bar,
        # worst by 0.31.
        check("...and NO token share passes the 0.05 bar in either "
              "arm on any seed ({} against {}), because forcing is "
              "capped at each token's own published share"
              .format(" ".join(str(o[3]) for o, _ in rows),
                      " ".join(str(n[3]) for _, n in rows)),
              all(n[3] == 0 and o[3] == 0 for o, n in rows))

        check("...and the worst single share miss stays small in the "
              "forced arm ({}), so the gain is not paid for out of "
              "the published marginals"
              .format(" ".join("%.4f" % n[2] for _, n in rows)),
              all(n[2] < 0.05 for _, n in rows))

        # ---- THE SIZE GRID STILL DECIDES THE SIZE -----------
        check("...and the mean set size is unchanged ({:.3f} against "
              "{:.3f}), because a forced token DISPLACES the weakest "
              "unforced pick rather than extending the row - the "
              "published size grid still decides the length"
              .format(rows[0][0][4], rows[0][1][4]),
              abs(rows[0][0][4] - rows[0][1][4]) < 0.15)

        # ---- THE BUDGET IS A REAL BOUND ---------------------
        b = _force_budget(np.array([0.25, 0.5, 1e-6]), 1000)
        check("the budget is each token's published share in ROWS "
              "({}), which is the bound no weight can take back - "
              "the solve can drive a token's UNFORCED picks to zero "
              "and the forced ones remain"
              .format(list(int(x) for x in b)),
              list(int(x) for x in b) == [250, 500, 0])

        check("...and it refuses to budget anything for a token the "
              "contract barely publishes, so a near-absent token "
              "cannot be forced onto rows at all", int(b[2]) == 0)

        # ---- THE BROAD-TOKEN CAP ----------------------------
        G.COOC_FORCE_MIN = 0.90
        (_, _, w_b, ov_b, _), bpb, pb, gb = arm(
            0.90, 3, broad=True)
        s = gb["s"].fillna("")
        got_broad = float(np.nan_to_num(
            S.has_token(s, ";", "t0001").to_numpy(dtype=float)).mean())
        check("a BROAD implied token is not forced - its published "
              "share is {:.2f}, over the {:.2f} cap, and it keeps "
              "its own placement at {:.3f}, inside this project's "
              "own 0.05 share bar, rather than being dragged to the "
              "union of its impliers' rows. Forcing a 0.86-share "
              "token was measured here flipping a guarded NEGATIVE "
              "correlation to positive - an inversion manufactured "
              "by a fix for a close verdict"
              .format(float(pb[1]), COOC_FORCE_MAX_SHARE, got_broad),
              float(pb[1]) > COOC_FORCE_MAX_SHARE
              and abs(got_broad - float(pb[1])) < 0.05)

        check("...and the fixture that asks it is FEASIBLE - the "
              "shares sum to the mean set size ({:.2f} against "
              "{:.2f}) even after one is set to 0.80. The first "
              "version left the rest unscaled, so the CONTROL arm "
              "missed the broad token too at 0.6886, and I read "
              "that as the forcing dragging it"
              .format(float(pb.sum()), MEAN_SIZE),
              abs(float(pb.sum()) - MEAN_SIZE) < 0.1)

        # ---- ONE LEVEL, NOT A CASCADE -----------------------
        check("the tier forces ONE level only, so a chain of "
              "near-deterministic pairs cannot fill a row from a "
              "single pick and let the implication graph decide the "
              "size instead of the published grid",
              G.COOC_FORCE_DEPTH == 1)
    finally:
        G.COOC_FORCE_MIN = COOC_FORCE_MIN

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
