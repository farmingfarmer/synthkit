"""The key/released pair, verified - and the vendor's answer read
back against the key.

THE ARCHITECTURE THIS SERVES. One run produces two files and the
arrangement between them:

    key.csv        the full-fidelity k-anonymous synthetic. We
                   keep it. It is what our own questions are
                   asked against.
    generated.csv  the released file. Named relationships have
                   been re-pointed or removed by the dials, so a
                   model trained on it learns the wrong thing
                   about those relationships.
    cipher.json    which key row each released row took each
                   perturbed value from.

Every dial is a PERMUTATION of one column's values across rows.
That is why the pair works at all: each column's own
distribution is byte-identical in both files, so the released
file is not damaged data - it is the same data, rearranged - and
the rearrangement is exactly invertible.

WHAT THE CIPHER IS FOR, AND WHAT IT IS NOT FOR. It is the AUDIT
record: it proves what we changed, proves we changed nothing
else, and lets us restate any statistic of the released file as
the statistic of the key file. It is NOT a way to un-distort a
vendor's row-level predictions, and this file refuses to pretend
otherwise:

    a vendor's score for released row p is f(X_p, C_s) where C_s
    is some OTHER row's value of the perturbed column. We want
    f(X_p, C_p). No arrangement of the inputs produces that
    number, because f was never evaluated there. Only f can
    answer it.

So there are three routes to the value, and they are not equally
sound. `rehearse` measures which one your dial setting leaves
you, on your own data, BEFORE anything is sent:

  A  THE VENDOR RETURNS A MODEL, or lets us call it. Run it on
     key.csv in-house. Recovery is EXACT and the cipher is not
     needed for it at all. This is the route to ask for in the
     contract, and the one the dials were really built to make
     safe: they can have the released file to explore and train
     on, and we score the key file ourselves.
  B  THE VENDOR RETURNS ONLY PER-ROW SCORES. Approximate. A
     surrogate is fitted to (released rows -> their scores) and
     evaluated on the key rows. The error is real and is
     MEASURED here rather than assumed.
  C  THE VENDOR RETURNS AN AGGREGATE - a coefficient, an effect
     size, a ranking of drivers. The cipher gives the exact
     before/after of every statistic we moved, so a conclusion
     about a re-pointed pair can be restated. A conclusion about
     a pair we did not touch needs no correction.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fidelity_deck import read_frame          # noqa: E402

MIN_ROWS = 200


def load_pair(run):
    """key frame, released frame, cipher - or a refusal."""
    run = Path(run)
    for need in ("key.csv", "generated.csv", "cipher.json"):
        if not (run / need).exists():
            return None, None, None, (
                "no {} in {} - a key/released pair is written "
                "only by a run that actually set a dial "
                "(--poison / --suppress / --suppress-top). A "
                "full-fidelity run has one file and needs "
                "no cipher.".format(need, run))
    k = read_frame(run / "key.csv")
    g = read_frame(run / "generated.csv")
    c = json.loads((run / "cipher.json").read_text(
        encoding="utf-8"))
    return k, g, c, None


def verify(run):
    """Is the released file an exact permutation of the key?

    This is the claim that makes a re-pointed file a legitimate
    test article rather than corrupted data, so it is asserted
    rather than described: every perturbed column round-trips
    through the cipher EXACTLY, holds the identical multiset of
    values, and every column we did not touch is untouched.
    """
    k, g, c, err = load_pair(run)
    if err:
        print("STOP: " + err)
        return 1
    bad = []
    print("key/released pair in {}".format(run))
    print("  rows        key {}  released {}".format(len(k),
                                                     len(g)))
    if len(k) != len(g):
        bad.append("row counts differ - the pair is not aligned")
    if list(k.columns) != list(g.columns):
        bad.append("column lists differ")
    cols = c.get("columns") or {}
    print("  perturbed   {}".format(", ".join(cols) or "(none)"))
    for col, tf in cols.items():
        tf = np.asarray(tf, dtype=np.int64)
        rec = np.empty(len(g), dtype=object)
        rec[tf] = g[col].to_numpy()
        exact = np.array_equal(np.asarray(rec, dtype=str),
                               k[col].to_numpy().astype(str))
        a = np.sort(pd.to_numeric(k[col],
                                  errors="coerce").dropna())
        b = np.sort(pd.to_numeric(g[col],
                                  errors="coerce").dropna())
        same = (len(a) == len(b)
                and np.allclose(a, b, equal_nan=True))
        print("    {:<22} inverse exact {}   same multiset "
              "{}".format(col, "YES" if exact else "NO",
                          "YES" if same else "NO"))
        if not exact:
            bad.append("{}: the cipher does not invert".format(
                col))
        if not same:
            bad.append("{}: the distribution MOVED - a dial "
                       "must only rearrange".format(col))
    untouched = [x for x in g.columns if x not in cols]
    moved = [x for x in untouched
             if not np.array_equal(g[x].to_numpy().astype(str),
                                   k[x].to_numpy().astype(str))]
    print("  untouched   {} column(s), {} changed".format(
        len(untouched), len(moved)))
    if moved:
        bad.append("columns changed with no cipher entry: "
                   + ", ".join(moved[:6]))
    for r in (c.get("repointed") or []):
        _rch = r.get("reachable")
        _ask = float(r.get("requested", 0))
        _got = float(r.get("achieved", 0))
        print("    re-pointed {:<28} asked {:+.2f}  landed "
              "{:+.2f}".format(r.get("pair", "?"), _ask, _got))
        if _rch:
            # A GAP BETWEEN ASKED AND LANDED IS NOT AUTOMATICALLY
            # A MISS. Rank correlation is bounded by the tie
            # structure, and a reader who sees -0.60 asked and
            # -0.29 landed will hunt a sampler bug unless the
            # ceiling is printed beside it.
            _lo, _hi = float(_rch[0]), float(_rch[1])
            _bound = (_ask < _lo - 1e-9) or (_ask > _hi + 1e-9)
            print("      reachable band for this pair "
                  "[{:+.3f}, {:+.3f}]{}".format(
                      _lo, _hi,
                      "  <- the ask was outside it; the dial "
                      "went as far as the pair allows"
                      if _bound else ""))
    for r in (c.get("suppressed") or []):
        print("    suppressed {}".format(r))
    print("")
    if bad:
        print("NOT VERIFIED:")
        for b in bad:
            print("  - " + b)
        return 1
    print("VERIFIED. The released file carries every column's "
          "own distribution unchanged and differs from the key "
          "only by the recorded arrangement. That is the claim "
          "to make about it: a faithful rearrangement, not "
          "damaged data.")
    return 0


def _model():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        max_iter=120, random_state=0)


def _design(frame, target, drop):
    """Numeric design matrix - categoricals as codes."""
    use = [c for c in frame.columns
           if c != target and c not in drop]
    out = {}
    for c in use:
        v = pd.to_numeric(frame[c], errors="coerce")
        if v.notna().mean() < 0.5:
            v = pd.Series(
                pd.Categorical(frame[c].astype(str)).codes,
                index=frame.index).astype(float)
        out[c] = v
    return pd.DataFrame(out, index=frame.index), use


def rehearse(run, target, group_by=None, seed=0):
    """Which recovery route does this dial setting leave us?

    Measured in-house, before anything is sent, with a stand-in
    for the vendor's model: a model is trained on the KEY file,
    which is the thing a vendor would have built if we had given
    them our best data. Its scores on the key rows are TRUTH. Its
    scores on the RELEASED rows are what comes back to us.

    Two numbers come out of that, and they are the two axes the
    choice of dial setting actually trades between:

      PROTECTION      how much worse a model TRAINED on the
                      released file is at the real relationship.
                      High is the point of the exercise.
      RECOVERABILITY  how well route B puts a returned score
                      back onto the key rows. Low means ask for
                      route A in the contract.

    A dial setting that protects nothing is theatre; one that
    protects and is unrecoverable is fine PROVIDED the contract
    gets you the model. Nothing here chooses for you - the
    choice is a governance decision and it should be made
    against these two numbers rather than a feeling.
    """
    k, g, c, err = load_pair(run)
    if err:
        print("STOP: " + err)
        return 1
    if target not in k.columns:
        print("STOP: {} is not a column. Columns: {}".format(
            target, ", ".join(list(k.columns)[:14])))
        return 1
    drop = {group_by} if group_by else set()
    yk = pd.to_numeric(k[target], errors="coerce")
    live = yk.notna()
    if int(live.sum()) < MIN_ROWS:
        print("STOP: {} has {} present rows, under the {} a "
              "model needs. A share is not the test - rows "
              "are.".format(target, int(live.sum()), MIN_ROWS))
        return 1
    Xk, use = _design(k, target, drop)
    Xg, _ = _design(g, target, drop)
    rng = np.random.RandomState(seed)
    idx = np.flatnonzero(live.to_numpy())
    rng.shuffle(idx)
    cut = int(len(idx) * 0.7)
    tr, te = idx[:cut], idx[cut:]
    touched = [x for x in (c.get("columns") or {}) if x in use]
    target_dialed = target in (c.get("columns") or {})

    print("rehearsing {} on {}".format(target, run))
    print("  {} predictors, {} of them perturbed: {}".format(
        len(use), len(touched), ", ".join(touched) or "(none)"))
    print("  the target itself: {}".format(
        "PERTURBED" if target_dialed else "untouched"))
    yg = pd.to_numeric(g[target], errors="coerce")

    # PROTECTION is always measured, never asserted. Both models
    # are judged against the KEY file's own truth on rows neither
    # of them trained on: one learned from the key, one from the
    # released file. The gap is what the dials took away from a
    # vendor who trains on what we sent.
    def _r2(m, X):
        pr = m.predict(X.iloc[te])
        t = yk.iloc[te].to_numpy()
        ss = float(np.sum((t - t.mean()) ** 2)) or 1.0
        return 1.0 - float(np.sum((t - pr) ** 2)) / ss

    r_key = _r2(_model().fit(Xk.iloc[tr], yk.iloc[tr]), Xk)
    r_rel = _r2(_model().fit(Xg.iloc[tr], yg.iloc[tr]), Xg)
    print("")
    print("  PROTECTION - skill on OUR truth, on held-out rows")
    print("    a model trained on the key file       "
          "{:+.3f}".format(r_key))
    print("    a model trained on the released file  "
          "{:+.3f}".format(r_rel))
    print("    denied to a vendor who trains on it   "
          "{:+.3f}".format(r_key - r_rel))
    # THE WEAK CASE IS MEASURED, NOT INFERRED FROM THE PLAN. In
    # a flat table every column is a predictor of every other,
    # so "no perturbed predictor" almost never happens and a
    # structural branch testing for it would be dead code. What
    # does happen is a dial that moves a column this target
    # barely reads - visible only in the two numbers above.
    if r_key < 0.05:
        print("")
        print("  NOBODY CAN PREDICT THIS TARGET from the other "
              "columns ({:+.3f} on the key file itself), so "
              "there is nothing here to protect and nothing to "
              "recover. Pick a target a vendor would actually "
              "be asked to model.".format(r_key))
        return 0
    if (r_key - r_rel) < 0.02:
        print("")
        print("  THIS DIAL SETTING PROTECTS NOTHING ABOUT THIS "
              "TARGET: a vendor who trains on the released file "
              "keeps essentially all the skill ({:+.3f} of it "
              "denied). A model routes around a perturbed "
              "column through the others, so dialing a "
              "COVARIATE often buys very little - dial the "
              "OUTCOME and rehearse again.".format(
                  r_key - r_rel))

    print("")
    print("  RECOVERABILITY - getting a returned answer back "
          "onto the key rows")
    if not touched:
        # The dial is on the LABEL and the predictors are
        # faithful. A pre-trained vendor model reads only
        # predictors, so its answer about released row p is
        # already its answer about key row p - exact, and worth
        # distinguishing sharply from the case below, because
        # the observable ("no predictor perturbed") is the same
        # and the conclusion is the opposite.
        print("    EXACT, and no surrogate is involved. Every "
              "predictor this target reads is untouched, so a "
              "vendor model that SCORES our rows returns the "
              "key answer already.")
        print("")
        print("  THIS IS THE ARRANGEMENT TO PREFER: dial the "
              "OUTCOME, leave the covariates faithful. It "
              "denies the training value - measured above - and "
              "keeps the scoring value exactly. Dialing a "
              "covariate instead corrupts the input their model "
              "reads, which costs you value on the way in; "
              "rehearse that arrangement and compare the two "
              "numbers before choosing.")
        return 0

    # the stand-in vendor model: built on our best data, which is
    # what they would have built had we shared it
    vend = _model().fit(Xk.iloc[tr], yk.iloc[tr])
    truth = vend.predict(Xk.iloc[te])          # f on key rows
    returned = vend.predict(Xg.iloc[te])       # f on released
    sd = float(np.std(truth)) or 1.0
    naive = float(np.mean(np.abs(returned - truth))) / sd
    sur = _model().fit(Xg.iloc[tr], vend.predict(Xg.iloc[tr]))
    rec = float(np.mean(np.abs(sur.predict(Xk.iloc[te])
                               - truth))) / sd
    print("    taking the returned score untouched  "
          "{:.3f}   <- the error if you ignore the "
          "distortion".format(naive))
    print("    route B, surrogate                   "
          "{:.3f}".format(rec))
    print("    route A, their model run on the key  0.000   "
          "exact by construction")
    print("    (in sds of the score; lower is better)")
    print("")
    if naive < 0.05:
        # THE SAME OBSERVABLE, TWO OPPOSITE CONCLUSIONS AGAIN.
        # On the real extract a dialed span_days left
        # condition_count's returned score 0.007 sd from truth -
        # the model barely reads the column we moved - and the
        # tool said "ROUTE B DOES NOT WORK", which reads as the
        # distortion being too strong when it was too weak to
        # matter. Note the two numbers agree: a dial that denies
        # nothing (above) is the same fact as a return that
        # needs no recovery, and agreement between them is the
        # signal, not a coincidence.
        print("  NOTHING TO RECOVER HERE: the returned score is "
              "already {:.3f} sd from truth, because this "
              "target barely reads the column the dial moved. "
              "Take the return as-is - and note this is the "
              "SAME fact as the protection number above, which "
              "is why they agree.".format(naive))
    elif rec < naive * 0.5:
        print("  Route B recovers most of it here - {:.3f} sd "
              "closer to truth than the raw return.".format(
                  naive - rec))
    elif rec < naive:
        print("  Route B helps and does not rescue this: "
              "{:.3f} sd closer, still {:.3f} sd out. Get the "
              "model.".format(naive - rec, rec))
    else:
        print("  ROUTE B DOES NOT WORK AT THIS SETTING - the "
              "surrogate is no better than the raw return. A "
              "perturbed PREDICTOR corrupts the input their "
              "model reads, and no arrangement of inputs "
              "recovers a number their model never computed. "
              "Either get the model, or move the dial onto the "
              "OUTCOME instead.")
    return 0


def recover(run, scores_csv, score_col=None, out=None):
    """Apply route B to a real returned file, and say what it is.

    The returned scores must be in the row order of the released
    file - one per released row - because that is the only thing
    that relates them to anything.
    """
    k, g, c, err = load_pair(run)
    if err:
        print("STOP: " + err)
        return 1
    sc = read_frame(scores_csv)
    if len(sc) != len(g):
        print("STOP: {} has {} rows, the released file has {}. "
              "The returned scores must be one per released "
              "row, in that order.".format(scores_csv, len(sc),
                                           len(g)))
        return 1
    if score_col is None:
        num = [x for x in sc.columns
               if pd.to_numeric(sc[x],
                                errors="coerce").notna().mean()
               > 0.9]
        if len(num) != 1:
            print("STOP: say which column holds the score with "
                  "--score-col. Numeric columns: {}".format(
                      ", ".join(num) or "(none)"))
            return 1
        score_col = num[0]
    y = pd.to_numeric(sc[score_col], errors="coerce")
    Xg, use = _design(g, score_col, set())
    Xk, _ = _design(k, score_col, set())
    live = y.notna()
    if int(live.sum()) < MIN_ROWS:
        print("STOP: {} present scores, under {}.".format(
            int(live.sum()), MIN_ROWS))
        return 1
    sur = _model().fit(Xg[live.to_numpy()], y[live])
    est = sur.predict(Xk)
    o = Path(out or (Path(run) / "recovered.csv"))
    pd.DataFrame({"recovered_" + score_col: est}).to_csv(
        o, index=False)
    print("wrote {} - one estimate per KEY row, in key.csv's "
          "order.".format(o))
    print("")
    print("WHAT THIS IS: a surrogate fitted to the returned "
          "scores and read on the key rows. It is route B, so it "
          "is an ESTIMATE of what the vendor's model would have "
          "said about the key rows, not that number. Run "
          "`dialback.py rehearse` on this run to see the "
          "measured error of this route at this dial setting "
          "before anyone quotes it.")
    print("WHAT IT IS NOT: a reconstruction of real patients. "
          "Both files are synthetic; the key file is the "
          "k-anonymous reference, not the extract.")
    return 0


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        print("usage:")
        print("  dialback.py verify RUNDIR")
        print("  dialback.py rehearse RUNDIR --target COL "
              "[--group-by COL]")
        print("  dialback.py recover RUNDIR --scores FILE.csv "
              "[--score-col NAME] [--out FILE.csv]")
        return 2
    cmd, run = argv[0], argv[1]
    kw = {}
    i = 2
    while i < len(argv) - 1:
        if argv[i].startswith("--"):
            kw[argv[i][2:].replace("-", "_")] = argv[i + 1]
            i += 2
        else:
            i += 1
    if cmd == "verify":
        return verify(run)
    if cmd == "rehearse":
        if "target" not in kw:
            print("STOP: rehearse needs --target COL")
            return 2
        return rehearse(run, kw["target"],
                        group_by=kw.get("group_by"),
                        seed=int(kw.get("seed", 0)))
    if cmd == "recover":
        if "scores" not in kw:
            print("STOP: recover needs --scores FILE.csv")
            return 2
        return recover(run, kw["scores"],
                       score_col=kw.get("score_col"),
                       out=kw.get("out"))
    print("STOP: unknown command {} - verify / rehearse / "
          "recover".format(cmd))
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
