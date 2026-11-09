"""What kinds of pattern can the fitted path FIND, and which survive
GENERATION? Two different failures, scored separately.

    python scripts/bench_new_path.py [--patients 800] [--harder 1.0]
                                     [--seeds 1] [-o bench.json]

WHY THIS EXISTS. The hard part was already built and never pointed at
this path.

`make_tidy_fixture.py` plants fourteen relationships covering the
kinds that actually break a pattern finder - a U-shape with zero
linear correlation, an XOR with no main effect at all, a Simpson's
reversal where the pooled sign is the OPPOSITE of the truth, a lagged
cross-column pair, a three-way, a saturating knee, an effect present
in only a fifth of patients, an exact identity - plus noise columns
that must never be found. `score_discovery.py` scores recall BY KIND
against that truth.

Neither could see the new path. `score_discovery` read condnet models
and confirmation reports; `recall_sweep` drives condnet;
`pair_fidelity_sweep` uses the fixture but measures pair survival, not
which KINDS were found. So the question "why does it struggle with
complex patterns in real data" had no measured answer - not because
the benchmark was missing, but because nothing connected it.

TWO SCORES, because they fail independently:

  FOUND      discovery names the relationship at all. A miss here is
             a search that cannot represent the pattern
  SURVIVED   the pattern is still measurable in the generated data.
             A miss here is a search that found it and a sampler that
             could not carry it - which is the failure a per-column
             fidelity report cannot see

A kind can be found and not survive. That gap is the honest subject
of this file, and reporting one number for both would hide it.

WHAT SURVIVAL MEANS PER KIND, since one test does not fit them:
a monotone kind is checked by rank correlation; a U-shape and an XOR
by whether the child's mean differs across the parent's extremes in
the way planted; an identity by whether it still holds row-wise.
Every check is stated beside its number.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit import blueprint as B                    # noqa: E402
from synthkit.discover import discover                 # noqa: E402
from synthkit.generate import generate                 # noqa: E402


def num(frame, col):
    return pd.to_numeric(frame.get(col), errors="coerce")


def spearman(a, b):
    ok = a.notna() & b.notna()
    if int(ok.sum()) < 50:
        return float("nan")
    return float(a[ok].rank().corr(b[ok].rank()))


def curve_gap(frame, x, y):
    """Mean of y at the EXTREMES of x minus its mean in the middle.

    A U-shape and a threshold both have near-zero rank correlation, so
    a monotone test scores them as absent whether or not they are
    there. This one sees a bend."""
    xs, ys = num(frame, x), num(frame, y)
    ok = xs.notna() & ys.notna()
    if int(ok.sum()) < 100:
        return float("nan")
    xs, ys = xs[ok], ys[ok]
    q = np.quantile(xs, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    lo = ys[(xs >= q[0]) & (xs <= q[1])].mean()
    mid = ys[(xs > q[2]) & (xs <= q[3])].mean()
    hi = ys[(xs > q[4]) & (xs <= q[5])].mean()
    sd = float(ys.std()) or 1.0
    return float(((lo + hi) / 2.0 - mid) / sd)


def visit_order(frame):
    """The column THIS file declares its visit sequence by.

    THE TWO TABLES DECLARE IT DIFFERENTLY AND BOTH MUST BE READ
    AS THEY ARE WRITTEN. A source extract has no visit index, so
    its dates are the only sequence it carries. The generated file
    emits `visit_number` as 1..n in row order, which ASSERTS a
    sequence - and draws its date column from a marginal, which
    does not: measured on this fixture, only 0.8% of generated
    patients have their dates in row order while `visit_number` is
    in row order on 100% of them. So sorting the output by its
    date reads a lagged relationship at +0.31 where reading it by
    the column the file itself numbers reads +0.86.

    Which is a DEFECT IN THE OUTPUT and not a measurement choice -
    two columns in one file disagreeing about the order of a
    patient's visits is this project's own alarm - and it is a
    separate one from the lag: it is there with or without
    `--lags`, and a consumer sorting by date gets a different
    sequence from one sorting by the index. Named here, measured,
    and read the way each file writes it rather than papered over
    by picking whichever ordering flatters the number.
    """
    if "visit_number" in frame.columns:
        return "visit_number"
    return "visit_start_date"


def lag_corr(frame, parent, child, group="person_id",
             order=None):
    """`child_t` against `parent_(t-1)`, with the lag RECOMPUTED.

    THE EMITTED `x__prev` COLUMN MUST NOT BE TRUSTED HERE. With
    `--lags` the search sees `x__prev` and the blueprint then
    models it as an ordinary numeric, so generation DRAWS it
    rather than carrying the previous row's `x`. A test that read
    `y ~ x__prev` off the output would therefore pass on a file
    whose lag column contradicts its own `x` on the previous
    visit - the broken-identity class this project already
    records for `age_at_visit` against `year_of_birth`, where the
    pair correlation stayed strong while the identity held on 21%
    of rows.

    So the lag is rebuilt from the frame's OWN ordering, on both
    tables, which is the only form of the question that asks
    whether the TEMPORAL structure survived rather than whether a
    column with a suggestive name did.
    """
    order = order or visit_order(frame)
    if group not in frame.columns or order not in frame.columns:
        return float("nan")
    f = frame[[group, order, parent, child]].copy()
    f[parent] = pd.to_numeric(f[parent], errors="coerce")
    f[child] = pd.to_numeric(f[child], errors="coerce")
    _ord = pd.to_numeric(f[order], errors="coerce")
    if _ord.notna().all():
        f["_ord"] = _ord
    else:
        f["_ord"] = pd.to_datetime(f[order], errors="coerce")
    f = f.sort_values([group, "_ord"], kind="stable")
    f["_prev"] = f.groupby(group, sort=False)[parent].shift(1)
    return spearman(f["_prev"], f[child])


def survives(kind, truth, src, gen):
    """Is the planted pattern still measurable after generation?

    Returns (verdict, source_value, generated_value, how). `None`
    means this kind has no honest one-line test here and is reported
    as unscored rather than guessed at."""
    child = truth["child"]
    parents = truth["parents"]
    if child not in gen.columns:
        return (False, None, None, "the child column is not in the "
                                   "generated table")
    p0 = parents[0]
    if p0 not in gen.columns:
        return (False, None, None, "the parent column is not in the "
                                   "generated table")

    if kind == "lagged-cross-column":
        # THE SAME-VISIT PAIR CARRIES NO SIGNAL BY DESIGN, which
        # is what made this row unscored: the fixture plants
        # `y_t ~ x_(t-1)`, so `spearman(x_t, y_t)` reads 0.003 in
        # the SOURCE and the monotone branch correctly declined to
        # score it. The lag has to be rebuilt to ask the question
        # at all.
        s_ = lag_corr(src, p0, child)
        g_ = lag_corr(gen, p0, child)
        # AND THE SAME-VISIT PAIR IS NOW A FINDING RATHER THAN A
        # BLANK. With `--lags` the generated file came back
        # carrying a same-visit correlation of 0.97 between a
        # pair the SOURCE relates at 0.00, on every seed, where
        # WITHOUT `--lags` it correctly reads 0.00. A
        # relationship the data does not have is worse than a
        # missing one, because it reads as a finding - so an
        # INVENTED same-visit pattern fails this kind outright,
        # whatever the lag does.
        sv_s = spearman(num(src, p0), num(src, child))
        sv_g = spearman(num(gen, p0), num(gen, child))
        invented = (np.isfinite(sv_s) and np.isfinite(sv_g)
                    and abs(sv_s) < 0.1 and abs(sv_g) >= 0.3)
        if invented:
            return (False, round(sv_s, 3), round(sv_g, 3),
                    "INVENTED SAME-VISIT PATTERN: the source "
                    "relates this pair at {:+.3f} at the SAME "
                    "visit and the output at {:+.3f}. A "
                    "relationship the data does not have is "
                    "worse than a missing one".format(sv_s, sv_g))
        if not np.isfinite(s_) or abs(s_) < 0.05:
            return (None, s_, g_, "no LAGGED signal in the SOURCE "
                                  "to carry (the same-visit pair "
                                  "is near zero by design)")
        return (bool(np.isfinite(g_) and abs(g_) >= 0.5 * abs(s_)
                     and np.sign(g_) == np.sign(s_)),
                round(s_, 3), round(g_, 3),
                "the lag is RECOMPUTED from each table's own "
                "ordering, never read off the emitted __prev "
                "column, which generation draws rather than "
                "carries")

    if kind in ("linear", "saturating", "threshold",
                "heterogeneous",
                "subgroup-conditional", "three-way",
                "interaction-with-main-effect"):
        s = spearman(num(src, p0), num(src, child))
        g = spearman(num(gen, p0), num(gen, child))
        if not np.isfinite(s) or abs(s) < 0.05:
            return (None, s, g, "no monotone signal in the SOURCE to "
                                "carry, so nothing to score")
        return (bool(np.isfinite(g) and abs(g) >= 0.5 * abs(s)
                     and np.sign(g) == np.sign(s)),
                round(s, 3), round(g, 3),
                "rank correlation keeps its sign and at least half "
                "its size")

    if kind in ("nonlinear-u", "interaction-no-main-effect"):
        s = curve_gap(src, p0, child)
        g = curve_gap(gen, p0, child)
        if not np.isfinite(s) or abs(s) < 0.15:
            return (None, s, g, "no bend in the SOURCE to carry")
        return (bool(np.isfinite(g) and abs(g) >= 0.4 * abs(s)
                     and np.sign(g) == np.sign(s)),
                round(s, 3), round(g, 3),
                "the bend keeps its direction and 40% of its size")

    if kind == "exact-identity":
        return (None, None, None,
                "identities are scored by the tightness check in the "
                "fidelity report, not here")

    if kind == "simpsons-reversal":
        # THE POOLED SIGN IS THE OPPOSITE OF THE TRUTH, so scoring it
        # pooled would mark success on exactly the wrong answer.
        g_col = parents[1] if len(parents) > 1 else None
        if g_col is None or g_col not in gen.columns:
            return (None, None, None, "no stratum column to score by")
        def within(f):
            vals = []
            for _lvl, part in f.groupby(f[g_col].astype(str)):
                if len(part) < 60:
                    continue
                r = spearman(num(part, p0), num(part, child))
                if np.isfinite(r):
                    vals.append(r)
            return float(np.mean(vals)) if vals else float("nan")
        s, g = within(src), within(gen)
        if not np.isfinite(s) or abs(s) < 0.05:
            return (None, s, g, "no within-stratum signal in the "
                                "SOURCE")
        return (bool(np.isfinite(g) and np.sign(g) == np.sign(s)
                     and abs(g) >= 0.4 * abs(s)),
                round(s, 3), round(g, 3),
                "the WITHIN-stratum sign survives - the pooled sign "
                "is the opposite of the truth here")

    return (None, None, None, "no survival test for this kind")


def run(out_dir, patients, harder, seed, say,
        tangled=0, skew=0.0, lags=False):
    fx = out_dir / "fixture"
    fx.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_tidy_fixture.py"),
         "-o", str(fx), "--patients", str(patients),
         "--harder", str(harder), "--seed", str(seed),
         "--tangled", str(tangled),
         "--skew-planted", str(skew)],
        capture_output=True, text=True)
    if r.returncode != 0:
        say(r.stdout[-2000:] + r.stderr[-2000:])
        raise SystemExit("fixture generation failed")

    csvs = sorted(fx.glob("*.csv"))
    if not csvs:
        raise SystemExit("fixture wrote no csv")
    src = pd.read_csv(csvs[0], dtype=str, keep_default_na=False)
    truth = json.loads((fx / "ground_truth.json").read_text(
        encoding="utf-8"))
    say("fixture: {} rows x {} columns, {} planted relationships, "
        "{} noise columns".format(
            len(src), src.shape[1], len(truth["relationships"]),
            len(truth["noise_columns"])))

    lag_note = ""
    if lags:
        # THE SAME CONSTRUCTION `synthkit fit --lags` USES, called
        # rather than rebuilt - this instrument scores the FITTED
        # path, and a bench that built its own lag features would be
        # scoring a path nobody runs. `lagged-cross-column` is the
        # one kind that misses on EVERY seed, and the reason was that
        # the bench never built lags at all: a capability present in
        # the pipeline and wired to nothing in the instrument that
        # grades it.
        from synthkit.temporal import (add_lag_features,
                                       detect_time_column)
        rows = src.to_dict("records")
        tc, tk, _ev = detect_time_column(rows, "person_id")
        rows, info = add_lag_features(rows, "person_id", tc, tk)
        src = pd.DataFrame(rows)
        lag_note = "{} lag column(s) on time axis {}".format(
            info["added"], tc or "file order")
        say("--lags: " + lag_note)

    say("discovering (this is the slow part)")
    cat = discover(src, group_by="person_id", seed=seed + 1)
    (out_dir / "catalogue.json").write_text(json.dumps(cat, indent=1),
                                            encoding="utf-8")

    sc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "score_discovery.py"),
         "--truth", str(fx / "ground_truth.json"),
         "--model", str(out_dir / "catalogue.json")],
        capture_output=True, text=True)
    say(sc.stdout.strip() or sc.stderr.strip())

    # WHAT DISCOVERY FOUND, per planted relationship - recomputed here
    # rather than parsed out of the scorer's text, so the survival
    # table below lines up row for row.
    found = {}
    claim_pairs = set()
    for c in cat["claims"]:
        for p_ in (c.get("predictors") or []):
            a, b = c["child"], p_["column"]
            base = lambda z: (z.split("__prev")[0].split("__delta")[0]
                              .split("__has__")[0])
            claim_pairs.add(frozenset([base(a), base(b)]))
    for tr in truth["relationships"]:
        found[tr["child"]] = any(
            frozenset([tr["child"], p_]) in claim_pairs
            for p_ in tr["parents"])

    say("building and generating")
    bp = B.build(src, cat, group_by="person_id")
    rep = {}
    gen = generate(bp, n_patients=patients, seed=seed + 5,
                   report=rep)
    n_rel = len(bp.get("relationships") or [])
    n_drop = len(rep.get("edges_dropped") or [])

    # THE LAG DIAGNOSTIC, printed from the run rather than
    # guessed at. A lag column is a PERMUTATION of its own
    # column within each patient, so in the source `x` and
    # `x__prev` are related only as much as the column is
    # patient-stable - on this fixture, not at all. If the
    # OUTPUT relates them, every relationship drawn from
    # `x__prev` acquires a same-visit correlation with `x`
    # that the source does not have, which is the shape of
    # the fault rather than a story about it.
    if lags:
        from synthkit.temporal import LAG_SUFFIX
        say("")
        say("lag columns: a lag is a PERMUTATION of its own "
            "column, so x ~ x__prev measures how much the")
        say("engine invented between them. Anything drawn from "
            "x__prev inherits that against x.")
        # ABSENT IS A FINDING, NOT A BLANK. The first version
        # of this block printed its header and NO ROWS, because
        # not one lag column reaches the written file - which is
        # the whole answer and read as a broken diagnostic. A
        # blueprint claim whose parent is not in the output
        # cannot fire, so the kind is FOUND by the search and
        # unreachable by the sampler.
        _lag_src = [c for c in sorted(src.columns)
                    if c.endswith(LAG_SUFFIX)]
        _lag_gen = [c for c in _lag_src if c in gen.columns]
        say("  {} lag column(s) in the source frame, {} of them "
            "in the OUTPUT".format(len(_lag_src), len(_lag_gen)))
        if not _lag_gen:
            say("  NO lag column reaches the written file, so a "
                "claim with a lag parent cannot fire at")
            say("  generation however well the search found it.")
            for tr in truth["relationships"]:
                if tr["kind"] != "lagged-cross-column":
                    continue
                for r in (bp.get("relationships") or []):
                    if r["child"] != tr["child"]:
                        continue
                    # THE BLUEPRINT'S FIELD IS `parents`, A LIST
                    # OF NAMES. `predictors[].column` is the
                    # CATALOGUE's shape, and reading it here
                    # printed `<- []` for every record on a
                    # blueprint whose relationships plainly work -
                    # the ninth wrong-key fault in this project,
                    # in a diagnostic written the same day. It
                    # sent me looking for an empty-predictor bug
                    # that does not exist. The key is asserted
                    # against a LIVE built blueprint in
                    # smoke_bench, not remembered.
                    say("  blueprint {} <- {}".format(
                        r["child"], list(r.get("parents") or [])))
        for c in _lag_gen:
            base = c[:-len(LAG_SUFFIX)]
            if base not in src.columns:
                continue
            ss = spearman(num(src, base), num(src, c))
            gg = spearman(num(gen, base), num(gen, c))
            if not (np.isfinite(ss) and np.isfinite(gg)):
                continue
            flag = ("  INVENTED" if abs(ss) < 0.1
                    and abs(gg) >= 0.3 else "")
            say("  {:<34} source {:+.3f}  generated {:+.3f}{}"
                .format("{} ~ __prev".format(base[:24]),
                        ss, gg, flag))
    say("relationships in the blueprint {}, DROPPED to make the "
        "graph sampleable {} ({:.0%}) - the real extract drops 16 of "
        "26".format(n_rel, n_drop,
                    n_drop / float(max(n_rel, 1))))

    rows = []
    for tr in truth["relationships"]:
        ok, s, g, how = survives(tr["kind"], tr, src, gen)
        rows.append({"kind": tr["kind"], "child": tr["child"],
                     "parents": tr["parents"],
                     "found": bool(found.get(tr["child"])),
                     "survived": ok, "source": s, "generated": g,
                     "how": how})
    return {"rows": rows, "seed": seed, "harder": harder,
            "patients": patients, "relationships": n_rel,
            "edges_dropped": n_drop, "lags": lag_note}


# THE SWEEP THAT ATTRIBUTED THE DEBT, as data rather than prose.
#
# seed: (found_of_13, het_source_strength, het_found)
SEED_SWEEP = (
    (0, 11, 0.182, False),
    (1, 12, 0.247, True),
    (2, 11, 0.234, False),
    (3, 11, 0.230, False),
    (4, 11, 0.184, False),
)

# THE SAME FIVE SEEDS WITH `--lags`, measured 2026-11-08 once the
# bench could build the lag features at all. seed: found_of_13.
LAG_SWEEP = ((0, 12), (1, 13), (2, 12), (3, 12), (4, 12))

# WHAT THE EXTRA KIND COST BEFORE IT WAS FIXED, kept because the
# guard that stops it coming back is still the thing being
# tested. `discover._source` folded `x__prev` to `x`, so the
# LAG's curve was applied at the SAME visit and the output
# acquired a correlation the source does not have:
#   seed: (source same-visit, generated same-visit)
LAG_COST = ((0, 0.003, 0.971), (1, 0.000, 0.969),
            (2, -0.001, 0.973), (4, -0.004, 0.971))

# AND WHAT IT DOES NOW, same five seeds, after the parent is
# MARKED as read from the previous visit and the sampler reads it
# from the patient's previous row. The lagged relationship is
# FOUND and SURVIVES on every seed, landing within 0.03 of source
# - it was absent (-0.002) rather than degraded before.
#   seed: (source lagged, generated lagged, survived_of_scored)
LAG_SURVIVES = ((0, 0.944, 0.971, (9, 11)),
                (1, 0.944, 0.969, (11, 11)),
                (2, 0.946, 0.973, (10, 11)),
                (3, 0.947, 0.970, (9, 11)),
                (4, 0.945, 0.972, (10, 11)))


def seed_range_note():
    """What a single run's FOUND count does and does not mean.

    The sprint board carried "the bench reads 11/13 against a
    recorded 12/13 - still unattributed" as a debt, and a seed
    sweep attributed it. The only kind that moves is
    `heterogeneous`, which sits at discovery's inclusion
    threshold: a MISS at source 0.234 and a FIND at 0.247 are
    the same effect either side of the cut. Not a regression -
    the recorded 12/13 was the lucky seed of five.

    Returned as lines rather than printed, so a check can call
    this and see the real numbers instead of grepping the
    source for a sentence that may never execute.
    """
    counts = [c for _s, c, _v, _f in SEED_SWEEP]
    hits = [f for _s, _c, _v, f in SEED_SWEEP]
    strengths = [v for _s, _c, v, _f in SEED_SWEEP]
    found_at = [v for _s, _c, v, f in SEED_SWEEP if f]
    return [
        "THIS IS ONE SEED. Over seeds {}-{} the FOUND count "
        "runs {}-{} of 13, and the only kind that".format(
            SEED_SWEEP[0][0], SEED_SWEEP[-1][0],
            min(counts), max(counts)),
        "moves is `heterogeneous` - a marginal planted effect "
        "at discovery's inclusion threshold, found on",
        "{} of those {} seeds (source strength {}, found only "
        "at {}). A single run reading {} is".format(
            sum(1 for h in hits if h), len(SEED_SWEEP),
            " / ".join("{:.3f}".format(v) for v in strengths),
            " / ".join("{:.3f}".format(v) for v in found_at),
            min(counts)),
        "NOT a regression against a recorded {}. "
        "`lagged-cross-column` misses on every seed without".format(
            max(counts)),
        "--lags, and is found on every seed WITH it: {} of 13 "
        "over the same five seeds.".format(
            " / ".join(str(c) for _s, c in LAG_SWEEP)),
        "AND IT WAS NOT A FREE POINT UNTIL 2026-11-08. The "
        "output used to acquire a SAME-VISIT correlation of",
        "{:.2f}-{:.2f} on a pair the source relates at "
        "{:+.3f}-{:+.3f}, because the fold applied the LAG's".format(
            min(g for _s, _f, g in LAG_COST),
            max(g for _s, _f, g in LAG_COST),
            min(f for _s, f, _g in LAG_COST),
            max(f for _s, f, _g in LAG_COST)),
        "curve at the current visit - an invented relationship, "
        "which is worse than a missing one.",
        "NOW THE LAGGED RELATIONSHIP SURVIVES on every seed: "
        "source {:.2f}-{:.2f} against generated".format(
            min(x for _s, x, _g, _v in LAG_SURVIVES),
            max(x for _s, x, _g, _v in LAG_SURVIVES)),
        "{:.2f}-{:.2f}, and the invented pattern is refused by "
        "name so it cannot come back unnoticed.".format(
            min(g for _s, _x, g, _v in LAG_SURVIVES),
            max(g for _s, _x, g, _v in LAG_SURVIVES)),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--patients", type=int, default=600)
    ap.add_argument("--harder", default="1.0")
    ap.add_argument("--tangled", type=int, default=0)
    ap.add_argument("--skew-planted", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--lags", action="store_true",
                    help="build the per-patient lag features that "
                         "`synthkit fit --lags` builds, which is "
                         "what `lagged-cross-column` needs to be "
                         "findable at all")
    ap.add_argument("-o", "--out", default="")
    a = ap.parse_args()

    def say(m):
        print(m, flush=True)

    with tempfile.TemporaryDirectory() as td:
        res = run(Path(td), a.patients, a.harder, a.seed, say,
                  a.tangled, a.skew_planted, a.lags)

    say("")
    say("PATTERN KIND                        FOUND   SURVIVES  "
        "source -> generated")
    say("-" * 78)
    for r in res["rows"]:
        surv = ("  --  " if r["survived"] is None
                else ("  yes " if r["survived"] else "  NO  "))
        nums = ("" if r["source"] is None
                else "{} -> {}".format(r["source"], r["generated"]))
        say("{:<34}  {:<6} {:<9} {}".format(
            r["kind"][:34], "yes" if r["found"] else "NO",
            surv, nums))

    scored = [r for r in res["rows"] if r["survived"] is not None]
    say("")
    say("FOUND    {}/{}".format(sum(1 for r in res["rows"]
                                    if r["found"]), len(res["rows"])))
    say("SURVIVED {}/{} of the kinds with an honest survival test "
        "({} unscored)".format(
            sum(1 for r in scored if r["survived"]), len(scored),
            len(res["rows"]) - len(scored)))
    say("")
    say("A kind that is FOUND and does not SURVIVE is a sampler gap, "
        "not a search gap.")
    say("A kind that is not found cannot survive, and the two are "
        "fixed in different files.")
    # THIS IS ONE SEED, AND THE COUNT IS A RANGE.
    #
    # The sprint board carried "the bench reads 11/13 against a
    # recorded 12/13 - still unattributed" as a debt, and a seed
    # sweep attributed it: 11, 12, 11, 11, 11 over seeds 0-4,
    # with the ONLY varying kind `heterogeneous`, which sits at
    # discovery's inclusion threshold. Its own measured source
    # strength ran 0.182 / 0.247 / 0.234 / 0.230 / 0.184 and it
    # was found on exactly one of those - the 0.247 - so a miss
    # at 0.234 and a find at 0.247 are the same effect either
    # side of the cut. It is not a regression, and the recorded
    # 12/13 was the lucky seed.
    #
    # This project's own rule is to report the range and never a
    # point, so the bench says so itself rather than leaving the
    # next reader to re-derive it from a single run.
    say("")
    for _line in seed_range_note():
        say(_line)

    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1),
                               encoding="utf-8")
        say("wrote {}".format(a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
