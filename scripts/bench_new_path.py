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

    if kind in ("linear", "saturating", "threshold",
                "lagged-cross-column", "heterogeneous",
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
        tangled=0, skew=0.0):
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
            "edges_dropped": n_drop}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--patients", type=int, default=600)
    ap.add_argument("--harder", default="1.0")
    ap.add_argument("--tangled", type=int, default=0)
    ap.add_argument("--skew-planted", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("-o", "--out", default="")
    a = ap.parse_args()

    def say(m):
        print(m, flush=True)

    with tempfile.TemporaryDirectory() as td:
        res = run(Path(td), a.patients, a.harder, a.seed, say,
                  a.tangled, a.skew_planted)

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

    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1),
                               encoding="utf-8")
        say("wrote {}".format(a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
