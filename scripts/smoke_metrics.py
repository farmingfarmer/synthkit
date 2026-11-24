"""The metrics record a dashboard stacks.

WHY THESE CHECKS AND NOT OTHERS. The record exists so N runs can
be charted over time, which puts two failure modes ahead of
everything else.

A RENAMED OR NESTED METRIC BREAKS A CHART SILENTLY - it reads as
a flat line rather than as an error - so the core names are
pinned as a SUBSET (adding is free, renaming is deliberate) and
every value in `series` is asserted to be a scalar.

AND THE RECORD TRAVELS TO A DASHBOARD, so it must carry no row,
no bin count and no published value. That is checked
STRUCTURALLY, by refusing any array longer than the column count:
a histogram, a row slice and a per-patient series all look like
that, and a ban by NAME would only catch the ones somebody
thought to name.

IT IS DRIVEN AGAINST A REAL RUN, written by the real pipeline.
Every one of this project's twelve wrong-key faults was a module
reading another module's dict with names that sounded right,
while the checks built that dict BY HAND with the same invented
names - the fixture agreeing with the reader and both disagreeing
with the only thing that matters.
"""
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from synthkit import gate as G                         # noqa: E402
from synthkit import metrics as M                      # noqa: E402

PASS = FAIL = 0

# THE NAMES A DASHBOARD CHARTS. A SUBSET check, so adding a
# metric is free and renaming one is a deliberate edit here -
# which is the whole point, because a renamed key makes a chart
# read as a flat line rather than as an error.
CORE = (
    "gate_met", "gate_criteria_passed", "gate_criteria_total",
    "pairs_related", "pairs_direction_kept", "pairs_close",
    "pairs_inverted", "pairs_own", "pairs_own_close",
    "pairs_own_inverted", "columns_measured", "coverage_ok",
    "shapes_published", "shapes_tracked", "surfaces_compared",
    "surfaces_tracked", "dynamics_compared", "dynamics_tracked",
)


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def frame(n_pat=60, n_vis=6, seed=3):
    """A small cohort with a set column, so the token metrics are
    exercised rather than reading ABSENT for want of one."""
    rng = np.random.RandomState(seed)
    rows = []
    for p in range(n_pat):
        yob = 1950 + (p % 40)
        for v in range(n_vis):
            age = 2016 + v // 2 - yob
            a = rng.normal(10, 2)
            rows.append({
                "person_id": "P%03d" % p, "visit": v,
                "year_of_birth": float(yob),
                "age_at_visit": float(age),
                "height": float(170 - .25 * (age - 40)
                                + rng.normal(0, 4)),
                "a": float(a),
                "c": float(2 * a + rng.normal(0, 1)),
                # TWO TO FOUR TOKENS, not one to three. A set
                # column is only classed as one when the
                # separator appears in enough FILLED cells
                # (`SEP_SHARE`), so a vocabulary that often
                # yields a single token is not a set at all and
                # the token metrics read ABSENT for want of a
                # fixture rather than for want of a measurement.
                "conds": ";".join(sorted(set(
                    "c%02d" % (rng.randint(0, 6))
                    for _ in range(rng.randint(2, 5))))),
            })
    return pd.DataFrame(rows)


def thin_tokens(n_pat=80, n_vis=4, seed=7):
    """A vocabulary the k screen genuinely THINS.

    The fixture above publishes every token it finds, so its
    `set_tokens_published_share` reads 1.0 and the metric could
    be wrong forever without a check noticing. Here the tokens
    come from a MIDDLE BAND plus a rare tail, which is the shape
    the real extract has - its conditions publish 9.6% of what
    the source holds.

    THE MIDDLE BAND IS LOAD-BEARING. A pool where only ONE token
    clears the floor publishes no vocabulary at all, the column
    falls back to `levels` rather than `list`, and the token
    metrics then read ABSENT - measuring a different thing. The
    first version of this fixture did exactly that.
    """
    rng = np.random.RandomState(seed)
    band = ["m%02d" % i for i in range(8)]
    rows = []
    for p in range(n_pat):
        mine = band[p % len(band)], band[(p + 3) % len(band)]
        rare = "r%04d" % p
        for v in range(n_vis):
            rows.append({
                "person_id": "P%03d" % p, "visit": v,
                "x": float(rng.normal(10, 2)),
                "codes": ";".join(sorted(set(mine + (rare,)))),
            })
    return pd.DataFrame(rows)


def build_run(td, generate=True, frame_fn=None):
    """A real run directory, written by the real pipeline."""
    src = os.path.join(td, "src.csv")
    (frame_fn or frame)().to_csv(src, index=False)
    run = os.path.join(td, "run")
    cmd = [sys.executable, str(ROOT / "scripts" / "run_discovery.py"),
           "--src", src, "--group-by", "person_id", "--out", run]
    if generate:
        cmd.append("--generate")
    p = subprocess.run(cmd, capture_output=True, text=True,
                       cwd=str(ROOT), timeout=1800)
    return run, src, p


def main():
    with tempfile.TemporaryDirectory() as td:
        run, src, proc = build_run(td)
        check("the real pipeline wrote a run to read - without "
              "one every check below would be against a dict "
              "this file invented, which is how twelve wrong-key "
              "faults got in: exit {}".format(proc.returncode),
              proc.returncode == 0
              and os.path.exists(os.path.join(run,
                                              "fidelity.json")))

        rec = M.collect(run)
        sr = rec["series"]

        # ---- NOTHING IS RECOMPUTED -------------------------
        fid = json.load(io.open(os.path.join(run, "fidelity.json"),
                                encoding="utf-8"))
        v = G.assess(fid)
        check("the gate figures are the GATE's own, equal to the "
              "digit - a second implementation is how a dashboard "
              "and a release verdict come to disagree",
              sr["gate_met"] == bool(v["met"])
              and sr["gate_criteria_total"] == len(v["criteria"])
              and sr["gate_criteria_passed"] == sum(
                  1 for c in v["criteria"] if c["ok"]))
        s = fid["summary"]
        check("...and every pair and shape metric equals the "
              "run's own summary field rather than a re-derived "
              "one",
              sr["pairs_related"] == s["pairs"]
              and sr["pairs_close"] == s["pairs_close"]
              and sr["pairs_inverted"] == s["pairs_inverted"]
              and sr["shapes_tracked"] == s["shapes_ok"]
              and sr["dynamics_tracked"] == s["dynamics_ok"])
        check("...and the module imports no measure of its own - "
              "asserted on the source, because a convenience "
              "import is how the second implementation arrives",
              "from . import gate" in io.open(
                  str(ROOT / "synthkit" / "metrics.py"),
                  encoding="utf-8").read()
              and "import curvecheck" not in io.open(
                  str(ROOT / "synthkit" / "metrics.py"),
                  encoding="utf-8").read())

        # ---- THE DASHBOARD SURFACE -------------------------
        missing = [k for k in CORE if k not in sr]
        check("every core metric name is present, so a rename is "
              "a deliberate edit here rather than a chart that "
              "silently reads as a flat line: {}".format(
                  missing or "none missing"),
              not missing)
        nonscalar = [(k, type(x).__name__) for k, x in sr.items()
                     if not isinstance(x, (int, float, bool))
                     and x is not None]
        check("every value in `series` is a scalar or None - a "
              "nested value cannot be stacked across runs, which "
              "is the one thing this surface exists for: {}"
              .format(nonscalar[:3] or "all scalar"),
              not nonscalar)
        check("...and the counts come with their DENOMINATORS, "
              "never a share alone - three conclusions on this "
              "project were drawn from a percentage whose "
              "denominator had moved",
              sr["pairs_related"] is not None
              and sr["shapes_published"] is not None
              and sr["surfaces_compared"] is not None
              and sr["dynamics_compared"] is not None)

        # ---- THE STRUCTURAL PRIVACY CHECK ------------------
        ncols = rec["patterns"]["columns_published"]
        check("the record carries NO array longer than the column "
              "count - a histogram, a row slice and a "
              "per-patient series all look like that, and a ban "
              "by name would only catch the ones somebody "
              "thought to name",
              not M.long_arrays(rec, ncols + M.ARRAY_SLACK))
        planted = dict(rec)
        planted["oops"] = {"bins": list(range(ncols + 99))}
        check("...and the scan FIRES on a planted one, naming its "
              "path - a structural guard that cannot demonstrate "
              "its own fault is the shape this project has "
              "corrected more than any other: {}".format(
                  M.long_arrays(planted, ncols + M.ARRAY_SLACK)),
              any("oops" in x for x in M.long_arrays(
                  planted, ncols + M.ARRAY_SLACK)))
        check("...and `write` REFUSES such a record rather than "
              "shipping it with a warning, because a warning "
              "puts it on the dashboard anyway",
              _raises(lambda: M.check_record(planted))
              and not _raises(lambda: M.check_record(rec)))
        check("the withheld list TRAVELS as data, so a renderer "
              "cannot drop it - four statements of what this "
              "record deliberately does not carry",
              isinstance(rec["withheld"], list)
              and len(rec["withheld"]) >= 4
              and any("row" in x for x in rec["withheld"]))

        # ---- PRIVACY IS NOT A FIDELITY METRIC --------------
        check("the privacy posture is its OWN section and NOT in "
              "`series` - a privacy reading on a fidelity trend "
              "line invites somebody to optimise it, and k is a "
              "policy choice rather than a score",
              rec["privacy"]["k"] is not None
              and not [k for k in sr if "k_" == k[:2]
                       or k in ("k", "counted_in")])
        check("...and it carries the k rule in the contract's own "
              "words, counted in PATIENTS - one person seen two "
              "hundred times must not set a published bound "
              "alone",
              "patients" in (rec["privacy"]["counted_in"] or ""))

        # ---- THE RUN STAMP ---------------------------------
        r = rec["run"]
        check("the record names its run, its build and its "
              "source, so a dashboard cannot show a stale row as "
              "current: {} / {} / {}".format(
                  r["id"], r["build_id"], r["source_name"]),
              r["id"] == "run" and r["source_name"] == "src.csv"
              and r["rows_read"] == 360)
        check("...and `build_id` is a STRING beside the dict - "
              "`build` is {source, id, note}, and slicing it as a "
              "string prints a brace, which this project has on "
              "record as nearly thrown mid-demo",
              isinstance(r["build_id"], (str, type(None)))
              and isinstance(r["build"], (dict, type(None))))

        # ---- WHAT THE FIT FOUND IN THE SOURCE --------------
        pat = rec["patterns"]
        check("the patterns section reports what the fit found in "
              "the SOURCE - the half a reader asks 'was the "
              "original data worth modelling' of, which is a "
              "different question from fidelity and belongs "
              "apart: {} relationships, {} curves, {} tokens"
              .format(pat["relationships"], pat["effect_curves"],
                      pat["set_tokens_published"]),
              pat["relationships"] and pat["effect_curves"]
              and pat["set_tokens_published"])
        check("...and the claim skill is a RANGE rather than a "
              "mean - one number over claims of different "
              "strengths is the pooled-correlation fault this "
              "project withdrew a conclusion for",
              pat["claim_skill_min"] is not None
              and pat["claim_skill_max"] is not None
              and "claim_skill_mean" not in pat)

        # AND THE k SCREEN'S COST IS EXERCISED, not merely
        # reported. The fixture above publishes every token it
        # finds, so its share reads 1.0 and the metric could be
        # wrong forever; this one is built so the floor bites.
        tdt = os.path.join(td, "t")
        os.makedirs(tdt, exist_ok=True)
        runt, _, proct = build_run(tdt, frame_fn=thin_tokens)
        rt = M.collect(runt)["patterns"]
        check("the k screen's COST is a metric and the fixture "
              "reaches it - of the tokens the SOURCE holds, how "
              "many were published: {} of {}, share {}".format(
                  rt["set_tokens_published"],
                  rt["set_tokens_in_source"],
                  rt["set_tokens_published_share"]),
              proct.returncode == 0
              and rt["set_tokens_in_source"]
              and rt["set_tokens_published"]
              and rt["set_tokens_published_share"] < 0.9)
        check("...and the published count never EXCEEDS what the "
              "source held, which is the arithmetic that would "
              "catch the two counts being measured over "
              "different populations",
              rt["set_tokens_published"]
              <= rt["set_tokens_in_source"]
              and rt["set_tokens_above_k"]
              <= rt["set_tokens_in_source"])

        # ---- WHAT WAS TWEAKED, AND WHAT IT CANNOT ANSWER ---
        st = rec["settings"]
        check("the settings section records the seed and the "
              "dials the run was HOLDING, which is what makes two "
              "runs comparable at all",
              st["seed"] is not None and st["dials"] is not None)
        check("...and it SAYS that a setting's effect is a "
              "question about two runs, naming the tool - one run "
              "cannot answer it however many settings it records",
              "two runs" in (st["comparison_needs_two_runs"] or "")
              and "arm_compare" in (
                  st["comparison_needs_two_runs"] or ""))

        # ---- ABSENT IS NOT ZERO ----------------------------
        tdb = os.path.join(td, "b")
        os.makedirs(tdb, exist_ok=True)
        run2, _, proc2 = build_run(tdb, generate=False)
        check("a run that FITTED and never generated reads "
              "ABSENT, not zero - a dashboard charting a zero for "
              "a measurement nobody took is worse than a gap: "
              "exit {}".format(
                  proc2.returncode if proc2 else "n/a"),
              _absent_ok(run2))

        # ---- THE LIST A DASHBOARD IS BUILT AGAINST ---------
        cat = M.catalog()
        names = set(x["name"] for x in cat)
        check("every charted metric is DESCRIBED and every "
              "description names a metric that exists - a metric "
              "with no description is a chart nobody can read, "
              "and a description for a metric that is gone is a "
              "chart that reads as a flat line: {} metrics, {} "
              "descriptions".format(len(sr), len(cat)),
              names == set(sr) and len(cat) == len(sr))
        check("...and the descriptions are DERIVED from the same "
              "table `series` reads, so there is no second list "
              "to drift - asserted on the source, because a "
              "hand-kept copy is what went stale for this "
              "project's dependency table and its dial list",
              "_DIRECT" in io.open(
                  str(ROOT / "synthkit" / "metrics.py"),
                  encoding="utf-8").read()
              and all(len(x["means"]) > 20 for x in cat))
        p5 = subprocess.run(
            [sys.executable, "-m", "synthkit.cli", "metrics",
             "--names"], capture_output=True, text=True,
            cwd=str(ROOT), timeout=600)
        check("...and the list prints WITHOUT a run directory, "
              "because 'what metrics are there' is asked before "
              "anybody has a run to point at",
              p5.returncode == 0
              and "charted metric" in p5.stdout
              and "dynamics_tracked" in p5.stdout)
        p6 = subprocess.run(
            [sys.executable, "-m", "synthkit.cli", "metrics"],
            capture_output=True, text=True, cwd=str(ROOT),
            timeout=600)
        check("...and a bare invocation REFUSES with exit 2 "
              "rather than doing something plausible - a command "
              "that acts on no argument is the oldest failure "
              "shape in this project's record",
              p6.returncode == 2
              and "--names" in (p6.stdout + p6.stderr))

        # ---- THE CLI DOOR ----------------------------------
        p3 = subprocess.run(
            [sys.executable, "-m", "synthkit.cli", "metrics", run],
            capture_output=True, text=True, cwd=str(ROOT),
            timeout=600)
        check("the CLI writes metrics.json and says so - a "
              "capability reachable only from Python is this "
              "project's most repeated fault",
              p3.returncode == 0
              and "wrote" in p3.stdout
              and os.path.exists(os.path.join(run,
                                              "metrics.json")))
        p4 = subprocess.run(
            [sys.executable, "-m", "synthkit.cli", "metrics", td],
            capture_output=True, text=True, cwd=str(ROOT),
            timeout=600)
        check("...and a directory that is not a run is REFUSED by "
              "name with exit 2, rather than a record of nulls "
              "that charts as zeros",
              p4.returncode == 2
              and "provenance" in (p4.stdout + p4.stderr))
        on_disk = json.load(io.open(
            os.path.join(run, "metrics.json"), encoding="utf-8"))
        check("...and what lands on disk is what `collect` "
              "returns, so the file a dashboard reads and the "
              "record this suite checks are the same object",
              on_disk["series"] == rec["series"]
              and on_disk["withheld"] == rec["withheld"])

    print()
    if FAIL:
        print("{} of {} checks failed.".format(FAIL, PASS + FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


def _raises(fn):
    try:
        fn()
    except ValueError:
        return True
    except Exception:
        return False
    return False


def _absent_ok(run2):
    if not run2 or not os.path.isdir(run2):
        return False
    rec = M.collect(run2)
    sr = rec["series"]
    return (rec["run"]["generated"] is False
            and sr["pairs_related"] is None
            and sr["gate_met"] is None
            and rec["patterns"]["relationships"] is not None)


if __name__ == "__main__":
    main()
