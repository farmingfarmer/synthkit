"""What does the k rule COST, separated from what the engine costs?

WHY THIS EXISTS. The spread misses on the real extract are reported
as "bound-explained" - 68% to 96% of the column's magnitude sitting
beyond the published bound - and that is correct and important,
because it stops a privacy decision being read as a sampler bug. But
it answers only half the question. It says the k rule is responsible
for a shortfall; it does not say how much spread was reachable and
how much of THAT the engine actually delivered.

Those are different numbers and they call for different work:

  ceiling   = sd(source clipped to the published bounds) / sd(source)
              A REFERENCE, not a maximum - and the first version of
              this docstring called it one, which is false. Nothing
              beyond the bounds may be emitted, since they are the
              mean of the k most extreme patients. But CLIPPING
              compresses the tail toward the cap, while a generator
              may PLACE mass at the cap, which is bimodal and
              carries more variance: measured on a heavy-tailed
              fixture, clipped 16.12 against an extremal placement
              at 20.29, both matching the published tail mean. So
              `delivered` above 1.0 is legitimate rather than noise.

  delivered = sd(generated) / sd(source clipped)
              How much of the reachable spread the engine actually
              produced. THIS is the engineering gap, and it is the
              only part a fidelity change can move.

A column at delivered ~ 1.00 is finished: the remaining shortfall is
the privacy rule and the only way to move it is to change the rule,
which is a governance decision rather than an engineering one. A
column at delivered ~ 0.60 has real headroom.

READ IT AS A RANGE AND NEVER AS A POINT, which took a measurement to
learn. On a HEAVY-TAILED column the spread is carried by the top
segment, which is 1% of the rows, so `delivered` is a ratio whose
numerator rests on a few hundred values under a convex placement.
Measured on one blueprint over eight GENERATE seeds, with nothing
else changed, it runs 0.421 to 0.801 - a range 0.38 wide, which is
most of the distance to 1.00. A single reading therefore cannot
support a conclusion about the engine, and this project has twice
recorded a day lost to exactly that. A column's shortfall is
established by a PAIRED arm at the same seed, or by a seed sweep
reported as a range.

AND THE SAME IS TRUE OF A REPRODUCTION. A fixture reading 0.469
beside a real-extract 0.437 looks like the fault reproduced; with a
statistic this noisy it is two draws agreeing by chance, and the
fixture was additionally too small. At 9,000 rows its top segment
holds about 70 values, and the realised mean of the placement over
them is UNBIASED but spans a range of 0.70 across five generate
seeds (1.414 / 0.922 / 0.717 / 1.228 / 0.750); at four times the
rows that range is 0.15. Nothing undershoots - a low reading is a
draw. Match the ROW COUNT before reading a tail reproduction, and
read the segment over seeds before reading it at all.

NO VALUE IS PUBLISHED. Standard deviations, ratios and column names
only - the same discipline the deck follows, because a report about
a real extract is itself a release.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sayn import annotate                               # noqa: E402
from synthkit.intake import read_table                  # noqa: E402


def _say(n):
    return annotate(str(n))


def bounds_of(spec):
    """The published floor and cap of a numeric column, or None.

    They are `v[0]` and `v[-1]` of the quantile grid - the same two
    numbers `_pin_to_bounds` pins generated values back inside - and
    each is the mean of the k most extreme patients rather than one
    person's reading.
    """
    m = (spec or {}).get("marginal") or {}
    if m.get("type") != "quantiles":
        return None
    v = m.get("v") or []
    if len(v) < 2:
        return None
    try:
        return float(v[0]), float(v[-1])
    except (TypeError, ValueError):
        return None


def price(src_df, gen_df, bp, min_rows=50):
    cols = (bp.get("columns") or {})
    rows = []
    for name, spec in cols.items():
        b = bounds_of(spec)
        if b is None or name not in src_df.columns \
                or name not in gen_df.columns:
            continue
        lo, hi = b
        s = pd.to_numeric(src_df[name], errors="coerce").dropna()
        g = pd.to_numeric(gen_df[name], errors="coerce").dropna()
        if len(s) < min_rows or len(g) < min_rows:
            continue
        sd_s = float(s.std())
        if not np.isfinite(sd_s) or sd_s <= 0:
            continue
        sd_c = float(s.clip(lo, hi).std())
        sd_g = float(g.std())
        outside = float(((s < lo) | (s > hi)).mean())
        rows.append({
            "column": name,
            "sd_source": sd_s,
            "sd_reachable": sd_c,
            "sd_generated": sd_g,
            "ceiling": sd_c / sd_s,
            "delivered": (sd_g / sd_c) if sd_c > 0 else float("nan"),
            "overall": sd_g / sd_s,
            "rows_outside_bounds": outside,
        })
    return rows


def main(argv):
    if len(argv) < 2 or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("usage: k_price.py RUNDIR SRC.csv [--top N]")
        return 0 if (argv and argv[0] in ("-h", "--help")) else 2
    run, src = Path(argv[0]), argv[1]
    top = 12
    if "--top" in argv:
        top = int(argv[argv.index("--top") + 1])
    for f in ("blueprint.json", "generated.csv"):
        if not (run / f).exists():
            print("STOPPED: {} has no {}".format(run, f))
            return 2
    bp = json.loads((run / "blueprint.json").read_text(
        encoding="utf-8"))

    # THE SOURCE MUST BE THE ONE THAT PRODUCED THE RUN. A whole
    # session here was lost to measuring a generated file against an
    # extract that never produced it, and every number looked
    # reasonable.
    prov = {}
    if (run / "provenance.json").exists():
        prov = json.loads((run / "provenance.json").read_text(
            encoding="utf-8"))
    want = ((prov.get("source") or {}).get("rows_read"))
    src_df, how = read_table(src)
    if want and len(src_df) != want:
        print("STOPPED: this run recorded reading {} rows and the "
              "source handed in holds {}. Refusing: a run measured "
              "against an extract that never produced it looks "
              "entirely reasonable."
              .format(_say(want), _say(len(src_df))))
        return 2
    gen_df, _ = read_table(str(run / "generated.csv"))

    rows = price(src_df, gen_df, bp)
    if not rows:
        print("NO NUMERIC COLUMN CARRIES PUBLISHED BOUNDS in this "
              "blueprint, so there is no k price to read here. "
              "That is a finding about the run, not a clean bill.")
        return 3

    print("WHAT THE k RULE COSTS, AND WHAT THE ENGINE COSTS")
    print("  read as      {}".format(how))
    print("  columns      {}".format(_say(len(rows))))
    print()
    print("  ceiling   = how much spread the RULE still allows")
    print("  delivered = how much of THAT the engine produced")
    print()
    print("  {:<34} {:>9} {:>10} {:>9} {:>8}".format(
        "column", "ceiling", "delivered", "overall", "outside"))
    worst = sorted(rows, key=lambda r: r["delivered"])[:top]
    for r in worst:
        print("  {:<34} {:>9.3f} {:>10.3f} {:>9.3f} {:>7.1%}".format(
            r["column"][:34], r["ceiling"], r["delivered"],
            r["overall"], r["rows_outside_bounds"]))
    print()
    ceil_mean = float(np.mean([r["ceiling"] for r in rows]))
    deliv_mean = float(np.mean([r["delivered"] for r in rows]))
    at_ceiling = sum(1 for r in rows if r["delivered"] >= 0.95)
    print("  mean ceiling   {}".format(_say(round(ceil_mean, 4))))
    print("  mean delivered {}".format(_say(round(deliv_mean, 4))))
    print("  columns already AT the ceiling (delivered >= 0.95): "
          "{} of {}".format(_say(at_ceiling), _say(len(rows))))
    print()
    # A POINT READING OF A NOISY STATISTIC INVITES THE CONCLUSION IT
    # CANNOT SUPPORT. Measured at 0.421 to 0.801 over eight generate
    # seeds on one blueprint, so the warning travels with every
    # reading rather than living in a docstring nobody opens.
    print("  THIS IS ONE DRAW. On a heavy-tailed column `delivered`")
    print("  rests on the top 1% of rows, and over eight generate")
    print("  seeds on one blueprint it ran {} to {} - a range"
          .format(_say(0.421), _say(0.801)))
    print("  wider than most columns' distance to 1.00. Establish a")
    print("  shortfall with a PAIRED arm at the same seed, or a seed")
    print("  sweep reported as a range. Never from this one number.")
    print()
    if deliv_mean >= 0.95:
        print("  THE ENGINE IS AT THE RULE'S CEILING. The remaining")
        print("  spread shortfall is the privacy posture, and the")
        print("  only way to move it is to change the rule - a")
        print("  governance decision, not an engineering one.")
    else:
        print("  THERE IS ENGINEERING HEADROOM: the engine is")
        print("  delivering {} of the spread the rule still".format(
            _say(round(deliv_mean, 3))))
        print("  allows, so a fidelity change can move it without")
        print("  touching the privacy rule at all.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
