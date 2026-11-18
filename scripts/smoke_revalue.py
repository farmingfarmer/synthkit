"""A copied column carries its OWN published marginal.

WHY THIS EXISTS. An `==` constraint between a column present on
EVERY row and one present on a fifth of them is NOT a
whole-column identity: the two are equal where BOTH are present.
Copying the dense one wholesale hands the sparse one the DENSE
column's distribution, masked afterwards - so the sparse column
is never drawn from its own published grid at all, which is
exactly what `marg_draw` not holding it proves.

On the real extract that is the worst marginal on the file.
Measured, `procedure_quantity` against its own source marginal:

    copying (what ships)                      0.4002   dir 132/151
    --copy-coverage-guard (refuse the copy)   0.0254   dir 131/151
    re-valuing from its own grid              0.0255   dir 135/151

The guard and the re-value reach the SAME marginal. The guard
throws the ORDER away; this keeps it, and the order is worth
three relationship directions.

AND IT COSTS NO IDENTITY THAT WAS BEING DELIVERED. The shipping
default already breaks `procedure_count == procedure_quantity` on
90.1% of rows, because the bound-pinning pass clamps the copied
column to its own published cap. This does not lose an equality;
it stops pretending there was one.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from synthkit.discover import discover                 # noqa: E402
import synthkit.blueprint as B                         # noqa: E402
import synthkit.generate as G                          # noqa: E402
from synthkit.marginals import bands, shares, total     # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def digest(df):
    return hashlib.sha256(
        df.to_csv(index=False).encode()).hexdigest()[:16]


def num(fr, c):
    return pd.to_numeric(fr[c].replace("", None), errors="coerce")


def copied_fixture():
    """A sparse column EQUAL to a dense one where present, with
    INFORMATIVE presence - so its own marginal is genuinely
    different from the one it is copied from.

    Random presence would make the two marginals the same and the
    fixture could not tell a working re-value from a no-op, which
    is the shape of the real column: `procedure_quantity` is
    present on 19% of rows and sits 0.4002 off its own source
    marginal.
    """
    r = np.random.RandomState(4)
    npat, vis = 500, 8
    n = npat * vis
    dense = np.round(np.exp(r.normal(1.4, 0.9, n))).astype(int)
    pkeep = 1.0 / (1.0 + np.exp((dense - 4.0) / 2.0))
    keep = r.rand(n) < (0.19 * pkeep / pkeep.mean())
    return pd.DataFrame({
        "person_id": np.repeat(np.arange(npat), vis).astype(str),
        "dense": dense,
        "sparse": ["" if not k else str(int(v))
                   for k, v in zip(keep, dense)],
        "z": np.round(r.normal(0, 1, n), 3)})


def plain_fixture():
    r = np.random.RandomState(8)
    npat, vis = 300, 6
    n = npat * vis
    x = r.normal(10, 3, n)
    return pd.DataFrame({
        "person_id": np.repeat(np.arange(npat), vis).astype(str),
        "x": np.round(x, 3),
        "y": np.round(2.0 * x + r.normal(0, 1, n), 3)})


def main():
    df = copied_fixture()
    bp = B.build(df, discover(df, group_by="person_id"),
                 group_by="person_id")

    # THE FIXTURE MUST CONTAIN THE THING THE CHECK IS ABOUT.
    cp = G._copy_partners(bp)
    src_sparse = num(df, "sparse")
    check("the engine DETECTS the copy on this fixture, without "
          "which every number below would be about nothing: {}"
          .format(cp), cp.get("sparse") == "dense")
    check("...and the two columns genuinely differ in COVERAGE, "
          "which is what makes the `==` not a whole-column "
          "identity (dense {:.3f}, sparse {:.3f})".format(
              float(num(df, "dense").notna().mean()),
              float(src_sparse.notna().mean())),
          float(src_sparse.notna().mean()) < 0.5)
    check("...and the copied column's OWN marginal really is "
          "different from the one it is copied from - random "
          "presence would make them the same and this fixture "
          "could not tell a working pass from a no-op (means "
          "{:.2f} against {:.2f})".format(
              float(src_sparse.dropna().mean()),
              float(num(df, "dense").mean())),
          abs(float(src_sparse.dropna().mean())
              - float(num(df, "dense").mean())) > 1.0)

    check("the pass is OFF by default, so no run changes "
          "behaviour until somebody asks", G.REVALUE_COPIED is False)

    rep_off, rep_on = {}, {}
    g_off = G.generate(bp, seed=11, report=rep_off, revalue_copied=False)
    g_plain = G.generate(bp, seed=11)
    g_on = G.generate(bp, seed=11, report=rep_on, revalue_copied=True)

    check("...and with it off the output is BYTE-IDENTICAL to a "
          "generate that never heard of the flag ({})".format(
              digest(g_off)),
          digest(g_off) == digest(g_plain))
    check("...and the report carries nothing - absent rather than "
          "an empty list, so a reader cannot mistake 'not asked "
          "for' for 'asked for and did nothing'",
          "revalued_copied" not in rep_off)

    edges = bands(src_sparse.dropna())
    ssh = shares(src_sparse.dropna(), edges)
    d_off = total(ssh, shares(num(g_off, "sparse").dropna(), edges))
    d_on = total(ssh, shares(num(g_on, "sparse").dropna(), edges))
    check("the copied column moves TOWARD its own source marginal "
          "- the whole point ({:.4f} -> {:.4f}, {:.1f}x)".format(
              d_off, d_on, d_off / d_on if d_on else 0.0),
          d_on < d_off * 0.6)

    # THE NEIGHBOURING PROPERTIES, asserted in the same
    # measurement - this project's own rule, because a fix that
    # lands one number by breaking another is not a fix.
    c_off = float(num(g_off, "sparse").notna().mean())
    c_on = float(num(g_on, "sparse").notna().mean())
    check("...and COVERAGE is identical, because only rows that "
          "survive the presence mask are re-valued ({:.6f} both)"
          .format(c_on), abs(c_off - c_on) < 1e-9)

    r_off = float(pd.Series(num(g_off, "sparse")).corr(
        num(g_off, "dense"), method="spearman"))
    r_on = float(pd.Series(num(g_on, "sparse")).corr(
        num(g_on, "dense"), method="spearman"))
    check("...and the rank relationship to the column it was "
          "copied FROM survives, because the values are placed at "
          "the ranks the copy produced rather than redrawn in a "
          "new order ({:+.4f} -> {:+.4f})".format(r_off, r_on),
          r_on > 0.9)

    others = [c for c in g_off.columns if c != "sparse"]
    check("...and every other column is BYTE-IDENTICAL, including "
          "the one it was copied from - the pass touches the copy "
          "TARGET and nothing else",
          all(g_off[c].equals(g_on[c]) for c in others))

    rv = rep_on.get("revalued_copied") or []
    n_pres = int(num(g_on, "sparse").notna().sum())
    check("...and the run RECORDS what it did, naming the column "
          "and where it was copied from: {}".format(rv),
          len(rv) == 1 and rv[0]["column"] == "sparse"
          and rv[0]["copied_from"] == "dense")

    # THE ROW COUNT IS THE HONEST PART, and it is what the
    # presence restriction actually buys. Re-valuing every row and
    # letting the mask select afterwards reaches the SAME marginal
    # here - 0.1124 against 0.1202, a difference well inside this
    # measure's own noise floor of about 0.057 at 775 rows -
    # because an `==` copy forces the column's own marginal to BE
    # the source restricted to its present rows, so the two
    # constructions agree by arithmetic.
    #
    # What it is NOT free to skip is the REPORT: without the
    # restriction the run says it re-valued every row in the
    # frame when only the surviving fifth reaches the file, and a
    # record that overstates what a pass did is the fault this
    # project refuses everywhere else.
    check("...and the recorded row count is the number of rows "
          "that actually REACH the file, not the number the pass "
          "happened to write over ({} recorded, {} present)"
          .format(rv[0]["rows"] if rv else None, n_pres),
          bool(rv) and rv[0]["rows"] == n_pres)

    # INERT WHERE ITS MECHANISM IS ABSENT, asserted as a digest
    # rather than argued: a blueprint with no copy at all must
    # generate identically with the flag ON.
    pdf = plain_fixture()
    pbp = B.build(pdf, discover(pdf, group_by="person_id"),
                  group_by="person_id")
    check("a blueprint with NO copied column generates "
          "BYTE-IDENTICALLY with the flag ON, so the change acts "
          "only where its mechanism lives",
          not G._copy_partners(pbp)
          and digest(G.generate(pbp, seed=5))
          == digest(G.generate(pbp, seed=5, revalue_copied=True)))

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
