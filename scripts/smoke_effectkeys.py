"""Every field discovery sets on an effect must survive into the
blueprint, or be dropped BY NAME.

WHY THIS EXISTS. `blueprint.build` WHITELISTS the effect fields: it
constructs each parent's effect dict key by key. So a field
discovery computes and stores is dropped in SILENCE - no error, no
absence note, just a missing key. The record carries it:

    "The weight was solved, stored on the catalogue, and never
     reached the blueprint on its first run - no error, no absence
     note, just a missing key. That is the value-wired-to-nothing
     fault in a new place ... what they need is a check asserting
     that every key discovery sets on an effect survives into the
     blueprint."

It was recorded rather than fixed, because the field that exposed
it had been reverted. This is that check.

IT IS NOT "every name matches". One field is deliberately RENAMED -
discovery's `class_of_interest` is published as `of_class` - and a
check that forbade renames would be asserting a convention nobody
agreed to. What it forbids is a field that is NEITHER published NOR
on an explicit list, so adding one to discovery turns this red with
the field named, and the author then has to choose: publish it,
rename it, or say in the list why it stays behind.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import synthkit.blueprint as B  # noqa: E402
import synthkit.discover as D  # noqa: E402

PASS = FAIL = 0

# DISCOVERY'S NAME -> THE BLUEPRINT'S. A rename is a decision, so it
# is written down rather than inferred from a near match.
RENAMED = {"class_of_interest": "of_class"}

# FIELDS DELIBERATELY NOT PUBLISHED, each with the reason, because
# "it is not in the blueprint" and "we decided not to publish it"
# are different facts and only one of them is a defect.
NOT_PUBLISHED = {
    # `monotone` is a BOOLEAN restating what `shape` already says -
    # "rising" and "saturating" are monotone, "u-shaped" and
    # "non-monotone" are not - and nothing reads it off a published
    # blueprint. Publishing it would add a field to every artifact
    # that no consumer consults, which is the value-wired-to-nothing
    # fault pointed the other way. Derived, not dropped.
    "monotone": "restates `shape`; no consumer reads it off the "
                "blueprint",
}


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def frame(seed=7, n_pat=90, n_vis=9):
    """Shapes chosen so the effect dicts are richly populated.

    A frame whose curves are all straight lines publishes a thin
    effect and the check then passes by finding little - which is
    this project's worst kind of clean. So there is a bend, a
    threshold, an interaction and a partially-covered parent.

    And a CATEGORICAL parent, because `grid_kind` is the field this
    check found missing and the only curve whose grid is not
    numeric is one over a categorical column. Without it the two
    readings of `measure_shapes` below agree and prove nothing -
    a fixture must contain the thing the check is about, which is
    the mistake this suite made on its first run.
    """
    rng = np.random.RandomState(seed)
    rows = []
    for p in range(n_pat):
        lvl = rng.normal(0, 1)
        for v in range(n_vis):
            a = rng.normal(10 + 2 * lvl, 2)
            b = rng.normal(0, 1)
            c = rng.choice([0.0, 1.0])
            arm = str(rng.choice(["ctl", "lo", "hi"]))
            bend = -0.35 * (a - 10) ** 2
            step = 4.0 if b > 0.4 else 0.0
            byarm = {"ctl": 0.0, "lo": 3.0, "hi": 7.0}[arm]
            rows.append({
                "person_id": "P%03d" % p,
                "visit": v,
                "a": a,
                "b": b,
                "c": c,
                # a CATEGORICAL parent: `arm` shifts y by level, so
                # discovery publishes a curve whose grid holds
                # LABELS rather than numbers
                "arm": arm,
                # a bend in `a`, a threshold in `b`, a product
                "y": float(bend + step + 3.0 * b * c + byarm
                           + rng.normal(0, 1)),
                # present on about half the rows, so the
                # absent-side response has something to describe
                "sparse": (float(rng.normal(5, 1))
                           if rng.rand() < 0.5 else np.nan),
                "z": float(2.0 * a + rng.normal(0, 1)),
            })
    return pd.DataFrame(rows)


def effect_keys_from(cat):
    """Every key any predictor's effect dict carries."""
    seen = set()
    for cl in cat.get("claims") or []:
        for p in cl.get("predictors") or []:
            eff = p.get("effect")
            if isinstance(eff, dict):
                seen.update(eff.keys())
    return seen


def published_keys_from(bp):
    seen = set()
    for r in bp.get("relationships") or []:
        for _, eff in ((r.get("evidence") or {}).get("effect")
                       or {}).items():
            if isinstance(eff, dict):
                seen.update(eff.keys())
    return seen


def main():
    df = frame()
    cat = D.discover(df, group_by="person_id")
    bp = B.build(df, cat, group_by="person_id", time_col="visit")

    dkeys = effect_keys_from(cat)
    pkeys = published_keys_from(bp)

    # THE CHECK CANNOT PASS BY FINDING NOTHING.
    check("discovery publishes effect dicts at all on this frame, "
          "without which every assertion below would hold "
          "vacuously: {} field(s)".format(len(dkeys)),
          len(dkeys) >= 6)
    check("...and the blueprint publishes effect dicts at all: {} "
          "field(s)".format(len(pkeys)),
          len(pkeys) >= 6)

    # THE PROPERTY. Each discovery field is published under its own
    # name, published under a DECLARED new name, or on the
    # deliberately-not-published list with a reason.
    unaccounted = sorted(
        k for k in dkeys
        if k not in pkeys
        and RENAMED.get(k) not in pkeys
        and k not in NOT_PUBLISHED)
    check("every field discovery sets on an effect reaches the "
          "blueprint, is renamed by declaration, or is on the "
          "not-published list - the whitelist in `build` drops an "
          "unlisted one in SILENCE: {}".format(
              ", ".join(unaccounted) or "all accounted for"),
          not unaccounted)

    # THE RENAME IS ASSERTED IN BOTH DIRECTIONS, so the map cannot
    # go stale: a declared rename whose SOURCE discovery no longer
    # sets is a line nobody will remove, and one whose TARGET the
    # blueprint no longer carries is a hole the map hides.
    for src, dst in RENAMED.items():
        check("the declared rename {} -> {} is live on both sides, "
              "so the map is a statement about today's code rather "
              "than a comment about an old one".format(src, dst),
              src in dkeys and dst in pkeys)

    # THE REVERSE DIRECTION IS REPORTED AND NOT ASSERTED. A key the
    # blueprint carries and discovery does not set is legitimate -
    # `build` computes some itself - so this prints rather than
    # fails, because a check that forbade it would forbid the
    # blueprint doing its own work.
    extra = sorted(k for k in pkeys if k not in dkeys
                   and k not in RENAMED.values())
    print("    note: {} blueprint-only effect field(s): {}".format(
        len(extra), ", ".join(extra) or "none"))

    # AND THE CHECK MUST BE ABLE TO FAIL. Demonstrated rather than
    # claimed: a field planted on the catalogue's own effect dicts
    # must come back unaccounted, because `build` has no line for
    # it.
    planted = "a_field_nobody_published"
    for cl in cat.get("claims") or []:
        for p in cl.get("predictors") or []:
            if isinstance(p.get("effect"), dict):
                p["effect"][planted] = 1.0
    bp2 = B.build(df, cat, group_by="person_id", time_col="visit")
    pkeys2 = published_keys_from(bp2)
    dkeys2 = effect_keys_from(cat)
    check("the check CATCHES the fault it was written for - a field "
          "planted on discovery's own effect dict is not published "
          "and is named as unaccounted",
          planted in dkeys2 and planted not in pkeys2)

    # AND THE FIELD THAT WAS FOUND MISSING IS ASSERTED TO MOVE A
    # MEASUREMENT, because a field published and read by nobody is
    # the same fault pointed the other way. `grid_kind` is read by
    # `measure_shapes` and by the deck, both defaulting to
    # "numeric" when absent - so with it absent the skip for a
    # CATEGORICAL parent could not fire and those curves were
    # absorbed by `skipped_thin_rows`, sending a reader after a
    # thin cohort when the cause is a parent nobody can bin.
    import copy

    import synthkit.curvecheck as CC

    gid = df["person_id"]
    with_kind = CC.measure_shapes(df, df, bp, gid, 10)
    stripped = copy.deepcopy(bp)
    for rel in stripped.get("relationships") or []:
        for _, e in ((rel.get("evidence") or {}).get("effect")
                     or {}).items():
            e.pop("grid_kind", None)
    without = CC.measure_shapes(df, df, stripped, gid, 10)

    check("the fixture HAS a curve over a categorical parent, "
          "without which the two readings below would agree and "
          "prove nothing",
          (with_kind.get("skipped_categorical_parent") or 0) >= 1)
    check("publishing grid_kind moves the skip OUT of thin-rows and "
          "into its own named reason - {} thin / {} categorical "
          "with it, against {} thin / {} without".format(
              with_kind["skipped_thin_rows"],
              with_kind.get("skipped_categorical_parent"),
              without["skipped_thin_rows"],
              without.get("skipped_categorical_parent") or 0),
          (without["skipped_thin_rows"]
           - with_kind["skipped_thin_rows"]
           == (with_kind.get("skipped_categorical_parent") or 0)))

    for label, r in (("with", with_kind), ("without", without)):
        tot = (r["compared"] + (r.get("not_readable") or 0)
               + r["skipped_scaffolding"]
               + r["skipped_absent_columns"]
               + r["skipped_thin_rows"] + r["skipped_few_bins"]
               + (r.get("skipped_categorical_parent") or 0))
        check("the shapes identity still balances {} the field - "
              "published {} against the buckets' {} - so a new skip "
              "reason did not leave a term unaccounted".format(
                  label, r["published"], tot),
              tot == r["published"])

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
