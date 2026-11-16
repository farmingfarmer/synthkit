"""The surface census, which printed a NEGATIVE count on a real run.

`peek surfaces` reports `published = measured + token + other`.
It counted every surface touching a token indicator as
unmeasurable - true while `compare` refused those columns, and
false since the blueprint's own vocabulary is used to rebuild
them. On a real 800-patient run it then printed:

    73 surface(s) published: 47 measured, 64 on token-indicator
    columns ..., -38 otherwise unmeasured

The arithmetic balances because one term is NEGATIVE, which is
this project's own totals rule satisfied by a number that cannot
exist. A reader doing the sum would have found it consistent.

This suite holds the census to two properties: the categories are
DISJOINT, and no count can be negative. The view had no coverage
of any kind before this - the same gap `peek.py` has already been
caught with once.
"""
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import peek as PK                                       # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def surf(child, a, b):
    return {"child": child,
            "parents": [a, b],
            "evidence": {"interaction": {
                "pair": [a, b],
                "grid_a": [0.0, 1.0], "grid_b": [0.0, 1.0],
                "response": [[0.0, 1.0], [1.0, 0.0]]}}}


def plant(d, rels, measured, unreadable=None, name="r"):
    run = Path(d) / name
    run.mkdir(parents=True, exist_ok=True)
    (run / "blueprint.json").write_text(
        json.dumps({"relationships": rels, "columns": {}}),
        encoding="utf-8")
    (run / "fidelity.json").write_text(json.dumps({
        "surfaces": {"surfaces": measured,
                     "unreadable": unreadable or []},
        "generation": {"edges_dropped": []}}), encoding="utf-8")
    return str(run)


def census(out):
    """(published, measured, not_readable, token, other).

    NOT READABLE is its own term because it is a THIRD answer: a
    surface whose own sampling noise reaches the bar is neither
    measured nor an instrument gap, and folding it into the
    residual reads as "unmeasured for some other reason" - the
    bare question mark this view exists to remove.
    """
    for ln in out.splitlines():
        if "surface(s) published:" in ln:
            import re
            n = re.findall(r"(-?\d+)", ln)
            # The 0.35 in the surrounding prose is not a count.
            n = [x for x in n if "." not in x]
            return tuple(int(x) for x in n[:5])
    return None


def main():
    with TemporaryDirectory() as d:
        # A TOKEN SURFACE THAT WAS MEASURED ANYWAY. This is the
        # exact shape that produced the negative: since the
        # vocabulary fix a `__has__` pair CAN be compared, so
        # counting it as unmeasurable double-counts it.
        rels = [surf("x", "a", "b"),
                surf("s__has__t", "s__has__u", "a"),
                surf("y", "s__has__v", "b")]
        measured = [
            {"child": "x", "pair": ["a", "b"],
             "gap_sd": 0.1, "tracks": True},
            {"child": "s__has__t", "pair": ["a", "s__has__u"],
             "gap_sd": 0.2, "tracks": True}]
        run = plant(d, rels, measured)
        buf = io.StringIO()
        with redirect_stdout(buf):
            PK.surfaces_view(run)
        out = buf.getvalue()
        c = census(out)
        check("the census line is printed at all, so this suite is "
              "not passing on a view that produced nothing",
              c is not None and len(c) == 5)
        pub, meas, unread, tok, other = c
        check("published counts every surface in the blueprint "
              "({} of 3)".format(pub), pub == 3)
        check("measured counts the ones with a gap record ({} of "
              "2), including the TOKEN surface that was compared - "
              "a surface that was measured is not an instrument "
              "gap whatever its columns are called".format(meas),
              meas == 2)
        check("the token bucket holds only the UNMEASURED "
              "token-touching surface ({}), so the categories are "
              "disjoint".format(tok), tok == 1)
        check("...and the residual is therefore ZERO rather than "
              "negative ({})".format(other), other == 0)
        check("no surface is unreadable in this run, and that term "
              "reads ZERO rather than being absent - a term that "
              "vanishes cannot be summed against the total ({})"
              .format(unread), unread == 0)
        check("the categories sum to published, which is "
              "the totals rule and is only meaningful once no "
              "term can be negative",
              meas + unread + tok + other == pub)
        check("and no term is negative, which is the fault this "
              "suite exists for - a real run printed -38",
              min(c) >= 0)

        # THE OLD BEHAVIOUR, REPRODUCED, so the fix is shown to
        # correct something rather than asserted to.
        _tok_old = sum(
            1 for r in rels
            if any(x.startswith("s__has__")
                   for x in ([r["child"]]
                             + r["evidence"]["interaction"]["pair"])))
        check("counting EVERY token-touching surface as "
              "unmeasurable - the old rule - gives {} and a "
              "residual of {}, which is the negative the real run "
              "printed".format(_tok_old, pub - meas - _tok_old),
              pub - meas - _tok_old < 0)

        # AND A RUN WITH NO SURFACES SAYS SO rather than printing
        # a census of zeros that reads as a measurement.
        run2 = plant(d, [], [])
        buf = io.StringIO()
        with redirect_stdout(buf):
            PK.surfaces_view(run2)
        out2 = buf.getvalue()
        check("a blueprint publishing no surface says so in "
              "words, because a census of zeros reads as a "
              "measurement",
              "no interaction surfaces are published" in out2)

        # AND THE NEW TERM MUST FIRE, because a term asserted
        # only at ZERO is untested - the whole point of counting
        # it apart is that it can be non-empty, and a run where it
        # is empty cannot show the arithmetic still balances.
        run3 = plant(d, rels,
                     [{"child": "x", "pair": ["a", "b"],
                       "gap_sd": 0.1, "tracks": True}],
                     unreadable=[{"child": "s__has__t",
                                  "pair": ["a", "s__has__u"],
                                  "cells_compared": 9,
                                  "noise_floor_sd": 0.42}],
                     name="r3")
        buf = io.StringIO()
        with redirect_stdout(buf):
            PK.surfaces_view(run3)
        out3 = buf.getvalue()
        c3 = census(out3)
        check("with a surface the measure could not READ, the new "
              "term is NON-ZERO and the identity still holds - "
              "published {} = measured {} + not readable {} + "
              "token {} + other {}".format(*c3) if c3 else "no "
              "census line",
              c3 is not None
              and c3[2] == 1
              and c3[1] + c3[2] + c3[3] + c3[4] == c3[0]
              and min(c3) >= 0)
        check("...and it is NOT double counted in the token "
              "bucket, which would make the residual negative "
              "again - the categories stay disjoint with a third "
              "term in them",
              c3 is not None and "AND THAT RESIDUAL IS NEGATIVE"
              not in out3)
        check("...and the surface itself is named NOT READABLE "
              "with its own floor, so a reader is told WHY rather "
              "than finding it in a residual",
              "NOT READABLE" in out3 and "0.42" in out3)

        # THE NEGATIVE IS CALLED OUT IF IT EVER HAPPENS AGAIN.
        src = io.open(ROOT / "scripts" / "peek.py",
                      encoding="utf-8").read()
        check("the view refuses to let a negative residual pass "
              "silently if the categories ever overlap again",
              "RESIDUAL IS NEGATIVE" in src)

        # THE STALE REASON. The per-surface line used to say a
        # token column "exists only inside the search, never in
        # the written files" - true before the blueprint's own
        # vocabulary was used to rebuild them, and the reason the
        # bucket went from 67 to a handful.
        check("...and an unmeasured token surface is explained by "
              "the OUTPUT carrying too few of that token, not by "
              "the column being unrebuildable",
              "carries too few of" in out)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
