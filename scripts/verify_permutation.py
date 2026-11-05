"""Is this file a REARRANGEMENT of that one, or different data?

WHY THIS EXISTS. Everything the repair and the output dials
claim rests on one property: the file they write holds the same
values as the file they read, in a different order. That is
what makes the privacy posture survive a repair - the published
bounds, the point masses and the k-anonymous extremes are
exactly what they were, because reordering values already
released discloses nothing new - and it is what makes a cipher
able to undo the change at all.

It was checked BY HAND. A permutation claim is the kind of
sentence that sounds obviously true and is wrong whenever an
arithmetic step crept in: this project has already caught a
dial re-rendering 22.29 as 22.289999999999999, which is not a
rearrangement of anything. So the claim gets a tool.

WHAT IT CHECKS, AND IN WHICH ORDER:
  ROWS-SAME        the file did not gain or lose a row
  COLS-SAME        no column appeared or disappeared
  CHANGED          exactly which columns differ, by name
  MULTISET-SAME    each changed column holds the same values
  ORDER-CHANGED    ...and genuinely in a different order
  OTHERS-IDENTICAL every untouched column byte-for-byte

BOTH SIDES ARE READ FROM DISK, as strings. Comparing a written
file against an in-memory frame reads `array_equal False` on a
file that is a perfect permutation, because a round trip moves
the last bits of a float - the claim being made is about two
files, so it is measured between two files.

A column whose values all happen to be equal cannot show
ORDER-CHANGED, and saying so beats quietly passing: a constant
column is reported as UNORDERABLE rather than counted as proof.
"""
import sys
from pathlib import Path


def compare(a_path, b_path):
    """Returns (verdict, lines). Verdict is True only if every
    difference is a pure rearrangement."""
    import pandas as pd
    rd = dict(dtype=str, keep_default_na=False,
              encoding="utf-8-sig", low_memory=False)
    a = pd.read_csv(a_path, **rd)
    b = pd.read_csv(b_path, **rd)
    lines, ok = [], True

    rows_same = len(a) == len(b)
    lines.append("ROWS-SAME        {}  ({} and {})".format(
        rows_same, len(a), len(b)))
    cols_same = list(a.columns) == list(b.columns)
    lines.append("COLS-SAME        {}".format(cols_same))
    if not (rows_same and cols_same):
        lines.append("STOP: a file that gained or lost rows or "
                     "columns is not a rearrangement of "
                     "anything, and nothing below would mean "
                     "what it says.")
        return False, lines

    changed = [c for c in a.columns if not a[c].equals(b[c])]
    lines.append("CHANGED          {}".format(
        ", ".join(changed) if changed else "(nothing)"))
    if not changed:
        lines.append("the two files are IDENTICAL - which is a "
                     "pass for the permutation question and "
                     "means the operation changed nothing")
        return True, lines

    for c in changed:
        same_multiset = sorted(a[c]) == sorted(b[c])
        ok = ok and same_multiset
        lines.append("  {:<28} MULTISET-SAME {}".format(
            c, same_multiset))
        if not same_multiset:
            # NAME WHAT APPEARED. "not a permutation" sends
            # somebody reading the whole column; a value that is
            # in one file and not the other names the step that
            # introduced it - a re-render, a rounding, a clip.
            extra = sorted(set(b[c]) - set(a[c]))[:3]
            if extra:
                lines.append(
                    "  {:<28} VALUES NOT IN THE ORIGINAL: {}"
                    .format("", ", ".join(repr(x)
                                          for x in extra)))
        if len(set(a[c])) <= 1:
            lines.append("  {:<28} ORDER-CHANGED UNORDERABLE - "
                         "the column is constant, so a "
                         "reordering cannot be shown".format(""))
        else:
            lines.append("  {:<28} ORDER-CHANGED {}".format(
                "", True))

    untouched = [c for c in a.columns if c not in changed]
    lines.append("OTHERS-IDENTICAL {}  ({} column(s))".format(
        True, len(untouched)))
    lines.append("")
    lines.append(
        "VERDICT: {}".format(
            "every difference is a PURE REARRANGEMENT - the "
            "values released are exactly the values that were "
            "published, in a different order"
            if ok else
            "NOT a permutation - at least one column holds "
            "values the original does not, so the published "
            "bounds and point masses are no longer what this "
            "file was graded on"))
    return ok, lines


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        print("usage: verify_permutation.py ORIGINAL.csv "
              "REARRANGED.csv")
        return 2
    for p in argv[:2]:
        if not Path(p).exists():
            print("STOP: {} does not exist".format(p))
            return 1
    ok, lines = compare(argv[0], argv[1])
    for ln in lines:
        print(ln)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
