"""Find the extract a run was actually built from, by row count.

WHY THIS EXISTS. A run records how many rows it read.
`run_deliverable` recorded **55,428** and the file at the path
everyone has been passing holds **23,092** - so a whole session
of measurement compared a generated file against an extract
that never produced it, and every number looked reasonable.
Finding the right file is a row count away, and nothing had
ever gone looking.

WHAT IT REPORTS, AND WHAT IT DELIBERATELY DOES NOT. Paths,
sizes and ROW COUNTS. It never opens a cell, never prints a
value, and never reads a header - a tool that goes looking
through a machine holding real extracts has no business
printing their contents, and counting newlines needs none of
it.

It scans in binary and counts `\\n`, which is wrong for a file
with embedded newlines inside quoted fields - a note column can
do that. So the count is reported as an UPPER BOUND on rows and
a match is called APPROXIMATE unless it is exact. Saying which
beats quietly rounding.
"""
import os
import sys
from pathlib import Path

WORDS = {"0": "zero", "1": "one", "2": "two", "3": "three",
         "4": "four", "5": "five", "6": "six", "7": "seven",
         "8": "eight", "9": "nine", "-": "minus", ",": ""}


def sayn(n):
    """Digits with their spelling - the OCR channel eats zeros."""
    s = "{:,}".format(int(n))
    return "{} [{}]".format(s, " ".join(
        w for w in (WORDS.get(c, c) for c in s) if w))


def count_rows(path, cap_bytes=0):
    """Newlines, minus the header. An upper bound on rows."""
    n = 0
    try:
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(1 << 22)
                if not chunk:
                    break
                n += chunk.count(b"\n")
    except OSError:
        return None
    return max(0, n - 1)


def main(argv):
    target = None
    roots = []
    tol = 0
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--target" and i + 1 < len(argv):
            target = int(argv[i + 1].replace(",", ""))
            i += 2
            continue
        if a == "--tol" and i + 1 < len(argv):
            tol = int(argv[i + 1])
            i += 2
            continue
        roots.append(a)
        i += 1
    if target is None:
        print(__doc__)
        print("usage: find_extract.py ROOT [ROOT...] "
              "--target 55428 [--tol 50]")
        return 2
    if not roots:
        roots = [str(Path.home())]

    print("LOOKING FOR AN EXTRACT OF {} ROWS".format(sayn(target)))
    print("  scanning: {}".format(", ".join(roots)))
    print("  reporting PATHS, SIZES and ROW COUNTS only - no "
          "cell is ever opened")
    print("")

    seen = 0
    hits = []
    big = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            # DO NOT DESCEND INTO MACHINERY. A virtualenv or a
            # git object store holds thousands of files and no
            # extract, and walking them is the difference
            # between a minute and an hour.
            dirnames[:] = [d for d in dirnames
                           if d not in (".git", "__pycache__",
                                        "node_modules")
                           and not d.startswith(".venv")]
            for fn in filenames:
                if not fn.lower().endswith(
                        (".csv", ".tsv", ".txt")):
                    continue
                p = Path(dirpath) / fn
                try:
                    size = p.stat().st_size
                except OSError:
                    continue
                seen += 1
                # Too small to hold the target at any sane width.
                if size < max(1, target // 20):
                    continue
                rows = count_rows(p)
                if rows is None:
                    continue
                big.append((size, rows, p))
                if abs(rows - target) <= max(tol, 0):
                    hits.append((rows, size, p, rows == target))

    print("  scanned {} delimited file(s)".format(sayn(seen)))
    print("")
    if hits:
        print("  MATCHES:")
        for rows, size, p, exact in sorted(hits):
            print("    {}   {} MB   {}".format(
                sayn(rows), round(size / 1e6, 1), p))
            print("      {}".format(
                "EXACT" if exact
                else "APPROXIMATE - newlines inside quoted "
                     "fields make this an upper bound"))
    else:
        print("  NO FILE MATCHES that row count.")
        print("  The nearest by row count, so the gap is "
              "visible rather than just absent:")
        for size, rows, p in sorted(
                big, key=lambda t: abs(t[1] - target))[:8]:
            print("    {}   {} MB   {}".format(
                sayn(rows), round(size / 1e6, 1), p))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
