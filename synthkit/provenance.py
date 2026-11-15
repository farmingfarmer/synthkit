"""Can this run be traced to the file that produced it?

WHY THIS EXISTS. A whole session once compared a run against an
extract that never produced it - the file had 23,092 rows and the
run had read 55,428 - and every number looked perfectly
reasonable, which is the whole danger. The guard added afterwards
compares ROW COUNTS, and that is weaker than it reads: the data
machine holds SIX files called `tidy_visits.csv` and all six have
exactly 55,428 rows. A run recorded the BASENAME and nothing
else, so it could not be traced to which of them produced it, and
a paired arm rested on somebody's inference about a directory.

So a run records a CONTENT DIGEST of its source. Two things about
that are deliberate:

A DIGEST OF THE INPUT FILE'S BYTES IS PORTABLE, unlike a digest
of generated output. This project records a pinned digest of a
CSV of floats breaking across a pandas release and across a
Python patch version, because float FORMATTING moved. The bytes
of a file somebody handed us do not move, so the comparison is
exact on any stack.

AND THE PATH IS NOT RECORDED, only the containing directory's
NAME. A full path on the data machine carries the operator's
username, and provenance travels with a run - this project keeps
an explicit list of things that must not travel for exactly that
reason. The directory name is the distinguishing information
without the identifying part.

AND THE GUARD SAYS WHICH TEST IT APPLIED. An older run has no
digest to compare, so the check falls back to the row count -
and a reader who is not told that believes they got the exact
check. Absent is not a mismatch; it is a weaker answer, and it
says so.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

DIGEST_CHARS = 32


def file_digest(path, chunk: int = 1 << 20) -> Optional[str]:
    """sha256 of the file's bytes, truncated, or None."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            while True:
                b = fh.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()[:DIGEST_CHARS]
    except OSError:
        return None


def source_record(path, rows: int, columns: int,
                  patients: Optional[int] = None,
                  max_rows: Optional[int] = None
                  ) -> Dict[str, Any]:
    """What a run should record about the file it read."""
    p = Path(path)
    rec: Dict[str, Any] = {
        "name": p.name,
        "rows_read": int(rows),
        "columns_read": int(columns),
        "max_rows": max_rows or None,
        # the containing directory's NAME, never the path - six
        # files share the basename on the machine that holds the
        # data, and a full path carries a username
        "parent_name": (p.parent.name or None),
        "digest": file_digest(p),
        "digest_note": ("sha256 of the source file's BYTES, "
                        "truncated. Portable across stacks, "
                        "unlike a digest of generated output, "
                        "whose float formatting moves with the "
                        "library version."),
    }
    if patients is not None:
        rec["patients"] = int(patients)
    return rec


def _src_block(prov: Dict[str, Any]) -> Dict[str, Any]:
    s = prov.get("source")
    return s if isinstance(s, dict) else {}


def check_source(run, path, rows: int) -> Tuple[bool, str, str]:
    """Is `path` the file this run was built from?

    Returns (ok, strength, message). `strength` is "exact" when a
    digest was compared, "rows" when only the row count was
    available, and "none" when the run recorded nothing to
    compare - so a caller can print which answer it got rather
    than implying the strong one.
    """
    prov = Path(run) / "provenance.json"
    if not prov.exists():
        return True, "none", ("this run has no provenance.json, "
                              "so nothing can confirm the source")
    try:
        p = json.loads(prov.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True, "none", ("this run's provenance.json cannot "
                              "be read, so nothing can confirm "
                              "the source")
    src = _src_block(p)
    want_d = src.get("digest")
    if want_d:
        got = file_digest(path)
        if got is None:
            return False, "exact", (
                "{} could not be read to compare against the "
                "digest this run recorded".format(Path(path).name))
        if got != want_d:
            where = src.get("parent_name")
            return False, "exact", (
                "{} is NOT the file this run was built from - the "
                "content digests differ. The run read a file of "
                "this name{}; six files of the same name and the "
                "same row count exist on that machine, which is "
                "why the row count alone cannot tell them "
                "apart.".format(
                    Path(path).name,
                    " in a directory called {!r}".format(where)
                    if where else ""))
        return True, "exact", (
            "source confirmed by content digest{}".format(
                " (directory {!r})".format(src["parent_name"])
                if src.get("parent_name") else ""))

    want_r = src.get("rows_read") or p.get("rows_read")
    if want_r:
        if int(want_r) != int(rows):
            return False, "rows", (
                "{} has {} rows but this run recorded reading {} "
                "- that is not the extract this run was built "
                "from".format(Path(path).name, int(rows),
                              int(want_r)))
        return True, "rows", (
            "source matches on ROW COUNT only - this run predates "
            "the content digest, and several files can share a "
            "row count, so this is weaker than an exact match")
    return True, "none", ("this run recorded nothing about its "
                          "source that can be compared")


def guard(run, path, rows: int, out=None) -> bool:
    """Print the verdict and return whether to proceed."""
    ok, strength, msg = check_source(run, path, rows)
    line = ("  source: {}".format(msg) if ok
            else "STOPPED: {}".format(msg))
    if out is None:
        import sys
        out = sys.stdout if ok else sys.stderr
    print(line, file=out)
    if ok and strength == "rows":
        print("  (pass a run made since the digest was recorded "
              "for an exact check)", file=out)
    return ok
