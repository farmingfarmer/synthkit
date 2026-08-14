"""Which commit produced this run.

WHY THIS EXISTS. The machine holding the extract is not a git
checkout. `docs/WINDOWS.md` section 1 is "Pull the repo (no git
needed)": it downloads a zipball, extracts it and renames the folder,
so `git log` there returns nothing and there is no way to ask the
machine what it is running.

That matters because the release gate is "known working". Code is
authored where there is no data, proven where there is, and only then
published. A run that proves a fix has to name the tree it proved, or
the gate is a guess - and a stale extract of the previous build looks
exactly like a current one. Several rounds of output were read against
fixes the running code did not yet contain, and nothing in the run
said so.

`gui.build_fingerprint` already hashes every module's bytes, which
answers "are these two runs the same code". It cannot answer "which
commit", because a content hash has no history behind it. The two are
complements: the fingerprint detects drift, this names the build.

WHERE THE ANSWER COMES FROM, in order:

  git        `git rev-parse HEAD`, on the development machine, where
             the checkout is authoritative and no file can go stale
  zipball    `BUILD_ID.txt`, written by the sync ritual on the data
             machine from the extracted folder's name - GitHub stamps
             the commit into it, and the rename in that ritual was
             throwing the only provenance the machine ever receives
  unknown    said plainly. A build id that quietly reports something
             is worse than one that reports nothing: the whole point
             is to be believed.

THE OWNER IS DELIBERATELY DISCARDED. That folder is named
`<owner>-<repo>-<sha>`, and provenance.json travels with the output to
people who should not receive a personal account handle - the same
reason the sidecar records the source file by NAME and never by path,
since a path on that machine carries a work login. Only the sha is
identifying, and only the sha is kept.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

# The filename the sync ritual writes at the repo root. Gitignored:
# on a checkout it would be stale the moment anything is committed,
# and git is the better answer there anyway.
STAMP = "BUILD_ID.txt"

# Enough to identify a commit, short enough to read in a log line.
# GitHub's zipball folder and `git rev-parse` do not agree on length,
# so two ids are compared on their COMMON PREFIX, not for equality.
WIDTH = 12


def _from_git(root: Path) -> Optional[str]:
    try:
        r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                           capture_output=True, text=True)
    except (OSError, ValueError):
        return None
    if r.returncode != 0:
        return None
    sha = (r.stdout or "").strip()
    return sha[:WIDTH] or None


def _from_stamp(root: Path) -> Optional[str]:
    """The sha out of `<owner>-<repo>-<sha>`, and nothing else.

    Split from the RIGHT once: a repository name may contain hyphens
    and the owner certainly may, but the sha is always the last field
    and never contains one."""
    p = root / STAMP
    try:
        raw = p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    line = raw.strip().splitlines()[0].strip() if raw.strip() else ""
    if not line:
        return None
    sha = line.rsplit("-", 1)[-1].strip()
    if not sha:
        return None
    return sha[:WIDTH]


def build_id(root: Optional[Path] = None) -> Dict[str, Any]:
    """`{"source": git|zipball|unknown, "id": str|None, "note": str}`."""
    root = Path(root) if root is not None \
        else Path(__file__).resolve().parent.parent

    sha = _from_git(root)
    if sha:
        return {"source": "git", "id": sha,
                "note": "git rev-parse HEAD on the machine that ran it"}

    sha = _from_stamp(root)
    if sha:
        return {"source": "zipball", "id": sha,
                "note": "{} written by the sync ritual from the "
                        "extracted folder's name; the owner and repo "
                        "are discarded on purpose".format(STAMP)}

    return {"source": "unknown", "id": None,
            "note": "NOT A GIT CHECKOUT AND NO {} - this run cannot "
                    "say which commit produced it. Re-run the sync "
                    "ritual in docs/WINDOWS.md section 1, which writes "
                    "that file.".format(STAMP)}


def describe(root: Optional[Path] = None) -> str:
    """One line for the top of a run, where the settings are echoed."""
    b = build_id(root)
    if b["id"] is None:
        return "build: UNKNOWN - this run cannot say which commit " \
               "produced it (no git, no {})".format(STAMP)
    return "build: {} (from {})".format(b["id"], b["source"])
