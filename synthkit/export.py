"""One screened file that can leave the machine holding the extract.

WHY THIS EXISTS. The development machine has no clinical data and the
data machine has no authoring on it, so every fix is written blind
against an invented fixture and judged from whatever the operator can
get back. What came back, for four rounds running, was PHOTOGRAPHS OF
A MONITOR - which is how a column got paired with the wrong note, at
an angle, and the pairing had to be withdrawn.

That is the real bottleneck in this project. It is not that the fixes
are hard; it is that the measurements cannot travel. Half of
CONVENTIONS.md exists because a fixture invented here could not
reproduce a shape that only exists there, and the way out of that is
not better guessing - it is getting the MEASUREMENTS off that machine.

WHAT TRAVELS, and why it is allowed to. `fidelity.json`,
`provenance.json` and `findings.txt` are aggregates: a mean, a spread,
a correlation, a count of columns that passed. `blueprint.json` is
NOT included even though it is designed to be publishable, and neither
is `catalogue.json` or `generated.csv` - not because they are unsafe,
but because this file exists to answer "what did the run measure",
and a bundle that quietly grows to hold a generator is a bundle
nobody re-reads before sending.

WHAT THE SCREEN IS FOR. It is not a privacy proof and does not claim
to be one - k-anonymity lives in `blueprint.py` and the disclosure
work lives in the attack scripts. It checks the two ways an
aggregates-only file has actually gone wrong here, and REFUSES on
either:

  A PATH          `provenance.json` records the source by NAME on
                  purpose, because a path on that machine carries a
                  work login. Anything path-shaped is a regression.
  A ROW           every array in these files is per-column or
                  per-column-pair. Nothing is per-ROW. So an array
                  longer than the table has column pairs is either a
                  new per-row field or a leak, and either way a human
                  should look before it travels.

A MISSING BUILD ID IS A WARNING, NOT A REFUSAL, and the first version
of this got that wrong. It refused, which blocked exporting a run
made before the build stamp existed - so the only way to send any
measurement was a fresh 35-minute run, which is the round trip this
module exists to remove. The file is not UNSAFE without a build id;
it is only harder to READ, and this project's own rule for that case
is to say UNKNOWN plainly rather than to refuse. So the warning is
STAMPED INTO THE BUNDLE, where it travels with the numbers and cannot
be lost the way a line of console output can.

The screen REFUSES rather than redacts. Silently removing something
teaches nobody, and the next bundle contains it again."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

PARTS = ("fidelity.json", "provenance.json")
TEXT = ("findings.txt",)

# Windows drive paths, UNC shares and POSIX absolutes. `C:\Users\...`
# is the shape that actually carries the login.
PATHISH = re.compile(r"(?:[A-Za-z]:[\\/])|(?:\\\\[A-Za-z0-9_-]+\\)"
                     r"|(?:/(?:home|Users)/)")


def _walk(o: Any, path: str = ""):
    """Every leaf and every array, with the path that reached it."""
    if isinstance(o, dict):
        for k, v in o.items():
            for x in _walk(v, "{}/{}".format(path, k)):
                yield x
    elif isinstance(o, list):
        yield ("list", path, len(o))
        for i, v in enumerate(o):
            for x in _walk(v, "{}[]".format(path)):
                yield x
    else:
        yield ("leaf", path, o)


def warnings(bundle: Dict[str, Any]) -> List[str]:
    """Reasons this is harder to READ. It still travels."""
    out = []
    build = ((bundle.get("provenance") or {}).get("build") or {})
    if not build.get("id"):
        out.append(
            "NO BUILD ID: this run cannot say which commit produced "
            "it, so its numbers cannot be tied to a known tree. The "
            "measurements are still valid; what is unknown is which "
            "code made them.")
    return out


def screen(bundle: Dict[str, Any]) -> List[str]:
    """Reasons this must not travel. Empty list means it may."""
    problems = []

    # The cap is STRUCTURAL, not a guess: everything in here is one
    # entry per column or per ordered column pair.
    n_col = int(((bundle.get("provenance") or {}).get("source")
                 or {}).get("columns_read") or 0)
    cap = max(n_col * n_col, 64)

    for kind, path, val in _walk(bundle):
        if kind == "list":
            if val > cap:
                problems.append(
                    "{} holds {} entries, more than the {} column "
                    "pairs this table can have - that is a per-ROW "
                    "field, not an aggregate".format(path, val, cap))
        elif isinstance(val, str) and PATHISH.search(val):
            problems.append(
                "{} looks like a filesystem path ({!r}) - the source "
                "is recorded by name on purpose, because a path on "
                "that machine carries a work login"
                .format(path, val[:60]))
    return problems


def collect(run_dir) -> Dict[str, Any]:
    """Read a run directory into one dict. Missing parts are named."""
    d = Path(run_dir)
    out: Dict[str, Any] = {
        "bundle": "synthkit-diagnostics",
        "contains": ("aggregate measurements only - no record, no "
                     "row, and no generated data. See `note` on each "
                     "part for what its numbers mean."),
        "missing": [],
    }
    key = {"fidelity.json": "fidelity",
           "provenance.json": "provenance"}
    for name in PARTS:
        p = d / name
        if p.exists():
            out[key[name]] = json.loads(p.read_text(encoding="utf-8"))
        else:
            out["missing"].append(name)
    for name in TEXT:
        p = d / name
        if p.exists():
            out["findings_txt"] = p.read_text(encoding="utf-8")
        else:
            out["missing"].append(name)
    return out


def build(run_dir) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """(bundle, refusals, warnings). Warnings ride INSIDE the bundle."""
    b = collect(run_dir)
    warn = warnings(b)
    if warn:
        b["warnings"] = warn
    return b, screen(b), warn
