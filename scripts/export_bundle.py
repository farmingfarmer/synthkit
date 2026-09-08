"""Make one screened file the measurements can travel in.

    python scripts/export_bundle.py --run RUNDIR --out diagnostics.json

Run this on the machine holding the extract, after a run. It reads
that run's aggregate artifacts, screens them, and writes ONE file -
about 100KB - that answers "what did the run measure" without
carrying a record, a row, or a path.

If the screen refuses, it names what it found and writes nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from synthkit.export import build           # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # Echo what arrived: this terminal has eaten a flag three times.
    print("run  {}".format(a.run))
    print("out  {}".format(a.out))
    print()

    bundle, problems, warns = build(a.run)
    if bundle.get("missing"):
        print("NOT IN THAT DIRECTORY: {}".format(
            ", ".join(bundle["missing"])))
        print("(a run without --generate writes no fidelity.json)")
        print()
    for w in warns:
        print("WARNING: {}".format(w))
        print("  (stamped into the bundle, so it travels with the "
              "numbers rather than")
        print("   living in a console line somebody has to remember)")
        print()
    if problems:
        print("REFUSED - nothing was written. {} reason(s):".format(
            len(problems)))
        for p in problems:
            print("  - {}".format(p))
        return 1

    Path(a.out).write_text(json.dumps(bundle, indent=1),
                           encoding="utf-8")
    n = Path(a.out).stat().st_size
    print("wrote {} ({:.0f} KB)".format(a.out, n / 1024.0))
    prov = bundle.get("provenance") or {}
    print("build      {}".format(
        (prov.get("build") or {}).get("id") or "UNKNOWN"))
    print("source     {} ({} rows, {} columns)".format(
        (prov.get("source") or {}).get("name"),
        (prov.get("source") or {}).get("rows_read"),
        (prov.get("source") or {}).get("columns_read")))
    print()
    print("This holds aggregate measurements only - no record, no "
          "row, no path, and")
    print("no generated data. It is safe to attach to a message in "
          "the way a chart of")
    print("the same numbers would be; it is not a privacy "
          "certificate, and the release")
    print("decision for a COHORT is still the one in "
          "docs/ and the attack scripts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
