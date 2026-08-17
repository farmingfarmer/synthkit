"""Moved into the package; this is a shim over `synthkit.pipeline`.

    synthkit fit --src TIDY.csv --out DIR        (preferred)
    python scripts/run_discovery.py --src ...    (still works)

WHY THE SHIM STAYS. `docs/WINDOWS.md` names this path, and that ritual
is pasted by hand into a terminal that has mangled commands three
times; changing it and the guide together would strand any operator
running an older copy of the guide against a newer checkout. Several
smoke suites also import helpers from here by name.

WHY THE MODULE MOVED. `scripts/` is not installed. Everything here was
reachable only from a git checkout, so `pip install synthkit` shipped
every part of the product except the one the README leads with.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Re-exported for the suites that import them by name. Underscored
# names do not travel through `import *`, so they are listed.
from synthkit.pipeline import (compare, render,          # noqa: E402,F401
                               render_verdicts, verdicts,
                               main, say, die,
                               _pair_fidelity, _will_drop_full,
                               _report_types)

if __name__ == "__main__":
    sys.exit(main())
