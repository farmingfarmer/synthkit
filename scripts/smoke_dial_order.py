"""A dial set AFTER the stage it configures reaches nothing.

WHY THIS EXISTS. `--extra-knots` was wired with the generate-time
flags, so `blueprint.EXTRA_KNOTS` was assigned at line 940 while
`B.build` runs at 679. The run printed "publishing 9 extra interior
quantile(s)", the blueprint was built without them, and a real arm
on the 800-patient extract came back IDENTICAL to its control on
every column - about half an hour of machine time spent measuring a
dial that never engaged.

Nothing could have caught it. The flag's own line said it fired,
and every check in its suite set the module constant DIRECTLY
rather than through the CLI, so all sixteen stayed green.

This is a STATIC check over `pipeline.py`: an assignment to a
BLUEPRINT module constant must appear before the blueprint is
built, and an assignment to a GENERATE module constant before
generate is called. It is cheap, it is exact about the fault, and
it does not need a fixture that reaches each dial's mechanism -
which is the thing that makes a per-dial behavioural audit hard.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


# The import alias a block uses is arbitrary (`_bp`, `_bp_xk`,
# `_gen`), so the STAGE is identified by what the alias was bound
# to, not by its name.
BP_IMPORT = re.compile(
    r"import\s+synthkit\.blueprint\s+as\s+(\w+)")
GEN_IMPORT = re.compile(
    r"import\s+synthkit\.generate\s+as\s+(\w+)")
ASSIGN = re.compile(r"^\s*(\w+)\.([A-Z][A-Z0-9_]*)\s*=", re.M)


def main():
    src = (ROOT / "synthkit" / "pipeline.py").read_text(
        encoding="utf-8")

    bp_aliases = set(BP_IMPORT.findall(src))
    gen_aliases = set(GEN_IMPORT.findall(src))
    check("pipeline.py imports the blueprint module under at "
          "least one alias, without which this audit would pass "
          "by finding nothing",
          bool(bp_aliases))
    check("...and the generate module likewise",
          bool(gen_aliases))

    build_at = src.find("bp = B.build(df, cat,")
    gen_at = src.find("g = generate(bp,")
    check("the blueprint BUILD site is found - the audit is "
          "anchored on it, so a rename must break this rather "
          "than silently disable it",
          build_at > 0)
    check("...and the GENERATE call site likewise", gen_at > 0)

    late_bp, late_gen = [], []
    for m in ASSIGN.finditer(src):
        alias, const = m.group(1), m.group(2)
        at = m.start()
        if alias in bp_aliases and at > build_at:
            late_bp.append((const, at))
        if alias in gen_aliases and at > gen_at:
            late_gen.append((const, at))

    check("every BLUEPRINT dial is set before the blueprint is "
          "built - one set after it prints its message and "
          "reaches nothing: {}".format(
              ", ".join(c for c, _ in late_bp) or "none late"),
          not late_bp)
    check("every GENERATE dial is set before generate is called: "
          "{}".format(
              ", ".join(c for c, _ in late_gen) or "none late"),
          not late_gen)

    # THE AUDIT MUST BE ABLE TO FIRE. A static scan that matches
    # nothing passes trivially, which this project calls the worst
    # kind of clean.
    early_bp = [m.group(2) for m in ASSIGN.finditer(src)
                if m.group(1) in bp_aliases and m.start() < build_at]
    early_gen = [m.group(2) for m in ASSIGN.finditer(src)
                 if m.group(1) in gen_aliases and m.start() < gen_at]
    check("the scan FINDS blueprint dials at all, so a pass means "
          "they are early rather than that none were seen: {}"
          .format(", ".join(sorted(set(early_bp)))),
          len(early_bp) >= 2)
    check("...and generate dials likewise: {}".format(
        ", ".join(sorted(set(early_gen)))),
          len(early_gen) >= 2)

    # AND IT MUST CATCH THE REAL FAULT, demonstrated rather than
    # claimed: the same scan run over a copy with one assignment
    # moved after the build.
    broken = src.replace(
        "        _bp_xk.EXTRA_KNOTS = _xk", "        pass", 1)
    ins = broken.find("    say(\"generating\")")
    broken = (broken[:ins]
              + "    _bp_xk.EXTRA_KNOTS = _xk\n"
              + broken[ins:])
    b_build = broken.find("bp = B.build(df, cat,")
    caught = [m.group(2) for m in ASSIGN.finditer(broken)
              if m.group(1) in bp_aliases and m.start() > b_build]
    check("the audit CATCHES the fault it was written for - the "
          "same scan over a copy with EXTRA_KNOTS moved after the "
          "build names it",
          "EXTRA_KNOTS" in caught)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
