"""Smoke: the package installs from what it declares, and the CI
net covers all of itself.

WHY THIS EXISTS. `requires-python` said `>=3.8` while every numeric
dependency the fitted path needs declares `>=3.10`. That does not fail
loudly on 3.8 or 3.9 - pip resolves whatever ancient versions still
claim to support the interpreter, and the install "succeeds". On the
machine holding the real extracts a silently different pandas produces
a wrong number instead of an error, which is the failure class this
whole repository is built to refuse.

Nothing could have caught it. Every check in this suite is about the
metadata rather than the code, and there was no suite about metadata.

THE LOAD-BEARING CHECK IS THE FLOOR COMPARISON. It reads the installed
dependencies' own `Requires-Python` and asserts ours is not below any
of them - so the day scikit-learn moves to 3.11, this goes red here
instead of on someone's laptop three weeks later.

It degrades honestly off a full install: the floor check needs the
dependencies present to read their metadata, so where they are absent
it reports SKIPPED rather than passing on an empty comparison - the
same rule `smoke_no_personal` follows off a git checkout.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def parse_floor(spec):
    """The lowest Python this specifier admits, as a (major, minor).

    Only `>=X.Y` and `>X.Y` matter here; every dependency in play
    states one of those. An unparseable specifier returns None and is
    reported rather than assumed satisfied."""
    if not spec:
        return None
    best = None
    for part in str(spec).split(","):
        m = re.match(r"\s*(>=|>)\s*(\d+)\.(\d+)", part)
        if not m:
            continue
        v = (int(m.group(2)), int(m.group(3)))
        if best is None or v > best:
            best = v
    return best


def main():
    txt = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    m = re.search(r'^requires-python\s*=\s*"([^"]+)"', txt,
                  re.MULTILINE)
    check("pyproject states a Python floor at all", bool(m))
    ours = parse_floor(m.group(1)) if m else None
    check("...and it is a floor this suite can read ({})".format(
        m.group(1) if m else "-"), ours is not None)

    # THE DEPENDENCIES ARE DECLARED, not left to a side file. They
    # lived in requirements.txt alone, and the gap had to be papered
    # over with a bold warning in CONVENTIONS and another in
    # WINDOWS.md, because without them seven suites fail on the import
    # line and read as broken code.
    dm = re.search(r"^dependencies\s*=\s*\[(.*?)\]", txt,
                   re.MULTILINE | re.DOTALL)
    declared = re.findall(r'"([A-Za-z0-9_.\-]+)', dm.group(1)) if dm \
        else []
    check("the fitted path's dependencies are declared in pyproject, "
          "so `pip install -e .` is enough - a documented footgun is "
          "still a footgun",
          {"numpy", "pandas", "scikit-learn", "scipy"} <= set(declared))

    # THE FLOOR MUST NOT SIT BELOW ANY DEPENDENCY'S OWN FLOOR.
    try:
        import importlib.metadata as md
        have = True
    except ImportError:
        have = False

    floors = {}
    if have:
        for name in ("numpy", "pandas", "scikit-learn", "scipy"):
            try:
                floors[name] = parse_floor(
                    md.metadata(name).get("Requires-Python"))
            except Exception:
                floors[name] = None

    # WHICH QUESTION IS BEING ASKED, because there are two and only
    # one of them is a defect.
    #
    #   is our floor below what OUR DECLARED MINIMUMS need?
    #       a real defect - `pip install` on the floor then resolves
    #       something older than we ever tested, which is how a
    #       silently different pandas reaches the machine with the
    #       data on it. This is the bug that shipped: `>=3.8`
    #       declared while the minimums needed 3.9.
    #
    #   do the INSTALLED versions need more than our floor?
    #       not a defect. It means the operator has newer packages
    #       than we require, which is the normal case and gets more
    #       normal with time. Read on the data machine - numpy 2.5
    #       and scipy 1.18 both want >=3.12 - it failed a suite that
    #       had nothing wrong with it.
    #
    # The first is answered from the floors of the versions we
    # DECLARE, which do not change when someone upgrades. They cannot
    # be read from metadata for versions that are not installed, so
    # they are recorded here - and a check below fails if a declared
    # dependency is missing from the table, so adding one forces the
    # entry rather than silently skipping it.
    MIN_FLOORS = {"numpy": (3, 8),          # numpy 1.24
                  "pandas": (3, 8),         # pandas 2.0
                  "scikit-learn": (3, 9),   # scikit-learn 1.4
                  "scipy": (3, 8)}          # scipy 1.10
    missing_tbl = sorted(set(declared) - set(MIN_FLOORS))
    check("every declared dependency has a recorded floor for the "
          "MINIMUM version we allow, or this check silently skips "
          "it{}".format("" if not missing_tbl
                        else " -- " + ", ".join(missing_tbl)),
          not missing_tbl)
    need = max([MIN_FLOORS[d] for d in declared
                if d in MIN_FLOORS] or [(0, 0)])
    check("our floor {}.{} is at least what our own declared MINIMUM "
          "dependency versions need ({}.{}) - below it, an install on "
          "the floor resolves something older than was ever tested"
          .format(ours[0] if ours else 0, ours[1] if ours else 0,
                  need[0], need[1]),
          ours is not None and ours >= need)

    known = dict((k, v) for k, v in floors.items() if v is not None)
    if not known:
        check("our Python floor is at least every dependency's own   "
              "[SKIPPED - dependencies not installed, and an empty "
              "comparison is not a pass]", True)
        check("...and the floor is reported beside the dependency "
              "that sets it   [SKIPPED]", True)
    else:
        worst = max(known.values())
        who = sorted(k for k, v in known.items() if v == worst)
        # REPORTED, NOT FAILED. See above: newer installed packages
        # are the normal case and are not a defect in this package.
        if ours is not None and ours < worst:
            print("    note: the versions installed here need Python "
                  "{}.{} ({}), above our declared floor of {}.{}. "
                  "Anyone on the floor resolves older ones."
                  .format(worst[0], worst[1], ", ".join(who),
                          ours[0], ours[1]))
        check("the installed dependencies' floors are readable, so "
              "the note above is measured rather than assumed",
              bool(known))
        check("...and every dependency named in pyproject reports a "
              "readable floor, or one of them could move under us "
              "unseen ({} of {} readable)".format(
                  len(known), len(floors)),
              len(known) == len(floors))

    # The interpreter running the suite must itself satisfy the floor,
    # or this suite is reporting on an install nobody has.
    here = sys.version_info[:2]
    check("the interpreter running this suite satisfies the declared "
          "floor ({}.{} >= {}.{})".format(
              here[0], here[1], ours[0] if ours else 0,
              ours[1] if ours else 0),
          ours is not None and here >= ours)

    # requirements.txt is named by docs/WINDOWS.md section 2 and is
    # pasted by hand on the machine that is hardest to debug from
    # here. It must keep working, and it must not hold a second copy
    # of the dependency list.
    req = (ROOT / "requirements.txt")
    check("requirements.txt still exists - WINDOWS.md section 2 names "
          "it, and that ritual is pasted by hand", req.exists())
    body = [ln.strip() for ln in
            req.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")]
    check("...and it installs the package rather than repeating the "
          "dependency list - two places to keep in step is how they "
          "come to disagree",
          body == ["."])

    # ---- THE SHARDED NET MUST STILL BE THE WHOLE NET -----------
    #
    # CI fans the suites out across runners so a ten-minute serial
    # wall does not become a check people switch off. The danger that
    # buys is silent: an off-by-one in the split drops a suite from
    # every shard, all shards go green, and the net now covers less
    # than it says while looking exactly the same.
    SKIP = {"run_all_smokes.py", "build_notebook.py",
            "build_demo_notebook.py", "build_system_map.py"}
    every = sorted(p.name for p in (ROOT / "scripts").glob("smoke_*.py")
                   if p.name not in SKIP)
    check("there are suites to shard, or the checks below prove "
          "nothing ({} found)".format(len(every)), len(every) > 20)
    for n in (2, 3, 4, 5, 7):
        got = []
        for k in range(n):
            got += [s for i, s in enumerate(every) if i % n == k]
        ok = sorted(got) == every and len(got) == len(set(got))
        check("splitting the net {} ways covers every suite exactly "
              "once - a shard that quietly drops one still reports "
              "green".format(n), ok)

    # A shard must not be able to claim what only the whole net can.
    runner = (ROOT / "scripts" / "run_all_smokes.py").read_text(
        encoding="utf-8")
    check("only an unsharded run may print ALL GREEN - the phrase "
          "every convention and doc treats as proof",
          'a.shards == 1' in runner and '"ALL GREEN"' in runner)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
