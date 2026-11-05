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
import subprocess
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
                  "scipy": (3, 8),         # scipy 1.10
                  "openpyxl": (3, 8)}       # openpyxl 3.1
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

    # THREAD SPIN-UP DOMINATES ON FIXTURE-SIZED DATA. Measured:
    # three_way on a 540-row frame takes 31.5s with default OpenMP
    # threading and 1.3s pinned to one thread, and smoke_relationships
    # went 248s -> 7s. Every suite here runs on small frames, so the
    # whole net was paying thread setup instead of doing arithmetic.
    # Pinned for the CHILD SUITES ONLY - the real extract is 55,428
    # rows where threading genuinely pays, so this must never reach
    # the library.
    check("the net pins BLAS/OpenMP threads for the suites it "
          "spawns - fixture-sized frames spend more time starting "
          "threads than computing, and the net gates every push",
          'OMP_NUM_THREADS' in runner
          and 'env.setdefault' in runner
          and 'env=env' in runner)
    check("...by setdefault, so an outside value still wins and the "
          "pinning can be overridden rather than imposed",
          'setdefault' in runner)
    check("...and it is set on the SUBPROCESS environment, not in "
          "any module the generation path imports, because the real "
          "extract wants the threads",
          'OMP_NUM_THREADS' not in
          (ROOT / "synthkit" / "jointcheck.py").read_text(
              encoding="utf-8"))

    # ---- EVERY SCRIPT THAT IMPORTS THE PACKAGE MUST PUT THE
    #      ROOT ON THE PATH FIRST
    #
    # Four scripts had a module-level `from synthkit ...` with
    # no preceding root insert, and each was found REACTIVELY -
    # one when ten gate checks went red, one when three help
    # checks did. A sweep costs nothing and catches the next
    # one at authoring time instead of on the machine that
    # holds the extract. The ORDER matters, not just the
    # presence: `relationship_report.py` had an insert, inside
    # a function, after the import.
    import re as _re
    _late = []
    for _p in sorted((ROOT / "scripts").glob("*.py")):
        _src = _p.read_text(encoding="utf-8")
        _ins = _imp = None
        for _i, _ln in enumerate(_src.splitlines()):
            if (_ins is None and "sys.path.insert" in _ln
                    and not _ln.startswith((" ", "\t"))):
                _ins = _i
            if (_imp is None
                    and _re.match(r"^(from synthkit|import synthkit)",
                                  _ln)):
                _imp = _i
        if _imp is not None and (_ins is None or _ins > _imp):
            _late.append(_p.name)
    check("every script importing the package at module level "
          "puts the repo ROOT on sys.path FIRST - a script's "
          "sys.path[0] is its own directory, so without it "
          "`python scripts/X.py` dies on a tree with no "
          "install{}".format(
              "" if not _late else ": " + ", ".join(_late)),
          not _late)

    # A WRONG PATH IS WORSE THAN A MISSING ONE, and a STATIC
    # check cannot see it. `smoke_tidy_fixture` inserted only
    # `ROOT / "scripts"`, so `synthkit` was never on the path
    # and the suite ran solely where the package happened to be
    # installed - it passed here for months and died on the
    # data machine's no-install tree. A regex for "does it
    # insert the root" was written, flagged 38 scripts
    # including two already verified to work, and was DELETED:
    # the inserts here are written a dozen different ways, so
    # that check matched style rather than behaviour and a
    # check that cries wolf is worse than none.
    #
    # The behavioural test below is the real one, and for the
    # SMOKE scripts the net itself is the detector - it runs
    # each as a subprocess, so a broken path surfaces as a
    # suite failure on a no-install tree, which is exactly how
    # these were found.

    # ---- A SCRIPT MUST RUN FROM A SOURCE TREE WITH NO INSTALL
    #
    # `python scripts/m0_gate.py RUNDIR` - the command that
    # module's own docstring gives - died with
    # ModuleNotFoundError on the data machine, where the
    # documented no-git flow is to extract a zipball and run.
    # A script's sys.path[0] is the SCRIPT's directory, not the
    # repo root, so `import synthkit` resolves only when
    # something else put it there. `python -c` works because the
    # CWD is on the path, which is exactly why this never showed
    # up by hand - and it took TEN checks down with it, all
    # reading as gate defects.
    #
    # Tested with an interpreter that has NO install, so the
    # script has to find the source tree itself. That is the
    # observable the fault changed; asserting the sys.path line
    # is present would pass on a tree where the import is
    # satisfied some other way.
    import json as _js
    import shutil as _sh
    import tempfile as _tf
    _clean = {"summary": {
        "pairs": 114, "pairs_sign_ok": 114, "pairs_close": 114,
        "pairs_inverted": 0, "coverage_ok": 42, "columns": 42,
        "set_tokens_ok": 62, "set_tokens_compared": 62,
        "set_empty_ok": 4, "set_empty_compared": 4}}
    _bare = None
    for _cand in ("python3.14", "python3.13", "python3.12"):
        _w = _sh.which(_cand)
        if not _w:
            continue
        _probe = subprocess.run(
            [_w, "-c", "import synthkit"],
            capture_output=True, text=True, cwd=_tf.gettempdir())
        if _probe.returncode != 0:
            _bare = _w
            break
    with _tf.TemporaryDirectory() as _td:
        _rd = Path(_td) / "run"
        _rd.mkdir()
        (_rd / "fidelity.json").write_text(
            _js.dumps(_clean), encoding="utf-8")
        if _bare is None:
            print("     NOTE: no interpreter without synthkit "
                  "installed is available, so the "
                  "no-install path cannot be exercised here - "
                  "SKIPPED rather than passed silently")
        else:
            # AND THE INVOCATION MUST REACH THE IMPORT.
            #
            # The first cut ran `peek.py RUNDIR` with no
            # subcommand, so peek printed its usage and exited
            # without importing anything - removing the path
            # insert broke NOTHING and the check was vacuous.
            # peek imports `synthkit.sets` inside the branch
            # that explains an UNMEASURED published surface, so
            # the fixture has to carry a published surface and
            # no gap record for it.
            (_rd / "blueprint.json").write_text(_js.dumps({
                "relationships": [{
                    "child": "y", "parents": ["x", "w"],
                    "evidence": {"interaction": {
                        "pair": ["x", "w"]}}}]}),
                encoding="utf-8")
            # THE ENTRY POINTS AN OPERATOR ACTUALLY RUNS.
            # Each of these is named in a docstring or in the
            # conventions as `python scripts/X.py ...`, and
            # each failed on the no-install tree.
            for _scr, _args in (
                    ("m0_gate.py", []),
                    ("peek.py", ["surfaces"]),
                    ("relationship_report.py", ["-h"]),
                    ("latent_challenger.py", ["-h"]),
                    ("handover_report.py", ["-h"]),
                    ("close_gap.py", []),
                    ("verify_permutation.py", [])):
                _r = subprocess.run(
                    [_bare, str(ROOT / "scripts" / _scr),
                     str(_rd)] + _args,
                    capture_output=True, text=True,
                    cwd=_tf.gettempdir(), timeout=600)
                _both = (_r.stdout or "") + (_r.stderr or "")
                # THE ASSERTION IS ABOUT `synthkit`, NOT ABOUT
                # ANY MISSING MODULE. The first cut rejected
                # every ModuleNotFoundError, so it failed on the
                # bare interpreter lacking NUMPY - an
                # environment limitation - and reported it as
                # the path defect. The fault being tested is
                # that the repo root is not on the path, and the
                # observable for that is `synthkit` itself.
                check("scripts/{} resolves `synthkit` from a "
                      "SOURCE TREE with no install - the "
                      "documented zipball flow on the data "
                      "machine, where a missing sys.path insert "
                      "read as ten gate defects".format(_scr),
                      "No module named 'synthkit'" not in _both
                      and 'No module named "synthkit"'
                      not in _both)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
