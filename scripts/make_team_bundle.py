"""Assemble the SELF-CONTAINED team bundle: everything a teammate
needs to run synthkit on a locked-down Windows machine, in one
folder - the test kit, the (enterprise-clean) source, offline
dependency wheels, install steps one command per line, and an
honest page about the optional AI backends.

WHY THE SOURCE IS FILTERED HERE TOO. The bundle travels the same
roads the enterprise repo does, so it carries the same exclusion
list - internal runbooks, the lab notebook (a one-line stub
remains, because ten code comments reference it by name), demo
scripts, and the personal-scan suite whose patterns are
themselves the thing being scanned for. The bundle SCANS ITSELF
before it will finish: a bundle that cannot prove it is clean
does not ship.

WHY WHEELS. `pip install -e .` needs the network; a teammate's
machine may not have it. `pip download --platform win_amd64
--only-binary=:all:` fetches Windows wheels from anywhere, so the
install becomes `pip install --no-index --find-links wheels`.

WHY OLLAMA IS OPTIONAL, STATED RATHER THAN BUNDLED. The entire
measure route - types, generate, verdict, dashboard - uses NO
language model. Only Describe-from-English needs one, and the
bench offers four backends; at the hospital the sanctioned path
is the cloud one already in the dropdown. Shipping a multi-GB
model installer inside every kit would make the common case pay
for the rare one; the page gives the exact steps instead,
including the fully-offline model-copy route.
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ONE exclusion list. KECK_PUSH_COMMANDS (the operator's manual
# export) names the same files; if this list changes, that file
# must change with it - stated here because the duplication is a
# known risk, accepted so the manual path has no import step.
EXCLUDE_FILES = [
    "RUNBOX.md", "RUNBOX.html", "CONVENTIONS.md",
    "docs/SPRINT.md", "docs/WINDOWS.md",
    "docs/DEMO_LLM_WINDOWS.md",
    "docs/demo_script.md", "docs/demo_script_v2.md",
    "docs/demo_script_clinic.md", "docs/demo_prompt.md",
    "scripts/smoke_no_personal.py",
]
EXCLUDE_DIRS = ["docs/knowledge_pack", ".git"]

CONVENTIONS_STUB = (
    "# Lab notebook\n\n"
    "Code comments reference CONVENTIONS.md, the project lab\n"
    "notebook. It lives in the internal development repo. The\n"
    "practices it records - fail-first checks, measured\n"
    "decisions, predictions on record - are summarized in\n"
    "docs/CODE_TOUR.md.\n")

SCAN = re.compile(
    r"farmingfarmer|marunycz|amarunyc"      # noqa: bundle-scan, noqa: personal-scan
    r"|runbox|outbox|thinkpad"              # noqa: bundle-scan, noqa: personal-scan
    r"|m3 max|macbook|sprint\.md"           # noqa: bundle-scan, noqa: personal-scan
    r"|CLAUDE\.md", re.I)                   # noqa: bundle-scan
SCAN_OK = re.compile(r"anthropic\.claude|claude-sonnet|claude-3")
SCAN_EXT = {".py", ".md", ".yml", ".toml", ".txt", ".html",
            ".ipynb", ".bat"}


def scan_tree(root: Path):
    """Every clean-rule hit in the tree, as (path, lineno, line).
    The bundle refuses to finish while this returns anything."""
    hits = []
    for f in sorted(root.rglob("*")):
        if not f.is_file() or f.suffix not in SCAN_EXT:
            continue
        # the scanner's own file is rule-text: the exclusion
        # list NAMES the excluded files and the pattern NAMES
        # the banned words - the same reason the personal-scan
        # suite stays out of the bundle entirely. This file
        # ships (the enterprise side rebuilds kits with it), so
        # it is exempted by name rather than excluded.
        if f.name == "make_team_bundle.py":
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.split("\n"), 1):
            # a check that ASSERTS an exclusion held must name
            # the excluded file; the marker exempts exactly
            # those lines, the same pattern the personal scan
            # uses for its own scan terms
            if "noqa: bundle-scan" in line:
                continue
            if SCAN.search(line) and not SCAN_OK.search(line):
                hits.append((str(f.relative_to(root)), i,
                             line.strip()[:90]))
    return hits


WALK_SKIP_DIRS = {".git", "__pycache__", ".venv", ".venv313",
                  ".venv314", ".pytest_cache", "keck_stage"}
WALK_SKIP_SUFFIX = {".pyc", ".zip"}
# a CSV is only ever TRACKED under these roots; one anywhere else
# is a stray output - possibly real-derived - and the build must
# STOP, not sweep it into a bundle that travels
CSV_ROOTS = ("data/", "docs/")


def _list_files():
    """The tree to export. `git ls-files` where git exists; on
    the data machine the checkout is a ZIPBALL EXTRACT with no
    git at all, and the first cut returned an empty list there -
    an empty source tree whose self-scan passes trivially, which
    is the worst kind of clean. The fallback walks the extract
    (a zipball IS the tracked tree), skipping the junk an
    installed checkout grows, and REFUSES by name any .csv
    outside the tracked data roots."""
    try:
        r = subprocess.run(["git", "ls-files"],
                           capture_output=True, text=True,
                           cwd=str(ROOT))
        files = r.stdout.split() if r.returncode == 0 else []
    except OSError:
        # the data machine has no git PROGRAM, not merely no
        # .git directory - a bare subprocess call there is a
        # crash, not an empty list
        files = []
    if files:
        return files
    out = []
    for f in sorted(ROOT.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(ROOT).as_posix()
        parts = set(rel.split("/"))
        if parts & WALK_SKIP_DIRS or f.suffix in WALK_SKIP_SUFFIX:
            continue
        if ".egg-info" in rel:
            continue
        if f.suffix == ".csv" and not rel.startswith(CSV_ROOTS):
            sys.exit("refusing: {} is a .csv outside the "
                     "tracked data roots - a stray output, "
                     "possibly real-derived, must never ride a "
                     "bundle. Move it out of the tree and "
                     "re-run.".format(rel))
        out.append(rel)
    if not out:
        sys.exit("found no files to export - run this from the "
                 "synthkit checkout (git or zipball extract).")
    return out


def build_source(dst: Path):
    tracked = _list_files()
    excluded = set(EXCLUDE_FILES)
    for rel in tracked:
        if rel in excluded:
            continue
        if any(rel.startswith(d + "/") or rel == d
               for d in EXCLUDE_DIRS):
            continue
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, out)
    # the stub that keeps ten comment references honest
    (dst / "CONVENTIONS.md").write_text(CONVENTIONS_STUB,
                                        encoding="utf-8")
    # BUILD_SHA.txt is export-subst: a zipball arrives with the
    # sha ALREADY substituted, and staging that file verbatim
    # commits OUR commit's hash into the team repo - whose own
    # archives then name a commit that is not in it. A tracked
    # stamp is the PLACEHOLDER; a substituted one is an archive
    # artifact, not a source file. Found by the verify clone's
    # own smoke_buildid, two checks red on the data machine.
    stamp = dst / "BUILD_SHA.txt"
    if stamp.exists():
        stamp.write_text("$Format:%H$\n", encoding="utf-8")
    # scrub the two in-flight items the manual export also scrubs
    gi = dst / ".gitignore"
    if gi.exists():
        gi.write_text("\n".join(
            l for l in gi.read_text(encoding="utf-8").split("\n")
            if "CLAUDE" not in l and "local pointer" not in l
            and "neutral. " not in l), encoding="utf-8")
    sg = dst / "scripts" / "smoke_gui.py"
    if sg.exists():
        sg.write_text(sg.read_text(encoding="utf-8").replace(
            '("Mac", "MacBook", "ThinkP'  # noqa: bundle-scan, noqa: personal-scan
            'ad", "M3")', "()"),
            encoding="utf-8")


LAUNCH_BAT = '@echo off\r\nsetlocal\r\ncd /d %~dp0\r\necho ================================================\r\necho  synthkit - one-time setup, then the bench opens\r\necho ================================================\r\nwhere python >nul 2>nul\r\nif errorlevel 1 goto :nopython\r\nif not exist .venv (\r\n  echo [1/3] creating a private Python environment...\r\n  python -m venv .venv\r\n  if errorlevel 1 goto :fail\r\n)\r\nif not exist .venv\\ok.marker (\r\n  echo [2/3] installing synthkit from the offline wheels, no network needed...\r\n  .venv\\Scripts\\python -m pip install --no-index --find-links wheels -e synthkit_src\r\n  if errorlevel 1 goto :fail\r\n  echo ok> .venv\\ok.marker\r\n)\r\necho [3/3] starting the bench - your browser opens on the HOME page...\r\n.venv\\Scripts\\python -m synthkit.cli gui\r\npause\r\nexit /b 0\r\n:nopython\r\necho Python was not found on this machine.\r\necho Install Python 3.12 or newer, then run this file again.\r\npause\r\nexit /b 1\r\n:fail\r\necho.\r\necho A step failed - read the message above; a photo of this\r\necho window is enough for whoever supports the kit.\r\npause\r\nexit /b 1\r\n'
CHECKS_BAT = '@echo off\r\nsetlocal\r\ncd /d %~dp0\r\nif not exist .venv\\ok.marker (\r\n  echo Run START_SYNTHKIT.bat first - it installs everything.\r\n  pause\r\n  exit /b 1\r\n)\r\ncd synthkit_src\r\n..\\.venv\\Scripts\\python -u scripts\\run_all_smokes.py\r\necho.\r\necho The words ALL GREEN are the verdict.\r\npause\r\n'

INSTALL_MD = """# Install, offline, one command per line (Windows)

THE EASY WAY: double-click START_SYNTHKIT.bat. It creates a
private environment, installs synthkit from the offline wheels,
and opens the bench in your browser - first run takes a minute,
after that it just opens. RUN_CHECKS.bat proves the install:
expect the words ALL GREEN. Everything below is the same thing
done by hand, for anyone who prefers a terminal.

You need Python 3.10 or newer (`python --version` to check).
The OFFLINE wheels in this folder cover Python 3.12 and 3.14 on
Windows; any other version needs network access for the four
numeric packages. Everything else is in this folder. From a terminal opened HERE:

    cd synthkit_src
    pip install --no-index --find-links ..\\wheels -e .
    python scripts\\run_all_smokes.py

Expect 68 suites, ALL GREEN (the words are the verdict; the
check count grows with every build). Then:

    python -m synthkit.cli gui

Your browser opens on the HOME page, which explains the rest.
The kit folder beside this file (kit\\START_HERE.md) walks every
box in order on invented data with a printed answer key.

If the wheels folder is missing or your Python version is not
covered, and you DO have network access:

    pip install -e .

does the same thing online.
"""

OLLAMA_MD = """# The optional AI backends - what needs one, what does not

NOTHING in the measure route uses a language model. Check types,
generate synthetic data, judge it at the gate, draw the
dashboard - all of it runs from the wheels in this bundle alone.

ONE feature uses an AI backend: Describe-from-English (step 01
of the create route), which turns a paragraph into a data
recipe. Four backends are offered in its dropdown:

1. Hospital cloud (AWS Bedrock) - the sanctioned path at work.
   Needs your AWS credentials configured; nothing to install.
2. Ollama app - a local model runner, if approved on your
   machine. Install from ollama.com, then, with network:
       ollama pull mistral-small3.1
   Fully offline instead: on any machine that already has the
   model, copy the entire `.ollama\\models` folder (user home
   directory) onto this machine at the same path, then install
   the Ollama app from its offline installer.
3. Local AI server (llama.cpp / LM Studio) - point the bench at
   its address.
4. Anthropic API - needs a key and network.

No backend at all? The spec presets that carry a compiled
recipe still validate and generate, and the whole measure route
is untouched.
"""


def main():
    ap = argparse.ArgumentParser(
        description="Assemble the self-contained team bundle.")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--wheels", choices=["win", "here", "skip"],
                    default="win",
                    help="win: Windows wheels (works from any "
                         "OS); here: this machine's platform; "
                         "skip: no wheels folder")
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    if out.exists() and any(out.iterdir()):
        sys.exit("refusing to write into non-empty {} - give a "
                 "fresh directory, so an old bundle cannot leak "
                 "into a new one".format(out))
    out.mkdir(parents=True, exist_ok=True)

    print("source (filtered) ...")
    build_source(out / "synthkit_src")
    print("test kit ...")
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts/make_testkit.py"),
         "-o", str(out / "kit")],
        capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("make_testkit failed:\n" + r.stdout[-800:]
                 + r.stderr[-800:])
    if a.wheels != "skip":
        print("wheels ({}) ...".format(a.wheels))
        # setuptools and wheel are BUILD dependencies: a modern
        # venv ships neither, so `pip install --no-index -e .`
        # fails while a machine with quiet network access never
        # shows it. Measured both ways on a fresh 3.14 venv.
        base = [sys.executable, "-m", "pip", "download",
                "-d", str(out / "wheels"),
                "numpy>=1.24", "pandas>=2.0",
                "scikit-learn>=1.4", "scipy>=1.10",
                "setuptools", "wheel"]
        if a.wheels == "win":
            # wheels are ABI-specific: the kit that shipped only
            # cp312 could not install on the data machine, which
            # had installed Python 3.14. Cover both; INSTALL.md
            # names them.
            runs = [base + ["--platform", "win_amd64",
                            "--only-binary=:all:",
                            "--python-version", v]
                    for v in ("312", "314")]
        else:
            runs = [base]
        for cmd in runs:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                sys.exit("pip download failed:\n" + r.stderr[-800:])
    # the easiest thing to run is the thing you double-click:
    # the launcher makes the venv, installs offline, starts the
    # bench, and PAUSES on every exit so an error is readable
    # rather than a vanished window.
    (out / "START_SYNTHKIT.bat").write_bytes(
        LAUNCH_BAT.encode("ascii"))
    (out / "RUN_CHECKS.bat").write_bytes(
        CHECKS_BAT.encode("ascii"))
    (out / "INSTALL.md").write_text(INSTALL_MD, encoding="utf-8")
    (out / "OLLAMA_OPTIONAL.md").write_text(OLLAMA_MD,
                                            encoding="utf-8")

    hits = scan_tree(out / "synthkit_src")
    if hits:
        for h in hits[:10]:
            print("DIRTY: {}:{}: {}".format(*h))
        sys.exit("the bundle scanned DIRTY - refusing to "
                 "finish; fix the exclusion list")
    print("self-scan: CLEAN")
    print("bundle at {} - zip the folder and send it".format(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
