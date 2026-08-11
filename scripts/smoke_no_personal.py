"""Smoke: nothing tracked names a person, a machine, or a personal
account.

This repository is mirrored to an enterprise remote. Anything committed
here reaches colleagues, so a laptop model, a personal GitHub handle or
a private email address in a comment is a small, permanent leak of
something that was never the point.

THE HARD PART IS THAT "claude" IS ALSO A PRODUCT NAME. The system calls
Anthropic models through Bedrock, so `anthropic.claude-sonnet-4-6-v1:0`
is a model identifier the code needs and `Llama / Mistral / Claude` is
interface copy naming which vendors are supported. Banning the string
outright would break the build to fix a documentation problem. What is
banned is the ASSISTANT and the machines it runs on.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PASS = FAIL = 0

# Personal to one operator: a handle, a surname, a work login, a
# private mailbox. None of these describe the software.
PERSONAL = (r"farmingfarmer", r"marunycz", r"amarunyc",   # noqa: personal-scan
            r"NE100\d{4}")                                  # noqa: personal-scan
# A machine is never a requirement. The GUI already bans these in its
# own labels; the same reasoning applies to every tracked file.
MACHINES = (r"\bMacBook\b", r"\bM3 Max\b", r"\bThinkPad\b",   # noqa: personal-scan
            r"\bmy Mac\b", r"\bthis Mac\b")                  # noqa: personal-scan
# The assistant, as distinct from the model the product calls.
ASSISTANT = (r"Claude Code", r"Co-Authored-By: Claude")  # noqa: personal-scan


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def tracked():
    out = subprocess.run(["git", "ls-files"], cwd=str(ROOT),
                         capture_output=True, text=True).stdout
    return [ROOT / p for p in out.split("\n") if p.strip()]


def scan(pats, files):
    hits = []
    rx = re.compile("|".join(pats))
    for f in files:
        try:
            body = f.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        for i, line in enumerate(body.splitlines(), 1):
            # A line that DEFINES the ban list is not a reference to
            # what it bans, and neither is a rule explaining why the
            # name is banned. Without this the scan reports itself.
            if "noqa: personal-scan" in line:
                continue
            m = rx.search(line)
            if m:
                hits.append("{}:{}: {}".format(
                    f.relative_to(ROOT), i, m.group(0)))
    return hits


def main():
    files = tracked()
    check("there are tracked files to scan, or this suite proves "
          "nothing", len(files) > 50)

    for name, pats in (("a personal handle, surname, login or "
                        "mailbox", PERSONAL),
                       ("a machine model - a laptop is never a "
                        "requirement", MACHINES),
                       ("the assistant that helped write it, as "
                        "opposed to the model the product calls",
                        ASSISTANT)):
        hits = scan(pats, files)
        check("no tracked file names {}{}".format(
            name, "" if not hits else " -- " + "; ".join(hits[:5])),
            not hits)

    # The counterpart: the product's own vendor references MUST
    # survive. A scrub that removes them breaks the build, and this
    # check fails if someone runs a blunt find-and-replace.
    body = "\n".join(
        f.read_text(encoding="utf-8", errors="ignore")
        for f in files if f.suffix in (".py", ".yml", ".ipynb"))
    check("model identifiers the product needs are still present - "
          "banning the vendor's name outright would break the build "
          "to fix a documentation problem",
          "anthropic.claude" in body and "claude-sonnet" in body)

    check("the tracked conventions file exists and is the neutral "
          "one", (ROOT / "CONVENTIONS.md").exists())
    check("...and the assistant-specific file is NOT tracked",
          not any(f.name == "CLAUDE.md" for f in files))

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
