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


# WHERE THE MESSAGE SCAN STARTS, and why it is not the first commit.
#
# THE FILE SCAN WAS HALF THE SURFACE. A commit message reaches the
# enterprise remote exactly like a tracked file does, and it is the
# harder half to correct because fixing one rewrites published
# history. But 52 of the 178 commits before this line already name the
# assistant in a trailer, and four name a laptop. Scanning all of them
# would paint the suite permanently red, which teaches everyone to
# ignore it - worse than no guard at all.
#
# So history through this commit is GRANDFATHERED, deliberately and in
# writing, and everything after it is held to the same list the
# tracked files are. Rewriting 178 commits to clean a string that has
# been in the log for months is not worth what it costs; keeping the
# next 178 clean is.
GRANDFATHERED_THROUGH = "a095f1fb072e7d3bd75ecde316acc7b1d2ddd202"


def commit_messages():
    """Commit messages after the grandfathered baseline.

    Fails loudly rather than scanning nothing if that baseline is not
    an ancestor of HEAD - a rewritten or shallow history would
    otherwise turn this whole check into a silent no-op, which is the
    failure mode the rest of this repository is built to avoid."""
    ok = subprocess.run(
        ["git", "merge-base", "--is-ancestor", GRANDFATHERED_THROUGH,
         "HEAD"], cwd=str(ROOT), capture_output=True)
    if ok.returncode != 0:
        return None
    out = subprocess.run(
        ["git", "log", "--format=%H%x00%B%x01",
         "{}..HEAD".format(GRANDFATHERED_THROUGH)],
        cwd=str(ROOT), capture_output=True, text=True).stdout
    msgs = []
    for chunk in out.split("\x01"):
        if "\x00" not in chunk:
            continue
        sha, body = chunk.split("\x00", 1)
        msgs.append((sha.strip()[:9], body))
    return msgs


def scan_messages(pats, msgs):
    rx = re.compile("|".join(pats))
    hits = []
    for sha, body in msgs:
        for line in body.splitlines():
            if "noqa: personal-scan" in line:
                continue
            m = rx.search(line)
            if m:
                hits.append("{}: {}".format(sha, m.group(0)))
    return hits


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

    # THE SAME BAN, APPLIED TO COMMIT MESSAGES. They reach the mirror
    # too, and a message is the harder half to fix: correcting one
    # means rewriting history that has already been pushed.
    msgs = commit_messages()
    check("the grandfathered baseline is still an ancestor of HEAD - "
          "if it is not, this scan would cover nothing and say so by "
          "passing", msgs is not None)
    msgs = msgs or []
    check("there are commits after the baseline to scan ({}), or the "
          "checks below prove nothing".format(len(msgs)), len(msgs) > 0)
    for name, pats in (("a personal handle, surname, login or "
                        "mailbox", PERSONAL),
                       ("a machine model", MACHINES),
                       ("the assistant that helped write it",
                        ASSISTANT)):
        hits = scan_messages(pats, msgs)
        check("no COMMIT MESSAGE names {}{}".format(
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

    # OS METADATA IS A PROVENANCE LEAK TOO, and a text scan cannot
    # see it. `docs/.DS_Store` was tracked: a binary Finder index
    # holding the folder's layout and file listing from one person's
    # desktop. It carries no banned string, so every check above
    # passed while it sat in the tree.
    junk = [f.relative_to(ROOT) for f in files
            if f.name in (".DS_Store", "Thumbs.db", "desktop.ini")
            or f.name.endswith(".swp")]
    check("no operating-system metadata file is tracked - a binary "
          "index of one person's folders passes every text scan{}"
          .format("" if not junk
                  else " -- " + ", ".join(str(j) for j in junk)),
          not junk)

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
