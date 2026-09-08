"""Smoke: a run can name the commit that produced it, on a machine
with no git.

The release gate is "proven against a real extract". That gate is a
guess unless the proving run names the tree it proved, and the machine
holding the extract is not a git checkout - `docs/WINDOWS.md` section
1 downloads a zipball, extracts it and renames the folder, which threw
away the commit hash GitHub stamps into that folder's name.

TWO CHECKS HERE ARE LOAD-BEARING.

The first is that an absent answer says so. A build id that quietly
reports something plausible is worse than one that reports nothing,
because the only reason to have it is to be believed - the same rule
as the getattr ban and the empty-scan gate in `smoke_no_personal`.

The second is that the OWNER NEVER SURVIVES. The extracted folder is
named `<owner>-<repo>-<sha>`, `provenance.json` travels with the
output, and the sidecar already records the source file by name rather
than path for exactly this reason. Only the sha identifies a build;
the account handle in front of it is a leak with no purpose.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit.build_id import (ARCHIVAL, STAMP,     # noqa: E402
                               build_id, describe)

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def is_repo(d: Path) -> bool:
    r = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"],
                       capture_output=True)
    return r.returncode == 0


def main():
    import tempfile

    # ---- the development machine --------------------------------
    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True)
    if head.returncode == 0:
        sha = head.stdout.strip()
        b = build_id(ROOT)
        check("in a git checkout the answer comes from git, which "
              "cannot go stale the way a written file can",
              b["source"] == "git")
        check("...and it is actually HEAD, not something that merely "
              "looks like a sha", b["id"] and sha.startswith(b["id"]))
    else:
        check("in a git checkout the answer comes from git   "
              "[SKIPPED - not a checkout]", True)
        check("...and it is actually HEAD   [SKIPPED]", True)

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)

        # THE FIXTURE MUST CONTAIN THE THING THE CHECK IS ABOUT. Every
        # check below is about behavior OFF a git checkout, so if
        # this directory were inside one they would all pass while
        # measuring the git path instead.
        check("the fixture directory is genuinely not a git checkout, "
              "or every check below silently measures the wrong path",
              not is_repo(d))

        # ---- nothing to go on ------------------------------------
        b = build_id(d)
        check("with no git and no stamp the build is UNKNOWN rather "
              "than a plausible-looking guess",
              b["source"] == "unknown" and b["id"] is None)
        check("...and it says how to fix it, naming the ritual that "
              "writes the file",
              STAMP in b["note"] and "WINDOWS.md" in b["note"])
        check("...and the line printed at the top of a run says "
              "UNKNOWN in words, not by omission",
              "UNKNOWN" in describe(d))

        # ---- the zipball the data machine actually gets -----------
        # A real extracted folder name. The owner and repo both carry
        # hyphens, which is why the sha is taken from the RIGHT.
        owner = "some-account-name"
        repo = "a-repo-with-hyphens"
        sha40 = "1f4c9ab77e3d5602bb18aa9c0e5d7431f0a2c6b9"
        (d / STAMP).write_text(
            "{}-{}-{}\n".format(owner, repo, sha40), encoding="utf-8")

        b = build_id(d)
        check("off a checkout the stamp written by the sync ritual is "
              "used - the zipball's folder name is the only provenance "
              "that machine ever receives",
              b["source"] == "zipball")
        check("...and the sha survives it, taken from the RIGHT so a "
              "hyphenated owner or repo cannot eat it",
              b["id"] and sha40.startswith(b["id"]))

        # THE LEAK CHECK, ON A SECOND FIXTURE, and the second fixture
        # is the entire point.
        #
        # Run against a mutant that keeps the whole folder name, this
        # check PASSED while the code leaked. The id is truncated to
        # WIDTH, and `some-account-name` cut to twelve characters is
        # `some-account`, which does not contain the owner as a
        # substring - so the guard could not fire on the fixture that
        # was supposed to trigger it. It sat beside a check that DID
        # catch the mutant and looked like it was working.
        #
        # A SHORT owner survives truncation, so the leak is visible in
        # the id itself. Same mutant, and now this check is what fails.
        short_owner, short_repo = "acct", "kit"
        (d / STAMP).write_text(
            "{}-{}-{}\n".format(short_owner, short_repo, sha40),
            encoding="utf-8")
        b2 = build_id(d)
        blob = repr(b2) + describe(d)
        check("the OWNER does not survive into anything the run "
              "publishes - provenance.json travels, and an account "
              "handle in it is a leak with no purpose",
              short_owner not in blob)
        check("...and neither does the repository name",
              short_repo not in blob)
        check("...while the sha still does, or the two checks above "
              "would pass on a build id that reported nothing at all",
              b2["id"] and sha40.startswith(b2["id"]))

        # ---- a stamp that arrived empty or broken ------------------
        (d / STAMP).write_text("\n", encoding="utf-8")
        check("an empty stamp reports UNKNOWN rather than an empty "
              "build id that reads as an answer",
              build_id(d)["source"] == "unknown")

        # A truncated download is how this file will most often be
        # wrong: the ritual's own size-check exists because a rejected
        # token writes a 106-byte error page instead of an archive.
        (d / STAMP).write_text("   ", encoding="utf-8")
        check("...and so does one holding only whitespace",
              build_id(d)["source"] == "unknown")

    # ---- it is gitignored, so it can never be committed stale ----
    ig = (ROOT / ".gitignore")
    check("the stamp is gitignored - committed, it would be wrong on "
          "every machine but the one that wrote it, and would carry "
          "an account handle into a tracked file",
          ig.exists() and STAMP in ig.read_text(encoding="utf-8"))

    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", STAMP],
                             capture_output=True, text=True)
    check("...and it is not tracked right now",
          not tracked.stdout.strip())

    # ---- THE MECHANISM THAT REMOVES THE HAND STEP ----------------
    #
    # A step the operator has to remember is a step that gets skipped,
    # and it was skipped on three runs in a row - each one reporting
    # UNKNOWN, which is the answer this module exists to avoid having
    # to give. GitHub builds a zipball by running `git archive`, and
    # `git archive` substitutes `$Format:` placeholders in files marked
    # export-subst. So the sha can ride INSIDE the download.
    #
    # ARCHIVING THIS REPO IS THE ONLY HONEST TEST OF THAT. The
    # question is whether `.gitattributes` marks the file, and a
    # temporary fixture has no .gitattributes to get wrong. Written
    # against a tree where the mark was missing, this fails.
    if head.returncode == 0:
        tracked_sha = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", ARCHIVAL],
            capture_output=True, text=True)
        check("{} is TRACKED - the opposite of the stamp, and it has "
              "to be: an untracked file never reaches an archive at "
              "all".format(ARCHIVAL), bool(tracked_sha.stdout.strip()))

        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "extracted"
            d.mkdir()
            tar = Path(td) / "a.tar"
            arc = subprocess.run(
                ["git", "-C", str(ROOT), "archive", "--format=tar",
                 "HEAD", "-o", str(tar)], capture_output=True)
            check("`git archive` runs on this repo - the same command "
                  "GitHub runs to build the zipball the data machine "
                  "downloads", arc.returncode == 0)
            subprocess.run(["tar", "-xf", str(tar)], cwd=str(d),
                           capture_output=True)
            check("the fixture is an EXTRACTED ARCHIVE and not a "
                  "checkout, or every check below measures the git "
                  "path instead", not is_repo(d))
            got = ""
            if (d / ARCHIVAL).exists():
                got = (d / ARCHIVAL).read_text(
                    encoding="utf-8").strip()
            check("the archive carries {} - which happens only if "
                  ".gitattributes marks it export-subst".format(
                      ARCHIVAL), bool(got))
            check("...and git SUBSTITUTED the real sha into it, so a "
                  "download names its own commit with nobody having "
                  "run a step", got == sha)

            b3 = build_id(d)
            check("...so a run from that extracted tree answers from "
                  "the archive rather than reporting UNKNOWN",
                  b3["source"] == "archive")
            check("...with the sha of the commit archived, not "
                  "something that merely looks like one",
                  b3["id"] and sha.startswith(b3["id"]))
            check("...and the line at the top of a run names it",
                  "UNKNOWN" not in describe(d))

            # AN UNSUBSTITUTED PLACEHOLDER IS NOT AN ANSWER. In a
            # checkout the file still reads `$Format:%H$`, and
            # publishing that as a build id would be exactly the
            # plausible-looking guess this module refuses.
            (d / ARCHIVAL).write_text("$Format:%H$\n", encoding="utf-8")
            check("an unsubstituted placeholder reports UNKNOWN - it "
                  "means this is a checkout, not an archive",
                  build_id(d)["source"] == "unknown")
            (d / ARCHIVAL).write_text("not-a-sha\n", encoding="utf-8")
            check("...and so does a file somebody edited by hand: 40 "
                  "hex characters or it is not an archive stamp",
                  build_id(d)["source"] == "unknown")

            # PRECEDENCE. On the development machine git is
            # authoritative and cannot go stale; the archive file can.
            (d / ARCHIVAL).write_text(
                "1f4c9ab77e3d5602bb18aa9c0e5d7431f0a2c6b9\n",
                encoding="utf-8")
            check("in a real checkout git still wins over the archive "
                  "file, which is the one that can be stale",
                  build_id(ROOT)["source"] == "git")
    else:
        for lbl in ("archive carries the sha", "git wins over it"):
            check("{}   [SKIPPED - not a checkout]".format(lbl), True)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
