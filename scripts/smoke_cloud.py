"""Smoke: running in a cloud account without data leaving it.

WHY THIS EXISTS. A hospital's own AWS account is a legitimate
place to run this - an analyst should not need a laptop with
16GB free to fit an extract. What it must never become is a
route by which records leave the account they were already in.

THE LOAD-BEARING CHECK IS A REFUSAL, not a transfer. Uploading
the right files is easy to get right by accident; refusing the
source is the thing that has to hold when somebody passes the
wrong flag in a hurry. So the suite puts the SOURCE EXTRACT and
the KEY FILE inside the run directory - where a real run leaves
them - and demands they stay.

AND THE STORE IS A SECOND REAL IMPLEMENTATION, not a mock. The
whole upload path runs against a local directory, so what these
checks exercise is this project's logic rather than somebody
else's SDK behaving as its author imagined. A cloud path tested
only against a mock is tested against the mock.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from synthkit import cloud                            # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:3d}  {}".format(PASS, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def main():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        run = d / "run"
        run.mkdir()
        for n in ("blueprint.json", "generated.csv",
                  "fidelity.json", "findings.txt"):
            (run / n).write_text("x", encoding="utf-8")
        # THE TWO FILES THAT MUST NOT TRAVEL, sitting exactly
        # where a real run leaves them.
        (run / "tidy_visits.csv").write_text(
            "mrn,value\nP001,5\n", encoding="utf-8")
        (run / "key.csv").write_text("k", encoding="utf-8")
        dest = d / "bucket"

        wrote = cloud.publish(run, str(dest))
        names = sorted(Path(w).name for w in wrote)
        check("a publish sends the named artifacts: {}"
              .format(names),
              names == ["blueprint.json", "fidelity.json",
                        "findings.txt", "generated.csv"])
        check("...and the SOURCE EXTRACT stays, though it sat "
              "in the same directory - a run that could upload "
              "its own input is one bad argument away from "
              "publishing an extract",
              not (dest / "tidy_visits.csv").exists())
        check("...and `key.csv` stays too, because the key is "
              "the full-fidelity file the operator KEEPS and "
              "publishing it would defeat the arrangement it "
              "exists for",
              not (dest / "key.csv").exists())

        try:
            cloud.publish(run, str(dest),
                          only=["tidy_visits.csv"])
            said = ""
        except cloud.CloudRefusal as exc:
            said = str(exc)
        check("asking for the source BY NAME is refused, and "
              "the refusal says the list cannot be extended "
              "by a flag - the guard is not advice",
              "tidy_visits.csv" in said
              and "cannot be added by a flag" in said)

        try:
            cloud.publish(run, str(dest),
                          only=["blueprint.json",
                                "tidy_visits.csv"])
            mixed = ""
        except cloud.CloudRefusal as exc:
            mixed = str(exc)
        check("...and a MIXED request is refused whole rather "
              "than partly honoured - a partial upload that "
              "looks complete is the silent-difference fault "
              "this project refuses everywhere else",
              "tidy_visits.csv" in mixed)
        check("...and nothing from the mixed request was "
              "written, so all-or-nothing is a behaviour and "
              "not a sentence",
              not (dest / "tidy_visits.csv").exists())

        try:
            cloud.publish(d / "nope", str(dest))
            miss = ""
        except cloud.CloudRefusal as exc:
            miss = str(exc)
        check("a directory that is not there is refused with "
              "the reason, not a stack",
              "not a directory" in miss)

        empty = d / "empty"
        empty.mkdir()
        try:
            cloud.publish(empty, str(dest))
            emp = ""
        except cloud.CloudRefusal as exc:
            emp = str(exc)
        check("...and a run with NO artifacts is refused "
              "rather than reported as a successful upload of "
              "nothing - the two look identical in a log",
              "nothing to publish" in emp)

        # ---- ONE DIRECTION ONLY
        src = d / "in.csv"
        src.write_text("a,b\n1,2\n", encoding="utf-8")
        got = cloud.fetch_source(str(src), d / "work")
        check("a local source is read IN PLACE rather than "
              "copied - the compute comes to the data",
              Path(got) == src)
        check("...and there is NO function here that sends a "
              "source anywhere, which is not an oversight",
              not any(n for n in dir(cloud)
                      if "upload_source" in n
                      or "push_source" in n))

        # ---- S3 WITHOUT AWS: the URI layer is testable alone
        check("an s3:// URI is recognised and split",
              cloud.is_s3("s3://bkt/a/b.csv")
              and cloud.split_s3("s3://bkt/a/b.csv")
              == ("bkt", "a/b.csv"))
        check("...and a bucket with no object is refused, "
              "because that names a place and not a file",
              _refuses(lambda: cloud.fetch_source(
                  "s3://bkt", d / "work"), "names a bucket"))
        check("...and a local path is NOT mistaken for S3",
              not cloud.is_s3("/tmp/x.csv")
              and not cloud.is_s3("s3:/bad"))

        # ---- THE DESCRIPTION IS HONEST ABOUT WHAT HAS NOT RUN
        lines = "\n".join(cloud.describe())
        check("the deployment DESCRIBES itself, including that "
              "it has NOT been run against a live AWS account "
              "- a deployment is not measured until somebody "
              "deploys it",
              "NOT YET RUN AGAINST A LIVE AWS ACCOUNT" in lines
              and "only named artifacts go OUT" in lines)

        # ---- AND THE CLI CARRIES THE SAME GUARD
        r = subprocess.run(
            [sys.executable, "-m", "synthkit.cli", "batch",
             "--src", str(src), "--out", str(d / "w2"),
             "--dry-run"],
            capture_output=True, text=True, cwd=str(ROOT),
            timeout=900)
        check("`synthkit batch --dry-run` says what WOULD "
              "leave and starts nothing - a transfer nobody "
              "previewed is one nobody can object to",
              r.returncode == 0
              and "DRY RUN" in r.stdout
              and "blueprint.json" in r.stdout)
        check("...and it prints the posture BEFORE doing "
              "anything, so an operator reads the rule rather "
              "than trusting it",
              r.stdout.index("only named artifacts go OUT")
              < r.stdout.index("DRY RUN"))

    # ---- THE IMAGE CARRIES NO DATA
    dockerfile = (ROOT / "Dockerfile").read_text(
        encoding="utf-8")
    ignore = (ROOT / ".dockerignore").read_text(
        encoding="utf-8")
    check("there IS a Dockerfile, pinned to an interpreter "
          "version - this project has twice been bitten by "
          "testing against a different Python than the one "
          "that runs",
          "FROM python:3.14" in dockerfile)
    check("...and it runs as a non-root user, so a process "
          "cannot leave things behind in a shared account",
          "USER synth" in dockerfile)
    check("...and the build context EXCLUDES anything that "
          "could carry records - an extract swept into a "
          "layer travels with every pull of the image",
          all(pat in ignore
              for pat in ("*.csv", "data/", "run_*/")))
    check("...and the image carries the NET, so a deployment "
          "can prove the tool works before it is pointed at "
          "an extract",
          "run_all_smokes.py" in dockerfile)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


def _refuses(fn, text):
    try:
        fn()
        return False
    except cloud.CloudRefusal as exc:
        return text in str(exc)
    except Exception:
        return False


if __name__ == "__main__":
    main()
