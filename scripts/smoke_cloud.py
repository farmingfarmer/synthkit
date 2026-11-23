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
import io
import json
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
        lines0 = cloud.describe()
        check("the deployment DESCRIBES itself - what may leave, "
              "which direction data moves, and what is still "
              "owed. This check used to pin the sentence "
              "\"NOT YET RUN AGAINST A LIVE AWS ACCOUNT\", which "
              "a real upload later made false; a check written "
              "around a claim goes red when the claim is "
              "corrected, so it pins the PROPERTY instead",
              any("only named artifacts" in x.lower()
                  for x in lines0)
              and any(x.startswith("STILL OWED:") for x in lines0))

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
    # THE IMAGE MUST NOT PIN THE THREAD COUNT.
    #
    # It did: `ENV OMP_NUM_THREADS=1`, which is the TEST-RUN
    # setting. Fixture frames spend more time starting threads
    # than computing - worth 35x to the net - but a real extract
    # is 55,428 rows where the threads genuinely pay, so an ENV
    # here makes every production fit in the container
    # single-threaded. The net sets it on its own subprocesses
    # by setdefault, and an image ENV is exactly the outside
    # value that beats it. `smoke_packaging` guards the module
    # against this generalisation; the image needed it too.
    check("the image does NOT pin OMP_NUM_THREADS - that is "
          "the test-run setting, and in the image it would "
          "make every production fit single-threaded on an "
          "extract where the threads genuinely pay",
          "OMP_NUM_THREADS" not in dockerfile.split(
              "ENV ")[-1].splitlines()[0])
    check("...while still setting PYTHONUNBUFFERED, so a long "
          "fit's progress reaches the log rather than sitting "
          "in a buffer - a run nobody can watch is one nobody "
          "can tell from a hang",
          "PYTHONUNBUFFERED=1" in dockerfile)
    # AND EVERY PATH THE IMAGE COPIES MUST EXIST. A Dockerfile
    # naming a missing file fails at BUILD time, and nobody has
    # built this one - so the cheapest real check on a
    # deployment nobody has run is that it refers to things
    # that are there.
    _copies = []
    for _ln in dockerfile.splitlines():
        _ln = _ln.strip()
        if _ln.startswith("COPY "):
            _parts = _ln.split()[1:-1]
            _copies.extend(_parts)
    check("every path the image COPYs exists in the repo, so "
          "the build cannot fail on a name - checked here "
          "because no one has built it yet: {}"
          .format(", ".join(_copies)),
          bool(_copies)
          and all((ROOT / c).exists() for c in _copies))
    check("...and the ENTRYPOINT names a console script the "
          "package actually declares, or the container starts "
          "and immediately cannot find itself",
          "synthkit" in (ROOT / "pyproject.toml").read_text(
              encoding="utf-8").split("[project.scripts]")[-1]
          if "[project.scripts]" in
          (ROOT / "pyproject.toml").read_text(encoding="utf-8")
          else False)

    # ---- RENAMING ON THE WAY OUT ----
    # A recipient asked for the synthetic file under a name of
    # their choosing and the allow-list refused it - the guard
    # working, because it cannot know an arbitrary filename is the
    # synthetic output rather than the extract. The check runs on
    # the SOURCE so the guard is exactly as strong.
    import tempfile as _tf
    with _tf.TemporaryDirectory() as _d:
        rd = Path(_d) / "run"
        rd.mkdir()
        (rd / "generated.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        (rd / "data_dictionary.csv").write_text("col\na\n", encoding="utf-8")
        (rd / "key.csv").write_text("secret\n", encoding="utf-8")
        (rd / "tidy_visits.csv").write_text("real\n", encoding="utf-8")
        dst = Path(_d) / "bucket"

        # WRAPPED, because a mutation that makes this raise would
        # kill the suite before the laundering check below ever
        # runs - a crashed suite emits no FAIL line, and a filtered
        # count then reads zero and looks like a check that cannot
        # fail. Caught exactly that way.
        try:
            wrote = cloud.publish(
                rd, str(dst),
                only=["generated.csv", "data_dictionary.csv"],
                as_name={"generated.csv": "confidential_x.csv"})
        except Exception as _e:
            wrote = []
            print("    (rename raised: {})".format(str(_e)[:70]))
        check("a published file can be RENAMED on the way out: "
              "{}".format(", ".join(w.split("/")[-1] for w in wrote)),
              any(w.endswith("confidential_x.csv") for w in wrote)
              and any(w.endswith("data_dictionary.csv") for w in wrote))
        check("...and the renamed file is the only one that moved "
              "under that name - nothing else was published",
              len(wrote) == 2)

        # THE GUARD MUST NOT BE DEFEATABLE BY A RENAME. This is the
        # whole reason the check runs on the source.
        bad = ""
        try:
            cloud.publish(rd, str(dst), only=["key.csv"],
                    as_name={"key.csv": "generated.csv"})
        except cloud.CloudRefusal as e:
            bad = str(e)
        check("the key file is STILL refused when renamed to an "
              "allowed name - the check is on the source, so a "
              "rename cannot launder it: {}".format(bad[:52]),
              "key.csv" in bad)
        bad = ""
        try:
            cloud.publish(rd, str(dst),
                    only=["tidy_visits.csv"],
                    as_name={"tidy_visits.csv": "generated.csv"})
        except cloud.CloudRefusal as e:
            bad = str(e)
        check("...and so is the source extract, which is the one "
              "that would matter most: {}".format(bad[:46]),
              "tidy_visits.csv" in bad)

        # A RENAME OF SOMETHING NOT BEING PUBLISHED IS A MISTAKE.
        bad = ""
        try:
            cloud.publish(rd, str(dst),
                    only=["generated.csv"],
                    as_name={"fidelity.json": "x.json"})
        except cloud.CloudRefusal as e:
            bad = str(e)
        check("renaming a file that is not being published is "
              "REFUSED rather than ignored - a silent no-op is how "
              "somebody believes a file went out under a name it "
              "never had: {}".format(bad[:46]),
              "fidelity.json" in bad)

    # ----------------------------------------------------------
    # AND THE DOC MUST NAME THE SUITE THAT IS EXPECTED TO FAIL
    # IN THE IMAGE. The net inside the container reports 137 of
    # 138 green with `smoke_gui` red - the local bench, which is
    # not the deployment surface - and a deployer told to prove
    # the tool works, who then sees a red suite, will reasonably
    # conclude the opposite. This project has already had a
    # teammate stopped by exactly that.
    _doc = (ROOT / "docs" / "AWS.md").read_text(encoding="utf-8")
    _flat = " ".join(_doc.split())
    check("the AWS doc NAMES the one suite expected to fail in "
          "the image, and says anything else is a real finding - "
          "a red suite with no explanation reads as a broken tool",
          "smoke_gui" in _flat
          and "expected" in _flat
          and "real finding" in _flat)

    # A DOCUMENTED COMMAND NOBODY RAN. `docs/AWS.md` told a
    # deployer to prove the tool works before pointing it at an
    # extract with
    #
    #   docker run --rm IMAGE python .../run_all_smokes.py
    #
    # and that does not run: the image's ENTRYPOINT is
    # `synthkit`, so `python` is read as a subcommand and
    # argparse refuses it. The one command a deployment starts
    # with, wrong in the one document it reads. Found by running
    # it, not by reading it.
    #
    # The check is on the SHAPE rather than on the text, so a
    # future command that runs python in this image has to
    # override the entrypoint too.
    ep = "synthkit" in dockerfile.split("ENTRYPOINT")[-1][:40]
    check("the image's ENTRYPOINT is the console script, which is "
          "what makes a bare `python ...` argument a subcommand "
          "rather than an interpreter", ep)
    doc_aws = (ROOT / "docs" / "AWS.md").read_text(
        encoding="utf-8")
    for where, text in (("docs/AWS.md", doc_aws),
                        ("Dockerfile", dockerfile)):
        flat = " ".join(text.split())
        bad = []
        for seg in flat.split("docker run")[1:]:
            head = seg[:160]
            if " python " in head and "--entrypoint" not in head:
                bad.append(head[:70])
        check("every `docker run` in {} that runs python also "
              "overrides the entrypoint, without which the "
              "documented command refuses itself: {}"
              .format(where, "; ".join(bad) or "all override it"),
              not bad)

    # THE PREFLIGHT, THE DIAGNOSIS AND THE POLICY.
    #
    # A deployment fails for one of about eight reasons and the
    # raw error names none of them - the deliverable's own upload
    # died on intercepted TLS, whose message is about a
    # certificate rather than the variable to set. So the
    # classification is a PURE function on an exception, tested
    # on planted ones with no account, no credentials and no
    # boto3, for the same reason the store is an interface: a
    # diagnosis that can only be exercised by failing against a
    # real bucket is one nobody will exercise.
    # ----------------------------------------------------------
    for exc, want in (
            (_Boom("[SSL: CERTIFICATE_VERIFY_FAILED] certificate "
                   "verify failed", cls="SSLError"),
             "tls-intercepted"),
            (_Boom("An error occurred (AccessDenied) when calling "
                   "PutObject: requires "
                   "x-amz-server-side-encryption",
                   code="AccessDenied"), "sse-required"),
            (_Boom("Unable to locate credentials",
                   cls="NoCredentialsError"), "no-credentials"),
            (_Boom("The provided token has expired",
                   code="ExpiredToken"), "expired"),
            (_Boom("Access Denied", code="AccessDenied"), "denied"),
            (_Boom("no such bucket", code="NoSuchBucket"),
             "no-bucket"),
            (_Boom("redirect", code="PermanentRedirect"),
             "wrong-region"),
            (_Boom("Could not connect to the endpoint URL",
                   cls="EndpointConnectionError"), "unreachable"),
            (_Boom("something nobody has seen"), "unknown")):
        got = cloud.diagnose(exc)
        check("a {} failure is named {!r} rather than handed back "
              "as a traceback".format(want.replace("-", " "), want),
              got == want)
        check("...and its remedy is in the same sentence as the "
              "cause, for {}".format(want),
              len(cloud.CAUSE_TEXT.get(got, "")) > 40)

    # ORDER MATTERS, and this is the one pair that can swallow
    # each other: a bucket policy requiring encryption refuses
    # the put AS an access-denied, so the specific test must run
    # first or somebody is sent to the IAM console for a missing
    # header.
    check("an encryption-required refusal is NOT reported as a "
          "plain permissions problem, though it arrives as one",
          cloud.diagnose(_Boom(
              "AccessDenied: x-amz-server-side-encryption is "
              "required", code="AccessDenied")) == "sse-required")

    # READ-ONLY MEANS READ-ONLY. Writing an object into somebody
    # else's bucket is an outward action, so the default must not
    # do it - asserted on the CALLS the stand-in saw, not on the
    # report's own say-so.
    f = _FakeS3()
    rep = cloud.preflight("s3://b/pre", client=f)
    check("a preflight is READ-ONLY by default - it listed and "
          "did not put: calls were {}".format(f.calls),
          rep["ok"] and rep["wrote"] is False
          and "put" not in f.calls)
    check("...and it SAYS the write test was skipped and how to "
          "run it, because a step that vanishes reads as a step "
          "that passed",
          any(st["ok"] is None and "--write" in st["detail"]
              for st in rep["steps"]))

    f = _FakeS3()
    rep = cloud.preflight("s3://b/pre", client=f, allow_write=True)
    check("with --write it puts, reads back and DELETES the "
          "canary, leaving nothing behind: {}".format(
              list(f.obj) or "empty"),
          rep["ok"] and rep["wrote"] and not f.obj)
    check("...and the bytes that went out are the module's own "
          "fixed CANARY_BODY, not something assembled from the "
          "run - a preflight that wrote a sample of the data "
          "would be the exact thing this module prevents",
          f.put_bodies == [cloud.CANARY_BODY]
          and len(cloud.CANARY_BODY) < 80)

    # A PUT THAT SUCCEEDS AND A GET THAT DIFFERS is the
    # silent-difference fault, and no permission problem explains
    # it. It must FAIL rather than be reported as a write that
    # worked.
    f = _FakeS3(corrupt=True)
    rep = cloud.preflight("s3://b/pre", client=f, allow_write=True)
    check("a canary that does not read back byte for byte FAILS, "
          "because a put that succeeds and a get that differs is "
          "not a healthy destination",
          rep["ok"] is False
          and any(st["step"].startswith("the canary reads back")
                  and st["ok"] is False for st in rep["steps"]))

    # AN OBJECT LEFT IN A HOSPITAL BUCKET IS SOMEBODY'S CLEANUP,
    # so a failed delete names it rather than being absorbed.
    f = _FakeS3(fail_on={"delete"})
    rep = cloud.preflight("s3://b/pre", client=f, allow_write=True)
    check("a failed delete NAMES the object left behind, so "
          "nobody has to go looking for it",
          rep["ok"] is False
          and any("LEFT BEHIND" in st["detail"]
                  for st in rep["steps"]))

    f = _FakeS3(fail_on={"list"})
    rep = cloud.preflight("s3://b/pre", client=f)
    check("a destination that cannot be listed STOPS before any "
          "write is attempted: calls were {}".format(f.calls),
          rep["ok"] is False and "put" not in f.calls)

    # ENCRYPTION. The header is ours to send; the key and the
    # policy belong with the bucket.
    f = _FakeS3()
    cloud.preflight("s3://b/pre", client=f, allow_write=True,
                    sse="aws:kms", kms_key="key-1")
    check("the encryption mode and key are actually SENT on the "
          "put, not merely accepted as arguments: {}".format(
              f.last_extra),
          (f.last_extra or {}).get("ServerSideEncryption")
          == "aws:kms"
          and (f.last_extra or {}).get("SSEKMSKeyId") == "key-1")
    f = _FakeS3()
    cloud.preflight("s3://b/pre", client=f, allow_write=True)
    check("...and nothing is sent when none was asked for, so "
          "the bucket's own default applies",
          not (f.last_extra or {}))
    check("a KMS key without aws:kms is REFUSED rather than "
          "ignored - a key read and not used is worse than no "
          "key, because somebody believes it was applied",
          _refuses(lambda: cloud.S3Store(
              "b", client=_FakeS3(), kms_key="key-1"),
              "read and not used"))
    check("...and an unknown encryption mode is refused by name",
          _refuses(lambda: cloud.S3Store(
              "b", client=_FakeS3(), sse="rot13"),
              "not a server-side encryption mode"))
    with tempfile.TemporaryDirectory() as td:
        check("...and asking for encryption on a LOCAL directory "
              "is refused, because a setting that cannot be "
              "applied must not be silently ignored",
              _refuses(lambda: cloud.store_for(td, sse="AES256"),
                       "local directory, which has none"))

    # THE POLICY IS DERIVED FROM WHAT THIS CODE DOES.
    pol = cloud.iam_policy("s3://src/extracts/t.csv",
                           "s3://art/synthkit/run1")
    acts = sorted(a for st in pol["Statement"]
                  for a in st["Action"])
    check("the generated policy grants exactly the three actions "
          "this tool performs: {}".format(acts),
          acts == ["s3:GetObject", "s3:ListBucket",
                   "s3:PutObject"])
    check("...and NOT s3:* or s3:DeleteObject, because nothing "
          "here deletes an artifact",
          not any(a in ("s3:*", "s3:DeleteObject") for a in acts))
    check("...and the resources are the one extract and the one "
          "prefix, not the whole bucket",
          any(st["Resource"].endswith("/extracts/t.csv")
              for st in pol["Statement"])
          and any(st["Resource"].endswith("/synthkit/run1/*")
                  for st in pol["Statement"]))
    check("...and it says what it does NOT cover, because a "
          "policy that looks complete is worse than one naming "
          "its own gaps",
          len(cloud.policy_notes()) >= 3
          and any("KMS" in x for x in cloud.policy_notes()))
    check("a policy over nothing is refused rather than returned "
          "empty",
          not cloud.iam_policy("", "")["Statement"])

    # ----------------------------------------------------------
    # THE STATUS IS ONE SOURCE WITH TWO RENDERERS.
    #
    # The module used to say "nothing here has run against a live
    # AWS account", which stopped being true the day the
    # deliverable was uploaded to a real bucket - and a stale
    # claim in the one place a governance reviewer trusts is
    # worse than no claim.
    # ----------------------------------------------------------
    lines = cloud.describe()
    check("the deployment no longer makes the absolute claim it "
          "outgrew - a publish into a real bucket HAS happened",
          not any("NOT YET RUN AGAINST A LIVE AWS" in x
                  for x in lines))
    check("...and it separates what is VERIFIED from what is "
          "STILL OWED, because conflating them is the thing most "
          "likely to be assumed",
          any(x.startswith("VERIFIED:") for x in lines)
          and any(x.startswith("STILL OWED:") for x in lines))
    check("...and the owed list names the two that matter: a "
          "`batch` fit in the account, and a TASK ROLE",
          any("batch" in x and "end to end" in x
              for x in cloud.STATUS_OWED)
          and any("TASK ROLE" in x for x in cloud.STATUS_OWED))

    doc = (ROOT / "docs" / "AWS.md").read_text(encoding="utf-8")
    # WHITESPACE IS NORMALISED BEFORE EVERY PHRASE MATCH. A
    # sentence a markdown line break splits cannot be grepped
    # however correctly the page reads - this project has paid
    # for that six times.
    flat = " ".join(doc.split())
    for x in cloud.STATUS_VERIFIED + cloud.STATUS_OWED:
        check("the AWS doc states the status line {!r}... so the "
              "module and the document cannot disagree about a "
              "deployment's state".format(x[:46]),
              " ".join(x.split()) in flat)

    # ----------------------------------------------------------
    # THE CLI DOORS. `publish` did not exist: `cloud.publish` was
    # reachable only from `batch` after a fresh fit, so a run
    # already on disk could not be published through the tool at
    # all - and the rename a recipient asked for, which publish
    # grew a parameter for, had no path to reach it.
    # ----------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        run = Path(td) / "run"
        dest = Path(td) / "dest"
        run.mkdir()
        dest.mkdir()
        for nm in ("blueprint.json", "fidelity.json",
                   "generated.csv"):
            (run / nm).write_text("{}", encoding="utf-8")
        (run / "tidy_visits.csv").write_text("real,data",
                                             encoding="utf-8")
        (run / "key.csv").write_text("keep,this", encoding="utf-8")

        rc, out = _cli(["publish", str(run), str(dest),
                        "--dry-run"])
        check("`synthkit publish --dry-run` lists what would "
              "leave", rc == 0 and "generated.csv" in out)
        check("...and what STAYS, by name, which is the line a "
              "governance reviewer reads",
              "key.csv" in out.split("what STAYS")[-1]
              and "tidy_visits.csv" in out.split("what STAYS")[-1])
        check("...and it sends nothing", not list(dest.iterdir()))

        rc, out = _cli(["publish", str(run), str(dest),
                        "--only", "tidy_visits.csv"])
        check("asking the CLI for the source BY NAME is refused "
              "with the rule, not a traceback",
              rc == 2 and "not on the list" in out)
        rc, out = _cli(["publish", str(run), str(dest),
                        "--only", "key.csv",
                        "--as-name", "key.csv=cohort.csv"])
        check("...and renaming the KEY does not help it travel, "
              "because the check runs on the SOURCE name",
              rc == 2 and "not on the list" in out)
        check("...and nothing landed from either refusal",
              not list(dest.iterdir()))

        rc, out = _cli([
            "publish", str(run), str(dest), "--as-name",
            "generated.csv=confidential_synthetic_cohort.csv"])
        check("a published file can be renamed through the CLI, "
              "which is the request that made `as_name` exist",
              rc == 0
              and (dest / "confidential_synthetic_cohort.csv"
                   ).exists()
              and not (dest / "generated.csv").exists())
        rc, out = _cli(["publish", str(run), str(dest),
                        "--as-name", "generated.csv"])
        check("a rename that is not OLD=NEW is refused by name",
              rc == 2 and "OLD=NEW" in out)
        rc, out = _cli(["publish", str(run), str(dest),
                        "--as-name",
                        "generated.csv=sub/dir/x.csv"])
        check("...and a rename naming a PATH is refused, because "
              "the destination prefix comes from the URI",
              rc == 2 and "names a path" in out)

    # BATCH FORWARDS THE FIT'S OWN FLAGS. It could pass only
    # --group-by and --max-rows, so the one entrypoint a
    # deployment uses could not reproduce the run we ship.
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "t.csv"
        src.write_text("person_id,x\nA,1\n", encoding="utf-8")
        rc, out = _cli(["batch", "--src", str(src), "--out",
                        str(Path(td) / "w"), "--dry-run",
                        "--seed", "11", "--revalue-copied"])
        check("`batch` accepts the fit's own flags rather than "
              "refusing them, so a deployment can set the dials "
              "the deliverable was made with", rc == 0)
        rc, out = _cli(["batch", "--src", str(src), "--out",
                        str(Path(td) / "w2"), "--src",
                        "s3://b/other.csv"])
        check("...and a forwarded flag THIS command already "
              "controls is refused BY NAME rather than winning "
              "or losing silently",
              rc == 2 and "--src" in out)

    # AND A FORWARDED FLAG MUST REACH THE FIT AND CHANGE IT.
    # Asserting the message would pass on a command that prints
    # "forwarding" and drops the argument - a flag read and not
    # used is this project's oldest recorded fault. So: two runs
    # one seed apart, through the real command, and the OUTPUT
    # must differ and the provenance must record the seed it was
    # given.
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "s.csv"
        rows = ["person_id,v,x,y"]
        for pid in range(26):
            for vis in range(6):
                rows.append("P%02d,%d,%.3f,%.3f" % (
                    pid, vis, 10 + pid * 0.2 + vis * 0.1,
                    3 + vis * 0.4 + pid * 0.05))
        src.write_text("\n".join(rows) + "\n", encoding="utf-8")
        outs = {}
        for seed in ("11", "37"):
            dest = Path(td) / ("d" + seed)
            dest.mkdir()
            rc, _ = _cli(["batch", "--src", str(src), "--out",
                          str(Path(td) / ("r" + seed)),
                          "--group-by", "person_id",
                          "--publish", str(dest), "--seed", seed])
            outs[seed] = (rc, dest)
        check("a batch run with a forwarded --seed completes",
              all(rc == 0 for rc, _ in outs.values()))
        recs = {}
        for seed, (_, dest) in outs.items():
            pr = json.loads(
                (dest / "provenance.json").read_text(
                    encoding="utf-8"))
            recs[seed] = str((pr.get("settings") or {}).get("seed"))
        check("...and the run RECORDS the seed it was forwarded, "
              "so the setting is not merely accepted: {}"
              .format(recs),
              recs.get("11") == "11" and recs.get("37") == "37")
        a = (outs["11"][1] / "generated.csv").read_bytes()
        b = (outs["37"][1] / "generated.csv").read_bytes()
        check("...and the OUTPUT differs between the two, which "
              "is the only thing that distinguishes a forwarded "
              "flag from a printed one",
              a != b)

    # EVERY NEW DOOR ANSWERS --help, which is the recorded
    # argparse trap: a %-formatted help string raises eagerly on
    # 3.14 and took down every invocation of a script once.
    for cmd in ("publish", "cloud-check", "cloud-policy",
                "batch"):
        rc, out = _cli([cmd, "--help"])
        check("`synthkit {} --help` renders and exits 0"
              .format(cmd), rc == 0 and "usage" in out.lower())

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


class _Boom(Exception):
    """A planted provider failure.

    The class NAME and the error CODE are what `diagnose` reads,
    so both are settable here - which is what lets every cause be
    exercised with no boto3, no credentials and no account.
    """

    def __init__(self, msg, code=None, cls=None):
        super().__init__(msg)
        if code:
            self.response = {"Error": {"Code": code}}
        if cls:
            self.__class__ = type(cls, (_Boom,), {})


class _FakeS3:
    """A second store implementation, not a mock of boto3.

    A cloud path tested only against a mock is tested against the
    mock. This behaves like an object store - it keeps bytes, it
    can be made to fail one operation, and it RECORDS the calls
    it saw, which is what lets "read-only means read-only" be
    asserted on behaviour rather than on the report's own say-so.
    """

    def __init__(self, fail_on=None, corrupt=False,
                 code="AccessDenied"):
        self.obj = {}
        self.fail_on = set(fail_on or ())
        self.corrupt = corrupt
        self.code = code
        self.calls = []
        self.last_extra = None
        self.put_bodies = []

    def _maybe(self, op):
        self.calls.append(op)
        if op in self.fail_on:
            raise _Boom("denied on " + op, code=self.code)

    def list_objects_v2(self, **kw):
        self._maybe("list")
        return {"KeyCount": 0}

    def put_object(self, Bucket, Key, Body, **extra):
        self._maybe("put")
        self.last_extra = extra
        self.put_bodies.append(Body)
        self.obj[Key] = Body
        return {}

    def get_object(self, Bucket, Key):
        self._maybe("get")
        body = self.obj[Key]
        if self.corrupt:
            body = body + b"x"
        return {"Body": io.BytesIO(body)}

    def delete_object(self, Bucket, Key):
        self._maybe("delete")
        self.obj.pop(Key, None)
        return {}

    def upload_file(self, local, Bucket, Key, ExtraArgs=None):
        self._maybe("put")
        self.last_extra = ExtraArgs or {}
        data = Path(local).read_bytes()
        self.put_bodies.append(data)
        self.obj[Key] = data
        return {}


def _cli(argv):
    """Run the real CLI in-process and capture what it printed."""
    import contextlib
    from synthkit import cli
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), \
                contextlib.redirect_stderr(buf):
            rc = cli.main(argv)
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else 1
    return rc, buf.getvalue()


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
