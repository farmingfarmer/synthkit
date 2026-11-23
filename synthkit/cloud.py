"""Running in a cloud account without the data leaving it.

WHAT THIS IS FOR. A hospital's own AWS account, under its own
agreements, is a legitimate place to run this - an analyst
should not have to find a laptop with 16GB free to fit an
extract. What it must never become is a route by which records
leave the account they were already in.

THE RULE, AND IT IS ENFORCED RATHER THAN DOCUMENTED:

ONLY NAMED ARTIFACTS TRAVEL. The uploader takes an ALLOW-LIST
and refuses everything else BY NAME. The source extract is not
on it and cannot be put on it by passing a flag - a run that
could upload its own input is one bad argument away from
publishing an extract, and this project's whole structure
exists to make that impossible rather than unlikely.

THE STORE IS AN INTERFACE, NOT A LIBRARY CALL. S3 is one
implementation and a local directory is another, so the whole
path is exercised by the checks with no network, no
credentials and no mocking of somebody else's SDK. A cloud
path tested only against a mock is tested against the mock.

WHAT HAS AND HAS NOT RUN LIVE IS A LIST, NOT A SENTENCE. The
module used to say "nothing here has run against a live AWS
account", which was true when written and stopped being true
the day the deliverable was uploaded to a real bucket in the
dev account - and a stale claim in the one place a governance
reviewer trusts is worse than no claim. `STATUS` carries the
verified lines and the owed ones separately, `describe()`
prints them, and a check asserts the AWS doc says the same -
two places stating a deployment's status is how they come to
disagree.
"""
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict

# WHAT MAY LEAVE. Everything a run writes that is an aggregate,
# a synthetic file, or a report. The SOURCE is absent and the
# absence is the point: `generated.csv` is synthetic by
# construction and `key.csv` is NOT here either, because the
# key is the full-fidelity file the operator KEEPS - uploading
# it would defeat the arrangement it exists for.
ALLOWED = (
    "blueprint.json",
    "catalogue.json",
    "fidelity.json",
    "findings.txt",
    "generated.csv",
    "generated_repaired.csv",
    "generated_dialled.csv",
    "provenance.json",
    "cipher.json",
    "tablespec.json",
    "deck.html",
    "report_card.html",
    "signoff.html",
    # THE DICTIONARY IS A DELIVERABLE AND WAS NOT PUBLISHABLE.
    # It ships in the handover package and a recipient asked for
    # it by name, and `publish` refused - the list had never been
    # reconciled against what the packer actually sends. It
    # discloses nothing `blueprint.json` does not already: column
    # names, types, coverage, ranges, and the token vocabulary the
    # k rule already screened. Added deliberately rather than by
    # loosening the check.
    "data_dictionary.html",
    "data_dictionary.csv",
    "schema.json",
)

S3_RE = re.compile(r"^s3://([^/]+)/?(.*)$")

# WHAT IS VERIFIED, AND WHAT IS OWED. Kept as data rather than
# prose because it changes, and because the AWS doc and this
# module must not be able to say different things about it. Each
# line is a FACT somebody can check, not a reassurance.
STATUS_VERIFIED = (
    "A real upload into a real bucket in the dev account has "
    "happened: the 800-patient deliverable's dataset and data "
    "dictionary, from the data machine, under the account's own "
    "credentials.",
    "It needed AWS_CA_BUNDLE, because the network intercepts "
    "TLS. Verification was never disabled.",
    "The allow-list, the all-or-nothing refusal and the rename "
    "rule are exercised by `smoke_cloud` against a second store "
    "implementation, with no network and no credentials.",
    "The image builds, runs as non-root, carries the net, and "
    "holds no csv, no data/ and no run_* directory.",
    "A full `batch` fit runs end to end INSIDE THE CONTAINER: "
    "source in, six allow-listed artifacts out, the source not "
    "among them, with a forwarded --seed reaching the fit. What "
    "that does NOT establish is the account - see the owed list.",
)
STATUS_OWED = (
    "`synthkit batch` has NOT been run inside the ACCOUNT. It "
    "runs end to end in the container locally; what is owed is "
    "the same run against an s3:// source and prefix, which is a "
    "different thing and the one that matters.",
    "No run has used a TASK ROLE. The credentials so far were a "
    "pasted session, which is not what a deployment will use.",
    "The container has NOT been pushed to ECR, and no task "
    "definition exists.",
    "No bucket requiring server-side encryption has been written "
    "to, so the --sse path is code and checks rather than a "
    "measurement.",
)

# SERVER-SIDE ENCRYPTION MODES WE CAN ASK FOR. The KEY and the
# POLICY belong with the bucket - that is the account owner's to
# write - but the HEADER is ours to send, and a bucket whose
# policy requires it rejects a plain put with an access-denied
# that reads like a permissions problem.
SSE_CHOICES = ("", "AES256", "aws:kms")


class CloudRefusal(Exception):
    """A refusal that names the file and the rule."""


def is_s3(uri: str) -> bool:
    return bool(S3_RE.match(str(uri or "")))


def split_s3(uri: str) -> Tuple[str, str]:
    m = S3_RE.match(str(uri))
    if not m:
        raise CloudRefusal("{!r} is not an s3:// URI".format(uri))
    return m.group(1), m.group(2)


class LocalStore:
    """A directory pretending to be a bucket.

    Not a mock of S3 - a second real implementation of the same
    two operations. The checks drive the whole upload path
    through it, so what they exercise is this module's logic
    rather than somebody else's SDK behaving as its author
    imagined.
    """

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, local: Path, key: str) -> str:
        dest = self.root / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(local).read_bytes())
        return str(dest)

    def get(self, key: str, local: Path) -> str:
        src = self.root / key
        if not src.exists():
            raise CloudRefusal("no object at {}".format(key))
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        Path(local).write_bytes(src.read_bytes())
        return str(local)


class S3Store:
    """The real one. Constructed only when boto3 is present."""

    def __init__(self, bucket: str, client=None, sse: str = "",
                 kms_key: str = ""):
        if client is None:
            try:
                import boto3
            except ImportError:
                raise CloudRefusal(
                    "reading or writing s3:// needs the "
                    "`boto3` package, which is not installed. "
                    "Run: pip install boto3")
            client = boto3.client("s3")
        self.bucket = bucket
        self.client = client
        self.sse = str(sse or "")
        self.kms_key = str(kms_key or "")
        if self.sse and self.sse not in SSE_CHOICES:
            raise CloudRefusal(
                "{!r} is not a server-side encryption mode this "
                "knows: pick one of {}".format(
                    self.sse, ", ".join(c for c in SSE_CHOICES if c)))
        if self.kms_key and self.sse != "aws:kms":
            raise CloudRefusal(
                "a KMS key was given but the encryption mode is "
                "{!r} - pass --sse aws:kms, or drop the key. A key "
                "that is read and not used is worse than no key, "
                "because somebody believes it was applied."
                .format(self.sse or "none"))

    def extra_args(self) -> Dict[str, str]:
        """The encryption headers, or nothing.

        A BUCKET POLICY CAN REQUIRE THESE AND REJECT A PLAIN PUT.
        That is the likeliest way a first deployment fails against
        a hospital bucket - `s3:PutObject` denied with a condition
        on `s3:x-amz-server-side-encryption` - and it reads as a
        permissions problem rather than a missing header. So the
        mode is ours to pass, even though the KEY and the POLICY
        belong with the bucket.
        """
        if not self.sse:
            return {}
        extra = {"ServerSideEncryption": self.sse}
        if self.kms_key:
            extra["SSEKMSKeyId"] = self.kms_key
        return extra

    def put(self, local: Path, key: str) -> str:
        extra = self.extra_args()
        if extra:
            self.client.upload_file(str(local), self.bucket, key,
                                    ExtraArgs=extra)
        else:
            self.client.upload_file(str(local), self.bucket, key)
        return "s3://{}/{}".format(self.bucket, key)

    def get(self, key: str, local: Path) -> str:
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(local))
        return str(local)


def store_for(uri: str, client=None, sse: str = "",
              kms_key: str = ""):
    """The store a URI names, and the prefix inside it."""
    if is_s3(uri):
        bucket, prefix = split_s3(uri)
        return (S3Store(bucket, client=client, sse=sse,
                        kms_key=kms_key), prefix)
    if sse or kms_key:
        raise CloudRefusal(
            "server-side encryption was asked for but {!r} is a "
            "local directory, which has none - a setting that "
            "cannot be applied must not be silently ignored"
            .format(uri))
    return LocalStore(uri), ""


def refuse_unless_allowed(names: Iterable[str]) -> None:
    """Every name must be on the list, or nothing goes.

    ALL OR NOTHING, deliberately. Uploading the permitted
    files and skipping the rest would leave an operator
    believing the whole run had shipped, and a partial
    upload that looks complete is the silent-difference
    fault this project refuses everywhere else.
    """
    bad = [n for n in names if Path(n).name not in ALLOWED]
    if bad:
        raise CloudRefusal(
            "these are not on the list of things allowed to "
            "leave the account: {}. The list is: {}. The "
            "SOURCE EXTRACT is deliberately absent from it "
            "and cannot be added by a flag - a run that could "
            "upload its own input is one bad argument away "
            "from publishing an extract."
            .format(", ".join(sorted(Path(b).name for b in bad)),
                    ", ".join(ALLOWED)))


def publish(run_dir, dest_uri: str, client=None,
            only: Optional[List[str]] = None,
            as_name: Optional[Dict[str, str]] = None,
            sse: str = "", kms_key: str = "") -> List[str]:
    """Upload a run's artifacts, and nothing else.

    Returns the destinations written, so a caller can print
    exactly what left - a transfer that does not say what it
    moved is one nobody can audit afterwards.

    `as_name` RENAMES ON THE WAY OUT, and the check still runs on
    the SOURCE. A recipient asked for the synthetic file under a
    name of their choosing, and the allow-list refused it - which
    was the guard working, because it cannot know that an
    arbitrary filename is the synthetic output rather than the
    extract. Checking the source and letting the destination
    differ keeps the guard exactly as strong: `key.csv` is still
    refused, and renaming it does not help.

    A rename whose SOURCE is not on the list is refused the same
    way an unrenamed one is, and the returned list names what was
    actually written rather than what was asked for.
    """
    run = Path(run_dir)
    if not run.is_dir():
        raise CloudRefusal("{} is not a directory".format(run))
    names = ([n for n in only] if only
             else [p.name for p in sorted(run.iterdir())
                   if p.is_file() and p.name in ALLOWED])
    if not names:
        raise CloudRefusal(
            "nothing in {} is on the allow-list, so there is "
            "nothing to publish. Either the run did not write "
            "its artifacts or this is the wrong "
            "directory.".format(run))
    refuse_unless_allowed(names)
    missing = [n for n in names if not (run / n).exists()]
    if missing:
        raise CloudRefusal(
            "asked to publish {} which {} not in {}".format(
                ", ".join(missing),
                "is" if len(missing) == 1 else "are", run))
    store, prefix = store_for(dest_uri, client=client, sse=sse,
                              kms_key=kms_key)
    ren = dict(as_name or {})
    # A RENAME OF SOMETHING THAT IS NOT BEING PUBLISHED IS A
    # MISTAKE, not a no-op. Silently ignoring it is how somebody
    # believes a file went out under a name it never had.
    stray = [k for k in ren if k not in names]
    if stray:
        raise CloudRefusal(
            "asked to rename {} on the way out, but {} not being "
            "published from {}".format(
                ", ".join(sorted(stray)),
                "it is" if len(stray) == 1 else "they are", run))
    out = []
    for n in names:
        dest = ren.get(n, n)
        key = "{}/{}".format(prefix.rstrip("/"), dest) if prefix \
            else dest
        out.append(store.put(run / n, key))
    return out


def fetch_source(uri: str, local_dir, client=None) -> str:
    """Bring the source INTO the compute, never the reverse.

    One direction only. There is no function here that sends a
    source anywhere, and that is not an oversight.
    """
    local_dir = Path(local_dir)
    if not is_s3(uri):
        p = Path(uri)
        if not p.exists():
            raise CloudRefusal("{} does not exist".format(uri))
        return str(p)
    bucket, key = split_s3(uri)
    if not key:
        raise CloudRefusal(
            "{!r} names a bucket but no object - give the full "
            "path to the file".format(uri))
    store = S3Store(bucket, client=client)
    dest = local_dir / Path(key).name
    return store.get(key, dest)


# WHAT GOES WRONG ON A FIRST DEPLOYMENT, AND WHAT TO DO.
#
# Every entry is a failure somebody will meet, with the remedy
# in the same sentence. The first one is not hypothetical: the
# upload of the 800-patient deliverable failed on exactly it,
# because the network intercepts TLS, and the raw error names a
# certificate rather than the thing to set.
#
# The classification is a PURE FUNCTION on an exception, keyed on
# the type NAME and the error code, so it is tested on planted
# inputs with no account, no credentials and no boto3 - which is
# the same reason the store is an interface. A diagnosis that can
# only be exercised by failing against a real bucket is one
# nobody will exercise.
CAUSES = (
    ("tls-intercepted",
     "the TLS connection was refused by certificate validation. "
     "On a network that intercepts TLS, boto3 does not trust the "
     "intercepting root: export AWS_CA_BUNDLE pointing at a PEM "
     "holding the machine's trusted roots. Do NOT disable "
     "verification - this is confidential data in flight."),
    ("sse-required",
     "the bucket policy REQUIRES server-side encryption and the "
     "put did not ask for it. Pass --sse aws:kms with the "
     "bucket's key, or --sse AES256. This reads as a permissions "
     "problem and is a missing header."),
    ("no-credentials",
     "no credentials were found at all. In a task this means the "
     "task role is not attached; on a laptop it means the "
     "environment or profile is not set. Nothing here reads or "
     "stores a credential."),
    ("expired",
     "the credentials were found and have EXPIRED. Re-issue the "
     "session; a stale token reads identically to a missing "
     "permission."),
    ("denied",
     "the identity was accepted and the action was DENIED. The "
     "role needs the actions this tool performs and no others - "
     "`synthkit cloud-policy` prints exactly that policy."),
    ("no-bucket",
     "there is no such bucket from this identity. Either the "
     "name is wrong or it belongs to another account."),
    ("wrong-region",
     "the bucket is in a different region from the client. Set "
     "AWS_REGION, or AWS_DEFAULT_REGION, to the bucket's own."),
    ("unreachable",
     "the endpoint could not be reached. In a private subnet "
     "with no egress this needs an S3 VPC endpoint; otherwise it "
     "is a route or a DNS problem."),
    ("unknown",
     "the cause is not one this knows. The error is printed "
     "verbatim below rather than summarised, because a guess "
     "here sends somebody after the wrong thing."),
)
CAUSE_TEXT = dict(CAUSES)


def _error_code(exc) -> str:
    """The provider's own code for a failure, or ''."""
    resp = getattr(exc, "response", None)
    if isinstance(resp, dict):
        err = resp.get("Error") or {}
        if isinstance(err, dict):
            return str(err.get("Code") or "")
    return ""


def diagnose(exc) -> str:
    """Name the cause of a cloud failure, or say it is unknown.

    Returns a key into CAUSE_TEXT. Pure: it reads the exception's
    TYPE NAME, the provider's error code and the message, and
    touches no network.
    """
    name = type(exc).__name__
    code = _error_code(exc)
    msg = "{} {}".format(exc, code)
    low = msg.lower()
    # ORDER MATTERS. An SSE-required refusal arrives AS an
    # access-denied, so the specific test runs first or the
    # generic one swallows it and sends somebody to the IAM
    # console for a missing header.
    if ("ssl" in name.lower() or "certificate" in low
            or "certificate_verify_failed" in low):
        return "tls-intercepted"
    if ("server-side encryption" in low
            or "x-amz-server-side-encryption" in low
            or "serversideencryption" in low):
        return "sse-required"
    if "nocredential" in name.lower() or "partialcredential" in \
            name.lower() or "unable to locate credentials" in low:
        return "no-credentials"
    if "expired" in low or code in ("ExpiredToken",
                                    "ExpiredTokenException",
                                    "RequestExpired"):
        return "expired"
    if code in ("NoSuchBucket",) or "nosuchbucket" in low:
        return "no-bucket"
    if code in ("PermanentRedirect", "IllegalLocationConstraint"
                                     "Exception", "301") or \
            "wrong region" in low or "permanentredirect" in low:
        return "wrong-region"
    if code in ("AccessDenied", "AccessDeniedException",
                "InvalidAccessKeyId", "SignatureDoesNotMatch",
                "403") or "access denied" in low:
        return "denied"
    if ("endpointconnection" in name.lower()
            or "connecttimeout" in name.lower()
            or "could not connect" in low
            or "name or service not known" in low):
        return "unreachable"
    return "unknown"


# WHAT THE CANARY CONTAINS. A fixed constant, exposed so a check
# can assert the bytes that go out are THESE bytes and not
# something assembled from the run - a preflight that wrote a
# sample of the data would be the exact thing this module exists
# to prevent.
CANARY_BODY = b"synthkit preflight object - safe to delete\n"


def preflight(dest_uri: str, client=None, sse: str = "",
              kms_key: str = "", allow_write: bool = False,
              canary: str = "synthkit-preflight.txt") -> Dict:
    """Prove the destination works BEFORE a fit spends an hour.

    READ-ONLY BY DEFAULT. Writing an object into somebody's
    bucket is an outward action, so the write test is opt-in and
    the report says it was skipped and how to run it. A preflight
    that quietly writes is a preflight nobody can hand to a
    reviewer.

    The canary carries a fixed harmless line and no record, and it
    is DELETED afterwards. If the delete fails the report says so
    by name, because an object left behind in a hospital bucket
    is somebody's cleanup.

    Returns {"steps": [...], "ok": bool, "wrote": bool}. Each
    step is {"step", "ok", "detail"} with `ok` None where the
    step was skipped - absent is not failure, which is the
    distinction this project draws everywhere.
    """
    steps: List[Dict] = []

    def add(step, ok, detail):
        steps.append({"step": step, "ok": ok, "detail": detail})

    if not is_s3(dest_uri):
        add("destination is s3", None,
            "{!r} is a local directory - there is nothing to "
            "preflight, and a local publish needs no "
            "credentials".format(dest_uri))
        return {"steps": steps, "ok": True, "wrote": False,
                "local": True}

    bucket, prefix = split_s3(dest_uri)
    add("destination is s3", True,
        "bucket {}, prefix {!r}".format(bucket, prefix or ""))

    try:
        store, prefix = store_for(dest_uri, client=client, sse=sse,
                                  kms_key=kms_key)
    except CloudRefusal as e:
        add("client and encryption settings", False, str(e))
        return {"steps": steps, "ok": False, "wrote": False,
                "local": False}
    add("client and encryption settings", True,
        "encryption " + (
            "{}{}".format(sse, " with a KMS key" if kms_key else "")
            if sse else "not requested - the bucket's own default "
                        "applies"))

    # CAN WE SEE THE BUCKET AT ALL. One list of one object: the
    # cheapest call that distinguishes a wrong name from a denied
    # action from an unreachable endpoint.
    try:
        store.client.list_objects_v2(Bucket=bucket, MaxKeys=1)
        add("the bucket is reachable and listable", True,
            "list_objects_v2 returned")
    except Exception as e:  # noqa: BLE001
        cause = diagnose(e)
        add("the bucket is reachable and listable", False,
            "{}: {} [{}]".format(cause, CAUSE_TEXT[cause], e))
        return {"steps": steps, "ok": False, "wrote": False,
                "local": False}

    if not allow_write:
        add("a write test", None,
            "SKIPPED - writing an object into this bucket is an "
            "outward action and is opt-in. Re-run with --write to "
            "put, read back and delete {!r}.".format(canary))
        return {"steps": steps, "ok": True, "wrote": False,
                "local": False}

    key = "{}/{}".format(prefix.rstrip("/"), canary) if prefix \
        else canary
    body = CANARY_BODY
    try:
        extra = store.extra_args()
        store.client.put_object(Bucket=bucket, Key=key, Body=body,
                                **extra)
        add("a canary object can be written", True,
            "put s3://{}/{}".format(bucket, key))
    except Exception as e:  # noqa: BLE001
        cause = diagnose(e)
        add("a canary object can be written", False,
            "{}: {} [{}]".format(cause, CAUSE_TEXT[cause], e))
        return {"steps": steps, "ok": False, "wrote": False,
                "local": False}

    # READ IT BACK AND COMPARE THE BYTES. A put that succeeds and
    # a get that returns something else is the silent-difference
    # fault, and it is exactly what an encryption or a proxy
    # misconfiguration can produce.
    try:
        got = store.client.get_object(Bucket=bucket, Key=key)
        back = got["Body"].read()
        same = back == body
        add("the canary reads back byte for byte", same,
            "identical" if same else
            "the bytes returned differ from the bytes written, "
            "which no permission problem explains")
    except Exception as e:  # noqa: BLE001
        cause = diagnose(e)
        add("the canary reads back byte for byte", False,
            "{}: {} [{}]".format(cause, CAUSE_TEXT[cause], e))
        same = False

    try:
        store.client.delete_object(Bucket=bucket, Key=key)
        add("the canary is deleted", True, "removed")
    except Exception as e:  # noqa: BLE001
        cause = diagnose(e)
        add("the canary is deleted", False,
            "LEFT BEHIND at s3://{}/{} - {}: {} [{}]".format(
                bucket, key, cause, CAUSE_TEXT[cause], e))

    ok = all(st["ok"] is not False for st in steps)
    return {"steps": steps, "ok": ok, "wrote": True,
            "local": False}


def iam_policy(src_uri: str = "", publish_uri: str = "") -> Dict:
    """The least-privilege policy this tool needs, and no more.

    WHY THIS IS GENERATED RATHER THAN WRITTEN. `docs/AWS.md` says
    the task role is the account owner's to write, and that is
    still true - but handing somebody "write a policy" is handing
    them a guess. The ACTIONS are a property of this code: it
    GETS one object and PUTS several, and it lists a prefix to
    preflight. Anything else in a policy came from somewhere
    other than this tool.

    Deliberately absent: `s3:DeleteObject` on the publish prefix,
    because nothing here deletes an artifact; `s3:*`; and any
    resource outside the two prefixes named.
    """
    def arn(uri):
        if not is_s3(uri):
            return None, None
        b, k = split_s3(uri)
        return b, k

    stmts: List[Dict] = []
    sb, sk = arn(src_uri)
    if sb:
        stmts.append({
            "Sid": "ReadTheExtract",
            "Effect": "Allow",
            "Action": ["s3:GetObject"],
            "Resource": "arn:aws:s3:::{}/{}".format(sb, sk)})
    pb, pk = arn(publish_uri)
    if pb:
        pref = (pk.rstrip("/") + "/*") if pk else "*"
        stmts.append({
            "Sid": "WriteTheArtifacts",
            "Effect": "Allow",
            "Action": ["s3:PutObject"],
            "Resource": "arn:aws:s3:::{}/{}".format(pb, pref)})
        stmts.append({
            "Sid": "PreflightTheDestination",
            "Effect": "Allow",
            "Action": ["s3:ListBucket"],
            "Resource": "arn:aws:s3:::{}".format(pb),
            "Condition": {"StringLike": {
                "s3:prefix": [(pk.rstrip("/") + "/*")
                              if pk else "*"]}}})
    return {"Version": "2012-10-17", "Statement": stmts}


def policy_notes() -> List[str]:
    """What the generated policy does NOT cover, said out loud."""
    return [
        "These are the actions THIS TOOL performs. Anything else "
        "in a role came from somewhere else.",
        "s3:DeleteObject is absent on the publish prefix because "
        "nothing here deletes an artifact. The preflight canary "
        "needs it - run the preflight read-only, or grant it on "
        "the canary key alone.",
        "KMS is absent. If the bucket requires aws:kms the role "
        "also needs kms:GenerateDataKey and kms:Decrypt on that "
        "key, which belongs with the key's own policy.",
        "No VPC, subnet, task definition or bucket policy is "
        "here. Writing those blind would be guessing at "
        "somebody's network.",
    ]


def describe() -> List[str]:
    """What this deployment will and will not do, in lines."""
    return [
        "SOURCE comes IN to the compute; only named artifacts "
        "go OUT.",
        "What may leave: " + ", ".join(ALLOWED) + ".",
        "The source extract is NOT on that list and cannot be "
        "added by a flag.",
        "`key.csv` is not on it either - the key is the "
        "full-fidelity file the operator keeps, and "
        "publishing it would defeat the arrangement it "
        "exists for.",
        "Everything runs in the account that already holds "
        "the data; nothing in this module sends a record "
        "anywhere.",
    ] + ["VERIFIED: " + x for x in STATUS_VERIFIED] \
        + ["STILL OWED: " + x for x in STATUS_OWED]
