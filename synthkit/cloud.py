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

NOTHING HERE HAS RUN AGAINST A LIVE AWS ACCOUNT. Said at the
top of the module rather than discovered later: this is the
code and the guard, verified locally, and the first real
deployment is still owed a run. This file's rule is that a
number is not measured until somebody measures it, and the
same goes for a deployment.
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

    def __init__(self, bucket: str, client=None):
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

    def put(self, local: Path, key: str) -> str:
        self.client.upload_file(str(local), self.bucket, key)
        return "s3://{}/{}".format(self.bucket, key)

    def get(self, key: str, local: Path) -> str:
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(local))
        return str(local)


def store_for(uri: str, client=None):
    """The store a URI names, and the prefix inside it."""
    if is_s3(uri):
        bucket, prefix = split_s3(uri)
        return S3Store(bucket, client=client), prefix
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
            as_name: Optional[Dict[str, str]] = None) -> List[str]:
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
    store, prefix = store_for(dest_uri, client=client)
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
        "NOT YET RUN AGAINST A LIVE AWS ACCOUNT - this is the "
        "code and the guard, verified locally.",
    ]
