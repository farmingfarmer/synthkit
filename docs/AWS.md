# SynthKit in a hospital AWS account

**Nothing in this document has been run against a live AWS
account.** It is the code, the guard and the shape, verified
locally and checked by `smoke_cloud`. The first real deployment
is still owed a run, and this project's rule is that a thing is
not measured until somebody measures it — a deployment included.

---

## The shape, and why it is that shape

The extract already lives somewhere. The question a deployment
answers is not "how do we get the data to the tool" but "how do
we get the tool to the data" — so the compute runs **in the
account that already holds the extract**, and the only things
that leave are artifacts somebody chose to publish.

```
  ┌─────────────────── the account that holds the data ───────────────────┐
  │                                                                        │
  │   s3://…/extract.csv  ──►  ECS / Batch task  ──►  work dir            │
  │        (never leaves)        synthkit batch        blueprint.json     │
  │                                                     generated.csv      │
  │                                                     fidelity.json      │
  │                                                     findings.txt       │
  │                                                          │             │
  │                                                          ▼             │
  │                                              s3://…/artifacts/         │
  └────────────────────────────────────────────────────────────────────────┘
```

One direction for data. `synthkit/cloud.py` has a
`fetch_source` and no counterpart that sends a source anywhere;
that absence is the design, not an omission.

---

## What may leave, enforced rather than documented

`cloud.ALLOWED` is an allow-list, and `publish()` refuses
anything not on it **by name**:

> these are not on the list of things allowed to leave the
> account: `tidy_visits.csv`. … The SOURCE EXTRACT is
> deliberately absent from it and cannot be added by a flag — a
> run that could upload its own input is one bad argument away
> from publishing an extract.

Two refusals are worth calling out:

- **The source extract.** It sits in the same directory as the
  artifacts during a run, so "upload the run directory" is the
  natural mistake. It is refused.
- **`key.csv`.** The key is the full-fidelity synthetic file
  the operator **keeps** — the half of the key/released pair
  that never travels. Publishing it would defeat the whole
  arrangement, so it is off the list too.

A request naming both an allowed file and a refused one is
refused **whole**. A partial upload that looks complete is the
silent-difference fault this project refuses everywhere else.

---

## Running it

```
synthkit batch --src s3://BUCKET/extract.csv \
               --out /work/run1 \
               --group-by person_id \
               --publish s3://BUCKET/artifacts/run1 \
               --dry-run
```

`--dry-run` prints the posture and what *would* leave, and
starts nothing. A transfer nobody previewed is one nobody can
object to. Drop the flag to run the fit.

## The container

`Dockerfile` pins `python:3.14-slim`, because this project has
twice been bitten by testing against a different interpreter
than the one that runs — pandas 3 turned a `FutureWarning` into
a `TypeError`, and 3.14 turned a latent help-string bug into a
crash on every invocation.

It runs as a non-root user, and `.dockerignore` excludes
`*.csv`, `data/` and `run_*/`: an extract swept into a layer
would travel with every pull of the image, which is the worst
possible way for a record to leave an account.

The net ships inside the image, so a deployment can prove the
tool works **before** it is pointed at an extract:

```
docker run --rm IMAGE python /opt/synthkit/scripts/run_all_smokes.py
```

---

## What this does NOT do, named

- **No IAM, VPC or Terraform here.** The task role, the subnet
  and the bucket policy are the account owner's to write, and
  writing them blind would be guessing at somebody's network.
  What this gives them is a container with one entry point and
  a stated data flow to review.
- **No cross-account anything.** Every path in this module is
  within one account by construction.
- **No encryption configuration.** S3 server-side encryption
  and KMS keys are bucket policy, which belongs with the bucket.
- **No secrets.** `boto3` picks up the task role; nothing here
  reads or stores a credential.
- **Not run against a live account.** Stated again because it
  is the thing most likely to be assumed.

## The question a governance reviewer will ask

*"Can this send our data somewhere?"*

The honest answer is the allow-list and the direction of
`fetch_source`: the source comes in, named artifacts go out,
the list cannot be extended by a flag, and `smoke_cloud` proves
it by putting an extract and a key file in the run directory
and demanding they stay. Three mutations — allowing the source,
honouring a mixed request partly, and adding `key.csv` to the
list — each turn the suite red.

That is a check, not a promise.
