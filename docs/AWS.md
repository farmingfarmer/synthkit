# SynthKit in a hospital AWS account

## Where this actually stands

Not a reassurance — a list somebody can check. These lines live
in `synthkit/cloud.py` as `STATUS_VERIFIED` and `STATUS_OWED`,
`synthkit batch` and `synthkit publish` print them before doing
anything, and `smoke_cloud` asserts this document says the same.
Two places stating a deployment's status is how they come to
disagree.

**Verified.**

- A real upload into a real bucket in the dev account has happened:
  the 800-patient deliverable's dataset and data dictionary, from
  the data machine, under the account's own credentials.
- It needed AWS_CA_BUNDLE, because the network intercepts TLS.
  Verification was never disabled.
- The allow-list, the all-or-nothing refusal and the rename rule are
  exercised by `smoke_cloud` against a second store implementation,
  with no network and no credentials.
- The image builds, runs as non-root, carries the net, and holds no
  csv, no data/ and no run_* directory.
- A full `batch` fit runs end to end INSIDE THE CONTAINER: source
  in, six allow-listed artifacts out, the source not among them,
  with a forwarded --seed reaching the fit. What that does NOT
  establish is the account - see the owed list.
- Both architectures build and run: linux/amd64 and linux/arm64. An
  unpinned build gives the BUILDER's, and Fargate defaults to
  X86_64, so the task definition names it rather than inheriting it.
- The build context excludes csv and data at ANY depth, verified by
  scanning the built image rather than by reading .dockerignore - a
  csv was found inside it until today.
- A fit at the extract's scale was MEASURED: 576 s and 961 MB peak
  on 55,200 rows and 107 searched columns, which is where the task's
  cpu and memory defaults come from.

**Still owed.**

- `synthkit batch` has NOT been run inside the ACCOUNT. It runs end
  to end in the container locally; what is owed is the same run
  against an s3:// source and prefix, which is a different thing and
  the one that matters.
- No run has used a TASK ROLE. The credentials so far were a pasted
  session, which is not what a deployment will use.
- The container has NOT been pushed to ECR, and no task definition
  exists.
- No bucket requiring server-side encryption has been written to, so
  the --sse path is code and checks rather than a measurement.

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

**Preflight first.** A failure here is cheap and a failure after
a forty-minute fit is not.

```
synthkit cloud-check s3://BUCKET/artifacts/run1
synthkit cloud-check s3://BUCKET/artifacts/run1 --write
```

Read-only by default — writing an object into somebody's bucket
is an outward action, so the canary put is opt-in, and when it
runs the object is read back byte for byte and deleted. Every
step names its own cause and the remedy in the same sentence:
missing credentials, an expired session, a denied action, the
wrong region, an unreachable endpoint, intercepted TLS, or a
bucket policy that requires server-side encryption. That last
one arrives *as* an access-denied, which reads like a
permissions problem and is a missing header.

**The policy the role needs**, generated from what this code
actually does rather than written from memory:

```
synthkit cloud-policy --src s3://BUCKET/extracts/tidy_visits.csv \
                      --publish s3://BUCKET/artifacts/run1
```

It grants `s3:GetObject` on the one extract, `s3:PutObject` on
the one prefix, and `s3:ListBucket` conditioned on that prefix.
Nothing else. It prints what it does **not** cover underneath —
KMS, `s3:DeleteObject`, the VPC, the task definition, the bucket
policy — because a policy that looks complete is worse than one
that names its own gaps.

**The fit.**

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

Anything `batch` does not recognise is **forwarded to the fit**,
so a deployment can set the dials the deliverable was made with:

```
synthkit batch ... --publish s3://BUCKET/artifacts/run1 \
               --seed 11 --revalue-copied
```

It could previously pass only `--group-by` and `--max-rows`, so
the one entrypoint a deployment uses could not reproduce the run
we ship. A forwarded flag this command already controls is
refused by name, and the run prints what it forwarded.

**Publishing a run that already exists.**

```
synthkit publish /work/run1 s3://BUCKET/artifacts/run1 --dry-run
synthkit publish /work/run1 s3://BUCKET/artifacts/run1 \
         --as-name generated.csv=cohort.csv
```

`--dry-run` lists what would leave **and what stays**. The
allow-list check runs on the source name, so renaming a refused
file does not help it travel. This door did not exist: `publish`
was reachable only from `batch` after a fresh fit, and the
rename a recipient asked for had no path to reach it.

**Server-side encryption.** `--sse AES256` or `--sse aws:kms`
with `--sse-kms-key-id`, on both `batch` and `publish`. The key
and the bucket policy belong with the bucket; the *header* is
ours to send, and a bucket that requires it rejects a plain put.
A KMS key passed without `--sse aws:kms` is refused rather than
ignored.

## The container

`Dockerfile` pins `python:3.14-slim`, because this project has
twice been bitten by testing against a different interpreter
than the one that runs — pandas 3 turned a `FutureWarning` into
a `TypeError`, and 3.14 turned a latent help-string bug into a
crash on every invocation.

**The architecture is a choice, and it is not ours to inherit
from whoever builds.** An unpinned `docker build` produces an
image for the *builder's* architecture — this one is built on an
arm64 laptop, so a plain build gives `linux/arm64`, and ECS
Fargate defaults to `X86_64` unless the task definition sets
`runtimePlatform.cpuArchitecture: ARM64`. A mismatch fails with
`exec format error`, which reads as a broken image rather than a
wrong platform. Measured today: **both targets build and run**.

```
docker buildx build --platform linux/amd64 -t synthkit:amd64 --load .
docker buildx build --platform linux/arm64 -t synthkit:arm64 --load .
```

So the ask is one line in the task definition, not a code
change: **tell us which architecture the task will run, or set
`runtimePlatform` to match the image you push.** This project has
twice been bitten by testing against a different interpreter than
the one that runs; this is the same class one layer down.

It runs as a non-root user, and `.dockerignore` excludes
`*.csv` and `data/` **at any depth**, plus `run_*/` at the root:
an extract swept into a layer would travel with every pull of the
image, which is the worst possible way for a record to leave an
account.

Those patterns were top-level only until today, and
`docs/e2e/generated_patients.csv` was found **inside the built
image** — the stated protection did not hold for any file in a
subdirectory. Found by looking in the image, not by reading the
file. The recursive form of `run_*/` is a trap and is
deliberately absent: Docker strips the trailing slash, so
`**/run_*/` becomes `**/run_*` and matches
`scripts/run_all_smokes.py`, which silently removed the net and
the runner from the image. Root-only is sufficient anyway,
because a run directory's record-bearing content is its CSV.

The net ships inside the image, so a deployment can prove the
tool works **before** it is pointed at an extract. The
`--entrypoint` override is not optional: the image's entrypoint
is `synthkit`, so without it `python` is read as a subcommand and
argparse refuses it. The line below was wrong in this document
until somebody ran it.

```
docker run --rm --entrypoint python IMAGE \
    /opt/synthkit/scripts/run_all_smokes.py
```

**It reports one failing suite, and that is expected.**
`smoke_gui` is the local bench — not part of the deployment
surface. Two of its checks cannot pass in a container: the
campaign-compile endpoint wants to write outside `/work`, which
a non-root user cannot do, and its vendor-exam arm comes back
without a vendor. Measured in this image: **137 of 138 suites
green, `smoke_gui` the only failure, on the check "a vendor model
can be graded on the same exam".** Anything else failing is a
real finding. Saying so here because a deployer told to prove the
tool works, who then sees a red suite, will reasonably conclude
the opposite.

---

## How big a task, and how long a timeout

Nobody here could answer this before today, and it is the second
question an infrastructure review asks. **Measured** on a
fixture built to the extract's own scale — 55,200 rows, a
900-code condition vocabulary, four set columns, two dates, the
vitals and counts — which expands to **107 searched columns**:

| | measured |
|---|---|
| wall clock | **576 s** (9.6 min) |
| peak resident memory | **961 MB** |
| CPU consumed | 1,052 s user + 1,921 s system |

Read the last row before sizing the task. The fit used about
**5× parallelism**, so on 2 vCPU the same work takes roughly
three to five times the wall clock — call it **30–50 minutes**,
and set the task timeout well above that rather than at it.

The defaults `synthkit cloud-task` emits are **2048 CPU units
and 8192 MiB**: nearly NINE times the measured peak - 8 GiB
against 961 MB - because the real
extract is wider than this fixture — 44 columns and a
1,827-token published vocabulary — and this project has once
died allocating 1.5 GiB inside a single step. 4 GB would
probably hold and has not been measured; 8 GB has headroom and
costs pennies for a ten-minute task.

**What this is not.** It is a measurement of a fixture at the
extract's scale on an arm64 laptop, not of your extract on
Fargate. The numbers are a starting point with their method
stated, which is strictly more than a guess and strictly less
than a measurement of the thing itself.

```
synthkit cloud-task --image ACCOUNT.dkr.ecr.REGION.amazonaws.com/synthkit:v1 \
                    --src s3://BUCKET/extracts/tidy_visits.csv \
                    --publish s3://BUCKET/synthkit/run1 \
                    --arch X86_64 \
                    --fit-arg=--seed --fit-arg=11 \
                    --fit-arg=--revalue-copied
```

Our half of the task definition is filled in: the image, the
command, the entrypoint behaviour, the architecture, the working
directory, the ephemeral volume, the CPU and the memory. Yours
is left as explicit `REPLACE_ME` placeholders — the two role
ARNs, the log group and the region — and every one is listed
underneath the output. A template whose gaps are silent is worse
than no template, because somebody will apply it.

---

## What this does NOT do, named

- **No VPC or Terraform here.** The subnet, the task
  definition and the bucket policy are the account owner's to
  write, and writing them blind would be guessing at somebody's
  network. What this gives them is a container with one entry
  point, a stated data flow, and a generated least-privilege
  policy for the two S3 operations it performs.
- **No KMS key policy.** If the bucket requires `aws:kms` the
  role also needs `kms:GenerateDataKey` and `kms:Decrypt` on
  that key, which belongs with the key.
- **No cross-account anything.** Every path in this module is
  within one account by construction.
- **No encryption configuration beyond the header.** The key
  and the bucket policy belong with the bucket; `--sse` sends
  the mode, because a bucket that requires it would otherwise
  reject every put with an error that reads like a permissions
  problem.
- **No secrets.** `boto3` picks up the task role; nothing here
  reads or stores a credential.
- **Not yet run end to end in the account.** A publish into a
  real bucket has happened; a `batch` fit under a task role has
  not. The two are separated at the top of this document
  because conflating them is the thing most likely to be
  assumed.

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
