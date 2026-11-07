# Where synthkit sits against the other synthetic data tools

*Compiled 2026-11-07. Read the next section before any number
in this file.*

## What this document is, and what it is not

Our own numbers here are MEASURED, by this repo, on fixtures or
on the real extract, and every one of them is reproducible from
a command named beside it.

Everything about another vendor is READ FROM THEIR PUBLIC
DOCUMENTATION or from peer-reviewed literature. We have not run
their software. A capability is listed as theirs when they
document it, and that is a weaker kind of statement than our
own measurements — a documented feature is a claim, not a
reading. Where we could not establish something, this file says
"not found in public documentation" rather than "they do not
have it", because absence of evidence in a search is not
evidence of absence. That distinction is the same one this
project draws everywhere else between absent and
known-to-be-zero.

Nothing here is a competitive teardown for a sales deck. It is
a build list with the reasoning attached.

## The market moved in 2026

Before comparing features, the landscape itself changed, and a
comparison against a company that no longer sells independently
is worth little:

- **Gretel** was acquired by NVIDIA.
- **Hazy** was acquired by SAS.
- **Syntho** acquired the MOSTLY AI brand (June 2026); MOSTLY
  AI's SDK was open-sourced under Apache 2.0.
- **SDV** (Synthetic Data Vault) moved from MIT to the Business
  Source License — which matters for anyone who assumed it was
  permanently permissive.

The practical read: the independent tabular-synthesis vendors
consolidated into larger platforms, and the open-source options
either loosened (MOSTLY AI's SDK) or tightened (SDV).

## What they have that we do not

These are the real gaps, in the order I would spend on them.

### 1. Differential privacy — they ship it, we do not

MOSTLY AI documents differentially private synthetic data
generation. We do not have DP at all, and this file will not
pretend the difference is cosmetic: **k-anonymity on published
aggregates is not differential privacy**, which this project
states in its own privacy posture already.

What we have instead is a k-anonymous publication rule counted
in PATIENTS rather than rows, plus an attack battery. That is a
defensible posture and it is not the same guarantee. A
governance board that asks for an epsilon cannot be answered by
this tool today.

*Honest nuance:* DP on tabular synthesis carries a real utility
cost, and the literature on it is a survey-sized subject. The
gap is that we cannot offer the choice.

### 2. Sequential / longitudinal modelling as a first-class model

MOSTLY AI documents a sequential scenario; the research
frontier (RawMed, PATH, SynEHR) treats a patient's whole table
as the unit of synthesis.

Our position is split and should be stated that way:

- The **rules engine** models within-patient persistence,
  missing-clustering, and lagged cross-column effects with
  `--lags`. That is more than flattening.
- The **neural engine** carries patient IDENTITY (invented,
  person-level columns reproduced at 0.90–0.96) but models no
  trajectory. Its own gate line reads **dynamics tracked
  17/87** — structural, and we report it rather than hide it.

The literature's framing is the one to borrow: *flattening
fails to preserve temporal coherence even when it maintains
valid marginal distributions.* That is exactly our 17/87.

### 3. HIPAA Expert Determination certification

Tonic documents support for both HIPAA Safe Harbor and Expert
Determination. We have a Safe Harbor *inventory* and detectors;
we have **no certification and no expert determination**, and a
hospital compliance office will ask for one.

### 4. Platform surface

Managed deployment, database connectors, format-preserving
encryption, tokenization, subsetting, a no-code UI for
non-technical analysts, CI/CD integrations. Tonic Structural in
particular is a mature data-masking product. We have a CLI, a
bench, and self-contained HTML artifacts.

### 5. Generator breadth

Diffusion-based tabular models (TabDDPM, TabSyn), CTGAN,
Bayesian-network synthesizers, and LLM-based approaches are all
in the literature and in products. We have two engines: a
k-screened rules engine and a denoising autoencoder.

## What we have that I could not find in their documentation

These are listed with our measured numbers, and with an honest
note where the idea exists in research but not as a shipped
vendor feature.

### 1. An attack battery with POSITIVE CONTROLS — and the
### literature says the common vendor metric is broken

This is the strongest item on the page, and it is not a matter
of taste.

The standard vendor privacy metric is distance-based: MOSTLY AI
documents calculating distances from synthetic records to their
nearest training neighbour, with holdout distances as a
reference. That family is called **Distance to Closest Record
(DCR)**.

A 2026 paper — *The DCR Delusion: Measuring the Privacy Risk of
Synthetic Data* (arXiv:2505.01524) — states that Membership
Inference Attacks are "widely considered the gold standard",
that practitioners rely on DCR as a proxy, and that datasets
**labelled as private by DCR were "highly vulnerable to MIAs"**,
with DCR measures "uninformative of actual membership inference
risk" and "flawed by design" across classical and diffusion
models alike.

We run a membership inference battery, on two surfaces, **with a
positive control that must fail**:

| arm | our measured reading |
|---|---|
| output-surface adversary, real extract, 3 seeds | 0.514 / 0.523 / 0.551 |
| republish control, same arm | **1.000 CAUGHT** |
| weights-leak adversary, real extract | 0.366 / 0.479 / 0.619 |
| blueprint engine, membership worst AUC | 0.52 (coin flip 0.50) |
| blueprint engine, leaking-generator control | **1.00 FAIL** |

The control is the point. **Our own output-surface adversary was
once blind at width — it read 0.500 on a verbatim republish** —
and the only reason we know that is the control arm. That is the
DCR-delusion failure mode, caught in our own instrument, by a
check designed to be able to fail.

### 2. An aggregates-only release posture

The rules engine publishes **no records at all**: the artifact
it emits is a blueprint of k-screened aggregates — bounds that
are the mean of the k most extreme patients' own extremes,
levels suppressed below k, set tokens screened by patient, with
the suppression counted and reported. Generation then samples
from that contract.

This is different in kind from training a model on records,
which is what every generative approach above does — and the
project says so about its own neural engine, whose privacy
model it describes as different in kind for exactly this reason.

### 3. The key / released cipher architecture

One run writes three things: `key.csv` (the full-fidelity
k-anonymous synthetic file, kept and never shared),
`generated.csv` (the released file), and `cipher.json`
(which key row each released row took each perturbed value
from).

Every release dial is a PERMUTATION, so the released file is an
exactly invertible rearrangement of the key — verified value by
value through the disk, with a mutation test proving the
inverse breaks if the cipher stops composing. Measured on the
real extract: a vendor who TRAINS on the released file goes
from **+0.690 to −0.055** skill (denied **+0.744**), while a
vendor who merely SCORES our rows returns the key answer
exactly, because every predictor it reads is untouched.

Searches of vendor documentation and the literature returned
nothing describing this arrangement. I state that as not found,
not as nonexistent.

### 4. Per-claim receipts with a LIMITS section

`synthkit receipts RUNDIR` emits one receipt per published
claim, assembled from the run's own artifacts and **recomputing
nothing** (asserted to the digit). Each states what it does NOT
establish: association rather than causation, one cohort, a
weak skill called weak, seed-fragile attribution named, the k
it was screened at, and — sharpest — that a claim whose child
is a search column **cannot be checked in the output at all**.

First real read: **88 claims, 64 of them on columns the
operator never receives.**

Vendor QA reports are dataset-level (accuracy, correlations,
distributions, coherence, privacy). A per-claim receipt that
argues against itself is a different artifact.

### 5. A vendor exam with a computable ceiling

Train-on-synthetic-test-on-real is the standard protocol and we
have it. What we also have is a **planted answer**: an outcome
is planted with known effects in standard deviations, so the
CEILING — what a model that saw the truth achieves — is
computable, and a vendor is graded between a floor and that
ceiling rather than against an arbitrary bar.

Measured end to end: ceiling 0.786 / baseline 0.758 on the
strong tier. The instrument also caught a **miscalibrated bar**
— a 0.700 target sitting above that tier's own 0.698 ceiling,
so no solver on earth could clear it, visible in one line only
because the ceiling was computable.

*Honest framing:* planted-ground-truth benchmarking is a known
research technique (synthetic benchmarks for explainability;
Synthetic Hospital for longitudinal EHR). What I did not find
is a commercial synthetic data platform shipping it as part of
the generator.

### 6. Multi-hop cross-table couplings with a stated ceiling

Referential integrity across tables is table stakes — MOSTLY AI
documents a referential integrity scenario and SDV handles
relational schemas. What we add is enforcing the measured
cross-table RELATIONSHIPS through the key assignment, out to
three hops, with the limit stated as arithmetic:

| | measured |
|---|---|
| one-hop coupling, enforced vs control | +0.917 vs −0.023 |
| two-hop, before / after | −0.000 / +0.981 |
| three-hop, before / after | +0.010 / +0.981 |
| rank-one refusal | named, with what cannot come with it |

The refusal is the unusual part: a single foreign key is one
choice per row, so a link carries only rank-one correlation
structure. Two children on different parents is rank two and no
assignment produces it — we report `carried` in advance and
NAME the pair we cannot enforce, rather than fitting rank one
to it and reporting success. Our first cut did fit it anyway
and landed one pair at +0.994 against a target of +0.914 while
losing the other entirely.

### 7. The fidelity report suppresses its own bins

A histogram of real data is a set of bin counts, and a bin
holding three patients describes a group small enough to
gossip about. Our fidelity deck suppresses source bins backed
by fewer than k patients and PRINTS the count it suppressed.
I did not find another tool documenting this.

## The scoreboard, stated plainly

| capability | synthkit | documented elsewhere |
|---|---|---|
| differential privacy | **no** | yes (MOSTLY AI) |
| sequential / longitudinal model | partial (rules engine); neural engine 17/87 | yes (MOSTLY AI; research frontier) |
| HIPAA Expert Determination | **no** | yes (Tonic) |
| automatic PHI/PII detection | yes, 18-category inventory | yes (Tonic) |
| referential integrity across tables | yes | yes (MOSTLY AI, SDV) |
| cross-table RELATIONSHIP enforcement, 3 hops | yes, measured | not found |
| TSTR utility evaluation | yes | yes (most) |
| membership inference with positive controls | yes, two surfaces | not found as a shipped default |
| DCR-style distance privacy metric | yes, plus nn-ratio | yes (MOSTLY AI) |
| aggregates-only release option | yes | not found |
| invertible key/released cipher | yes | not found |
| per-claim receipts with limits | yes | not found |
| computable ceiling vendor exam | yes | research, not vendor-shipped |
| k-suppressed fidelity histograms | yes | not found |
| managed platform / connectors / no-code UI | **no** | yes (most) |
| generator breadth (diffusion, GAN, LLM) | two engines | yes (most) |

## What I would build next, in order

1. **Differential privacy as an OPTION on the neural path.**
   The largest single gap, and the one a compliance office asks
   for by name. It must be an option with its measured utility
   cost printed beside it, not a default — and the cost has to
   be measured on our own fixtures before it is offered.
2. **Patient trajectory in the neural engine.** The gate
   already says 17/87. The literature's framing — that
   flattening loses coherence even when marginals survive — is
   our own finding stated in their words, and it is the clearest
   quality gap a clinical buyer would notice.
3. **Expert Determination readiness.** Not a model change: a
   documentation and evidence package. We already have most of
   the evidence; nobody has assembled it for a determination.
4. **Free-text PHI.** Still the M6 governance gate.
5. Everything else on the platform list is engineering we know
   how to do and have deliberately not spent on.

## What this file must not become

A page that lists only our wins. The first four items under
"what they have" are real, they are not small, and two of them
(DP and trajectory) are the things a sophisticated buyer will
test first. If a future edit of this file grows the wins
section and not the gaps section, that is the drift this
paragraph exists to catch.

## Sources

- [The DCR Delusion: Measuring the Privacy Risk of Synthetic Data (arXiv:2505.01524)](https://arxiv.org/abs/2505.01524)
- [MOSTLY AI — synthetic data quality assurance](https://mostly.ai/blog/synthetic-data-quality-assurance)
- [MOSTLY AI — differentially private synthetic data](https://mostly.ai/blog/differentially-private-synthetic-data-with-mostly-ai)
- [MOSTLY AI — evaluate generator quality](https://mostly.ai/docs/generators/evaluate-quality)
- [MOSTLY AI vs SDV, sequential scenario](https://mostly.ai/blog/a-comparison-of-synthetic-data-vault-and-mostly-ai-part-2-sequential-scenario)
- [MOSTLY AI vs SDV, referential integrity scenario](https://mostly.ai/blog/a-comparison-of-synthetic-data-vault-and-mostly-ai-part-iii-referential-integrity-scenario)
- [mostlyai-qa, the open-sourced QA package](https://github.com/mostly-ai/mostlyai-qa)
- [Tonic Structural](https://www.tonic.ai/products/tonic-structural)
- [Tonic — synthesizing healthcare data with HIPAA expert determination](https://www.tonic.ai/guides/hipaa-ai-compliance)
- [Tonic — healthcare de-identification and synthesis](https://www.tonic.ai/solutions/industry/healthcare)
- [Synthcity: a benchmark framework for synthetic data](https://proceedings.neurips.cc/paper_files/paper/2023/file/09723c9f291f6056fd1885081859c186-Paper-Datasets_and_Benchmarks.pdf)
- [A survey on deep learning approaches for tabular data generation (arXiv:2503.05954)](https://arxiv.org/pdf/2503.05954)
- [Tabular data synthesis with differential privacy: a survey (arXiv:2411.03351)](https://arxiv.org/pdf/2411.03351)
- [Generating multi-table time series EHR from latent space (RawMed, arXiv:2507.06996)](https://arxiv.org/pdf/2507.06996)
- [Synthetic Hospital: an open, verifiable, physician-validated longitudinal EHR benchmark (arXiv:2609.30027)](https://arxiv.org/abs/2609.30027)
- [Methods for generating and evaluating synthetic longitudinal patient data: a systematic review (arXiv:2309.12380)](https://arxiv.org/pdf/2309.12380)
- [Top 5 synthetic data generation platforms of 2026](https://guptadeepak.com/tools/top-5-synthetic-data-generation-tools-2026/)
- [Best synthetic data tools compared (Tonic)](https://www.tonic.ai/synthetic-data/best-synthetic-data-tools)
