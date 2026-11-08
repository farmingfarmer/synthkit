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

### 1. Differential privacy — they ship it end to end, we now
### ship it for the weights and say so

MOSTLY AI documents differentially private synthetic data
generation. **We now have DP-SGD training on the neural path**
(`--dp-epsilon`, DP-1, 2026-11-07) with a correct RDP
accountant for the subsampled Gaussian mechanism — but the
epsilon covers **the autoencoder weights only**, and five other
stages of the pipeline touch the real data unaccounted. The run
names them and refuses to call its output differentially
private, so this is a partial answer stated as one.

The measured price, 51,200 rows, 3 seeds, against a control arm
that is the same network with the noise and clipping off:

| epsilon | sigma | holdout MSE | × control |
|---|---|---|---|
| 8.0 | 1.54 | 0.0226 | **1.16** |
| 3.0 | 3.41 | 0.0346 | **1.78** |
| 1.0 | 9.57 | 0.0977 | **5.02** |
| 0.5 | 18.86 | 0.1745 | **8.96** |

And the cost is **dominated by row count**, not by epsilon — at
epsilon 1.0 it is 18.8× on 3,200 rows and 4.6× on 51,200,
because the signal-to-noise of a DP step goes as
n/(√T·√params). Anyone quoting a ratio must quote the cohort
size with it.

End to end on the real engine at epsilon 3 on a 4,200-row
fixture: reconstruction error 0.00462 → 0.02736, train/holdout
ratio 1.03 → 0.97 (both generalising), and the gate's close,
sign and inverted criteria **unmoved** — though a five-column
fixture cannot separate the arms, so that last one is a limit
of the fixture and not evidence that DP is free.

**What is still missing** is the end-to-end guarantee:
**k-anonymity on published aggregates is not differential
privacy**, and neither is DP on one component. The remaining
stages are mostly counts and quantiles, which have standard DP
mechanisms — so this is the hard part done and the rest named
rather than a finished answer.

What we have instead is a k-anonymous publication rule counted
in PATIENTS rather than rows, plus an attack battery. That is a
defensible posture and it is not the same guarantee. A
governance board that asks for an epsilon cannot be answered by
this tool today.

*Honest nuance:* DP on tabular synthesis carries a real utility
cost, and the literature on it is a survey-sized subject. The
gap is that we cannot offer the choice.

### 2. Sequential / longitudinal modelling as a first-class model

**MEASURED AND PARTLY ADDRESSED 2026-11-07 (DYN-1).** The
mechanism behind `dynamics tracked 17/87` turned out to be one
line: the neural engine draws a patient's visit deviations iid
and then PERMUTES them, so within-patient lag-1 is exactly
**zero** by construction — the patient's level survives and
their course is absent. `--trajectory ar:RHO` walks the
deviations as an AR(1) in latent space and recovers it: lag-1
0.742 → **0.831** on systolic (clearing the 0.15 bar it was
failing), 0.676 → 0.805, 0.651 → 0.789.

**BOTH RESIDUES ARE NOW CLOSED.** The first was a measured
price: a within-patient COUNTER column was harmed (icc 0.002
→ 0.365 against a source 0.000), because persisting the
deviations makes a counter look patient-specific — so the gate
read 1/4 on every arm, one criterion fixed and one broken. A
column whose value equals its position within the patient plus a
constant on 99% of rows is now reproduced BY POSITION: exact,
not modelled, and DETECTED from the source rather than named, so
the same holds for a column called `seq` or `encounter_no`.

The second was that a FIXED rho is dataset-dependent and can
overshoot. It is SOLVED now against the source's own
within-patient AR — and the objective took three tries, which is
the part worth borrowing. The criterion grades `within_lag1`:
the pooled lag-1 with the between-patient share divided out.
Solving for the two measurements it is DERIVED from moves the
graded number the wrong way, because `(lag1 - icc) / (1 - icc)`
amplifies both errors as icc approaches one:

| objective | solved rho | dynamics |
|---|---|---|
| pooled lag-1 | 0.520 | 2/4 |
| pooled lag-1 and icc | 0.500 | 2/4 |
| `within_lag1`, what the gate reads | 0.900 | **4/4** |

The working objective lands essentially on the measured latent
deviation autocorrelation (0.892), which the two wrong
objectives were steering away from.

Across two longitudinal shapes and three seeds the graded
columns go **1.17 → 2.17** on average, worse on **no** arm, with
`close` and `inverted` identical on every arm and a
cross-sectional control grading zero columns. It still ships
**off by default**, and the real-extract control pair has now
decided that rather than deferring it. Same extract, same seed,
same settings, one flag apart:

| arm | dynamics | close | sign | INVERTED |
|---|---|---|---|---|
| off | 21/93 | 133/148 | 145 | 1 |
| ar | **28/93** | 131/149 | 146 | 1 |

**Seven more graded dynamics columns, at a cost of about two
close pairs** — and the close denominator moved by one, so the
absolute counts are what compare. It lands at 87.92% against an
87.9% bar where the control passed at 89.86%. Trading a
comfortably-passing criterion for a marginally passing one to
gain a different one is an operator's decision, not a default.
The mechanism transfers; the no-cost claim did not — our own
nine-arm fixture sweep read "never worse on close" because its
fixtures relate only three pairs, beneath the resolution of a
two-pair cost.

The honest ceiling, measured: the achieved within-patient AR is
**non-monotone in rho**, peaking near 0.70 at about 0.40. Past
the peak a high rho stops producing a DEVIATION and starts
producing part of the patient's LEVEL, so a source course
stronger than about 0.55 cannot be matched. We say so rather
than quoting the fixtures where it clears.

That ceiling is not what binds on this extract, though: its
latent deviation lag-1 is **+0.351** over 45,608 consecutive
pairs, against 0.89–0.95 on every fixture the mechanism was
developed against, and the solver landed 0.300 near it. The real
patient course is WEAK, and our fixtures were asking a harder
question than the data poses — worth knowing before anyone reads
a longitudinal claim in either direction.

MOSTLY AI documents a sequential scenario; the research
frontier (RawMed, PATH, SynEHR) treats a patient's whole table
as the unit of synthesis.

Our position is split and should be stated that way:

- The **rules engine** models within-patient persistence,
  missing-clustering, and lagged cross-column effects with
  `--lags`. That is more than flattening.
- The **neural engine** carries patient IDENTITY (invented,
  person-level columns reproduced at 0.90–0.96) and, with
  `--trajectory ar`, a patient's COURSE as well. Its recorded
  gate line of **dynamics tracked 17/87** was measured with the
  mechanism absent; it is an option now rather than a structural
  limit, bounded by the ~0.40 ceiling above, and the
  real-extract number has not yet been re-measured with it on.
  We report the 17/87 as the standing figure until it has.

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

### 7. Linkage risk against information the recipient already
### holds — with the bar set by its own positive control

45 CFR 164.514(b)(1) asks whether information could identify
someone "alone **or in combination with other reasonably
available information**". Our own determination package cited
two adversaries for that element — one seeing generated rows,
one assuming the weights leaked — and **neither holds external
data**. So we were asking the regulation's question and
measuring something narrower. `synthkit.linkage` closes it.

The translation is the interesting part. Against synthetic
output a match is not an identification — nobody's record is
there — so calling one an identification would overstate the
risk as badly as ignoring it understates it. What survives: an
adversary holding quasi-identifiers for a person they can *name*
selects the released rows matching that combination. Does that
tell them this person's value better than the same procedure
applied to someone never in the cohort? Only that **excess**
belongs to the release.

And the bar is a fraction of what a **verbatim republish**
achieves on the same cohort with the same bands, because an
absolute one was above its own ceiling: measured over six seeds,
a republish produces only **0.026–0.104 sd** of excess, since a
selection of at least five rows averages the target's own value
with four other people's. A 0.10 absolute bar would have missed
most republishes. Three distinct ways a clean number is
worthless — no control, no power, negative excess — get three
distinct verdicts, because each takes a different action.

I did not find another tool documenting a linkage measurement
for synthetic data with a positive control, let alone one whose
threshold is calibrated against that control.

### 8. The fidelity report suppresses its own bins

A histogram of real data is a set of bin counts, and a bin
holding three patients describes a group small enough to
gossip about. Our fidelity deck suppresses source bins backed
by fewer than k patients and PRINTS the count it suppressed.
I did not find another tool documenting this.

## The scoreboard, stated plainly

| capability | synthkit | documented elsewhere |
|---|---|---|
| differential privacy | **partial, all five stages measured** — DP training of the weights ships; the other four have mechanisms with measured prices (a DP vocabulary selection at eps 3 keeps 12% of what the k rule publishes on an 800-patient cohort; the latent mixture's cost is a SHAPE cost that more epsilon cannot buy off). Not on by default, and the reason is a number | yes (MOSTLY AI) |
| sequential / longitudinal model | **partial, measured on the real extract** — rules engine models it; the neural engine's AR(1) course with a SOLVED rho moves dynamics 21/93 to 28/93 on the extract and costs about two close pairs, so it stays off by default with the price on record | yes (MOSTLY AI; research frontier) |
| HIPAA Expert Determination | **partial** — the evidence package is assembled and says what is still OWED (`synthkit determination`); no expert has made a determination, and the tool refuses to imply one | yes (Tonic) |
| automatic PHI/PII detection | yes, 18-category inventory | yes (Tonic) |
| referential integrity across tables | yes | yes (MOSTLY AI, SDV) |
| cross-table RELATIONSHIP enforcement, 3 hops | yes, measured | not found |
| TSTR utility evaluation | yes | yes (most) |
| membership inference with positive controls | yes, two surfaces | not found as a shipped default |
| DCR-style distance privacy metric | yes, plus nn-ratio | yes (MOSTLY AI) |
| aggregates-only release option | yes | not found |
| linkage risk vs external quasi-identifiers | yes, with a control-calibrated bar | not found |
| invertible key/released cipher | yes | not found |
| per-claim receipts with limits | yes | not found |
| computable ceiling vendor exam | yes | research, not vendor-shipped |
| k-suppressed fidelity histograms | yes | not found |
| managed platform / connectors / no-code UI | **no** | yes (most) |
| generator breadth (diffusion, GAN, LLM) | two engines | yes (most) |

## What I would build next, in order

1. ~~**Differential privacy as an OPTION on the neural path.**~~
   **DONE 2026-11-07 (DP-1)** — and done the way this item
   asked: an option, off by default, with its measured cost
   printed beside it (1.16× at epsilon 8, 1.78× at 3, 5.0× at
   1) and its scope stated rather than implied.

   **AND THE OTHER FIVE STAGES ARE NOW MEASURED TOO (DP-2).**
   Four of them are patient counts or empirical quantiles, so
   the gap was never a research problem — the mechanisms are
   standard and are built: Laplace counts, Laplace and Gaussian
   histograms, a selection with a raised floor, and quantiles
   and bounds by the exponential mechanism.

   What the measurement says is the useful part, and it cuts
   against adding DP everywhere. Releasing *which* items
   cleared a floor is a different query from releasing their
   counts — an item held by exactly k patients appears or
   vanishes on one person — so the floor has to RISE, from 10
   patients to between 27 and 536 depending on epsilon and
   tokens per patient. On an 800-patient clinical-shaped
   vocabulary:

   | floor | tokens kept | share | |
   |---|---|---|---|
   | 10 | 320 | 100% | the k rule |
   | 99 | 39 | **12%** | DP eps 3 |
   | 196 | 19 | **6%** | DP eps 1 |

   So the k-anonymous aggregate release is not a shortcut taken
   instead of DP. At this cohort size a DP *selection* at a
   usable epsilon publishes almost no vocabulary, and a
   generator with no vocabulary is not more private — it is
   broken. DP on the **weights** is affordable and ships; DP on
   the **selection** is affordable at a cohort size we do not
   have.

   **AND THE LATENT MIXTURE IS CLOSED TOO (DP-3)**, the one
   stage that is neither a count nor a quantile. DP-EM is not
   the route — every iteration is a release and they compose —
   so `dp_gaussian_fit` makes a single release instead: clip
   the codes, publish a noisy mean and covariance, sample a
   Gaussian, as a drop-in for the engine's own mixture.

   Its cost is the useful finding, and it is **not a budget
   cost**. On a two-lump latent fixture the gap mass a mixture
   leaves empty reads 0.020 at the source, 0.022 through the
   mixture, and **0.391 through the DP Gaussian at epsilon 1, 3
   and 10 alike** — while the mean and spread come back to
   three decimals at every epsilon. More budget buys nothing;
   the loss is the model class a single release forces. We
   report that rather than quoting the moments, which all pass.

   So all five stages now have mechanisms and a measured price.
   **What remains is a decision, not a build**: none of these
   is on by default, because the measured prices say DP is
   affordable on the weights and expensive on everything else
   at the cohort sizes we have. Whether the real latent cloud
   is lumpy enough for the mixture cost to matter is unmeasured
   and needs a run on the machine that holds the extract.
2. ~~**Patient trajectory in the neural engine.**~~
   **BUILT 2026-11-07 (DYN-1)**, and built the way this item
   asked — the literature's framing, that flattening loses
   coherence even when marginals survive, was exactly our
   17/87, and the cause was that the deviations were drawn iid
   and permuted so within-patient lag-1 was zero BY
   CONSTRUCTION. An AR(1) in latent space recovers it with the
   rho SOLVED against the source, worse on no arm across nine,
   at no cost to `close` or `inverted`. **What remains is the
   real-extract control pair**, which is what decides whether
   it becomes the default, and the ~0.40 within-AR ceiling for
   sources with a stronger course than that. That is now the
   trajectory item.
3. ~~**Expert Determination readiness.**~~ **BUILT 2026-11-07
   (ED-1)**, and built the way this item described it — not a
   model change but an evidence package. `synthkit
   determination RUNDIR` assembles the ten elements 45 CFR
   164.514(b)(1) asks for from a run's own artifacts plus the
   attack batteries' outputs, recomputing nothing, and marks
   each one present or **OWED**. It exits 3 while anything is
   owed, and it refuses to call itself a determination — that
   is a qualified person's to make, and a tool claiming one
   would be the same fault as describing an output as
   differentially private when only the weights are accounted.
   **What remains is a run of record with every element
   present**, which needs the two batteries and a written scope
   statement on the machine that holds the extract.
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
