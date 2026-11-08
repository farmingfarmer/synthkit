# Scope and residual risk — a TEMPLATE, not a determination

`synthkit determination RUNDIR --scope FILE` needs a written
scope statement, because residual risk is the one element of
45 CFR 164.514(b)(1) that a run cannot supply: it is a statement
about what the analysis covers, and a scope nobody wrote is not
a scope.

**This file is a template with the measured facts filled in and
every JUDGMENT marked `[REVIEW]`.** It is not a determination
and it is not a finished scope statement. The determination is
made by a person with appropriate knowledge of statistical
methods; this exists so that person starts from the measurements
rather than from a blank page, and so the measurements they are
starting from are cited rather than remembered.

Replace the bracketed items, delete what does not apply, and
check every `[REVIEW]` against your own reading of the run.

---

## What is being released

- **Artifact**: `generated.csv` from run `[RUNDIR]`, plus the
  fidelity and sign-off pages that describe it.
- **Not released**: the source extract, `key.csv` (the retained
  full-fidelity file), and the trained network. `synthkit.cloud`
  refuses each of these BY NAME, and refuses a mixed request
  whole rather than partly.
- **Release surface**: output only. `[REVIEW]` — if the trained
  network is to be shared, the numbers below do not apply: the
  weights-surface adversary reads materially higher than the
  output-surface one, and the two are different recipients.

## Cohort and parameters

- Patients: `[N]`. Rows: `[N]`. Columns: `[N]`.
- k (minimum patients behind any published value): `[K]`
- Token cap / level cap: `[CAP]` / `[CAP]`
- Read these from the run's own `provenance.json` contract
  rather than from memory. It records k, both caps, the
  denoising level, the train/holdout split, every enforcement
  tier, the seed, and the posture.

## The method

Patterns are learned and published as **k-screened
aggregates**; no published number is an individual's value. The
stored minimum and maximum are the MEAN of the k most extreme
patients' own extremes, because storing the true 0th and 100th
percentile publishes one person's smallest and one person's
largest value. k counts PATIENTS, never rows — one person seen
two hundred times could otherwise set a published bound alone.

## Measured risk

Fill from the run's own artifacts and the attack battery's
output. Each number needs its CONTROL beside it or it is not
evidence:

| question | measured | control | arm |
|---|---|---|---|
| was this person in the cohort? | `[AUC]` | `[republish AUC]` | output surface |
| ...if the weights also leaked | `[AUC]` | — | weights surface |
| can a sensitive field be guessed? | `[excess]` | `[leaky excess]` | — |
| does it reproduce records? | `[nn-ratio]` | `[noise-frame ratio]` | — |

`[REVIEW]` — a control that cannot catch a verbatim republish is
a check that cannot fail. This project has recorded a reassuring
0.500 that was worth nothing for exactly that reason. Confirm
each control CAUGHT its planted leak before relying on the
number beside it.

## Retained information

Over-suppression is the trivial way to make risk very small, so
a determination that does not state what was retained has not
described a trade. Cite the fidelity gate's criteria and the
train-on-synthetic-test-on-real scores, which are measured on
the SAME held-out real rows for both arms and so cannot be
flattered by a thinner source.

## Residual risk and the bounds of this analysis

`[REVIEW] — every line here is a judgment.`

- One cohort shape, at one k, on one extract. A floor, not a
  certificate, and not a release decision for a different
  cohort.
- **k-anonymity on what is published is NOT differential
  privacy.** Differential privacy is available on the
  autoencoder weights with a measured cost; the other stages
  have mechanisms whose measured price is high at this cohort
  size (a DP vocabulary selection at epsilon 3 keeps about 12%
  of what the k rule publishes on 800 patients). None is on by
  default, and the run's contract records which were used.
- The effect curves, interaction surfaces and dynamics have NOT
  been audited by the attack battery the way the marginals and
  relationships have.
- Attack evidence must come from THIS cohort and THESE
  settings. `synthkit determination` marks evidence from another
  dataset `FIXTURE` and evidence from different caps `CONFIG`,
  and neither counts toward readiness — a larger token cap
  publishes more vocabulary about members, so the two runs
  measure different things and neither is evidence for the
  other. `[REVIEW]` — measured on one extract the DIRECTION of
  the cap's effect was not established (opposite signs on two
  seeds), so do not assume the smaller-cap number is the
  conservative one.
- `[REVIEW]` — any linkage risk from data the anticipated
  recipient holds that this analysis did not consider.

## What this document is not

It is not a determination, not a certification, and not a
statement that the risk is very small. Those are conclusions a
qualified person reaches; this is the evidence and the scope
laid out so they can reach one.
