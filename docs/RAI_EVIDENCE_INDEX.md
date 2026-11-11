# RAI evidence index

**For Patrick.** Yesha asked you to own the RAI vendor
questionnaire for SynthKit, leaning on what you did for Echo. This
file is not an attempt at the questionnaire. It is an index of what
evidence exists, where it is, which command regenerates it, and —
the part that matters most for an honest answer — **what is not
established.** Nothing here needs a SynthKit run to read; every
figure is already measured.

Written 11 November 2026. Every figure is measured on our own
800-patient extract unless it says FIXTURE.

---

## The one-line posture

Real clinical records are used to learn **parameters**; generation
never copies a record. No published number describes fewer than
**ten patients**, and no published number is any individual's value
— a bound that looks like "the oldest patient" is the mean of the
ten most extreme patients. What leaves the secure environment is
the synthetic table, published statistics, and summaries.

`synthkit determination` prints this posture and refuses to call
itself a determination.

---

## Evidence that exists, and how to regenerate it

| what | reading | regenerate with |
|---|---|---|
| Expert-determination elements | **10 of 11 present** on the real cohort | `synthkit determination RUNDIR` |
| Membership inference, released surface | **0.514 / 0.523 / 0.551** over three seeds, with a verbatim-republish control at **1.000 CAUGHT** | `latent_challenger.py attack --csv` |
| Membership inference, model surface | worst **0.6253**, marginal | same |
| Attribute disclosure | excess **−0.0055**, leaky-generator control **+0.2289** | `attribute_disclosure.py` |
| Memorization | held-out / train reconstruction **1.1196**; a network with nothing to learn reads **9.7** | written by the run |
| Linkage | CLEAN; control **+0.0746**, honest release **+0.0084**, **11.3%** of what a republish achieves | `synthkit linkage --scan` |
| Safe Harbor coverage | **8 covered, 8 partial, 2 not-in-a-table, 0 not covered** | `synthkit scrub --inventory` |
| Deployment surface | verified end to end in the container; see `docs/AWS.md` | `docker run … run_all_smokes.py` |

Supporting documents: `docs/DETERMINATION_SCOPE.md` (the scope
template with measured facts filled in and every judgment marked
for review), `docs/COMPETITIVE.md` (how this compares to the
commercial tools, including where we are behind).

---

## What is NOT established — please do not let a form imply otherwise

- **This is not a determination.** 45 CFR 164.514(b)(1) puts that
  in a qualified person's hands. We have assembled the evidence
  such a person would need. The tool says so itself.
- **Residual risk and scope are owed.** That is the eleventh
  element, and it is the one a run cannot supply — somebody has to
  write the scope. `docs/DETERMINATION_SCOPE.md` is a template,
  not a scope.
- **The output must not be described as differentially private.**
  An epsilon is accounted for the model **weights** only. Five
  other stages touch the real data unaccounted — the published
  numeric bounds, the latent mixture fitted on real codes, the
  k-screened visit-count and set-size grids, and the identity
  detection. `synthkit`'s own DP output names all five.
- **k-anonymity on what is published is not differential privacy,**
  and the effect curves, interaction surfaces and dynamics have
  not been audited to the same standard as the bounds.
- **Not yet run against a live AWS account.** As of today the whole
  deployment path is verified inside the container — including a
  real publish that refuses to carry the source extract — but no
  run has touched a live account. The tool prints this caveat in
  its own output.
- **Fixture evidence is not evidence about our cohort.** The
  determination package marks anything measured on invented data
  as FIXTURE and refuses to count it toward readiness. If a
  questionnaire answer rests on a number, check it is not marked
  FIXTURE or CONFIG.
- **Cohort size bounds some answers.** Three-factor structure needs
  roughly 3,200 patients to establish; we have 800. Eight of the
  published two-factor patterns cannot be measured at all for the
  same reason.
- **One known fidelity defect, stated plainly.** Procedures per
  visit comes out wrong: 93% of synthetic visits carry exactly one
  procedure against 53% in the real records. It is not a privacy
  effect and it is being fixed.

---

## If you need something that is not here

The quickest route is `synthkit determination RUNDIR`, which
assembles the elements and prints which are present, which are
FIXTURE, which are CONFIG, and which are OWED — and exits non-zero
while anything is owed, so it cannot be mistaken for a pass.
