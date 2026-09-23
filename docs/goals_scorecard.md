# The eight goals: where each one stands

Dated 2026-09-01. Every number below was measured — most on the real
800-patient extract (55,428 rows x 44 columns), the rest on fixtures
built from its measured statistics. Percentages are judgments; the
evidence beside each is not. Milestone letters refer to the roadmap
agreed 2026-08-25.

The one-line summary: **the middle of the pipeline (discover ->
anonymize -> generate -> verify) is strong and measured; the edges
(intake breadth, PHI scrubbing, the vendor-facing loop run
end-to-end) are where the remaining work lives.**

---

## Goal 1 — Universal upload with auto schema mapping   **~55%**

**What works, measured**
- Single-table CSV end to end. Types detected in seconds
  (`synthkit types`), including the silent-fault parsers: dates,
  currency, percent, clock time. Long/EAV extracts detected and
  pivoted (`--long CONCEPT=VALUE`).
- The shape sweep (`scripts/shape_sweep.py`) runs the whole pipeline
  over 23 dataset shapes a customer could send — flat tables with no
  grouping column, free text, high-cardinality codes, 92% sparse,
  one entity holding half the rows, unicode, duplicates, boolean
  spellings. **22 of 23 come out clean**; the 23rd is a documented
  privacy consequence, predicted in rows before generation.
- Flat data comes back flat: no invented columns, and the one
  derived column (within-entity order) appears only where entities
  repeat, and is named in the run.

**What remains**
- Relational, multi-table intake with key auto-detection (M3):
  `relational.py` exists from the first engine but the fitted path
  is single-table. The M3 gate: 3 multi-table fixtures, keys
  auto-detected, referential integrity 100%.
- File formats beyond CSV (Excel, JSON lines) — small, unstarted.
- Document upload is the M6 free-text decision, not promised yet.

**Time**: relational is the block — est. 2–3 weeks (M3, plan Oct 17
holds). Formats: days.

---

## Goal 2 — Automatic de-identification   **~65%**

**What works, measured**
- Everything published is k-anonymous over PATIENTS, never rows:
  bounds are means of the k most extreme patients' extremes; visit
  counts go through the same rule; set tokens are screened by
  patients; levels below k are suppressed and the suppression
  reported.
- High-cardinality identifiers (codes, IDs, free text fields) now
  publish only their SHAPE and generate invented labels: 1,436
  distinct in, zero real labels republished.
- Attacked, with positive controls: membership worst AUC 0.52
  (coin flip 0.50) where a leaking generator scores 1.00 and FAILs;
  attribute-disclosure excess -0.009 against +0.145 for a
  republishing generator; a near-verbatim set republish (one token
  swapped) is caught at 0.998 where string comparison scored 0.500.

- ~~The structured PHI layer~~ SHIPPED 2026-09-18
  (`synthkit/scrub.py`, `synthkit scrub CSV [--apply OUT]`):
  names, SSNs, phones, emails, street addresses, birth dates and
  per-patient identifiers, detected from VALUES with headers only
  assisting — a renamed SSN column is caught, a column NAMED ssn
  holding lab values is not. Gated both ways in `smoke_scrub`:
  7/7 planted PHI columns caught AND 0/7 near-miss clinical
  controls flagged (visit dates, `Oxygen Therapy`-shaped labels,
  shared nine-digit accession numbers), because a detector that
  flags everything also catches everything. Both mutations kill
  the gate (lexicon emptied → name missed; threshold unreachable
  → zero findings). Free-text columns are reported OUT OF SCOPE
  by name, never silently skipped.

- ~~The real-extract read~~ DONE 2026-09-22, and it earned its
  place: one true catch (visit_id, a per-row identifier - the
  wording now states the measured ratio, about one per row, where
  the first version printed "one per person" beside 55,428
  distinct over 800 patients) and three detector faults the
  fixture could not reach - visit dates flagged as identifiers
  (the fixture's date control had 9 distinct values against a
  rule that needs 0.9x the patient count; the extract has 4,692),
  and active_drugs misread as free text when it is a
  semicolon-joined set. Fixed: dates are never identifiers, set
  columns are judged by their TOKENS (a set of email addresses is
  caught where the joined string matches nothing), each fault
  reproduced from the measured statistic and watched red first.
- ~~The confirming re-read~~ DONE 2026-09-23: the corrected scrub
  read the extract again — visit_id the lone catch, honestly
  worded ("about one per row"); both dates and active_drugs back
  in the clear count, 42 of 44 clear.

**What remains**
- Free text PHI is the M6 decision gate.
- Honest posture, unchanged: k-anonymity on what is published is
  NOT differential privacy; the attack results are a floor on one
  cohort shape, not a certificate.

**Time**: the real-extract reading is one command on the data
machine; the free-text decision is governance, not engineering.

---

## Goal 3 — Pattern analysis, explained with receipts   **~80%**

*(2026-09-09: +10 — the dashboard's pattern cards, SHAP attribution
on both tables with interaction ranking, per-run shape and surface
fidelity, and the self-explaining gate all landed.)*

**What works, measured**
- Discovery confirmed out of sample, split by patient. On the
  planted-truth fixture: **12/13 relationship kinds found, 0 noise
  edges** — including a U-shape with zero linear correlation, an
  XOR, a Simpson's reversal. The miss needs `--lags`, which exists.
- Effect curves per parent (including measured-alone curves so a
  correlated co-parent cannot flip a sign), interaction surfaces,
  presence-as-signal, heteroscedastic residual profiles.
- Receipts today: `findings.txt` (skill, parents, what was trimmed
  and what was re-applied, dropped-edge accounting with a total),
  `catalogue.json` with per-claim skill and importances.
- The atlas (docs/system_map.html) now covers the fitted path and
  the verification layer: 97 components, code extracted live by ast
  so the map cannot drift, with a plain-English walkthrough for
  every component — the build REFUSES if any narrative is missing.

**What remains**
- The M4 bar: per-claim receipt JSON as a formal artifact, and the
  cycle-survival gap — on a ring-shaped graph 27/42 relationships
  are dropped to make it sampleable (the real extract drops 16/26).
  The refinement sweeps recover much of this; the ceiling is not
  yet at the M4 target of >= 12/13 including lagged.
- The search cap: only 24 tokens per set column can carry a
  relationship; on the extract the excluded tokens hold 72% of
  `conditions` mass. Measured, reported by `peek cap`, unresolved —
  the fix is a runtime budget decision (521 slots needed vs 96).

**Time**: receipts est. 1 week; cycle ceiling is the open research
item of M4 (plan Oct 31) — est. 2–3 weeks, the least certain number
on this page.

---

## Goal 4 — Dials over every pattern   **~80%**

**What works, measured**
- `--dial` reaches patients.count, coverage, shift, scale,
  persistence, clustering, visits_scale. Requested-vs-achieved is
  written into findings.txt, because a dial can be capped by the k
  bound or undone by constraint repair — a silent difference is the
  failure this tool exists to refuse. Unknown dial names are errors.
- shift and scale act independently about the PUBLISHED center
  (they used to fight; measured and fixed).

**What remains**
- M5: every dial class verified requested-vs-achieved with its
  neighboring properties asserted (the shift/scale fight was found
  exactly because neighbors were not asserted).
- ~~Dials over relationships~~ SHIPPED 2026-09-18:
  `--dial "CHILD<-PARENT.strength=0..1"` weakens ONE edge.
  Measured on a two-parent child: dialing both directions of the
  target pair to 0 takes it 0.919 → 0.056 while the neighboring
  relationship on the same child survives at 0.555; strength=1.0
  is bit-identical to no dial; an unknown edge errors with a
  did-you-mean. The first cut scaled ALL parents and killed the
  neighbor — caught because the check asserted the neighboring
  property, the shift/scale lesson applied on day one.

**Time**: verification pass est. 1 week; relationship dials est.
1–2 weeks (M5, plan Nov 14 holds).

---

## Goal 5 — High-fidelity generation   **~85%**

**What works, measured — on the real extract**
- Coverage 42/42; center within 10% of spread on 30/33 (was 18/34);
  set token shares 62/62 (was 54 of 91 missing); set EMPTY rates
  4/4 including the zero point-mass fix; inverted relationships 0;
  direction kept 108/115 = 93.9%.
- Against rulers: synthkit keeps 118.7 sign-correct relationships
  where a Gaussian copula keeps 76.7. On the transfer bench, only a
  "no relationships" generator collapses model ranking (-0.48);
  faithful reads +0.86.
- Spread that varies with the prediction, informative missingness,
  patient-level constancy, constraints discovered and (opt-in)
  repaired by swapping, dates rendered after arithmetic.

**What remains**
- The last M0 criterion: `close` at 76.5% against the 87.9% bar —
  and the seed sweep (76.5/80.6/80.2) says the gap is REAL, not
  noise. The real-data reading is complete: shapes survive
  (8/9, 7/7, 7/7 across seeds), published surfaces mostly do not
  (2/6 — the application machinery works, 787_2 tracks despite
  losing both pair members; the residue is the ring-adjacency
  ceiling plus 3 marginal misses), close drift concentrates in
  set-token relationships, and two inversions appeared on seed 37
  only (that file does not circulate).
- The mechanisms live on the reproduction fixtures (pressure
  family, attribution triangle) with measured gates; the
  synchronized final pass fixed surfaces on the fixture (2/5 →
  0/5 seeds failing) and did NOT transfer to the extract — the
  prediction was on record, and `peek surfaces` now diagnoses
  from the run directory.

**Time**: ~2–4 weeks on the fixtures, then confirmation on the
next real refit. The close gap and the ring ceiling are the
project's two genuine unknowns; everything else on this goal is
measured.

---

## Goal 6 — Self-assessment for sign-off   **~100%**

*(2026-09-14: CLOSED — scripts/signoff.py ships the roll-up page;
read failing before it shipped, on a fixture shaped like the real
extract's current gate.)*

*(2026-09-09: +10 — the gate grew to eight criteria including shapes
and interaction surfaces, every criterion explains itself with an
inspect-it-yourself pointer, failures are enumerated in the
dashboard, and what-now guidance carries exact commands.)*

**What works, measured**
- `fidelity.json` + findings verdicts; `contradictions.find` (the
  report cannot disagree with itself), `find_blueprint` (the
  contract cannot disagree with itself — predicted a -20% row
  shortfall, run came out -27%); `invariants.find` (declared
  properties asserted row-wise, no source data needed);
  `scripts/m0_gate.py` (six criteria, honest exit code — currently
  5/6); `peek` views for post-hoc diagnosis without one-liners.
- The run explains its own limits in place: k-rule costs in tokens
  per row, per-column survival (patients per level vs the k floor),
  privacy-caused spread shortfalls attributed, invented columns
  named.

**What remains**
- M5's single sign-off artifact: one page a decision-maker signs,
  rolling up gate + attacks + contradictions + disobedience. All
  inputs exist; the rollup does not.

**Time**: est. 3–5 days.

---

## Goal 7 — Vendor evaluation against planted truth   **~90%**

**What works, measured**
- `semisynth.plant/verify`: effects declared in standard
  deviations, planted +0.9/-0.5 recovered at +0.86/-0.44, a
  no-effect column reads +0.03, intercept solved not centered,
  achieved-vs-requested reported.
- `bridge.py`: measured marginals cross into the evaluation
  TableSpec as quantiles (every decile within 0.03 of a source sd
  on a column skewed 2.83); what does not cross is written on the
  artifact.
- The campaign/evaluator/gui machinery from the first engine runs
  offline and is demo-ready.

**What remains**
- ~~M1~~ DONE 2026-09-11: the loop ran end to end on 55,428
  real-shaped rows; ceiling/baseline/vendor produced, and the
  as-specified bar exposed as sitting above its own ceiling.
- ~~Repeatability~~ DONE 2026-09-18: `synthkit exam RUNDIR
  --effect col=BETA -o DIR` is the whole loop as one command —
  bridge, plant, ladder, baseline, showdown, report card — with
  per-stage echo and sentence refusals. The five-command version
  lost a day to a one-letter path typo.
- A real vendor in the vendor seat.

**Time**: vendor seat is a scheduling question, not a build.

---

## Goal 8 — AutoML competitor and report card   **~65%**

**What works, measured**
- `autosolver.py`: the deliberately modest floor (stdlib logistic
  regression, structurally blinded — trains on a shifted-seed
  table, never sees test labels). The ceiling/ours/vendor line
  exists in the campaign machinery.

**What remains**
- The M1 gate half that belongs here: AutoML v1 beating the naive
  baseline on all 3 planted mechanisms, and the single report-card
  artifact. Also unrun rather than unbuilt — but "auto-generated
  competitor" beyond the stdlib floor (feature engineering, model
  selection) is genuinely unstarted.

**Time**: report card with the stdlib floor: inside M1's week.
A competitor worth the name: est. 2 weeks, after M1.

---

## The milestone table, honestly

| Milestone | Agreed | Status 2026-09-01 |
|---|---|---|
| M0 fidelity core | Aug 29 | **5/6 criteria**; `close` 76.5% vs 87.9% is the whole gap |
| M1 evaluation loop | Sep 12 | components done, assembled run not started |
| M2 structured PHI | Sep 26 | not started; holds if begun within ~a week |
| M3 relational intake | Oct 17 | first-engine module exists; fitted path single-table |
| M4 pattern ceiling | Oct 31 | 12/13 forest; ring survival is the open research item |
| M5 dials + sign-off | Nov 14 | dials live; verification pass + rollup artifact remain |
| M6 free text | Dec 5 | decision gate at M2, intentionally unpromised |

Projected completion, carrying the observed ~1-week M0 slip through
the near milestones and holding the far ones: **without free text,
mid-November; with free text (if M2 says yes), mid-December.** The
original plan's estimate-quality caveat stands in both directions:
the last runtime estimate on this project was wrong by 30x, upward.

## Weighted overall

## Proposed working order (updated 2026-09-23)

The 2026-09-09 order's five named steps ALL LANDED: the
end-to-end run (09-11), the seed sweep (09-10), the report card
(09-11), the sign-off page (09-14), the structured PHI scrub
(built 09-18, corrected by its first real read 09-22, confirmed
on re-read 09-23) — plus relationship dials and the one-command
exam runner (09-18). Durations are from now, deliberately not
dates:

1. **The team's try-to-break findings** from the kit handout —
   each becomes a check; the suite grows and the kit gets
   harder. ~days, interleaved with everything else.
2. **Goal 5 — the close-gap mechanisms**, on the reproduction
   fixtures where they now live (trim machinery: attribution
   flip, set-token drift; the ring ceiling is the research
   piece). ~2-4 weeks, then confirmation on the next real refit.
3. **Goal 1 — multi-table intake**, the last big unstarted
   piece. ~3-6 weeks.
4. **Goal 3 — per-claim receipts.** ~1 week.
5. **Goal 2 — the free-text decision gate**, governance first.
   ~2-3 weeks of engineering IF the gate says yes.
6. **Goal 4 — the per-class verification pass.** ~1-2 weeks.
7. **Goal 7 — a real vendor in the vendor seat**: scheduling,
   not building.
8. **Goal 8 and the research tail** — a challenger worth the
   name, the attribution-flip stabilization, the ring-adjacency
   ceiling: ongoing and unscheduled, because unknowns move dates
   and these are the genuine unknowns.

Equal-weighting the eight goals: **~78% complete** (goal 2
60→65 on 2026-09-23: the real-extract read, its three detector
lessons, and the confirming re-read). Core complete in roughly
**6-7 weeks from 2026-09-23** — early November — with free text,
if its gate says yes, adding ~6 weeks beyond. The unweighted
number understates the risk profile: what remains is mostly
breadth (multi-table, receipts, verification) plus two genuine
unknowns — the `close` gap and the cycle ceiling — and unknowns,
not assembly, are what move dates.
