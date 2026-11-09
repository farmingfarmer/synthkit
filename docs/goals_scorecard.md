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

## Goal 1 — Universal upload with auto schema mapping   **~95%**

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
- Multi-table intake, M3's gate, DONE 2026-11-04
  (`synthkit/multitable.py`, `synthkit schema`): three
  fixtures — a star, a chain and one whose headers lie — with
  keys auto-detected FROM THE VALUES and referential integrity
  reproduced. Orphans are kept rather than cleaned away: the
  star plants 20, the source measures 97.8%, and the generated
  set reads 97.8% with the orphans pointing at invented keys
  that belong to nobody. No real key is republished. The
  negative controls carry the weight — a continuous
  measurement is refused as a key (the first version made a
  lab value a primary key and would have linked tables on it),
  a two-value column is refused, and unrelated tables produce
  NO links and say so.
- ~~FITTING the linked tables~~ LANDED 2026-11-06 (MT-1, MT-2
  and `synthkit relate`). Cross-table couplings are measured in
  the source and enforced through the KEY ASSIGNMENT. Measured
  through the written files, one seed, one flag apart:
  source `+0.892`/`+0.875`, enforced `+0.871`/`+0.865`,
  `--no-couple` **`-0.028`/`-0.066`** — without enforcement the
  structure is destroyed, not weakened, and `--no-couple` ships
  as the named control arm. Two couplings on one link land
  jointly where enforcing the strongest alone left one behind
  (total error 0.029 against 0.171).
- **AND THE WHOLE THING HAD NO OPERATOR PATH UNTIL `relate`
  EXISTED.** `detect`, `cross_couplings` and `assign_keys` were
  called by nothing but their own suite — `synthkit schema`
  reported a schema it could not act on. The
  capability-wired-to-nothing fault, at its largest.

**What remains for M3**
- **The rank-one ceiling, which is arithmetic rather than a
  gap to close.** One foreign key is one choice per row, so the
  correlations a link can carry have rank one. Two children
  riding DIFFERENT parents is a rank-two target and no
  assignment produces it; the solve reports `carried` in
  advance and REFUSES, naming what cannot come with the pair it
  enforces. Fitting a rank-one approximation anyway was
  measured WORSE than enforcing the strongest (it landed one
  pair at +0.994 against a target +0.914 and lost the other
  entirely).
- A child's VALUES are never re-drawn from its parent — only
  which parent it points at. A lab value is attached to a visit
  whose length of stay fits; it is not generated from it.
- ~~Multi-hop couplings~~ LANDED 2026-11-06 (MT-3). A lab value
  riding the PERSON's age spans two links and was never
  searched for — not weakened, absent, with the run reporting
  "0 cross-table coupling(s)". Measured through the written
  files on a three-table chain: source `+0.976`, before
  `-0.000`, after **`+0.981`**, `--no-couple` control `-0.000`.
  The fix needs no second solver: once a visit is attached to a
  person that visit row CARRIES the person's age, so the
  inherited columns are materialised onto the parent frame and a
  two-hop coupling becomes an ordinary one-hop coupling on a
  `__via__` column. One function serves the source (real keys)
  and the generated set (invented keys), so detection and
  enforcement cannot disagree about what a two-hop coupling is.
  The link loop is topological now, and no `__via__` column
  reaches a written file.
- ~~Three or more hops are not measured.~~ LANDED 2026-11-07
  (MT-4). Three hops were UNREACHABLE, not weakened — on a
  person → visit → order → result chain a result riding the
  PERSON's age read source `+0.976`, output `+0.010`, with the
  run reporting "0 couplings". `with_inherited` is recursive
  now, bounded at three, so a grandparent's column arrives on
  the parent as a `__via__` column and is carried one level
  further: **`+0.981`**, control arm `+0.010`. The path is in
  the name (`__via__visit_id>person_id__age`) because it
  appears in the run's report, and a cycle is refused by a
  visited-set guard rather than truncated by the depth bound.
- ~~FOUR or more hops are not measured, and the rank-one limit
  compounds along a chain.~~ MEASURED 2026-11-08 (MT-5,
  `scripts/hop_depth.py`), and the answer is that neither was
  the gap. On a five-table chain with **only the DIRECT
  couplings enforced**, a FOUR-hop correlation lands at 93% of
  source with the control arm at zero — each link's own
  enforcement COMPOSES, so depth arrives for free:

      hops   source   coupled   control   share
         1    0.900     0.879     0.006    0.98
         2    0.810     0.769     0.003    0.95
         3    0.729     0.676     0.002    0.93
         4    0.654     0.607    -0.005    0.93

  At the shipped bound of 3 the same cells read 0.98 / 0.95 /
  0.93 / 0.92, so **the multi-hop detection adds nothing on a
  chain** — which does not make it useless: it is load-bearing
  exactly where composition CANNOT happen, on MT-3's shape where
  the intermediate table carries no numeric to compose through
  (source 0.897, one-hop 0.010, multi-hop 0.817, control
  −0.013). And what three reaches is FOUR links, not three: the
  old comment understated it by one, because `cross_couplings`
  climbs from the PARENT side. On a five-table chain where ten
  couplings exist, bounds 0–4 detect 4 / 7 / 9 / 10 / 10 —
  three is already complete and raising it finds nothing new.
- **AND THE BOUND WAS DOCUMENTED AS A SETTING AND FROZEN AT
  IMPORT.** `hops: int = VIA_HOPS` in the signature, and a
  default argument is evaluated once — so reassigning
  `multitable.VIA_HOPS` changed nothing and the two call sites
  that omit it were pinned at three forever, while the module's
  own comment said "deeper is a setting, not a default". The
  measurement script then printed "detection ran at VIA_HOPS =
  4" over a run that used three. Read at call time now, and the
  check asserts the OUTCOME — couplings found at bound 0/1/2 go
  3, 5, 6 where the frozen version gives 6, 6, 6.
- **AND THE CHAIN EXPOSED A REAL DEFECT: colinear targets were
  collapsing onto one number while `carried` read 1.00.** A
  chain's parent frame carries a column plus the same column
  from one and two links up, correlated 0.89 and 0.79 with each
  other, so one link's target is three different numbers against
  three near-twins — and the solve weighted the parent score by
  the SVD's right singular vector, which is the TARGET DIRECTION
  and not the SCORE WEIGHTS that reach it. The achieved spread
  collapsed from the target's 0.187 to **0.010**. Whitening each
  side through its own correlation matrix improves all ten
  multi-target links over five seeds: squared error per link
  0.0119 → **0.0019**, worst single pair 0.1328 → **0.0351**,
  and the DIRECT coupling's error 0.0705 → **0.0296**. A
  single-coupling link is byte-identical, pinned as a digest.
  `carried` was right throughout — the metric answered its
  question and the weights did not reach what it promised.
- ~~File formats beyond CSV~~ SHIPPED 2026-11-03:
  `synthkit.intake` reads .csv/.tsv/.txt, .jsonl/.ndjson/.json,
  .xlsx/.xlsm and .parquet, dispatching on the extension, with
  `--sheet` for workbooks. The load-bearing check is not that
  each format opens but that EVERY FORMAT PRODUCES THE IDENTICAL
  FRAME to the CSV path — the reader is part of the suspect path,
  and a second reader that disagrees about a blank or a number
  would put the folded-empties fault back invisibly. That check
  immediately caught one: pandas' own `dtype=str` renders an
  integral float 60.0 as "60" where the CSV path preserves
  "60.0", so Excel disagreed on exactly those cells. Integrality
  is now decided per COLUMN by one renderer. A format whose
  package is absent is refused with the install that would fix
  it; an absent worksheet is refused listing the real ones.
- Document upload is the M6 free-text decision, not promised yet.

**Time**: relational has landed through four hops. What is left
on this goal is the M6 free-text decision, which is governance
before it is engineering.

---

## Goal 2 — Automatic de-identification   **~97%**

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

- ~~ZIP codes and the coverage question~~ DONE 2026-11-06
  (PHI-4). A ZIP is the one kind whose value shape cannot carry
  the detection — five digits is a visit count, a lab result, a
  row id — so it requires one of three tells: ZIP+4, a
  preserved leading zero, or the header. Generalised to the
  first three digits WITH the half of the rule a plain
  truncation skips: the 17 sparse three-digit areas go to 000.
  Remedy is generalise, never drop, because geography carries
  catchment and travel distance. Web addresses and IP addresses
  (categories 14 and 15) were missing outright and are now in
  the value battery. Five mutations each turn the gate red, and
  the negative controls caught a real bug: `\bzip\b` cannot
  match `patient_zip`, because `_` is a word character.
- ~~"what does this actually reach?"~~ ANSWERED 2026-11-06.
  `scrub --inventory` prints all **18 Safe Harbor categories**
  with a status and note each — **8 covered, 6 partial, 2 NOT
  COVERED, 2 not-in-a-table** — counts derived from the rows,
  gaps NAMED rather than summed, and no CSV required because
  the question is asked before anybody has a table. A detector
  that finds nothing reads as a clean bill of health and is
  not one.

**What remains**
- ~~**An Expert Determination RUN OF RECORD.**~~ **IT EXISTS,
  2026-11-08: TEN OF ELEVEN ELEMENTS MEASURED ON THE REAL
  EXTRACT**, with the membership, attribute and linkage
  batteries all pointed at the extract rather than a fixture,
  and every piece of evidence verified as being about THIS
  cohort at THESE caps.
  Membership at the released configuration: worst **0.6253
  MARGINAL** (weights surface, never released) with the
  republish control at **1.000 CAUGHT**. Attribute disclosure:
  excess **−0.0055** with the leaky control at **+0.2289**.
  Memorization: held-out reconstruction **1.1196** of training
  against a 9.7 control on a frame with nothing to learn. The
  Linkage reads **CLEAN** at 11.3% of what a verbatim
  republish achieves (542 selections, control +0.0746, honest
  +0.0084) — found by SEARCHING the banding ladder, since both
  arrangements tried by hand failed at opposite ends. The one
  element still OWED is `residual`, a statement about scope
  that a person writes. **Readiness is not a determination** — that is a
  qualified person's to make, and the tool refuses to imply
  otherwise.
- What remains on that: the scope statement signed off by a
  qualified person. That is the only element a RUN cannot
  supply. The evidence package is built
  (ED-1/ED-2/ED-3/LK-1, `synthkit determination`)
  and now covers ELEVEN elements, including the statute's "in
  COMBINATION with other reasonably available information" half
  that the two attack adversaries do not reach — `synthkit.
  linkage` measures what the release tells someone who already
  knows who you are, as the EXCESS over a non-member arm, with
  the bar set as a fraction of what a verbatim republish
  achieves on the same cohort (an absolute 0.10 bar was above
  its own ceiling: a republish reaches only 0.026–0.104 sd).
  Evidence from another dataset reads **FIXTURE** and evidence
  from different caps reads **CONFIG**; neither counts toward
  readiness, because the two runs MEASURE DIFFERENT
  THINGS - a larger token cap publishes more vocabulary about
  members - and neither is evidence for the other. (Measured on
  the real extract, the DIRECTION is not established: the
  output-surface adversary read 0.498 at cap 60 against 0.561 at
  cap 800 on one seed and 0.521 against 0.485 on another.) It marks every element present or
  **OWED** — but no run has all
  ten present, because that needs the two attack batteries and
  a WRITTEN scope statement on the machine that holds the
  extract. The tool exits 3 while anything is owed and refuses
  to call itself a determination; what remains is assembling
  the evidence on a real run and putting it in front of a
  qualified person.
- **Differential privacy now has a mechanism and a measured
  price for ALL FIVE stages** (DP-1, DP-2, DP-3), and what
  remains is a DECISION rather than a build. The prices are the
  finding. A DP vocabulary **selection** at epsilon 3 keeps 12%
  of what the k rule publishes on an 800-patient cohort, 6% at
  epsilon 1 — so DP on the SELECTION is affordable at a cohort
  size this project does not have, which is why the
  k-anonymous aggregate release is not a shortcut taken instead
  of DP. And the latent mixture's cost is a SHAPE cost that
  more epsilon cannot buy off: 0.391 gap mass against a source
  0.020 at epsilon 1, 3 and 10 alike, while every moment comes
  back to three decimals. Only the WEIGHTS epsilon is
  affordable, and it ships.
- **What is owed on that is one measurement, not a build:**
  whether the real latent cloud is lumpy enough for the mixture
  cost to matter. The fixture was built lumpy on purpose so the
  cost would be visible at all, and quoting its 20x gap as a
  statement about the extract would be the
  measured-at-one-shape fault.
- Free text PHI is the M6 decision gate.
- ~~City and county names~~ SHIPPED 2026-11-06 (PHI-5,
  `geography`). Header-driven and the finding says so: a city
  name and a race category are the same shape, so no test on
  the VALUES separates them. The value gate refuses numbers
  under a city-ish header; a combined `city_state` field IS
  caught; a STATE is left alone because the rule asks for
  subdivisions SMALLER than a state. Remedy is DROP, because
  Safe Harbor names no coarsening for a place name and a
  city-to-state table is data this project cannot keep correct.
  **`ethnicity` ends in `city`** — a loose pattern would have
  deleted a covariate, so it was tested against `ethnicity`,
  `capacity`, `toxicity`, `specificity` and `downtown` before
  the detector was written. Five mutations each red.
- ~~Vehicle (12) and device (13) serial numbers are NOT
  covered~~ BOTH MOVED 2026-11-07 (PHI-6). That entry's
  reasoning — "a VIN is indistinguishable from any other
  alphanumeric code" — was wrong about the VIN: it is 17
  characters, excludes I/O/Q, and position nine is a **check
  digit** computed from the other sixteen, with a measured
  3.1% coincidence floor against a 50% share bar. And a device
  UDI has two halves: `device_serial` fires on a PRODUCTION
  identifier (serial, lot, expiry) and deliberately NOT on a
  device identifier alone, because a bare GTIN is a model code
  that identifies nobody and dropping it would remove the
  clinical fact rather than an identifier. **Every Safe Harbor
  category is now at least partially reached: 8 covered, 8
  partial, 2 not-in-a-table, zero NOT COVERED.**
- Still not reached, and named in the inventory's own notes:
  **license plates** (short, alphanumeric, state-varying, no
  checksum) and **free-form device serials** with no GS1 syntax
  and no check digit.
- City and county names (part of category 2) are ordinary short
  strings with no shape to find and no per-patient constancy a
  demographic does not also have.
- ~~No differential privacy at all~~ PARTLY ADDRESSED 2026-11-07
  (DP-1): `--dp-epsilon` trains the neural path's autoencoder
  with DP-SGD and an RDP accountant verified four ways against
  itself. Measured price at 51,200 rows, against a control arm
  that is the same network with the noise off — **1.16× at ε=8,
  1.78× at ε=3, 5.02× at ε=1** — and the cost is dominated by
  ROW COUNT, 18.8× on 3,200 rows against 4.6× on 51,200.
  **The epsilon covers the WEIGHTS only.** Five stages still
  touch the data unaccounted (k-screen and vocabulary, the
  published bounds, the latent mixture, the visit-count and
  set-size grids, the identity tiers); the module names them and
  the run REFUSES to describe its output as differentially
  private. Those remaining stages are mostly counts and
  quantiles, which have standard mechanisms — the hard part is
  done and the rest is named.
- Honest posture: k-anonymity on what is published is NOT
  differential privacy, and neither is DP on one component; the
  attack results are a floor on one cohort shape, not a
  certificate.

**Time**: the real-extract reading is one command on the data
machine; the free-text decision is governance, not engineering.

---

## Goal 3 — Pattern analysis, explained with receipts   **~95%**

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

- ~~Per-claim receipt JSON as a formal artifact~~ SHIPPED
  2026-11-06 (`synthkit/receipts.py`, `synthkit receipts
  RUNDIR`). One receipt per published claim, assembled from
  `catalogue.json` / `blueprint.json` / `fidelity.json` and
  **recomputing nothing** — asserted to the digit against those
  artifacts, because a receipt that re-derived its numbers could
  disagree with the gate reading the same run. Each carries a
  LIMITS section stating what it does NOT establish: association
  not causation, one cohort, a skill under 0.30 called weak
  rather than suppressed, seed-fragile attribution named, a sign
  that reverses when co-parents are held fixed, the k it was
  screened at, and that trims are not asserted because the trim
  record is prose. **A claim whose child is a search column says
  it cannot be checked in the output at all** — the column is
  dropped before the file is written. 24 checks, five mutations
  each red.

**What remains**
- ~~The M4 bar: >= 12/13 found including lagged.~~ MET
  2026-11-08, and the reason it had not been was the INSTRUMENT.
  `bench_new_path.py` scored `lagged-cross-column` as a
  permanent miss with its own note saying the kind "needs
  --lags, which exists" — it exists in the PIPELINE, and the
  bench called `discover` on the raw frame and never built the
  lag features, so the kind could not be found by construction.
  Over the same five seeds: 11/12/11/11/11 without, and
  **12/13/12/12/12 with**, adding exactly one kind on every seed
  (asserted as a property, so a future change cannot report a
  broad gain through this door).
- **AND THE FLAG WAS A TRADE UNTIL IT WAS FIXED.** On those
  same runs the output acquired a SAME-VISIT correlation of
  **0.97** on a pair the source relates at 0.00, where without
  the flag it correctly reads 0.00 — and the lagged relationship
  did not survive either (lag recomputed from each table's own
  ordering: source **+0.98**, output **−0.002**). An invented
  relationship is worse than a missing one, so the kind FAILED
  survival rather than scoring the point. Re-measured on five
  seeds after the fix: FOUND and SURVIVING on all five, source
  **0.944–0.947** against generated **0.969–0.973**, and
  SURVIVED went 9/11 → 10/11 on the paired seed with every other
  row byte-identical. Seed 1 reads 13/13 found and **11/11
  surviving**.
- **AND THE INVENTED HALF IS FIXED, same day.** The cause was
  `discover._source` folding `x__prev` to `x` so the catalogue
  reads in the operator's own names — right, and it also erases
  WHEN, which is the whole substance of a lagged finding. The
  claim RECORDS the engineered members in `via` and
  `blueprint.build` never read that field, so a relationship
  found on the previous visit was applied as a same-visit curve.
  Paired on a planted fixture, one file apart: same-visit
  **+0.7901 → −0.0106** against a source −0.0038, with the
  honest same-visit relationship beside it unmoved (+0.9645 →
  +0.9591). The refusal is reported in
  `excluded.lag_only_parents` with the figure that justified it.
  Two near-misses of my own, both caught by mutation: deciding
  on rank correlation alone deletes a U-shape (−0.032 on
  Spearman, 2.52 sd on the bend), and neither measure can see an
  interaction-only parent — on `c = 8ab` with `--lags`,
  `c <- ['a','b']` became `c <- ['a']` until a parent named by
  the claim's own `interaction.pair` was made exempt.
- **AND THE OTHER HALF LANDED THE SAME DAY: the lagged
  relationship SURVIVES.** No lag column needed to reach the
  written file at all — the curve `discover` publishes under the
  base name IS the lag's (on a fixture whose same-visit pair reads
  0.00, that curve comes back `increasing` with a response range
  of 37 over a child of sd 10), so the shape was right and only
  the column it was read from was wrong. The parent is MARKED in
  `evidence.lag_parents` rather than dropped, and the sampler
  reads it from the patient's previous row:

      pair                      source    before     after
      x~y (same visit)         −0.0038   +0.7901   +0.0149
      x_(t−1) ~ y (lagged)     +0.8871   −0.0020   +0.7870
      u~w (honest same-visit)  +0.9698   +0.9645   +0.9645

  An absent temporal relationship carried at **89% of source**,
  the invented pattern gone, the honest one bit-identical. And it
  is INERT where its mechanism is absent, asserted as a digest: a
  blueprint predating the field generates byte-identically to one
  carrying it empty, which is what makes a four-call-site change
  to the sampler core safe to ship.
- **WHAT THE MEASUREMENT EXPOSED INSTEAD is a separate defect in
  the output: `visit_number` is in row order on 100% of generated
  patients and `visit_start_date` on 0.8%.** Reading the lagged
  pair by the index gives +0.86 and by the date +0.31. Two columns
  in one file disagreeing about the order of a patient's visits is
  this project's own alarm, it is there with or without `--lags`,
  and the obvious fix was checked against the source first — the
  tidy fixture's own rows are monotone within a patient on only
  47.5% of patients, so the source does not encode visit order in
  its rows either. The direction is therefore to make
  `visit_number` follow the generated dates, and it is not free:
  the real extract declares `visit_start_date <-> visit_end_date`
  at skill 0.9999, so permuting one date column within a patient
  while the other stays would destroy a near-identity.
- ~~Next step, measured rather than assumed cheap.~~ DONE the
  same day. A patient's dates go in visit order now — **0.2% →
  100% of patients** — and the near-identity survives because ONE
  permutation serves every date column, so each row keeps its own
  pair: start~end reads **+0.9988 in both arms**, bit-identical,
  and with constraint enforcement on `end >= start` holds on 100%
  of rows after the reorder. The VALUES move, not the rows,
  because the persistence lives in the row layout and reordering
  rows would destroy the thing this repairs. The time axis is
  recorded in `patients.time_column` so the sampler does not need
  a second detection rule.
  The price is one number and it is stated: a planted date-driven
  NUMERIC reads +0.9730 in source, +0.9643 before and **+0.9029**
  after. The whole gate is IDENTICAL on both arms — seven
  criteria unmoved.
- **AND THE CLAIM THAT THE GATE WAS AFFECTED IS WITHDRAWN.** The
  first measurement read the output's course by DATE (0.4026
  against a source 0.6456, where row order gives 0.6647) and I
  was one step from recording that `dynamics tracked` had been
  understated by a measurement artifact. `pipeline.compare`
  already orders the generated frame by `visit_number`. Running
  the gate rather than reasoning about it is the only reason that
  did not ship.
- What remains on dynamics is named and NOT yet measurable here.
  A generated patient's visit dates are drawn from the GLOBAL
  marginal, so they spread across the cohort rather than
  clustering into a care episode — the within-patient
  autocorrelation read **0.74 against a source 0.98**. That 0.98
  came from a fixture built with cumulative gaps, where patients
  are episodes by construction; the repo's own extract-shaped
  fixture has `visit_start_date` at **icc 0.0062** with a mean
  within-patient span HALF the cohort's, so its patients are not
  episodes and an output that spreads them is faithful there.
  Whether a real patient's visits cluster is a property of the
  real extract and neither fixture can answer it. The route is
  written down — a patient-level anchor plus a k-screened
  within-patient GAP distribution, drawn as a cumulative sum —
  with its price named in advance: the date column's global
  marginal becomes a convolution rather than the published
  quantiles, which the gate measures.
- The search cap: only 24 tokens per set column can carry a
  relationship; on the extract the excluded tokens hold 72% of
  `conditions` mass. Measured, reported by `peek cap`, unresolved —
  the fix is a runtime budget decision (521 slots needed vs 96).

**Time**: receipts est. 1 week; cycle ceiling is the open research
item of M4 (plan Oct 31) — est. 2–3 weeks, the least certain number
on this page.

---

## Goal 4 — Dials over every pattern   **~95%**

**What works, measured**
- `--dial` reaches patients.count, coverage, shift, scale,
  persistence, clustering, visits_scale. Requested-vs-achieved is
  written into findings.txt, because a dial can be capped by the k
  bound or undone by constraint repair — a silent difference is the
  failure this tool exists to refuse. Unknown dial names are errors.
- shift and scale act independently about the PUBLISHED center
  (they used to fight; measured and fixed).

**What remains**
- ~~M5: every dial class verified requested-vs-achieved with
  its neighboring properties asserted~~ DONE 2026-11-03,
  including the first run ON THE REAL BLUEPRINT
  (`dial_matrix.py --blueprint`). Four columns measured
  against 88 relationships. `span_days.shift` and `.scale`
  read CLEAN — that column was the recorded NO-OP until the
  cyclic-sweep fix. THREE dials LEAK on the real graph where
  all of them were clean on the one-relationship fixture:
  `missing_clustering` moves its own column's persistence
  (+0.13), `patients.count` moves `age_at_visit.persistence`
  (+0.10) and a pair (+0.16), `patients.visits_scale` moves
  `span_days.spread` (+0.12 — a second column it appeared to
  move was measured as SEED NOISE, floor +0.17, once the
  matrix was taught to measure its own floor). TWO are
  genuine
  NO-OPs on real data — `span_days.persistence` and
  `span_days<-procedure_count.strength`.
- Per-pattern dials over every pattern KIND the inventory
  names, shipped 2026-11-03: relationship, effect curve,
  identity, set token and interaction surface all reachable
  by an invertible output permutation the cipher can undo.
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

## Goal 5 — High-fidelity generation   **~93%**

**What works, measured — on the real extract**

*2026-11-04 night, on `phase2_800` — the extract the run was
actually built from, confirmed by the provenance guard rather
than by reading a number off a screen. Every figure in this
block is from one file, one denominator:*

| criterion | before repair | after repair |
|---|---|---|
| direction kept | 145/148 = 98.0% | **146/148 = 98.6%** PASS |
| close (bar 87.9%) | 130/148 = 87.8% FAIL | **133/148 = 89.9%** PASS |
| inverted | 0 | 0 |

The repair reorders ONE column jointly against its partners.
`verify_permutation` reads both files from disk and returns PURE
REARRANGEMENT: four columns reordered, multisets identical, 40
columns byte-identical. A permutation cannot move a marginal, so
the published bounds, point masses and k-anonymous extremes are
exactly what they were.

*Earlier real-extract figures, kept for continuity:*
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
- **The close gap is no longer the open M0 item** — it clears on
  the repaired artifact and misses by 0.1 points without the
  repair. What remains on this goal is the *unrepaired* engine
  reaching the bar on its own. **The ring ceiling is REFUTED**
  (2026-11-07, RING-1): swept across skill over eight seeds the
  sweeps deliver adjacent correlation equal to the skill they
  were given — 0.97 → **0.949**, 0.99 → **0.980** — with no cap
  anywhere, and the shape gap at high skill is 0.059 sd against
  a 0.35 bar, which is the floor between two draws of the SAME
  blueprint. The recorded sentence compared this fixture at its
  own hardcoded skill of 0.75 against what a near-identity
  needs at 0.97. On the real extract
  `visit_start_date ↔ visit_end_date` is declared at skill
  0.9999 in both directions and comes out at rho **1.0000**.
  Pinned as checks; a genuine 0.78 clamp turns three red.
  Five further close misses sit on TOKEN INDICATORS, which are
  search scaffolding dropped before `generated.csv` is written
  and so unreachable by any rearrangement of the output; going
  after them through their set column was measured HARMFUL
  (close 46/57 -> 32/57) and is off by default.
- **`dynamics tracked` is NO LONGER STRUCTURAL** (2026-11-07,
  DYN-1). It read 17/87 with "the engine is row-level and models
  no patient trajectory" beside it. An AR(1) on the latent
  deviation gives a patient a course, the rho is SOLVED against
  the source's own within-patient AR rather than guessed, and a
  column that IS the within-patient index is reproduced by
  position. Across two longitudinal shapes and three seeds the
  graded columns go 1.17 -> 2.17 on average, worse on NO arm,
  with `close` and `inverted` identical on every arm and a
  cross-sectional control grading zero columns.
  **THE REAL-EXTRACT CONTROL PAIR HAS NOW DECIDED IT**, same
  extract and seed, one flag apart: dynamics **21/93 -> 28/93**,
  close 133/148 -> 131/149, sign 145 -> 146, INVERTED 1 both
  ways. Seven more graded columns at a cost of about two close
  pairs, landing at 87.92% against an 87.9% bar where the
  control passed at 89.86%. **Still OFF by default**, because
  trading a comfortably-passing criterion for a marginally
  passing one is an operator's decision; the price is on record
  so the choice can be made. The mechanism transferred; the
  no-cost claim did NOT - our own nine-arm sweep read "never
  worse on close" because its fixtures relate only three pairs.
  And the real course is WEAK: the extract's latent deviation
  lag-1 is +0.351 over 45,608 pairs against 0.89-0.95 on every
  fixture, so the measured ~0.40 ceiling is not what binds
  here.
- *Superseded, kept so the trail is readable:* `close` at 76.5%
  against the 87.9% bar —
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

**Time**: the RING CEILING is closed — refuted by measurement,
not solved, and the ~2–4 week estimate it carried is returned.
What remains is the *unrepaired* engine reaching the close bar,
plus one re-aimed question: since the sampler renders whatever
skill it is handed, a shape gap on a near-identical pair comes
from DISCOVERY's skill assignment or the curve. The September
`map_cuff` symptom cannot be re-measured — those columns are
not in the current blueprint, where the pressure variants
appear only as weak parents at skill 0.07–0.36.

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

## Goal 7 — Vendor evaluation against planted truth   **~92%**

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

## Goal 8 — AutoML competitor and report card   **~98%**

**What works, measured**
- `autosolver.py`: the deliberately modest floor (stdlib logistic
  regression, structurally blinded — trains on a shifted-seed
  table, never sees test labels). The ceiling/ours/vendor line
  exists in the campaign machinery.

**What remains**
- **THE M1 GATE'S OWN BAR IS MISCALIBRATED, and that is a
  correction rather than a shortfall.** It asked for AutoML v1
  "beating the naive baseline on all 3 planted mechanisms."
  Measured, that bar is wrong: on a LINEAR mechanism the naive
  baseline — stdlib logistic regression — *is* the right answer,
  so a model selector beating it there would mean beating the
  truth. The property worth asserting is NEVER WORSE, plus
  decisively better where the floor is blind, and that is what
  shipped and is pinned: 0 of 9 arms worse by >0.01, xor
  0.49–0.52 → **0.88–0.90**, curved 0.54–0.58 → **0.72–0.76**,
  and a correct TIE on linear where it names logistic regression
  the winner itself. This project has the precedent — the
  2026-09-11 real run's own finding was a bar demanding 0.700
  over a 0.698 ceiling, and the rule from it was to set bars
  beneath the ceiling or judge on gap. Same class of error, in
  our own gate.
- ~~The single report-card artifact from a real run~~ RUN
  2026-11-08. `CARD.html` from `run_dial1`'s four-arm showdown on
  the real extract, every figure read back spelled by the
  machine:

      tier            ceiling   floor   automl   vendor   bar
      strong-signal    0.7884  0.7611   0.7841   0.7611  PASS
      as-specified     0.7040  0.6872   0.6955   0.6872  FAIL
      weak-signal      0.6301  0.6248   0.6082   0.6248  PASS

  The exam has now run twice on the real extract on two
  DIFFERENT synthetic draws of the same cohort and reproduced
  itself — every tier within 0.006 and every GAP within 0.001 of
  September's 0.758/0.786, 0.683/0.698, 0.617/0.622. That is the
  computable ceiling's whole claim, and it had never been tested
  across two synthetic files of one cohort.
- **AND THAT RUN PUT MY OWN COMPETITOR BELOW THE STDLIB FLOOR ON
  THE WEAK TIER — 0.6082 against 0.6248 — against the "never
  worse than the floor" claim recorded below.** The cause is
  switching families on a difference the held-out selection
  slice cannot resolve: with little signal a bare argmax picks
  whichever family the noise favoured. Fixed by making the
  SIMPLEST family the incumbent and requiring a flexible one to
  beat it by `1/sqrt(min(pos, neg))` on the slice — the
  approximate standard error of an AUC, so the bar tightens as
  the slice grows. Worst weak-arm delta **-0.0123 → -0.0000**,
  with the xor and curved gains intact and the selector still
  naming different families by mechanism. My first attempt at
  that fix was UNREACHABLE CODE — it decided inside the loop and
  `FAMILIES` is ordered simplest-last, so every candidate was
  simpler than the incumbent and the branch never ran; both arms
  came out byte-identical including the -0.0123.
- **AND THE CARD NOW CALLS OUT A BAR TIGHTER THAN EVERY MEASURED
  GAP**, the milder form of the miscalibration it already names.
  The as-specified tier's 0.700 bar sits 0.004 under a ceiling
  of 0.704 — achievable, but by less than any solver on that
  tier left on the table — where in September it sat ABOVE its
  ceiling. Reported separately from the strict case, because
  they have the same fix for different reasons.
- What remains is narrow and named: the competitor has been
  measured on three planted mechanisms, three seeds each, and
  the real extract's three tiers. It is not swept across cohort
  SIZE, and a selection slice carved from 120 rows is where the
  margin rule is least tested.
- ~~A competitor beyond the stdlib floor~~ SHIPPED 2026-11-03:
  `synthkit.automl` tries three model families with light
  feature engineering (numerics as numerics, capped one-hots,
  and a MISSING INDICATOR per column, because being measured is
  itself a signal) and picks by held-out AUC on a slice carved
  from the TRAINING rows. Measured across three mechanisms and
  three seeds: **never worse than the floor** (0 of 9 arms worse
  by >0.01), and decisively better where the floor is blind —
  xor **0.49–0.52 → 0.88–0.90**, curved **0.54–0.58 →
  0.72–0.76** — while correctly TYING on a linear mechanism and
  naming logistic regression as the winner there. It records
  every family it tried with its score, refuses to hold a
  contest below 120 rows, and falls back to the floor saying
  **"this run is NOT a competitor result"** when scikit-learn is
  absent, so a report card cannot claim one that never ran.
  Registered as `automl`.

**Time**: both shipped and both run on the real extract.

---

## The milestone table, honestly

| Milestone | Agreed | Status 2026-09-01 |
|---|---|---|
| M0 fidelity core | Aug 29 | **CLEARS AFTER REPAIR, 2026-11-04 night.** On the extract the run was actually built from (`phase2_800`, accepted by the provenance guard): close **130/148 = 87.8% unrepaired**, one tenth of a point under the bar, and **133/148 = 89.9% after `repair_inversions --close`** - PASS. direction kept 145->146/148 = 98.6%. INVERTED 0. The repair is a verified PURE REARRANGEMENT of four columns, so the privacy posture is exactly what was published. Earlier readings of 87.6% / 76.5% were against a different cohort or the wrong source file; see the note below. |
| M1 evaluation loop | Sep 12 | **ASSEMBLED RUN DONE** - this line was stale. `docs/e2e/end_to_end.json` records a full pass: 24 score checks passed, attack verdict PASS, showdown ceiling 0.9104 against a vendor 0.7308 with `vendor_beats_ours` False. The 2026-09-11 real run went bridge -> plant -> three tiers -> showdown on 55,428 rows, and its own finding was a MISCALIBRATED BAR: the as-specified tier demanded 0.700 over a 0.698 ceiling, visible in one line only because the ceiling is computable from the planted answer. Remaining: `vendor_exam` route A is built and exercised, the vendor SEAT is scheduling rather than building. |
| M2 structured PHI | Sep 26 | not started; holds if begun within ~a week |
| M3 relational intake | Oct 17 | **detection DONE 2026-11-04** (`synthkit schema`, 3 fixtures, keys from the values, integrity reproduced). The remainder is now MEASURED rather than described: a lab value planted at 2.0x its visit's length of stay reads source **+0.906** and generated **+0.021** - fitting each table alone destroys the cross-table relationship. Pinned as a characterization check, so the day joint fitting lands it goes red and the claim gets updated with it. |
| M4 pattern ceiling | Oct 31 | 12/13 forest; ring survival is the open research item |
| M5 dials + sign-off | Nov 14 | **DONE 2026-11-03**, and the CIPHER CHAIN now runs end to end on the real extract (2026-11-05): released file VERIFIED as a faithful rearrangement (43 columns untouched, every distribution unchanged); a vendor who TRAINS on it goes from **+0.690 to -0.055** skill, **denied 0.744**; a vendor who SCORES our rows returns the key answer **EXACTLY**, no surrogate. The dial also reported its own ceiling honestly - asked -0.60, landed -0.24, reachable band [-0.284, +0.392] bounded by the four distinct values the dialled column holds, so the ask did not exist. |
| M6 free text | Dec 5 | decision gate at M2, intentionally unpromised |

Projected completion, carrying the observed ~1-week M0 slip through
the near milestones and holding the far ones: **without free text,
mid-November; with free text (if M2 says yes), mid-December.** The
original plan's estimate-quality caveat stands in both directions:
the last runtime estimate on this project was wrong by 30x, upward.

## Deployment — added 2026-11-04

Not one of the eight goals, and needed before any of them ship
to a hospital: `synthkit batch`, `synthkit/cloud.py`, a
`Dockerfile` and `docs/AWS.md`. The compute goes to the data —
`fetch_source` brings a source IN and there is no function that
sends one anywhere. Only allow-listed artifacts leave; the
source extract and `key.csv` are both refused by name and the
list cannot be extended by a flag. **Not run against a live AWS
account** — that is the next thing it needs, and it is stated in
the module, the doc and the tool's own output.

Reviewed 2026-11-05 and **one real defect found by reading it**:
the image set `ENV OMP_NUM_THREADS=1`, which is the TEST-RUN
setting. Fixture frames spend more time starting threads than
computing — worth 35x to the net — but a real extract is 55,428
rows where the threads genuinely pay, and the net sets the
variable on its own subprocesses by `setdefault` so an image ENV
is exactly the outside value that beats it. Every production fit
in that container would have run single-threaded, silently.
Removed, with a check that fails if it comes back, plus the two
cheapest real assertions about a deployment nobody has built:
every `COPY` path exists, and the `ENTRYPOINT` names a console
script the package actually declares — a Dockerfile naming a
missing file fails only at build time.

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

Equal-weighting the eight goals: **~95.6% complete** as of
2026-11-08 — 95, 97, 95, 95, 93, 100, 92, 98. Moved on
2026-11-08: goal 3 90→95 because M4's bar — twelve of thirteen
planted kinds FOUND including the lagged one — is met on all five
seeds, and the reason it had not been was the instrument rather
than the engine: the bench never built the lag features the
pipeline builds, so the one kind needing them could not be found
by construction. Turning them on then exposed a real defect it
had been hiding — a lag parent folded to its base column and
applied at the SAME visit, inventing a 0.97 correlation on a pair
the source relates at 0.00 — and with that fixed the lagged
relationship is FOUND and SURVIVES on five seeds of five. Also
goal 1 90→95 because FOUR-hop cross-table couplings
are measured and the two things left named on M3 both turned out
not to be the gap — a chain composes, so a four-hop correlation
lands at 93% of source with ONLY the direct couplings enforced,
and the bound of three is already complete on a five-table
chain. What the measurement DID find is a real defect it was not
looking for: colinear targets on one link were collapsing onto a
single achieved number while `carried` read 1.00, and whitening
the score weights through each side's own correlation matrix
improves every one of ten multi-target links. Also goal 8 90→98 because the report card of record now
EXISTS — `CARD.html` from the real extract's four-arm showdown,
and the exam reproduced itself across two different synthetic
draws of one cohort to within 0.006 a tier and 0.001 a gap —
and because that run put my own AutoML competitor BELOW the
stdlib floor on the weak tier (0.6082 against 0.6248), which is
now diagnosed, fixed to −0.0000, and pinned. And goal 2 96→97
because LINKAGE is measured on the real extract, which takes
the Expert Determination package to TEN of eleven elements: the
measurable band was found by SEARCH (`--scan`) after both
hand-tried band arrangements failed at opposite ends — one with
no power, one with nothing scored. Earlier the same day: goal 2
94→96 for the determination run of record itself, each element
with the positive control that makes it worth reading, and the
gaps named rather than counted. Previously moved on
2026-11-07: goal 2 92→94 for two measured pieces, both of
which SAY what they do not cover. The Expert Determination
EVIDENCE PACKAGE (`synthkit determination`) assembles the ten
elements 45 CFR 164.514(b)(1) asks for from a run's own
artifacts, marks each present or **OWED**, exits 3 while
anything is owed, and refuses to call itself a determination —
that is a qualified person's to make. And the four unaccounted
differential-privacy stages now have mechanisms with a measured
price: a DP vocabulary **selection** at epsilon 3 keeps 12% of
what the k rule publishes on an 800-patient cohort, which is
why the k-anonymous aggregate release is not a shortcut taken
instead of DP. Previously moved on
2026-11-06, each for a measured reason: goal 1 78→90 (the
cross-table couplings are enforced and REACHABLE —
`synthkit relate`, with `--no-couple` as the named control arm
reading −0.028 against a source +0.892, and the rank-one
ceiling stated as arithmetic rather than a gap, and TWO-HOP
couplings that were never searched for — source +0.976,
before −0.000, after +0.981); goal 2 65→90
(ZIP detection with the sparse-ZIP3 half of the rule, web and
IP addresses, and an inventory naming this tool's coverage
across all 18 Safe Harbor categories — verified on the real
extract with ZERO false positives); goal 3 82→90 (per-claim
receipts as a formal artifact, recomputing nothing, each
stating what it does NOT establish). Previously 86.5% on
2026-11-05 — 78, 65, 82, 95, 90, 100, 92, 90. Moved that week,
each for a measured reason and not a feeling: goal 1 75→78
(multi-table detection landed on real OMOP tables, and the
cross-table gap is now a number rather than a caveat); goal 3
80→82 (every declared edge is accounted for, so the gate's
denominator can be audited); goal 4 90→95 (the cipher chain ran
end to end on the real extract — verified rearrangement, denied
0.744, exact recovery, and both recovery routes measured); goal
5 85→90 (M0 `close` clears on the repaired artifact, 89.9%
against an 87.9% bar, and misses by 0.1 points without it);
goal 7 90→92 (route A exercised against a planted answer).
**The percentages are judgments; the numbers beside them are
not.** (Earlier: ~78%, goal 2 60→65 on 2026-09-23 — the
real-extract read, its three detector lessons, and the
confirming re-read.) Core complete in roughly
**6-7 weeks from 2026-09-23** — early November — with free text,
if its gate says yes, adding ~6 weeks beyond. The unweighted
number understates the risk profile: what remains is mostly
breadth (multi-table, receipts, verification) plus two genuine
unknowns — the `close` gap and the cycle ceiling — and unknowns,
not assembly, are what move dates.

## Research tail, added 2026-10-27

Two items from the UM-tool sizing question, which asked for the
smallest cohort carrying the patterns that matter and got the
answer that the trade runs the other way.

### Temporal predictive modelling — within-stay trends   **TODO**

The delivered engine is row-level: a synthetic patient is a genuine
person-shaped bundle of visits, but a patient's course over TIME is
not modelled. Measured consequence: within-patient persistence
falls 0.4952 -> 0.1568 on the extract, and it falls on a thin
cohort too (0.3939 -> 0.0869), so it is structural rather than a
sizing problem. Any rule depending on a trend inside one admission
-- "haemoglobin falls 2 g/dL in 24h" -- cannot be exercised at any
patient count.

**The approach, as proposed by the operator and worth recording
because it sidesteps the obvious objection.** Fit a model on the
REAL patients' courses that learns what happens next given a
patient's current state, then use it to extend each synthetic
patient forward from their first visit. The reason this beats
modelling whole trajectories: learning TRANSITIONS pools evidence
across the entire cohort, where a trajectory SHAPE would need k
patients sharing a near-identical course. "Haemoglobin tends to
fall after this diagnosis" is fittable from every patient who had
it, so the thin-cell arithmetic that kills three-way combinations
does not bind the same way.

Three gates, cheapest first, and the first can kill it outright:

1. **Is the signal in the extract at all?** The tidy file is one row
   per visit. If serial measurements inside a stay were collapsed at
   extraction, no model can synthesise what the source does not
   contain. One query answers it.
2. **Sample, do not predict.** A point prediction gives the
   conditional MEAN and every synthetic patient's labs then glide
   smoothly - nobody ever drops two points in a day, which is the
   rule the work exists to support. It needs the conditional SPREAD
   too, heteroscedastic, which is the residual-profile machinery
   this file already records (source grew 1.85x where generation was
   flat at 1.02x until it was published).
3. **A course is more identifying than a visit.** The weights-leak
   adversary already reads 0.776 FAIL at width on the row-level
   engine; a sequence model trained on 800 trajectories has few
   examples to interpolate between, so its nearest plausible path
   may be a specific real person's. Run the nn-ratio tripwire and
   the membership battery against generated trajectories BEFORE
   building, not after.

Recommended shape: a two-week spike that answers all three with
numbers, not a build. The rules engine's lag machinery is a crude
version that already works (persistence within 0.15 on 28/34
columns on the real extract) and carries two known defects - a lag
parent can be dropped silently by cyclic refinement, and the
within-patient lag-1 estimate was attenuated 0.453 against a true
0.700. Those are the starting points if the spike says go.

### Complex combination patterns   **IN PROGRESS**

Three-way structure reads 0.000 into the synthetic data, and the
three causes need separating because they imply different work:
thin cells below the k floor (more patients fixes it), the
denoising regularizer discarding thin structure BY DESIGN (the same
change that took the membership adversary 0.776 -> 0.569), and the
engine losing three-way even on ample fixture data (two-way XOR
survives every seed; the 3-way mostly dies). Only the first two are
data-quantity problems.

**Done 2026-10-27: the measure can now SEE the clinical shape.**
The combinations a UM tool asks about - "infection diagnosis AND
high WBC AND intravenous antibiotic" - are two set tokens and a
lab. Set columns reach the report as semicolon-joined strings, so
`pd.to_numeric` made them NaN; and even expanded, `three_way`'s
`nunique >= 8` floor rejected the 0/1 indicators. The pattern class
the page is most asked about was outside the page TWICE OVER.
Tokens now expand through `sets.vocabulary` - the same screener the
generator uses, so the report and the engine cannot disagree about
what is publishable - and genuine binary indicators are admitted
(a CONSTANT column is not). A planted three-way spanning two set
tokens and a lab is found, with both mutations watched red.

NOT yet known: whether any such combination exists in the real
extract. The next real run answers it, and the expansion count is
printed so an unexamined token is distinguishable from an absent
pattern.
