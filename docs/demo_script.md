# The demo: a doctor, a vendor, and an answer key
## Runbook (25-30 minutes, works fully offline on the Windows laptop)

**The frame (say first):** "A vendor is selling us a 30-day
readmission model for our heart-failure population. They claim
AUROC of at least 0.75. We cannot test them on real patients -
no answer key, all PHI. So we will build a synthetic population
where we KNOW every answer, hide the decisive risk factors
where they actually live - in the discharge notes - and make
both the vendor's model and our own best model take the same
exam."

### Act 1 - the ask (2 min)
Open `docs/demo_prompt.md`, read the paragraph aloud (or
paste it at station 01 and gesture). Key beats: specific
distributions (bimodal EF, zero-inflated priors, lognormal
creatinine), realistic mess (missing, typos, mixed dates,
outliers, duplicates), the note field carrying the signal
behind negation traps, and the outcome DECLARED as a 5-12%
minority.

### Act 2 - the contract (3 min)
Station 01 -> click preset "CHF readmission w/ notes (vendor
demo)" -> station 02. Show the spec card fingerprint ("same
fingerprint, same data, forever"). Click Validate, then
Semantic lint: the L1 line reads the REALIZED prevalence inside
the declared 5-12% band - "the spec means what the paragraph
said." Optionally type one space, watch the fingerprint change,
undo it.

### Act 3 - the population (3 min)
Station 03, backend `stub`, directory gui_runs/demo ->
Render data. Point at: null cells, a typo'd unit, a mixed date
format in the preview - then scroll a discharge_note cell:
"pt admits to missing several doses..." next to another
patient's "denies missing any medication doses". Terminal
aside (window 3): `type gui_runs\demo\dirty.csv | more` if
anyone wants the raw artifact. Line: "every one of these
corruptions is deliberate, seeded, and ledgered."

### Act 4 - our own model, built blind (5 min)
Station 04: goal `predict`, outcome `readmitted_30d`, bars
`auroc=0.6,gap_max=0.3` -> Compile ladder -> solver
`autosolver_hybrid` -> Run ladder (ticker counts; ~1-2 min).
While it runs, explain: "synthkit trains its own challenger on
a BLINDED training population - shifted seed, labels it never
sees at test time. The hybrid mines the notes: it learns which
phrases predict readmission from training labels alone, with
negated mentions kept as separate evidence." Result: tiers
pass, interval-annotated.

### Act 5 - the showdown (5 min)
Station 05: baseline `autosolver_hybrid`, vendor
`vendor_model:predict` (VendorCo RiskScore - "a competent
tabular model; not a strawman; it just can't read") -> Run
showdown. THE SLIDE lands as three lines, approximately:

    as-specified  ceiling 0.822 / baseline 0.724 / vendor 0.604  [bar: PASS] vendor loses to a stdlib baseline

(Rehearsed numbers from the demo machine itself; deterministic,
so these exact digits reappear. Note the strong-signal tier: the
vendor legitimately WINS there - amplified tabular signal - which
is worth saying aloud: "the instrument is not rigged; on the
realistic tier, not reading the notes costs them 0.12." The
[bar:] tag is the campaign's auroc>=0.6 threshold, separate from
the baseline comparison.)

Read it aloud: "This population supports AT MOST 0.82 - known
exactly, because we planted the truth. The vendor's claimed
0.75 is unreachable without reading the notes: they cap near
0.60. Our own free challenger, built in seconds, reaches 0.72.
Any vendor who wants this contract must beat the model we get
for free - on data where we know the answers."

### Act 6 - the close (3 min)
- `synthkit trials` on the campaign dir (window 3): append-only
  record - "every claim in this meeting is a file."
- The atlas (docs/system_map.html): one click into Truth
  Planning - "this is how the answer key is made."
- The cost line: "rendered here by a $0 local model on this
  laptop; the same spec runs against Bedrock's open-weight
  catalog when we want scale" (optional: re-render 8 docs at
  station 03 with backend `openai` if llama-server is up).

### Pre-demo checklist
- [ ] `synthkit version` matches the chip after hard-reload
- [ ] Preset chip loads; Validate green; lint INFO in-band
- [ ] One full rehearsal of Acts 3-5 (numbers stable: same
      seed, same data)
- [ ] vendor_model.py present in the repo root (rode the sync)
- [ ] Optional: llama-server tested for the Act-6 flourish
