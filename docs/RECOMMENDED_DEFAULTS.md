# What we recommend, before anyone asks us for anything

A request arrives: *"we want synthetic data for this cohort."*
This is what we hand back by default. Not a menu — a position. The
requester can argue us off it, and then we write down what they
changed and why.

Every figure here is measured on the 800-patient cohort. Where we
have not measured something, it says so.

---

## The defaults

| | Default | Why |
|---|---|---|
| **Smallest group we will describe** | **10 patients.** Never fewer. Raisable. | Below that, a published pattern starts describing individuals rather than a population. Counted in **patients, not visits** — somebody seen 200 times cannot push a pattern over the line alone. |
| **How much data we produce** | **Same size as the source cohort.** | More rows make results *steadier*, not *smarter*. Everything the data knows came from the real patients behind it. If a study needs more signal it needs more patients. |
| **What is released** | **The generated file only.** The model is destroyed after generation. | This is what makes the privacy argument work. A released model could reproduce patterns on demand; a released file cannot. |
| **The answer key** | **Withheld** whenever the data is being used to evaluate somebody's tool. | If the thing being predicted is in the file, a perfect score is unreadable — you cannot tell a good tool from a leaked answer. |
| **Identifier scan** | **Runs before anything else**, across all 18 regulated categories. | Every silent failure this tool has had was a column read as the wrong type. Cheap, and it runs in seconds. |
| **What ships with the file** | **The scorecard, the dictionary, and the limits** — always, in the same package. | A file without its measurements invites someone to assume it is as good as real data. |
| **De-identification sign-off** | **We assemble the evidence. We do not make the determination.** | Ten of the eleven required elements are measured. The eleventh is a written scope a qualified person signs. |

---

## PHI is not all treated the same, and that matters

A concern worth answering directly: *some identifying detail is
useful, and stripping all of it damages the data.* True, and the
tool already distinguishes three responses rather than deleting
everything it finds:

- **Drop** — direct identifiers. Names, record numbers, device
  serials. These carry no analytic value.
- **Generalise** — keep the variable, remove the precision.
  Service dates keep their **year** so the time axis survives.
  Ages over 89 become a single **"90 or above"** category rather
  than the column being deleted.
- **Review** — free-text notes. We report what we found and
  **do not act**. Redacting clinical text properly needs
  judgement we do not have, and a redactor that misses one
  identifier is worse than none because it hands back false
  confidence.

So the honest position is: we remove what identifies, we coarsen
what is both identifying and useful, and we refuse to quietly
destroy a column to make a scan look clean.

---

## What we need from a requester

1. **The data as CSV**, with a stated schema. Not a database table,
   not a live connection. One prescribed format keeps the surface
   small and reviewable.
2. **Which column, if any, their tool is meant to predict** — so we
   know what to withhold.
3. **What the data is for.** A tool being tested, a dashboard being
   built, and a study being powered need different things, and the
   defaults above move for the first two.

## What we will not do

- Go below ten patients for any published pattern. The tool refuses
  before it starts.
- Hand over the answer key and the explanatory data in the same file
  when a tool is being evaluated.
- Claim an Expert Determination.
- Describe the output as differentially private. We have built that
  method and measured its cost; it covers the model's training, not
  the whole pipeline, and we say so.
- Release the trained model.

## Decisions that are theirs, not ours

- Whether to raise the minimum group size above ten. It is a real
  trade and we will measure it for them, but the appetite is not
  ours to set.
- Whether the file may leave the account it was generated in.
- Whether a given use needs a signed determination.

---

## The cost, so nobody is surprised

The minimum-group rule is not free, and the cost lands almost
entirely on **rare codes** rather than on the overall shape of the
data.

Measured on the 800-patient cohort:

| | publishable | in the real data | kept |
|---|---|---|---|
| diagnoses | 1,050 | 10,975 | **9.6%** |
| procedures | 195 | 3,619 | **5.4%** |
| medications | 529 | 2,789 | 19.0% |
| routes of administration | 53 | 153 | 34.6% |

Numbers fare far better: across 31 numeric fields the rule leaves
**89%** of the natural spread available, and the engine delivers
**99.7%** of what is left.

**So: common clinical pictures come through well. Rare-condition
work does not, and no setting fixes that.** That is the single most
important sentence to give a requester up front.

---

## What we have not measured

**We have never swept the minimum group size.** Every figure above
is at ten. We cannot currently answer *"what would fifteen cost
us?"* with evidence, and we should not guess. That measurement is a
re-fit per value and nothing more — it should be run before anyone
negotiates the number.

One anchor while it is outstanding: the stricter formal privacy
method, at a conventional setting, would publish about **12%** of
the vocabulary we publish today. Our current position is already
considerably more conservative than it appears.
