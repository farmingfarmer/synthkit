# SynthKit — demo script for Yesha

**Assume she has seen nothing.** No jargon, no internals. Her lens is
executive: *input → output → impact*. Lead with the picture, land the
business value, stay honest about limits.

**Total: ~15 minutes talking, ~15 for her questions.**

---

## BEFORE SHE JOINS (2 minutes, do this first)

1. Bench is open at the **HOME** page. Logo glowing top-left.
2. Have these open in **browser tabs**, in this order, so you never
   hunt for anything live:
   - Tab 1: the bench (HOME)
   - Tab 2: `marketing/SynthKit_pullapart_loop.gif` (double-click it)
   - Tab 3: `marketing/SynthKit_showcase_slide.html`
3. Decide now: **you are not going to run a long job in front of her.**
   Everything below is instant.

---

## 1. THE HOOK — lead with the picture (90 seconds)

> **Switch to Tab 2 — the pull-apart loop. Let it run. Say nothing for
> three seconds.**

**Say:**

> "Every dot here is a patient. The blue ones are real. The red ones
> don't exist — SynthKit invented them.
>
> Watch what happens when I pull them apart. [*let the loop separate*]
> Same shape. Same structure. Same relationships between age, blood
> pressure, and glucose. But not one of those red patients is a real
> person, and not one real record was copied to make them."

> **Pause. Let it loop once more.**

> "That's the whole product in one picture. We learn the patterns. We
> never copy the records."

**Why this works:** she asked for razzle dazzle and a 10-second pitch.
This is both. Do not explain the math. Do not say "latent space."

---

## 2. WHAT IT IS — one sentence (30 seconds)

**Say:**

> "SynthKit is an internal toolkit that generates synthetic medical data
> that behaves like real medical data — so we can hand it to a vendor,
> or a researcher, or an intern, without handing over patient records."

**Then the two benefits, in her words:**

> "Two things that gets us. One: we can evaluate vendors on data that
> looks like ours, instead of their cherry-picked demo set. Two: our own
> people can build and test without waiting months for data access."

---

## 3. USER INPUT — "from nothing" (2 minutes)

> **Switch to Tab 1, the bench. Click `01 Describe` in the left rail.**

**Say:**

> "There are two ways in. The first one is: you just tell it what you
> want, in plain English."

> **Scroll to the big text box. Read your typed description aloud —
> slowly. It's already there.**

> "'A 1,000-patient heart failure cohort. Age, sex, blood pressure,
> HbA1c, kidney function, current medications. About 12% readmitted
> within 30 days, and readmission more likely for older patients with
> poor kidney function.'
>
> That's it. That's the input. No data required — this is creating a
> dataset from nothing but a sentence."

> **Point at the template chips above the box.**

> "And if someone doesn't want to write a paragraph, there are
> ready-made starting points. This one — [*point at CHF readmission*]
> — is the one I'll use, because it's already compiled and I don't want
> to make you watch a model think."

> **Click `CHF readmission w/ notes (vendor demo)`. It jumps to Step 2.**

**Say (as Step 2 appears):**

> "Whatever you typed becomes this — a formal, reviewable recipe. And
> this is the part compliance cares about: **nothing gets created until
> a human approves this recipe.** The AI drafts it. A person signs it."

---

## 4. THE OUTPUT — data on screen (2 minutes)

> **Click `03 Data` in the left rail. Scroll to `Create the data`.
> Click it. It's instant.**

**Say while it writes:**

> "And that's the dataset."

> **Point at the three stat cards.**

> "A thousand and twenty synthetic patients. Seven hundred deliberate
> errors — because real data is messy, and if we hand a vendor perfectly
> clean data we learn nothing about whether their tool can cope.
>
> And this one matters — [*point at the 9.3% card*] — we **asked** for a
> readmission rate between 5 and 12 percent, and the generated data came
> back at 9.3. It hit the target we set. That's the instrument working."

> **Scroll to the table.**

> "Here are the patients. Names, ages, ejection fraction, sodium,
> creatinine. Missing values where real data would have missing values.
> Downloadable as CSV or Excel right here — [*point at the buttons*] —
> plus an answer key, because we planted the truth on purpose and we know
> what the right answer is."

> **Say the line that lands with executives:**

> "None of these people exist. This file can go to a vendor today."

---

## 5. HOW GOOD IS IT — the proof (3 minutes)

> **Switch to Tab 3 — the showcase slide. Scroll slowly.**

**Say:**

> "Now, the obvious question: is it any good, or does it just look good?"

> **At the spinning cloud:**

> "Same picture from the top — real and synthetic patients plotted by
> age, blood pressure and glucose. They land on top of each other."

> **Scroll to the gate scorecard — the green MET.**

> "This is the part I want you to remember. We don't ask anyone to trust
> us. Every run is **graded**, automatically, against the original data.
> Six criteria. The relationships survive, the distributions match, the
> shapes of the curves hold.
>
> And it is built to **refuse**. If a run doesn't clear the bar, this
> says NOT MET, names exactly which check failed, and tells you what to
> do about it. Our answer is never 'trust us' — it's a number, next to
> the way to check the number."

> **Scroll to the distribution plots and the correlation heatmaps.**

> "Blue is the original, red is the synthetic — they trace each other.
> And these two grids are every relationship in the data, original on
> the left, synthetic on the right. Nearly identical."

---

## 6. PRIVACY — be precise, do not oversell (90 seconds)

**This is the part that gets you through compliance. Say it exactly:**

> "On privacy, I want to be careful with my words, because this is where
> vendors overclaim and get caught.
>
> Every number the tool publishes has to describe **at least ten
> patients** — never one person. Even the highest and lowest values in a
> column are averages across the ten most extreme patients, so no
> individual's actual value is ever exposed.
>
> And we attack our own output. We run a re-identification attack that
> tries to work out whether a specific person was in the source data. It
> scores about 50% — a coin flip. To prove the attack isn't just broken,
> we also run it against a generator that cheats and republishes real
> records — that one gets caught at 100%.
>
> What I will **not** say is that it's 'anonymized' or 'de-identified,'
> full stop. The honest claim is: group-level statistics, measured to
> resist that attack, on this cohort shape. That's defensible. The
> bigger claim isn't."

**If she pushes on it:** "It's a floor, not a certificate. And that's
the sentence I'd want us to put in front of compliance."

---

## 7. CLOSE — impact and the ask (60 seconds)

**Say:**

> "So: a sentence in, a usable dataset out, and a grade on the way that
> tells you whether to trust it.
>
> What it unlocks: vendor evaluations on data shaped like ours.
> Researchers and interns building without a data-access queue. And a
> way to answer 'can your tool handle our data' without our data ever
> leaving the building."

**Then the ask — pick the one you actually want:**

- "I'd like your read on taking this to compliance, and what you want
  the posture slide to say."
- "I'd like to pick two or three real vendor evaluations to run this on
  next."
- "I'd like Massey's help turning these visuals into the deck."

---

## IF SHE ASKS — short answers

**"Is this using real patient data?"**
> "Not in what I just showed you — that was invented demo data. When we
> point it at a real extract, the real data stays on the secured machine.
> Only the statistics travel, and only group-level ones."

**"How is this different from what vendors sell?"**
> "Two things. It runs inside our walls, so our data never leaves. And
> it grades itself and refuses — I haven't seen a vendor deck that shows
> you its own failures by name."

**"Can it do [other dataset]?"**
> "Yes — we've run it across 23 different dataset shapes. It's not
> tuned to one table."

**"How long does a real run take?"**
> "Minutes to a few hours depending on size. The 800-patient extract
> with 47 columns runs in about five minutes."

**"What can't it do?"**
> *(Answer this honestly — it buys you credibility for everything else.)*
> "Two things. It carries who a patient is, but it does not yet model a
> patient's course **over time** — the sequence of their visits. And the
> privacy result is one cohort shape at one setting; it's a floor, not a
> blanket guarantee. Both are written down, not hidden."

**"What do you need from me?"**
> *(Have your one ask ready — see Close above.)*

---

## RULES FOR YOURSELF

- **Do not say:** latent space, autoencoder, k-anonymity, blueprint,
  gate criteria by name. She does not care and it costs you the room.
- **Do say:** patterns, relationships, graded, refuses, group-level,
  nobody real.
- **Do not run anything slow on screen.** Everything above is instant.
- **If something breaks live:** go to Tab 3, the showcase page. It's
  static and it carries the whole story.
- **Let the loop run while you talk.** Silence over motion is fine.
