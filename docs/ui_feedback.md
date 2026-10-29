# UI feedback from the walkthrough — verbatim-sourced work list

## STATUS as of 2026-10-29 — what is done, what is left

DONE, with checks that were watched red first:

- P1 output-size lever — `synthkit/sizing.py`, the Generate panel's
  ladder, refusal rows shown with their reason. Sequencing as she
  asked: read the data, then offer bounds that belong to that file.
- P2 "deliberate corruptions" — replaced by requested-against-
  achieved with a confidence percentage, and it found a class the
  old count could not see (an op asked for that cannot apply to its
  column lands nothing and wrote no ledger row). Spec step 2 states
  the percentage and names which of the two causes is responsible.
- P2 "view the performance", "Read the columns", lag stated automatic.
- P3 gray out the route not taken — chosen at Home, `pickRoute`.
- P3 do not re-ask source/output on Generate — `carryForward()`.
- P3 merge Source and Generate — one station, measure route is now
  two numbered steps (Source 01, Verdict 02).
- P3 remove the seed control — hidden.
- P3 Dashboard density — the deck opens on a summary with
  "Show everything (N more sections)", inline so it travels into the
  station iframe and the shared file alike.
- P4 Fit / Learn / Duel folded under "testing an ML tool or vendor";
  Roadmap folded under "internal".
- P5 black out of the plots — `INK = "#2f3a44"`.

NOT DONE, and deliberately not decided here:

- **P3 privacy floor: the conflict is still open.** She said the
  floor is internal, "not something you're going to want to give to
  users ... we would want to have our own standard". The team demo
  asked for the opposite and it SHIPPED - a user-settable k with a
  refusal below 10. Both are reasonable and they contradict; picking
  one silently would bury a stakeholder disagreement inside a code
  change. Needs a decision, not an implementation.

NOT UI, still owed:

- The screenshot user guide (she gated it on the build settling;
  the build has just moved again).
- The 100-patient cohort drawn from the 5,000-patient set.



Extracted from the screen-share walkthrough transcript. Quotes are
hers. Nothing here is inferred; where a line is ambiguous it says so.

## PRIORITY 1 — the output-size lever (she named it the priority)

> "on this roadmap prioritize what I was saying about that lever to
> control the sample size ... that's going to be an immediate
> question. Like I see that already."

The shape she asked for, in three parts:

1. **Ratio control.** "Run in one to one and can you do one to ten?"
   — 800 patients in should not force 800 out.
2. **A fidelity percentage attached to each level.** "when you add
   the thresholds you put a percentage of fidelity on the bottom.
   Like ... if I gave you 50 records, I want 50, that'll be like
   very close, 97%. As opposed to I want 50,000 and then at that
   point you have like 20% fidelity."
3. **A refusal floor.** "If we can get up to 70%, only those are the
   ones we'll give ... That 50,000 we shouldn't even do."

And the sequencing that makes it possible:

> "you read the data, you understand the complexity of the patterns,
> and then as step two, based off of what I gave you, then you can
> curate what I can and can't do in terms of levers ... Here's low,
> here's high. This is all I can do and I can work between this.
> Alex, that would be amazing if we can put that in this."

So: **learn the patterns FIRST, then offer the dials bounded by what
that specific dataset can support, then generate.** Condensing and
generating become two steps rather than one.

## PRIORITY 2 — wording that confuses or alarms

- **"deliberate corruptions" must go.** "I don't know what deliberate
  corruption means and is this something the user needs to know?" and
  "in calling it a corruption ..." — she wants a **confidence /
  percentage of what was requested** instead. Her general note: "how
  we're wording all of this and how we're presenting it to the user
  is really important."
- **"see the fidelity" -> "view the performance."** "I would just say
  performance and view the performance or something."
- **"check the types"** — "rather be check the types maybe ... we'll
  work soon [on] what that is." Needs a plainer label.
- **Lag features must say they are automatic.** "maybe that's one
  thing you should put there. That this is like automatically
  determined ... Because I think that's not clear."

## PRIORITY 3 — remove and simplify

- **Gray out the route you did not choose.** "is there a way we can
  gray out that part, if you don't start with the dataset, you start
  with just the text ... that way you can't even click on those
  buttons." Chosen at Home.
- **Do not re-ask for source and output on Generate.** "why are we
  re-pointing the source and the output?" -> "a root output directory
  ... in that first part and then just skip that part here."
- **Merge Source and Generate.** "what is on this page? You wouldn't
  have to have anything. You could just have source and generate
  essentially the same station."
- **Remove the seed control.** "Don't give them that option. They
  don't need it ... keep it on the back."
- **Privacy floor is internal, not a user dial.** "this is something
  you're not going to want to give to users ... We would want to have
  our own standard."
  **CONFLICT, FLAGGED:** the team demo produced the opposite ask — a
  user-settable k with a floor, which shipped. One of these has to
  give; do not silently pick.
- **Dashboard is too dense.** "there's a lot here ... we can update,
  modify, get this to a filtered version."

## PRIORITY 4 — branding and framing

- **Fit / Learn / Duel are not general-purpose.** "this should be
  again, I'm thinking about branding ... it is not for all of the
  [neuro-agnostic]. This is specifically for **testing machine
  learning analytics tools**."
- **Roadmap is internal only.** "road map is of course only for us."

## PRIORITY 5 — cosmetic, but she chased it twice

- **Take the black out of the plots.** "just take the black out of
  those damn plots ... that can't be black. I will tell Alex. I will
  tell him to regenerate them now."

## Not UI, recorded so it is not lost

- **Nothing local. Everything on AWS**, S3 or Redshift, with the data
  movement question to go to the AWS team.
- **Two or three user personas** to be written (she and Patrick).
- **A screenshot user guide**, Salesforce-style, once the build is
  final.
- **Deliverable:** a synthetic **100-patient cohort drawn from the
  5,000-patient all-of-us set** - not from the original 100 - with
  the same underlying tables (encounters, labs, medications), then
  compare its per-person stats against the original 100 cohort.
  "I'm trying to pivot away from the other data set."
- **Deliverable:** share the 100-patient output in the chat before
  the compliance meeting.
- **Timing:** December task force review, not November.
