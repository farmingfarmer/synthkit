"""The explanation graph: every term in the bench, drillable.

WHY THIS EXISTS. The bench is full of words that carry real
machinery behind them - k-anonymity, the latent space, a
membership adversary, an interaction surface - and a reader who
does not already know them has no way in. Every one of those
words is now clickable: it opens a panel that explains the idea
in plain language, and the panel's own text carries more
clickable terms, so a reader can keep drilling until they reach
the level they wanted.

ONE REGISTRY, HERE. The bench renders it and any future page can
too; two copies of an explanation are how the interface and the
documentation come to say different things about the same
mechanism.

THE CONTRACT, CHECKED. Every `[[slug]]` inside a body must
resolve to another entry - a link that goes nowhere is worse than
no link, because the reader trusted it. `broken_links()` returns
them and the checks assert the list is empty, the same way the
atlas refuses to build when a component is missing.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

# slug -> (title, body). Bodies may reference other entries as
# [[slug]] and are rendered with those as further links.
TERMS: Dict[str, Dict[str, str]] = {

"k-rule": {"title": "The k rule",
 "body": "Nothing is published unless at least <b>k</b> different "
 "people stand behind it (k is 10 here). A value held by three "
 "patients never leaves the machine; a pattern shared by "
 "hundreds does. This one rule is why the synthetic data can be "
 "shared at all.<br><br>It is counted in [[patients-not-rows]], "
 "which matters more than it sounds. It shapes what the "
 "generator is even allowed to learn, so it also explains some "
 "apparent 'errors' - see [[privacy-not-a-fault]]."},

"patients-not-rows": {"title": "Patients, not rows",
 "body": "Every privacy count here is over <b>people</b>, never "
 "records. One person seen 200 times could otherwise supply the "
 "ten most extreme rows by themselves and still be called "
 "anonymous by a row-counting rule.<br><br>So the [[k-rule]] "
 "asks how many distinct patients stand behind a value, and the "
 "published minimum and maximum are the <i>mean of the k most "
 "extreme patients' own extremes</i> - never one person's "
 "actual number."},

"privacy-not-a-fault": {"title": "Privacy, not a fault",
 "body": "Sometimes a synthetic column is visibly narrower than "
 "the real one, and that is the [[k-rule]] working rather than "
 "the generator failing. If three patients out of four hundred "
 "hold the extreme values, those values cannot be published, so "
 "the spread legitimately shrinks.<br><br>The reports say so "
 "explicitly - they print what share of the column's magnitude "
 "sits beyond the published bound - because someone once spent "
 "a day hunting a sampler bug that was this rule doing its "
 "job."},

"rules-engine": {"title": "The rules engine",
 "body": "The original generator. It studies the real data, "
 "writes down what it found as a list of published rules and "
 "distributions - a <b>blueprint</b> - and then builds new "
 "records from that written recipe <i>alone</i>. It never "
 "touches a real record while generating.<br><br>That is its "
 "great strength: everything it publishes is already "
 "[[k-rule]]-screened, so it has passed a full "
 "[[membership-attack]] battery, and the blueprint is a file a "
 "human can read and audit. Its weakness is measured too: it "
 "carries none of the [[combination-effect]]s in the real data. "
 "Compare [[neural-engine]]."},

"neural-engine": {"title": "The neural engine",
 "body": "An autoencoder. It trains a small neural network to "
 "squeeze each record down to a handful of numbers - its "
 "[[latent-space]] - and expand it back, then invents new "
 "records by sampling that squeezed space. Step by step: "
 "[[how-it-learns]], then [[how-it-generates]].<br><br>Because it "
 "learns the data's shape rather than a list of rules, it "
 "carries structure the [[rules-engine]] loses: on the real "
 "extract it keeps direction on 97% of relationships against "
 "92.6%, shows 3 [[backwards]] relationships against 11, and "
 "carries all eight mined [[combination-effect]]s where the "
 "rules engine reads zero.<br><br>The trade is that it learns "
 "from the <i>records</i>, so it needs defending: see "
 "[[denoising]], [[k-aware-blur]] and [[membership-attack]]. It "
 "also needed [[patient-structure]] built into it."},

"how-it-learns": {"title": "How the autoencoder learns",
 "body": "In four steps, and no step is a black box.<br><br>"
 "<b>1. Every record becomes numbers.</b> A number column "
 "becomes two - its value, standardized, and a flag for whether "
 "it was missing at all. A category becomes one column per "
 "level. A date is parsed to days since an epoch and treated as "
 "a number. A [[set-column]] becomes its size plus one flag per "
 "token. On the real extract 44 columns become about 425 of "
 "these.<br><br><b>2. The k rule is applied BEFORE training, "
 "not after.</b> Numbers are clipped to the k-anonymous bound, "
 "levels held by fewer than k patients are folded away. The "
 "network never sees the extremes it could leak - see "
 "[[k-before-training]].<br><br><b>3. One network is trained to "
 "rebuild its own input.</b> It is deliberately pinched in the "
 "middle: 425 numbers in, down to a few dozen, back out to 425. "
 "The only way to get the record back through that pinch is to "
 "learn what records of this kind LOOK like - see "
 "[[the-bottleneck]]. It is trained on a corrupted copy of the "
 "input, which is [[denoising]].<br><br><b>4. New records come "
 "from the squeezed side.</b> See [[how-it-generates]]. And "
 "whether it learned rather than memorized is measured on "
 "people it never saw: [[reconstruction-gap]]."},

"the-bottleneck": {"title": "The bottleneck",
 "body": "The pinch in the middle of the network, and the whole "
 "reason this works.<br><br>The network is shaped wide, narrow, "
 "wide - about 425 numbers in on the real extract, down to a "
 "few dozen, back out to 425. Those few dozen numbers are the "
 "[[latent-space]], and everything the network wants to say "
 "about a record has to fit through them.<br><br>That is a "
 "constraint, not an inconvenience. There is not enough room to "
 "store a copy of any individual record, so the cheapest way to "
 "get records back out is to learn the structure they share - "
 "which is exactly the thing we want to keep and not the thing "
 "we must not copy. The width is set from the data's own width "
 "and bounded, because an 8-wide bottleneck on a 77-column "
 "frame was measured reconstructing only 61% of the "
 "relationships.<br><br>It is not the only defense. See "
 "[[k-before-training]], [[denoising]] and "
 "[[reconstruction-gap]]."},

"how-it-generates": {"title": "How new records are made",
 "body": "Nothing is copied and nothing is sampled from a list "
 "of real records.<br><br>After training, every training record "
 "is pushed as far as the [[the-bottleneck]] and the cloud of "
 "points that lands there is described by a statistical model - "
 "a mixture of gaussian blobs. To invent a record, a NEW point "
 "is drawn from that model, somewhere the real points tend to "
 "be but almost certainly not on top of any of them, and pushed "
 "through the second half of the network to expand back into a "
 "full record.<br><br>Patients are built the same way, one "
 "level up: a point is drawn for the PERSON, then each of their "
 "visits is that point plus a drawn deviation, so every visit "
 "of a generated patient shares a who. Identities are invented "
 "(S000001 upward), never a real person's. See "
 "[[patient-structure]].<br><br>Before a drawn point is used it "
 "is checked: if it lands where fewer than k real patients "
 "stand, it is nudged until it does - [[k-aware-blur]]. Then "
 "the decoded values are clipped back inside the published "
 "bounds."},

"k-before-training": {"title": "k applied before training",
 "body": "The order matters more than the rule.<br><br>The "
 "[[k-rule]] is applied to the data BEFORE the network sees it: "
 "numbers clipped to the k-anonymous bound (the mean of the k "
 "most extreme patients' own extremes, never one person's "
 "value), and any category level held by fewer than k patients "
 "folded away. A network cannot leak an extreme it was never "
 "shown.<br><br>Be clear about what this does and does not buy. "
 "It narrows what CAN be learned; it does not by itself prevent "
 "memorization, which happens at the record level. That is why "
 "there are three more things: [[denoising]] during training, "
 "[[k-aware-blur]] at sampling time, and two measurements after "
 "the fact - [[reconstruction-gap]] and "
 "[[membership-attack]].<br><br>This is a genuinely different "
 "posture from the [[rules-engine]], which never touches a "
 "record at generation at all. Said every time it comes up."},

"reconstruction-gap": {"title": "The overfitting check",
 "body": "15% of PATIENTS - never just rows, so no one appears "
 "on both sides - are held out and the network never trains on "
 "them. Afterwards both groups are pushed through the network "
 "and rebuilt, and the errors are compared.<br><br>A network "
 "that learned the SHAPE of the data rebuilds a stranger about "
 "as well as a member, so the ratio sits near 1.0. A network "
 "that memorized rebuilds its own rows far better and the ratio "
 "climbs - and that is the same signal a membership adversary "
 "reads, which is why this is a privacy measure and not only a "
 "quality one.<br><br>It has a [[positive-control]], because a "
 "number that cannot go bad proves nothing. Measured on a frame "
 "built with NOTHING to learn - every value pure noise, so any "
 "low training error IS memorization - an undefended "
 "wide-bottleneck network reads <b>9.7</b> while the shipping "
 "configuration reads <b>1.8</b>, because it cannot memorize "
 "even when memorizing is the only way to do well. On ordinary "
 "structured data it reads <b>1.16</b>, and <b>1.44</b> with "
 "[[denoising]] switched off.<br><br>It is REPORTED, not gated: "
 "see [[the-gate]] for what does gate."},

"set-column": {"title": "A set column",
 "body": "A cell holding several things at once - a visit's "
 "conditions, its drugs, its procedures - usually written "
 "joined by a separator.<br><br>Treating the whole joined "
 "string as a category is wrong and was wrong here: "
 "<code>t01;t03</code> and <code>t03;t14</code> share what "
 "matters and share no category. On the real extract that put "
 "thousands of distinct combinations against a 60-level cap, so "
 "nearly every row collapsed onto one meaningless value.<br><br>"
 "So a set is modeled as its SIZE plus one indicator per "
 "token that clears the [[k-rule]], and generation draws a size "
 "and then that many tokens. Two consequences a reader should "
 "know: generated sets are visibly SHORTER than real ones "
 "because the rare tokens cannot be published, and an empty set "
 "means the visit genuinely held nothing - which is different "
 "from a visit whose contents were all too rare to publish, and "
 "the two are reported apart."},

"latent-space": {"title": "The latent space",
 "body": "The handful of numbers the [[neural-engine]] squeezes "
 "each record into. Think of it as a compressed description: "
 "enough to rebuild the record approximately, far too few "
 "numbers to memorise it exactly.<br><br>New records are made by "
 "sampling points in that space and expanding them back out. "
 "Where a point lands decides what kind of record comes out, "
 "which is why [[patient-structure]] and [[k-aware-blur]] both "
 "work by controlling <i>where</i> the sampling happens."},

"patient-structure": {"title": "Patient structure",
 "body": "These records follow people across repeated visits - "
 "about 76 visits per patient in the real extract - so a "
 "generator that emits unlinked rows has lost something large, "
 "however good each row looks.<br><br>The [[neural-engine]] "
 "originally did exactly that. Now each record's "
 "[[latent-space]] code splits into the <b>patient's centre</b> "
 "and that <b>visit's deviation</b> from it: a synthetic patient "
 "draws one centre, gets a [[k-rule]]-screened number of visits, "
 "and all their visits are placed around that centre. Measured "
 "by [[between-patient-share]]."},

"between-patient-share": {"title": "Between-patient share",
 "body": "The share of a column's variation that sits "
 "<i>between</i> people rather than within one person's own "
 "visits. A year of birth is about 1.00 - it belongs to the "
 "person. A single blood-pressure reading is much lower - it "
 "belongs to the visit.<br><br>It is the one number that says "
 "whether a longitudinal table actually has people in it. A "
 "generator with no [[patient-structure]] returns ~0 for every "
 "column, because its rows belong to nobody."},

"denoising": {"title": "Denoising training",
 "body": "The [[neural-engine]] is trained to rebuild a "
 "<b>clean</b> record from a <b>deliberately corrupted</b> one. "
 "That denies it the option of memorising exact records, which "
 "is precisely what a [[membership-attack]] reads back out of "
 "the trained weights.<br><br>Measured: it took the "
 "weights-based adversary from 0.776 (a fail) to 0.569 (a pass) "
 "<i>while fidelity improved</i>. A defence that also improves "
 "quality is not a trade-off, so it is the default."},

"k-aware-blur": {"title": "The k-aware blur",
 "body": "Generation should be faithful on average but "
 "deliberately <i>un</i>faithful where fewer than k people stand "
 "behind the pattern. So before a sampled point in "
 "[[latent-space]] is turned into a record, we ask how many "
 "distinct patients sit near the combination of values it would "
 "become; if fewer than k, it is blurred until it is no longer "
 "in a lonely place.<br><br>Honest result: it costs nothing "
 "(+0.1% on average fidelity) and, measured on the available "
 "data, buys nothing visible either - rare records were already "
 "further from the synthetic output than common ones. It ships "
 "as insurance with that stated."},

"membership-attack": {"title": "Membership inference",
 "body": "The central privacy question: given the released data, "
 "can an attacker tell whether a particular person was in the "
 "original cohort? We run it as a real attack and report the "
 "worst result, where 0.50 is a coin flip.<br><br>The "
 "[[rules-engine]] reads 0.52. The [[neural-engine]] reads 0.569 "
 "with [[denoising]] - but only against the adversary that sees "
 "its <i>output</i>; an attacker who obtains the trained network "
 "itself is a different and harder question.<br><br>None of "
 "these numbers mean anything without a [[positive-control]]."},

"positive-control": {"title": "The positive control",
 "body": "A clean attack result proves nothing unless the same "
 "attack <i>catches</i> a generator you know is cheating. So "
 "every privacy measurement here is run twice: once on the real "
 "generator, and once on a deliberate leaker that republishes "
 "its own training records.<br><br>The leaker must fail. When it "
 "does not, the attack is blind and its reassuring number is "
 "worthless - which is exactly what we found at full width, and "
 "why an output-side attack that survives many columns is still "
 "an open item."},

"drift": {"title": "Drift",
 "body": "The <b>synthetic</b> number minus the <b>original</b> "
 "one, for a pair of columns. Both are "
 "[[rank-correlation]]s.<br><br>A drift of -0.36 on a pair the "
 "source measured at 0.49 means the relationship survived but "
 "arrived much weaker. The SIGN of the drift only tells you "
 "which way it moved; what matters is where it landed, which is "
 "why each row is labelled: [[faded]], [[overshot]], "
 "[[sign-lost]] or [[backwards]].<br><br>A pair drifting is not "
 "automatically a fault - see [[privacy-not-a-fault]]."},

"rank-correlation": {"title": "Rank correlation",
 "body": "Every 'original' and 'synthetic' number in these "
 "tables is a Spearman rank correlation: -1 to +1, measuring "
 "whether one column goes up when another does, without "
 "assuming the relationship is a straight line.<br><br>Rank was "
 "chosen because clinical columns are skewed and a Pearson "
 "correlation would be dragged around by a handful of extreme "
 "values.<br><br>It has one important blind spot: a U-shaped "
 "relationship has a rank correlation near ZERO even though the "
 "relationship is strong. That is why [[effect-curve]]s are "
 "measured separately - [[close]] alone would call a lost "
 "U-shape a pass."},

"faded": {"title": "Faded",
 "body": "The relationship still points the same way, but it "
 "arrived <b>weaker</b>: the source said 0.57 and the synthetic "
 "data says 0.10.<br><br>This is the gentlest of the four "
 "verdicts - nothing in the file is misleading about "
 "DIRECTION - but it fails [[close]], because an analyst "
 "sizing an effect from this data would understate it. Check "
 "the k-rule column first: if the pair's columns lose their "
 "magnitude to the published bound, the fading is "
 "[[privacy-not-a-fault]]."},

"overshot": {"title": "Overshot",
 "body": "The relationship points the same way but arrived "
 "<b>stronger</b> than the source: 0.29 becoming 0.65.<br><br>"
 "Rarer than [[faded]] and worth more suspicion, because a "
 "generator inventing strength is putting a finding in the file "
 "that the real data does not support. Usually it means a "
 "modelled effect was applied twice, or that a column the "
 "relationship depended on came out less noisy than it really "
 "is."},

"sign-lost": {"title": "Sign lost",
 "body": "The relationship <b>did not survive at all</b>: the "
 "source had it clearly one way and the synthetic data reads "
 "near zero or slightly the other way.<br><br>Distinct from "
 "[[backwards]] by degree, not kind. A pair reading 0.41 in the "
 "source and -0.14 here has lost its sign, but -0.14 is weak "
 "enough that nobody would call it a finding. It still fails "
 "[[direction-kept]].<br><br>When many of these appear at once "
 "on related columns, suspect one broken mechanism rather than "
 "many broken pairs - on the real extract, every such pair was "
 "a set token against its own count partner, and one fix moved "
 "all of them."},

"coverage": {"title": "Coverage",
 "body": "Whether each column is PRESENT - not missing - at the "
 "same rate as in the real data, within 5 percentage "
 "points.<br><br>It sounds like a minor bookkeeping check and "
 "it is the most valuable early warning here: a coverage miss "
 "almost always means a column was read as the WRONG TYPE "
 "upstream. A date read as a category came back 79.9% missing "
 "against 0% in the source, and coverage is what said so."},

"set-token-shares": {"title": "Set token shares",
 "body": "For a [[set-column]], whether each individual item "
 "appears on the same share of rows as it does in the real "
 "data.<br><br>Measured per token rather than per column "
 "because a set can have exactly the right average length and "
 "still be made of the wrong things. It is checked beside the "
 "EMPTY rate - the share of rows whose set is genuinely empty - "
 "because neither number can see the other's failure: a "
 "generator can spread the right total mass over too many rows "
 "and every share still reads correctly."},

"suppressed-bins": {"title": "Suppressed bins",
 "body": "A histogram of real data is a set of counts, and a "
 "bar holding three people describes a group small enough to "
 "gossip about. So bars backed by fewer than k patients are "
 "removed from THIS REPORT, and the number removed is printed "
 "rather than hidden.<br><br>A fidelity report on real data is "
 "itself a release. See [[k-rule]]."},

"skill": {"title": "Skill",
 "body": "How much of a column a model can predict from its "
 "drivers, measured on <b>patients held out of the fit</b> - "
 "never on the rows it learned from, which would flatter every "
 "number here.<br><br>It is what separates a relationship worth "
 "publishing from a coincidence. The same held-out idea is used "
 "one level up to ask whether the generator itself is "
 "memorizing: [[reconstruction-gap]]."},

"the-gate": {"title": "The gate",
 "body": "Eight criteria a finished run either meets or does "
 "not, with an honest exit code: [[direction-kept]], "
 "[[close]], [[backwards]], coverage, set-token shares, empty "
 "rates, [[effect-curve]] tracking and [[interaction-surface]] "
 "survival.<br><br>A gate is a <b>floor, not a certificate</b>. "
 "When it fails it prints what to do next, cheapest option "
 "first - and 'proceed with the number stated' is on that list, "
 "because releasing is a human decision. The single hard stop "
 "is [[backwards]]."},

"direction-kept": {"title": "Direction kept",
 "body": "Of the relationships the real data really contains, "
 "how many point the <b>same way</b> in the synthetic file? If "
 "something rises with age in reality, it should rise with age "
 "here.<br><br>This is the most forgiving of the relationship "
 "measures - it ignores strength entirely. Its stricter partner "
 "is [[close]], and its catastrophic failure is "
 "[[backwards]]."},

"close": {"title": "Close",
 "body": "Stricter than [[direction-kept]]: not just the same "
 "direction, but roughly the same <b>strength</b>. A "
 "relationship can point the right way and still arrive badly "
 "diluted, which would mislead anyone modelling on the "
 "file.<br><br>Some of the gap is [[privacy-not-a-fault]] - when "
 "a column loses its extremes to the [[k-rule]], the "
 "relationship it carries necessarily weakens."},

"backwards": {"title": "Backwards (inverted)",
 "body": "A relationship that runs the <b>opposite way</b> to "
 "reality - falling where the real data rises. This is worse "
 "than losing the relationship entirely, because a missing "
 "pattern reads as nothing while a reversed one reads as a "
 "<i>finding</i>, and someone may act on it.<br><br>It is the "
 "one criterion in [[the-gate]] that is absolute: the bar is "
 "zero, always, and the guidance says stop rather than offering "
 "options."},

"effect-curve": {"title": "Effect curve",
 "body": "A relationship is not one number, it is a <b>shape</b>. "
 "A threshold, a saturation and a straight line can all share "
 "the same correlation while meaning completely different "
 "things.<br><br>So each discovered relationship is drawn as a "
 "curve measured on the original and again on the synthetic, "
 "and the gap between them is reported in standard deviations. "
 "Rank correlation cannot see a U-shape at all; this can."},

"interaction-surface": {"title": "Interaction surface",
 "body": "Some effects only exist when two columns act "
 "<b>together</b> - the pair matters beyond either column "
 "alone. A published surface is the generator's record of one "
 "such pair.<br><br>Related but distinct: a "
 "[[combination-effect]] is the same idea measured directly "
 "from the data rather than from what the generator chose to "
 "publish. Nothing above two columns at a time is modelled, and "
 "the reports say so rather than leaving it implied."},

"combination-effect": {"title": "Combination effects",
 "body": "The strongest two-column interactions <b>mined from "
 "the real data itself</b>, rather than taken from what an "
 "engine chose to publish. Their parents are found by model "
 "importance, not correlation - an XOR-shaped relationship has "
 "almost no correlation with its own causes, which is exactly "
 "why it matters.<br><br>This is the measurement that separated "
 "the two engines: on the real extract the [[rules-engine]] "
 "carries <b>none</b> of the top eight, and the "
 "[[neural-engine]] carries them."},

"driver-attribution": {"title": "Driver attribution (SHAP)",
 "body": "For a given column, which other columns actually drive "
 "it, and in what proportion? Computed on the original and the "
 "synthetic separately - matching bars mean the synthetic file "
 "distributes the driving the way reality does.<br><br>It caught "
 "something no pairwise measure could: on the real extract the "
 "[[rules-engine]]'s output reassigned a column's main driver "
 "(94/6 in reality became 25/75). Each relationship survived "
 "individually; <i>who drove whom</i> had been rewritten."},

"seed-sweep": {"title": "Seed sweep",
 "body": "The same settings with a different random seed give "
 "measurably different results, so a single run cannot tell a "
 "real improvement from luck. Anything drawn from a heavy tail "
 "needs several seeds before it means anything.<br><br>Here that "
 "is a rule with scars behind it: a +6% row-count 'bias' turned "
 "out to be noise over 40 seeds, and a fix that looked perfect "
 "on one seed reproduced the original bug on another."},

"planted-truth": {"title": "Planted truth",
 "body": "To grade a model honestly you need to know the right "
 "answer in advance - and nobody knows the true answer in real "
 "clinical data. So the exam keeps the measured covariates and "
 "plants a <b>known</b> outcome on top, declared in standard "
 "deviations.<br><br>That makes the [[ceiling]] computable, "
 "which is the whole point. What a model is asked here is "
 "narrower than 'does this work on our data', and every "
 "artifact says so about itself."},

"ceiling": {"title": "The ceiling",
 "body": "The best score <b>any</b> solver could achieve given "
 "the planted answer and the noise around it. It is knowable "
 "only because of [[planted-truth]].<br><br>It changes what a "
 "result means: a solver scoring 0.70 is excellent against a "
 "0.71 ceiling and poor against a 0.95 one. It also catches "
 "miscalibrated tests - a demanded bar sitting <i>above</i> its "
 "own ceiling is a broken exam, not a failed solver, and the "
 "report card says which."},

"the-duel": {"title": "The duel",
 "body": "Both generators measured against the same source on "
 "one page: every column drawn three ways, a grid showing which "
 "engine goes wrong where, the [[combination-effect]]s each one "
 "carries, and [[between-patient-share]] per column.<br><br>It "
 "exists because a fidelity number with nothing to compare it "
 "against means very little. It is how we know the "
 "[[neural-engine]] is better, and it remains the ruler any "
 "future change is measured with."},

"dials": {"title": "Dials",
 "body": "Deliberate adjustments to the generated data - shift a "
 "column, scale it, change how much is missing, weaken one "
 "specific relationship - so a dataset can be shaped to a "
 "purpose rather than only copied.<br><br>Every dial reports "
 "<b>requested against achieved</b>, because a dial can be "
 "capped by the [[k-rule]] or undone by a repair, and a silent "
 "difference between what you asked for and what you got is the "
 "failure this tool most refuses."},
}


# What text in the interface should become a link to which
# entry. Longest phrases first at match time, so "membership
# inference" wins over "inference". Only the FIRST occurrence in
# a section is linked - a page speckled with the same link ten
# times is harder to read, not easier.
PHRASES: Dict[str, List[str]] = {
    "k-rule": ["k rule", "k-anonymous", "k-anonymity",
               "k floor", "k-screened"],
    "patients-not-rows": ["over PATIENTS", "over patients"],
    "privacy-not-a-fault": ["privacy, not a fault",
                            "beyond bound", "published bound"],
    "rules-engine": ["rules engine", "RULES engine", "blueprint"],
    "neural-engine": ["neural engine", "NEURAL engine",
                      "latent engine", "autoencoder"],
    "latent-space": ["latent space"],
    "how-it-learns": ["how it learns", "learns the patterns",
                      "measure the patterns"],
    "the-bottleneck": ["bottleneck", "condensed feature space",
                       "compressed"],
    "how-it-generates": ["how it generates", "then generate",
                         "draws records"],
    "k-before-training": ["k-screened frame",
                          "before the network sees"],
    "reconstruction-gap": ["reconstruction", "held-out",
                           "overfitting", "holdout"],
    "set-column": ["set column", "set columns"],
    "patient-structure": ["patient structure",
                          "within-patient dynamics"],
    "between-patient-share": ["between-patient share",
                              "belongs to the person"],
    "denoising": ["denoising"],
    "k-aware-blur": ["k-aware blur", "k-blur"],
    "membership-attack": ["membership inference",
                          "membership attack", "membership"],
    "positive-control": ["positive control"],
    "the-gate": ["the gate", "M0 gate"],
    "drift": ["drift"],
    "rank-correlation": ["rank correlation", "Spearman"],
    "faded": ["faded"],
    "overshot": ["overshot"],
    "sign-lost": ["sign lost", "direction lost"],
    "coverage": ["coverage"],
    "set-token-shares": ["set token shares", "token shares"],
    "suppressed-bins": ["suppressed", "bin(s) suppressed"],
    "skill": ["skill", "held-out patients"],
    "direction-kept": ["direction kept", "kept direction",
                       "keep their direction"],
    "close": ["kept strength", "land within 0.2"],
    "backwards": ["INVERTED", "inverted", "BACKWARDS"],
    "effect-curve": ["effect curve", "shapes tracked",
                     "pattern card"],
    "interaction-surface": ["interaction surface",
                            "two-variable surface"],
    "combination-effect": ["combination effect",
                           "mined interaction"],
    "driver-attribution": ["driver attribution", "SHAP",
                           "drivers, attributed"],
    "seed-sweep": ["seed sweep", "seed-swept", "2-3 seeds"],
    "planted-truth": ["planted truth", "planted answer",
                      "planted outcome"],
    "ceiling": ["ceiling"],
    "the-duel": ["the duel", "Duel"],
    "dials": ["dial", "dials"],
}


def broken_links() -> List[str]:
    """Every [[slug]] that resolves to nothing. A link that goes
    nowhere is worse than no link - the reader trusted it."""
    bad = []
    for slug, e in TERMS.items():
        for ref in re.findall(r"\[\[([a-z0-9-]+)\]\]", e["body"]):
            if ref not in TERMS:
                bad.append("{} -> {}".format(slug, ref))
    return sorted(bad)


def orphans() -> List[str]:
    """Entries nothing links to. Not an error - a term can be
    reachable by being marked up in the interface - but worth
    seeing, because an entry no path reaches is one nobody
    reads."""
    linked = set()
    for e in TERMS.values():
        linked.update(re.findall(r"\[\[([a-z0-9-]+)\]\]",
                                 e["body"]))
    return sorted(set(TERMS) - linked)


def embed_html(scope: str = "body") -> str:
    """A SELF-CONTAINED explain panel for a standalone page.

    WHY THE DECK NEEDS ITS OWN. The bench renders this registry
    into its own panel, but the Dashboard is served into a
    sandboxed iframe and is ALSO a file people send to each
    other - so a reader outside the bench, or inside the frame,
    could reach none of it. The panel travels with the page.

    The REGISTRY is what must not diverge and does not: one
    dictionary, rendered twice. The markup differs because the
    hosts differ, which is the honest reason for two renderers
    rather than a copied vocabulary."""
    import json as _json
    data = _json.dumps({"terms": TERMS, "phrases": PHRASES})
    return """
<style>
.xq{border-bottom:1px dotted #9aa7b0;cursor:help;color:inherit}
.xq::after{content:"\\00a0?";font-size:.72em;vertical-align:.32em;
  color:#8c1515;font-weight:700}
.xq:hover{border-bottom-style:solid;border-bottom-color:#8c1515;
  background:rgba(140,21,21,.07)}
#xqwrap{display:none;position:fixed;inset:0;z-index:900;
  background:rgba(23,34,44,.36)}
#xqwrap.on{display:block}
#xqbox{position:absolute;top:50%;left:50%;
  transform:translate(-50%,-50%);width:min(620px,92vw);
  max-height:78vh;overflow:auto;background:#fff;border-radius:14px;
  padding:26px 30px 30px;box-shadow:0 30px 80px rgba(23,34,44,.34)}
#xqbox h3{margin:2px 0 4px;font-size:21px}
#xqbox .xrule{height:2px;background:#8c1515;width:44px;
  margin:10px 0 14px}
#xqbody{font-size:15px;line-height:1.65;color:#1d2a35}
#xqtrail{font:600 11px/1.4 ui-monospace,Menlo,monospace;
  letter-spacing:.08em;text-transform:uppercase;color:#6e6e73;
  margin-bottom:6px}
#xqtrail a{color:#8c1515;cursor:pointer;text-decoration:none}
#xqclose{position:absolute;top:12px;right:14px;border:0;
  background:none;font-size:23px;cursor:pointer;color:#6e6e73}
</style>
<div id="xqwrap" onclick="if(event.target===this)xqClose()">
 <div id="xqbox"><button id="xqclose" onclick="xqClose()">&times;
 </button><div id="xqtrail"></div><h3 id="xqtitle"></h3>
 <div class="xrule"></div><div id="xqbody"></div></div></div>
<script>
var XQ=__DATA__, xqPath=[];
function xqBody(t){return t.replace(/\\[\\[([a-z0-9-]+)\\]\\]/g,
  function(m,k){var e=XQ.terms[k];return e?
    '<span class="xq" onclick="xqOpen(\''+k+'\')">'+
    e.title+'</span>':m;});}
function xqOpen(sl,push){
  var e=XQ.terms[sl];if(!e)return;
  if(push!==false)xqPath.push(sl);
  document.getElementById('xqtitle').textContent=e.title;
  document.getElementById('xqbody').innerHTML=xqBody(e.body);
  document.getElementById('xqtrail').innerHTML=xqPath.map(
    function(k,i){var nm=(XQ.terms[k]||{}).title||k;
      return i===xqPath.length-1?('<b>'+nm+'</b>'):
        '<a onclick="xqBack('+i+')">'+nm+'</a>';}).join(' &rsaquo; ');
  document.getElementById('xqwrap').classList.add('on');}
function xqBack(i){xqPath=xqPath.slice(0,i+1);
  xqOpen(xqPath[xqPath.length-1],false);}
function xqClose(){document.getElementById('xqwrap')
  .classList.remove('on');xqPath=[];}
document.addEventListener('keydown',function(e){
  if(e.key==='Escape')xqClose();});
/* FIRST OCCURRENCE PER SECTION, and never inside a control, a
   code block or the panel itself - a page speckled with the same
   link ten times is harder to read, not easier. */
function xqMark(){
  var pairs=[];
  Object.keys(XQ.phrases).forEach(function(sl){
    (XQ.phrases[sl]||[]).forEach(function(ph){
      pairs.push([ph,sl]);});});
  pairs.sort(function(a,b){return b[0].length-a[0].length;});
  var scopes=document.querySelectorAll('__SCOPE__');
  if(!scopes.length)scopes=[document.body];
  Array.prototype.forEach.call(scopes,function(sec){
    var used={};
    var w=document.createTreeWalker(sec,NodeFilter.SHOW_TEXT,{
      acceptNode:function(n){
        var p=n.parentElement;
        if(!p)return NodeFilter.FILTER_REJECT;
        if(p.closest('pre,code,script,style,svg,button,'+
          '.xq,#xqbox'))return NodeFilter.FILTER_REJECT;
        return n.nodeValue&&n.nodeValue.trim().length>2?
          NodeFilter.FILTER_ACCEPT:NodeFilter.FILTER_REJECT;}});
    var nodes=[];while(w.nextNode())nodes.push(w.currentNode);
    nodes.forEach(function(n){
      for(var i=0;i<pairs.length;i++){
        var ph=pairs[i][0],sl=pairs[i][1];
        if(used[sl])continue;
        var at=n.nodeValue.indexOf(ph);
        if(at<0)continue;
        var after=n.splitText(at);after.splitText(ph.length);
        var sp=document.createElement('span');
        sp.className='xq';sp.textContent=ph;
        sp.onclick=(function(k){return function(){xqOpen(k);};})(sl);
        after.parentNode.replaceChild(sp,after);
        used[sl]=1;return;}});});}
if(document.readyState!=='loading')xqMark();
else document.addEventListener('DOMContentLoaded',xqMark);
</script>
""".replace("__DATA__", data).replace("__SCOPE__", scope)


def as_json() -> Dict[str, Any]:
    """The registry in the shape the bench renders."""
    return {"terms": TERMS, "phrases": PHRASES}
