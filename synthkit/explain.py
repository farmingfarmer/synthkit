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
 "records by sampling that squeezed space.<br><br>Because it "
 "learns the data's shape rather than a list of rules, it "
 "carries structure the [[rules-engine]] loses: on the real "
 "extract it keeps direction on 97% of relationships against "
 "92.6%, shows 3 [[backwards]] relationships against 11, and "
 "carries all eight mined [[combination-effect]]s where the "
 "rules engine reads zero.<br><br>The trade is that it learns "
 "from the <i>records</i>, so it needs defending: see "
 "[[denoising]], [[k-aware-blur]] and [[membership-attack]]. It "
 "also needed [[patient-structure]] built into it."},

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


def as_json() -> Dict[str, Any]:
    """The registry in the shape the bench renders."""
    return {"terms": TERMS, "phrases": PHRASES}
