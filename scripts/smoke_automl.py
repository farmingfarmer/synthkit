"""Smoke: the competitor that makes a vendor claim mean something.

WHY THIS EXISTS. `autosolver` is the deliberately modest floor -
stdlib logistic regression, and honest about it. A vendor
beating that floor has shown nothing. `automl` is what we could
actually do: a few families, light feature engineering, chosen
on a held-out slice of the TRAINING rows.

THE TWO PROPERTIES THAT MATTER, and they pull against each
other. It must WIN where the floor is blind, or it is not a
competitor. And it must NEVER LOSE where the floor is already
right, or every tie becomes an argument about whether the
competitor was trying. Both are measured across mechanisms and
seeds, because a single seed on a model comparison is worth
nothing and this project has the scars to prove it.

AND IT MUST NOT CHEAT. The selection split is carved from the
training rows; a family chosen on the test set would score
beautifully and mean nothing. That is asserted by handing it
test rows whose labels contradict the training signal - a
selector peeking at them would pick differently.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from synthkit.automl import (                        # noqa: E402
    FAMILIES, LAST_CHOICE, MIN_ROWS_TO_SELECT, _auc, automl,
    describe)
from synthkit.autosolver import autosolver           # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:3d}  {}".format(PASS, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def planted(kind, n=3000, seed=0):
    """Mechanisms the floor can and cannot see.

    `linear` is what logistic regression is FOR - a competitor
    that cannot tie here is broken. `xor` and `curved` are
    invisible to it, which is where a competitor has to earn
    its name.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    c = rng.choice(["x", "y", "z"], n)
    if kind == "linear":
        z = 1.2 * a + 0.8 * b
    elif kind == "xor":
        z = 2.2 * np.sign(a) * np.sign(b)
    elif kind == "curved":
        z = 1.5 * (a ** 2) - 1.2
    else:
        raise ValueError(kind)
    z = z + 0.6 * (c == "x") + rng.normal(0, 0.5, n)
    y = (1 / (1 + np.exp(-z)) > rng.random(n)).astype(int)
    rows = [{"a": str(round(x, 3)), "b": str(round(w, 3)),
             "c": k} for x, w, k in zip(a, b, c)]
    return rows, [int(v) for v in y]


def arm(kind, seed):
    rows, y = planted(kind, seed=seed)
    cut = int(0.7 * len(rows))
    tr, trl, te, tel = rows[:cut], y[:cut], rows[cut:], y[cut:]
    f = _auc(tel, autosolver()(tr, trl, te))
    m = _auc(tel, automl(seed=seed)(tr, trl, te))
    chose = (LAST_CHOICE.get("automl") or {}).get("chosen")
    return f, m, chose


def main():
    try:
        import sklearn  # noqa: F401
        have = True
    except ImportError:
        have = False

    if not have:
        print("     NOTE: scikit-learn is not installed - the "
              "competitor falls back to the floor and the "
              "comparison checks are SKIPPED rather than "
              "passing on a competitor that never ran")
        rows, y = planted("xor", n=400)
        automl()(rows[:300], y[:300], rows[300:])
        note = LAST_CHOICE.get("automl") or {}
        check("...and the fallback SAYS it is not a competitor "
              "result, so a report card cannot claim one that "
              "never ran",
              "floor" in str(note.get("chosen", ""))
              and "NOT a competitor" in str(note.get("why", "")))
        print()
        print("All {} checks passed.".format(PASS))
        return

    # ---- IT WINS WHERE THE FLOOR IS BLIND
    gains = {}
    for kind in ("linear", "xor", "curved"):
        gains[kind] = [arm(kind, s) for s in (0, 1, 2)]

    chosen = {k: [c for _f, _m, c in v]
              for k, v in gains.items()}
    gains = {k: [(f, m) for f, m, _c in v]
             for k, v in gains.items()}
    xor = gains["xor"]
    check("the floor is genuinely BLIND to the xor mechanism - "
          "AUC {} across three seeds, which is a coin flip. A "
          "competitor measured against a floor that already "
          "solves the problem proves nothing"
          .format([round(f, 3) for f, _ in xor]),
          all(f < 0.60 for f, _ in xor))
    check("...and the competitor SEES it: {} against the "
          "floor's {}, a gain of {} on every seed"
          .format([round(m, 3) for _, m in xor],
                  [round(f, 3) for f, _ in xor],
                  [round(m - f, 3) for f, m in xor]),
          all(m - f > 0.25 for f, m in xor))
    cur = gains["curved"]
    check("the same on a curved mechanism - floor {}, "
          "competitor {}"
          .format([round(f, 3) for f, _ in cur],
                  [round(m, 3) for _, m in cur]),
          all(m - f > 0.10 for f, m in cur))

    # ---- AND IT NEVER LOSES WHERE THE FLOOR IS RIGHT
    lin = gains["linear"]
    check("on a LINEAR mechanism - what logistic regression is "
          "for - the competitor ties rather than losing: floor "
          "{}, competitor {}. A selector that cannot tie here "
          "would make every close result an argument about "
          "whether it was trying"
          .format([round(f, 3) for f, _ in lin],
                  [round(m, 3) for _, m in lin]),
          all(m >= f - 0.01 for f, m in lin))
    allarms = [x for v in gains.values() for x in v]
    worse = [(f, m) for f, m in allarms if m < f - 0.01]
    check("...and across all {} arms it is never worse than "
          "the floor by more than 0.01 - {} arms worse"
          .format(len(allarms), len(worse)),
          not worse)
    # THIS CHECK WAS WRITTEN AS A LITERAL `True` and left
    # that way for one run - a placeholder that reads as
    # coverage, which is the exact fault this project
    # forbids. It asserts the thing now: a selector that
    # always names the same family is a second floor with
    # extra steps.
    check("...and it CHOSE differently by mechanism rather "
          "than always naming one family - linear {} against "
          "xor {}, which is what makes it a selector"
          .format(sorted(set(chosen["linear"])),
                  sorted(set(chosen["xor"]))),
          len({c for v in chosen.values() for c in v}) > 1
          and set(chosen["linear"]) != set(chosen["xor"]))

    # ---- THE CHOICE IS MADE WITHOUT READING A TEST LABEL
    rows, y = planted("xor", seed=7)
    cut = int(0.7 * len(rows))
    automl(seed=7)(rows[:cut], y[:cut], rows[cut:])
    note = dict(LAST_CHOICE.get("automl") or {})
    check("the run RECORDS every family it tried with the "
          "held-out score, not only the winner - a competitor "
          "that reports one number cannot be argued with, and "
          "the point of this one is to be argued with",
          len(note.get("families_tried", [])) == len(FAMILIES)
          and all("held_out_auc" in t
                  for t in note["families_tried"]))
    check("...and the selection holdout is carved from the "
          "TRAINING rows - {} of them - with the record saying "
          "no test label was read"
          .format(note.get("n_select_holdout")),
          0 < note.get("n_select_holdout", 0) < cut
          and "no test label" in note.get("why", ""))

    # A SELECTOR THAT PEEKED WOULD PICK DIFFERENTLY. The test
    # rows are replaced with rows whose signal is REVERSED; a
    # choice made on them could not match the honest one.
    # DRASTICALLY DIFFERENT, not merely shuffled. The first
    # version swapped two columns, which left the test rows
    # statistically similar enough that a selector scoring on
    # them still ranked the families the same way - the
    # mutation that decided ENTIRELY on the test set passed
    # this check while failing three others. Constant rows
    # cannot be confused with the real ones by anything.
    rrows = [{"a": "0", "b": "0", "c": "z"}
             for _ in rows[cut:]]
    automl(seed=7)(rows[:cut], y[:cut], rrows)
    note2 = dict(LAST_CHOICE.get("automl") or {})
    check("...and the choice is IDENTICAL when the test rows "
          "are replaced wholesale - proof the test set played "
          "no part in it, which a held-out AUC on the wrong "
          "rows would quietly break",
          note2.get("chosen") == note.get("chosen")
          and note2.get("held_out_auc") == note.get(
              "held_out_auc"))

    # ---- IT IS DETERMINISTIC
    r1 = automl(seed=3)(rows[:cut], y[:cut], rows[cut:])
    r2 = automl(seed=3)(rows[:cut], y[:cut], rows[cut:])
    check("the same seed gives the same scores - a competitor "
          "that moves on re-run cannot be compared with "
          "anything",
          list(r1) == list(r2))

    # ---- IT REFUSES TO PRETEND ON TOO LITTLE DATA
    srows, sy = planted("xor", n=60, seed=1)
    automl(seed=1)(srows[:50], sy[:50], srows[50:])
    small = dict(LAST_CHOICE.get("automl") or {})
    check("below {} rows it does NOT hold a contest it cannot "
          "settle - it names one family and says why, because "
          "a selection made on a handful of rows is noise "
          "wearing a decision".format(MIN_ROWS_TO_SELECT),
          not small.get("families_tried")
          and "below the {}".format(MIN_ROWS_TO_SELECT)
          in small.get("why", ""))

    # ---- AND IT REACHES THE REGISTRY
    from synthkit.gui import _solver
    fn = _solver("automl")
    check("`automl` resolves as a registry name, so a campaign "
          "on the data machine can name it without a dotted "
          "path - a capability wired to nothing is this "
          "repo's named silent-failure class",
          callable(fn) and fn.__name__ == "synthkit-automl")

    # DESCRIBED AFTER A FULL-SIZE RUN. The previous call was
    # the too-little-data one, which holds no contest, so
    # asking it to render "what it beat" was asking for
    # something that correctly does not exist.
    automl(seed=7)(rows[:cut], y[:cut], rows[cut:])
    lines = describe()
    check("describe() renders the choice for a report card, "
          "naming the winner and what it beat",
          any("competitor:" in ln for ln in lines)
          and any("tried" in ln for ln in lines))

    # ---- THE SHOWDOWN'S FOURTH ARM
    #
    # A vendor clearing `autosolver` has shown nothing, so the
    # verdict is placed against the COMPETITOR where one ran.
    # The three placements are the product statement and each
    # is asserted, because a vendor between the floor and us
    # must not round up to a pass.
    from synthkit.autosolver import ShowdownResult, ShowdownTier

    def tier(comp, vend):
        return ShowdownTier(tier="t", ceiling=0.90,
                            baseline_auroc=0.62,
                            vendor_auroc=vend,
                            vendor_passed=True,
                            competitor_auroc=comp,
                            competitor_name="automl")

    check("a vendor ABOVE our competitor is told its claim is "
          "worth something",
          "worth something" in tier(0.80, 0.85).verdict)
    check("...a vendor BETWEEN the stdlib floor and our "
          "competitor is named as exactly that, rather than "
          "rounded up to beating a baseline nobody should be "
          "impressed by",
          "between the stdlib floor" in tier(0.80, 0.70).verdict
          and "worse than us" in tier(0.80, 0.70).verdict)
    check("...and a vendor below the floor still says so",
          "loses to a stdlib baseline"
          in tier(0.80, 0.55).verdict)
    check("...while a tier where NO competitor ran keeps the "
          "old wording - absent is not the same fact as "
          "scoring zero, and an old result must still load",
          tier(None, 0.70).competitor_auroc is None
          and "baseline" in tier(None, 0.70).verdict)

    res = ShowdownResult(title="t", vendor_name="acme",
                         tiers=[tier(0.80, 0.70)])
    txt = res.format_text()
    # THE TIER LINE, not the whole document - the header
    # names the vendor first, so searching the text found
    # "acme" before "competitor" and failed a line that was
    # already correct.
    _tl = txt.splitlines()[-1]
    check("the printed line puts the competitor BETWEEN the "
          "floor and the vendor, where the eye goes - at the "
          "end the floor would keep reading as the thing to "
          "beat: {}".format(txt.splitlines()[-1].strip()[:70]),
          _tl.index("competitor") < _tl.index("acme")
          and _tl.index("floor") < _tl.index("competitor"))

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
