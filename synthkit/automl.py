"""A competitor worth the name, for the vendor seat.

WHY THIS EXISTS. `autosolver` is the deliberately modest FLOOR -
stdlib logistic regression, full-batch gradient descent, zero
randomness - and it is honest about being the floor. But a
vendor evaluation grades a vendor against what WE could do, and
"we could do stdlib logistic regression" sets a bar nobody has
to clear. A vendor beating that floor has shown nothing; a
vendor beating a competent in-house attempt has shown
something.

WHAT IT DOES, AND WHAT IT REFUSES TO DO.

It tries a small set of model families with light feature
engineering and picks by an INTERNAL held-out split. It never
sees a test label - the solver contract hands it
`test_rows` with no labels, and the selection split is carved
out of the training rows only. A model selected on the test set
would score beautifully and mean nothing, which is the
in-sample fault this project already records in `vendor_exam`,
where a vendor above the ceiling is a broken exam rather than a
good score.

IT SAYS WHAT IT CHOSE. Every run records the families tried,
their internal scores, the winner, and how many features were
built. A competitor that reports only a number cannot be
argued with, and the point of this one is to be argued with.

IT DEGRADES, STATED. Without scikit-learn it falls back to the
stdlib floor and says so, rather than failing a campaign on an
import. The fallback is NOT silent: the recorded choice names
it, so a report card cannot quietly claim a competitor that
never ran.
"""
from typing import Any, Callable, Dict, List

# What the last call chose, for the report card to read. Keyed
# by tag so two solvers in one campaign do not overwrite each
# other - the same shape `autosolver.LAST_FIT` uses.
LAST_CHOICE: Dict[str, Any] = {}

# The families, in the order they are tried. Kept as data so the
# record can name one that was tried and lost, not only the
# winner.
FAMILIES = ("gradient-boosted trees", "random forest",
            "regularized logistic regression")

# How much of the training data is held out to CHOOSE with.
# Carved from the training rows only - never the test rows.
SELECT_FRACTION = 0.25

# Below this many rows there is nothing to select on, and a
# selection made on forty rows is noise wearing a decision. The
# floor is used instead, and the record says why.
MIN_ROWS_TO_SELECT = 120


def _have_sklearn():
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


def _numeric(v):
    """A float, or None. Never NaN - a nan sails past `is None`
    and poisons every comparison downstream, which this project
    has already paid for once in the membership adversary."""
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none", "null", "na"):
        return None
    try:
        f = float(s.replace(",", ""))
    except ValueError:
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


class Features:
    """Numbers as numbers, categories as capped one-hots, and a
    MISSING INDICATOR for every column that is ever absent.

    The indicator is not decoration. This project measured that
    being MEASURED is itself a signal - in an extract the test
    was ordered BECAUSE the patient was unwell - and a model
    that cannot see absence cannot use it.
    """

    MAX_LEVELS = 20

    def __init__(self):
        self.numeric: List[str] = []
        self.levels: Dict[str, List[str]] = {}
        self.absent: List[str] = []
        self.names: List[str] = []
        self.fill: Dict[str, float] = {}

    def fit(self, rows):
        if not rows:
            return self
        cols = []
        for r in rows:
            for c in r:
                if c not in cols:
                    cols.append(c)
        for c in cols:
            vals = [r.get(c) for r in rows]
            nums = [_numeric(v) for v in vals]
            got = [v for v in nums if v is not None]
            # A COLUMN WHOSE PRESENT VALUES ARE NUMBERS IS
            # NUMERIC, however often it is present. This repo
            # records that lesson four times - at low
            # cardinality, at 46.6% presence, at zero presence,
            # and in the attribution screen - so the test is on
            # the present values and not on how many there are.
            if got and len(got) >= 0.6 * max(
                    1, sum(1 for v in vals
                           if v not in (None, "", "nan"))):
                self.numeric.append(c)
                srt = sorted(got)
                self.fill[c] = srt[len(srt) // 2]
            else:
                seen = {}
                for v in vals:
                    k = "" if v is None else str(v)
                    seen[k] = seen.get(k, 0) + 1
                keep = [k for k, _ in sorted(
                    seen.items(), key=lambda kv: -kv[1])
                    if k != ""][:self.MAX_LEVELS]
                if keep:
                    self.levels[c] = keep
            if any(v in (None, "", "nan") for v in vals):
                self.absent.append(c)
        self.names = (
            list(self.numeric)
            + ["{}={}".format(c, lv)
               for c in self.levels for lv in self.levels[c]]
            + ["{}:absent".format(c) for c in self.absent])
        return self

    def encode(self, rows):
        out = []
        for r in rows:
            row = []
            for c in self.numeric:
                v = _numeric(r.get(c))
                row.append(self.fill.get(c, 0.0)
                           if v is None else v)
            for c in self.levels:
                v = "" if r.get(c) is None else str(r.get(c))
                for lv in self.levels[c]:
                    row.append(1.0 if v == lv else 0.0)
            for c in self.absent:
                v = r.get(c)
                row.append(1.0 if v in (None, "", "nan") else 0.0)
            out.append(row)
        return out


def _auc(y, p):
    """Rank AUC, ties averaged. Pure python so the floor path
    needs nothing installed."""
    pairs = sorted(zip(p, y))
    n = len(pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    pos = sum(1 for _, yy in pairs if yy)
    neg = n - pos
    if pos == 0 or neg == 0:
        return 0.5
    s = sum(ranks[k] for k, (_, yy) in enumerate(pairs) if yy)
    return (s - pos * (pos + 1) / 2.0) / (pos * neg)


def _split(n, frac, seed):
    """A deterministic held-out slice of the TRAINING rows.

    Deterministic because a competitor that scores differently
    on re-run cannot be compared with anything, and this file's
    own rule is that a single seed decides nothing.
    """
    import random
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    cut = max(1, int(round(n * frac)))
    return set(idx[:cut])


def _fit_family(name, xtr, ytr, seed):
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == "gradient-boosted trees":
        m = HistGradientBoostingClassifier(
            random_state=seed, max_iter=200,
            early_stopping=False)
    elif name == "random forest":
        m = RandomForestClassifier(
            random_state=seed, n_estimators=300,
            n_jobs=1, min_samples_leaf=2)
    else:
        m = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0))
    m.fit(xtr, ytr)
    return m


def _scores(model, x):
    if hasattr(model, "predict_proba"):
        return [float(p[1]) for p in model.predict_proba(x)]
    return [float(v) for v in model.predict(x)]


def automl(seed: int = 0, tag: str = "automl") -> Callable:
    """The campaign-contract competitor.

    Selects among families on a held-out slice of the TRAINING
    rows, then refits the winner on all of them.
    """
    def solve(train_rows, train_labels, test_rows):
        y = [1 if v else 0 for v in train_labels]
        note: Dict[str, Any] = {"seed": seed,
                                "families_tried": [],
                                "n_train": len(train_rows)}

        if not _have_sklearn():
            # STATED, NEVER SILENT. A report card must not be
            # able to claim a competitor that never ran.
            from .autosolver import LogisticBaseline
            note["chosen"] = "stdlib floor (scikit-learn absent)"
            note["why"] = ("scikit-learn is not installed, so "
                           "no family could be tried and the "
                           "floor was used. This run is NOT a "
                           "competitor result.")
            LAST_CHOICE[tag] = note
            model = LogisticBaseline()
            model.fit(train_rows, train_labels)
            return model.score(test_rows)

        feats = Features().fit(train_rows)
        note["n_features"] = len(feats.names)
        x_all = feats.encode(train_rows)

        n = len(train_rows)
        distinct = len(set(y))
        if n < MIN_ROWS_TO_SELECT or distinct < 2:
            # A SELECTION MADE ON FORTY ROWS IS NOISE WEARING A
            # DECISION. One family, said plainly.
            pick = FAMILIES[-1]
            note["chosen"] = pick
            note["why"] = (
                "only {} training rows ({} outcome classes) - "
                "below the {} needed to choose between "
                "families on a held-out slice, so the most "
                "stable family was used without a contest"
                .format(n, distinct, MIN_ROWS_TO_SELECT))
            model = _fit_family(pick, x_all, y, seed)
            LAST_CHOICE[tag] = note
            return _scores(model, feats.encode(test_rows))

        held = _split(n, SELECT_FRACTION, seed)
        xtr = [x_all[i] for i in range(n) if i not in held]
        ytr = [y[i] for i in range(n) if i not in held]
        xho = [x_all[i] for i in range(n) if i in held]
        yho = [y[i] for i in range(n) if i in held]
        note["n_select_holdout"] = len(xho)

        best = None
        for fam in FAMILIES:
            try:
                m = _fit_family(fam, xtr, ytr, seed)
                sc = _auc(yho, _scores(m, xho))
            except Exception as exc:        # never break a run
                note["families_tried"].append(
                    {"family": fam, "error": str(exc)[:120]})
                continue
            note["families_tried"].append(
                {"family": fam, "held_out_auc": round(sc, 4)})
            if best is None or sc > best[0]:
                best = (sc, fam)

        if best is None:
            from .autosolver import LogisticBaseline
            note["chosen"] = "stdlib floor (every family failed)"
            LAST_CHOICE[tag] = note
            model = LogisticBaseline()
            model.fit(train_rows, train_labels)
            return model.score(test_rows)

        sc, fam = best
        note["chosen"] = fam
        note["held_out_auc"] = round(sc, 4)
        note["why"] = (
            "best held-out AUC among {} families on {} rows "
            "carved from the TRAINING split - no test label was "
            "read to make this choice".format(
                len(note["families_tried"]), len(xho)))
        # REFIT ON EVERYTHING. The split existed to choose, not
        # to train - throwing away a quarter of the data after
        # the decision is made would hand the vendor a
        # handicap we invented.
        model = _fit_family(fam, x_all, y, seed)
        LAST_CHOICE[tag] = note
        return _scores(model, feats.encode(test_rows))

    solve.__name__ = "synthkit-automl"
    return solve


def describe(tag: str = "automl") -> List[str]:
    """What the last call chose, in lines a report card prints."""
    note = LAST_CHOICE.get(tag)
    if not note:
        return ["the competitor has not run yet"]
    out = ["competitor: {}".format(note.get("chosen", "?"))]
    if note.get("why"):
        out.append("  {}".format(note["why"]))
    for t in note.get("families_tried", []):
        if "error" in t:
            out.append("  tried {:<32} FAILED: {}".format(
                t["family"], t["error"]))
        else:
            out.append("  tried {:<32} held-out AUC {:.4f}"
                       .format(t["family"], t["held_out_auc"]))
    if note.get("n_features") is not None:
        out.append("  {} features from {} training rows"
                   .format(note["n_features"], note["n_train"]))
    return out
