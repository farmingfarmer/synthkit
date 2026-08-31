"""Find what explains each column, without being the thing that
generates it.

WHY THIS EXISTS. The previous engine was one object that both searched
for structure and generated data. Discovery therefore inherited every
constraint generation has - values discretised into bins, every
relationship given a direction, the graph kept acyclic - and paid for
them in the search. Three consequences, all measured:

  * one pass over 130 columns and 30,566 rows cost 916 seconds, of
    which ~570 survived a 90% row cut. The cost was the interpreter
    walking Python lists, not the statistics
  * a conditional probability table over ten bins cannot see a lagged
    effect or a pure interaction, so both had to be hand-manufactured
    as columns - 210 products to recover what a boosted tree gets for
    nothing
  * an exclusive-or is SYMMETRIC, so the search correctly found
    y <- a*b, b <- a*y and a <- b*y. All three are true and together
    they are a cycle. Only a GENERATOR needs to pick one; a catalogue
    of patterns does not

So this module answers one question per column - what predicts it, how
strongly, and does that hold on patients the search never saw - and
answers it with a model that handles non-linearity and interactions
natively. Nothing here is generative. Direction, acyclicity and
discretisation are the generator's problems, and it gets to solve them
against a finished catalogue instead of during the search.

CONFIRMATION IS NOT A SEPARATE STEP ANY MORE. Skill and importance are
both measured on held-out PATIENTS, so a relationship that only lived
in the patients the search looked at scores zero by construction. The
old pipeline needed a discover-then-confirm pass and a Bonferroni
correction over thousands of pairs; neither is needed when the number
being reported is already out-of-sample.

TWO THINGS DELIBERATELY EXCLUDED as predictors:

  the group id      it identifies the patient, and a model given it
                    memorises rather than explains
  a column's OWN
  lag or delta      x__prev predicts x almost perfectly in anything
                    that persists, and permutation importance would
                    hand it the whole budget and mask every
                    cross-column relationship behind it. Persistence
                    is a separate mechanism the generator already
                    models, so it is not a finding here
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)

from .dates import (DATE_ORIGIN, date_kind, from_datetime,
                    labeller, to_ordinal)
from . import sets as _sets
from .quantities import ordinal_spec, quantity_kind, to_number
from .shapes import (FLAT_SHARE, additive_departure, curve_centre,
                     describe, describe_joint, effect_curve,
                     joint_surface, surface_centre)

# HistGradientBoosting bins categoricals into at most 255 slots.
MAX_LEVELS = 200
# What a value becomes when it falls outside the cap. Named once so
# the blueprint can look for it without a second spelling of the same
# string - a column made entirely of this is a DESTROYED column, and
# the check that finds it must be looking for the same thing.
OTHER = "__other__"
LAG_SUFFIXES = ("__prev", "__delta")

# A DATE IS NOT A CATEGORY. Without the branch in `prepare` a date
# column arrived as text, failed the numeric test, and became a
# category capped at MAX_LEVELS with everything outside the top 200
# rewritten to `__other__`. On the 800-patient extract that was 88.1%
# of visit_start_date, and `coverage_generated` still read 1.0 with
# `coverage_delta` 0.0 - coverage counts PRESENCE, so the column was
# destroyed with every per-column check green. What a date means as a
# number lives in `dates.py`, because generation needs the inverse and
# must not import a modelling library to format a string.


def _is_numeric(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s):
        return True
    conv = pd.to_numeric(s, errors="coerce")
    present = s.notna() & (s.astype(str).str.strip() != "")
    if present.sum() == 0:
        return False
    return bool(conv[present].notna().mean() >= 0.99)


def _is_identifier(s: pd.Series, n_rows: int) -> bool:
    """A key, not a measurement.

    Present on essentially every row and near-unique. Given one, a
    model memorises the row instead of explaining it, and every
    importance it earns is leakage. The old engine learned this the
    hard way twice: the first guard counted occurrences instead of
    distinct holders, and the second returned early on numerics - so
    it passed a string visit_id in a fixture and did nothing at all
    against the real extract, where visit_id is an integer."""
    present = s.dropna()
    if len(present) < 0.99 * max(n_rows, 1):
        return False
    if not pd.api.types.is_numeric_dtype(present):
        return bool(present.nunique() >= 0.999 * len(present))
    v = present.to_numpy(dtype=float)
    return bool(np.all(np.mod(v, 1.0) == 0.0)
                and len(np.unique(v)) >= 0.999 * len(v))


def prepare(df: pd.DataFrame,
            group_by: Optional[str] = None,
            drop_identifiers: bool = True,
            ordinals: Optional[Dict[str, Any]] = None,
            k: int = 10,
            expand_by: Optional[str] = None):
    """Typed frame: numerics as float with NaN, DATES as days since
    DATE_ORIGIN, everything else as a capped category. Blanks become
    NaN rather than a level, so `missing` is one concept and not three
    spellings of it.

    Returns (frame, identifiers, dates, quantities, sets). FIVE, not
    four: a caller that has not been updated raises immediately, which
    is the same reason there were three and then four. A set column
    silently left unexpanded is a relationship quietly lost, and this
    file already records what that costs.

    `dates` maps each column that
    was read as a date to the format it was read under, which is what
    lets generation write it back as a date instead of as an ordinal
    float. THREE return values rather than two so a caller that has
    not been updated raises immediately: a date silently left as a
    number in the output file is exactly the kind of silent no-op this
    frame is supposed to make impossible.

    Built with a single concat. Assigning column by column leaves the
    frame fragmented and every later `.loc` pays for it."""
    cols, ident, dates, quantities, sets = {}, [], {}, {}, {}
    pending_sets = []
    expand_by = (_sets.EXPAND_BY if expand_by is None
                 else str(expand_by).lower())
    if expand_by not in ("frequency", "signal"):
        raise ValueError(
            "expand_by must be 'frequency' or 'signal', not "
            "{!r} - an unrecognised value would silently fall back "
            "to whichever branch happened to be first".format(
                expand_by))

    # A REAL COLUMN THAT LOOKS LIKE SCAFFOLDING IS A SILENT DROP.
    # `source_of` folds `foo__has__bar` into `foo`, and the blueprint
    # skips anything whose source is not itself - so a column
    # genuinely named that way would vanish from the output with every
    # check green. Reported, never risked.
    bad = _sets.collisions(df.columns)
    if bad:
        raise ValueError(
            "these column names collide with set-indicator "
            "scaffolding and would be silently dropped: {}. Rename "
            "them, or run with a copy that does.".format(
                ", ".join(sorted(bad))))

    gv = (df[group_by].astype(str).to_numpy()
          if group_by and group_by in df.columns
          else np.arange(len(df)))
    for c in df.columns:
        if c == group_by:
            continue
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            # already typed by the reader; no text to parse
            cols[c] = from_datetime(s)
            dates[c] = {"format": "%Y-%m-%d",
                        "parsed_format": "%Y-%m-%d",
                        "origin": DATE_ORIGIN,
                        "floored_to_day": True,
                        "unparsed_share": 0.0,
                        "ambiguous": False, "alternatives": []}
            if drop_identifiers and _is_identifier(cols[c], len(df)):
                ident.append(c)
                del cols[c]
                del dates[c]
            continue
        if s.dtype == object:
            s = s.astype(str).str.strip()
            s = s.mask(s.str.lower().isin(
                ["", "nan", "none", "null", "na", "n/a"]))
        # A DECLARED ORDER WINS OVER EVERY TEST BELOW. The operator
        # said what the order is; nothing here is entitled to a second
        # opinion, and no test on the strings could form one anyway.
        if ordinals and c in ordinals:
            q = ordinal_spec(ordinals[c], s)
            cols[c] = to_number(s, q)
            quantities[c] = q
        elif _is_numeric(s):
            cols[c] = pd.to_numeric(s, errors="coerce").astype(float)
        else:
            d = date_kind(s)
            q = quantity_kind(s) if d is None else None
            if d is not None:
                cols[c] = to_ordinal(s, d["parsed_format"])
                dates[c] = d
            elif q is not None:
                # A NUMBER WEARING PUNCTUATION IS STILL A NUMBER.
                # Currency, a percent sign and a clock all failed the
                # numeric test and fell to the category branch, where
                # a column with more distinct values than the cap
                # becomes the sentinel - 85% and 62% on a plainly
                # tabular file, both silent.
                cols[c] = to_number(s, q)
                quantities[c] = q
            else:
                s = s.astype("object")
                keep = s.value_counts().index[:MAX_LEVELS]
                s = s.where(s.isin(keep) | s.isna(), OTHER)
                cols[c] = s.astype("category")
                # A SET IS A BAD CATEGORY AND A GOOD SET OF
                # INDICATORS. The capped category above stays, because
                # it is what the blueprint builds the list marginal
                # from; the indicators are what anything LEARNS from.
                # Measured on a fixture shaped after the real extract:
                # the combination string carries r2 0.324 of a signal
                # its own token carries at 0.733.
                # THE FULL VOCABULARY IS TAKEN HERE AND THE CHOICE
                # OF WHICH TOKENS TO EXPAND IS DEFERRED, because that
                # choice is made by measuring each token against the
                # NUMERIC COLUMNS and those are still being built in
                # this loop. Expanding by frequency needed nothing
                # but the column itself, which is exactly why it was
                # done that way and exactly what was wrong with it.
                v = _sets.vocabulary(df[c], gv, k=k, cap=0)
                if v is not None:
                    sets[c] = v
                    pending_sets.append(c)
                    # PRESENT-AND-EMPTY IS A STATE, NOT A SPELLING OF
                    # MISSING - for a SET, and only for a set.
                    #
                    # The blanking above is correct for a category:
                    # "", "nan" and "null" are three ways of writing
                    # the same absence. A set is different. An empty
                    # list says the visit had no procedures, which is
                    # a fact about the visit, and folding it into NaN
                    # threw that fact away before anything could
                    # measure it - so generation gave every visit
                    # content. On the real extract that is 80.5% of
                    # `procedures` and 36.0% of `drug_routes`.
                    #
                    # It also left `coverage` disagreeing with the
                    # generated frame, which carries "" as a present
                    # string: source read 0.193 against a generated
                    # 1.0 on two columns. Two numbers describing one
                    # row disagreeing is the signal, again.
                    _blank = (df[c].notna()
                              & (df[c].astype(str).str.strip()
                                 == ""))
                    if bool(_blank.any()):
                        _cc = cols[c].astype("object")
                        _cc[_blank.to_numpy()] = _sets.EMPTY
                        cols[c] = _cc.astype("category")
        if drop_identifiers and _is_identifier(cols[c], len(df)):
            # A date UNIQUE ON EVERY ROW is still a key, and was
            # dropped as one before this branch existed. Typing it as
            # a number must not quietly promote it back into the
            # search, so the guard runs after the branch, not instead
            # of it.
            ident.append(c)
            del cols[c]
            dates.pop(c, None)
            quantities.pop(c, None)
    # NOW CHOOSE WHICH TOKENS BECOME SEARCH COLUMNS.
    #
    # `EXPAND_CAP` keeps the most common, and common is not
    # informative. On the real extract the cap leaves 1,026 of 1,050
    # `conditions` tokens unexaminable while the 24 kept carry only
    # 27.6% of the column's mass - and the budget cannot simply be
    # raised, because covering 80% of every set column needs 521
    # slots against the 96 available, taking the search from 138
    # columns to about 563.
    #
    # So the same budget is spent on the tokens most likely to carry
    # something. Measured on a fixture whose driver sits on 6% of
    # rows while thirty other tokens sit on 25%: by frequency it
    # ranks 31 of 31 and no cap keeps it; by signal it ranks 1.
    #
    # The tokens are all above the k floor either way, so this
    # changes what is SEARCHED and not what is published.
    if pending_sets:
        _num = dict((n_, v_) for n_, v_ in cols.items()
                    if pd.api.types.is_numeric_dtype(v_))
        _frame = (pd.concat(_num, axis=1) if _num
                  else pd.DataFrame(index=df.index))
        if _num:
            _frame.columns = list(_num)
        for c in pending_sets:
            v = sets[c]
            toks = v.get("tokens") or []
            if len(toks) > _sets.EXPAND_CAP:
                # THE CAP ALWAYS APPLIES; only the ORDER depends on
                # the rule. Putting the truncation inside the signal
                # branch meant the default expanded EVERY token - the
                # cap silently stopped existing, which the check on
                # "tokens found vs expanded" caught immediately.
                if expand_by == "signal":
                    order = _sets.rank_by_signal(df[c], toks, _frame,
                                                 gv)
                    rank = dict((t, i) for i, t in enumerate(order))
                    toks = sorted(
                        toks,
                        key=lambda t: rank.get(t["value"], len(rank)))
                toks = toks[:_sets.EXPAND_CAP]
            v = dict(v)
            v["tokens"] = toks
            v["tokens_returned"] = len(toks)
            v["expansion_chosen_by"] = expand_by
            sets[c] = v
            cols.update(_sets.expand(df[c], v))

        # PUT THE INDICATORS BACK WHERE THEY USED TO SIT.
        #
        # Deferring the expansion so the screen could see the numeric
        # columns also moved every `__has__` column to the END of the
        # frame - on the tidy fixture the first one went from position
        # 38 to 76. That fixture's claims came out identical either
        # way, so it looked harmless, and on the real extract it was
        # not: the same command on the same data trimmed 13
        # relationships over 18 columns where the previous build
        # trimmed 14 over 19, and `close` fell from 87/114 to 72/115.
        #
        # Discovery screens to the top predictors and breaks ties on
        # the order it meets them, so column position is not
        # cosmetic. Rebuilding the mapping in the ORIGINAL order costs
        # nothing and removes the difference rather than reasoning
        # about which orders are safe.
        _ordered = {}
        for _c in df.columns:
            if _c in cols:
                _ordered[_c] = cols[_c]
            for _d in _sets.expand_names(_c, sets.get(_c)):
                if _d in cols:
                    _ordered[_d] = cols[_d]
        for _c, _v in cols.items():
            if _c not in _ordered:
                _ordered[_c] = _v
        cols = _ordered

    if drop_identifiers and ident:
        # Dropping a key is not enough - anything ENGINEERED from it
        # carries the same information back in. `visit_id__prev` is a
        # near-unique integer too, but it is blank on each patient's
        # first visit, so its coverage falls below the guard's
        # threshold and it survives. Measured: visit_id was correctly
        # dropped and then reappeared as a predictor of age_at_visit,
        # because importance is reported per SOURCE column and the
        # source of visit_id__prev is visit_id.
        keys = set(ident)
        for c in [c for c in cols if _source(c) in keys]:
            ident.append(c)
            del cols[c]
            dates.pop(c, None)
            quantities.pop(c, None)
    out = pd.concat(cols, axis=1) if cols else pd.DataFrame(
        index=df.index)
    out.columns = list(cols)
    # A FOURTH RETURN VALUE, not a frame attribute. `DataFrame.attrs`
    # is dropped by most operations without saying so, and a renderer
    # that silently goes missing is how a column gets written out as a
    # bare number where the source had money - the same silent-no-op
    # class this file is built to avoid. An un-updated caller raises
    # here instead.
    return out, ident, dates, quantities, sets


def _source(col: str) -> str:
    """The real column an engineered feature stands for.

    `x__prev` and `x__delta` are both scaffolding built from `x`. The
    catalogue speaks in source names because that is what the reader
    has: `x moved this`, not `x__prev moved this`."""
    for suf in LAG_SUFFIXES:
        if col.endswith(suf):
            return col[:-len(suf)]
    # A SET INDICATOR IS NOT FOLDED, deliberately. `x__prev` stands
    # for `x` and has nothing of its own to say, but
    # `conditions__has__t03` names WHICH token moved the child - which
    # is the entire finding, and folding it to `conditions` would
    # throw away the answer while reporting the question. It is a
    # column in its own right: it gets a marginal, an effect curve
    # over two levels rather than a 201-level grid, and generation
    # derives it from the set it already drew.
    return col


# How many bins the residual profile is cut into. Few, because each
# one has to clear k PATIENTS before it can be published, and a
# profile that is mostly suppressed is worse than none.
SPREAD_BINS = 5


def _residual_spread(y_true, y_pred, groups, k: int = 10):
    """How the leftover spread changes with the predicted value.

    WHY. Generation shrinks the drawn value by `sqrt(1 - skill)`,
    which is ONE number for the whole column. Real clinical columns
    are heteroscedastic - lab variance grows with level, stay length
    varies far more for sick patients. Measured on a fixture whose
    noise sd grows 4x across the range, over six seeds: the source
    spread grows 3.7x and the generated grows 2.3x, with the top
    slice understated by 25% every time. Marginal spread reads 1.03
    throughout, so nothing catches it.

    Published as a MULTIPLIER on the overall residual sd rather than
    as an sd, so it travels with the column's own scale and a dial
    that widens the column does not have to be applied twice.

    OUT OF SAMPLE, on held-out patients: in-sample residuals are
    shrunk by the fit itself, and a spread model built from them would
    be confidently narrow.

    K-SCREENED BY PATIENTS, like every other published number. A bin
    carried by fewer than k patients publishes no multiplier, and its
    entry is 1.0 - the honest fallback, which is what the column does
    today."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(ok.sum()) < 100:
        return None
    y_true, y_pred = y_true[ok], y_pred[ok]
    g = np.asarray(groups)[ok] if groups is not None else None

    resid = y_true - y_pred
    overall = float(np.std(resid))
    if not np.isfinite(overall) or overall <= 0:
        return None

    edges = np.quantile(y_pred, np.linspace(0, 1, SPREAD_BINS + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return None
    idx = np.clip(np.searchsorted(edges, y_pred, side="right") - 1,
                  0, len(edges) - 2)

    centres, mults, published = [], [], 0
    for b in range(len(edges) - 1):
        m = idx == b
        if int(m.sum()) < 20:
            centres.append(float(np.mean(edges[b:b + 2])))
            mults.append(1.0)
            continue
        if g is not None and len(np.unique(g[m])) < k:
            centres.append(float(np.mean(edges[b:b + 2])))
            mults.append(1.0)
            continue
        sd = float(np.std(resid[m]))
        centres.append(float(np.mean(y_pred[m])))
        mults.append(round(min(max(sd / overall, 0.1), 4.0), 4))
        published += 1

    if published < 2:
        return None
    # FLAT MEANS NOTHING TO CARRY. Publishing a profile that says
    # "multiply by about one everywhere" adds a moving part and buys
    # nothing, and the fallback already behaves that way.
    if max(mults) / max(min(mults), 1e-9) < 1.25:
        return None
    return {"at": [round(c, 6) for c in centres],
            "multiplier": mults,
            "bins_published": published,
            "bins": len(mults),
            "bins_are_k_anonymous": k,
            "note": "residual sd relative to this column's overall "
                    "residual sd, by predicted value, measured on "
                    "held-out patients"}


def _set_family(col: str, columns) -> List[str]:
    """Everything derived from the same set column as `col`.

    ONE FAMILY, EXCLUDED FROM ITSELF, exactly as a column and its lags
    are. `conditions__n` predicts `conditions__has__t03` on any data
    at all: draw more tokens and any given token is likelier to be
    among them. That is arithmetic, and the first run with indicators
    turned it into twenty-two claims - every token "explained" by its
    own set size at skills of 0.02 to 0.47, crowding out the one
    finding that mattered.

    Siblings go too. Two tokens co-occurring IS a real thing to know,
    but `sets.py` says plainly that co-occurrence is not modelled and
    generation draws tokens independently - so discovering it would
    put a relationship in the report that nothing downstream can
    honour, and a finding that cannot be acted on reads exactly like
    one that can."""
    base = _sets.source_of(col)
    if base is None:
        return []
    return [c for c in columns
            if c != col and (_sets.source_of(c) == base or c == base)]


def _self_lags(col: str, columns) -> List[str]:
    """A column's own lag and delta - excluded, see module docstring."""
    return [c for c in columns
            if any(c == col + suf for suf in LAG_SUFFIXES)]


def _skill(kind, y_true, y_pred, base_pred) -> float:
    """How much better than knowing nothing, on held-out patients.

    Scaled so 0 means 'no better than the column's own marginal' and 1
    means perfect, for both kinds - a regression R-squared and a
    classification error reduction are otherwise not comparable, and
    the catalogue ranks them side by side."""
    if kind == "regression":
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - base_pred) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    err = float(np.mean(y_true != y_pred))
    base = float(np.mean(y_true != base_pred))
    return 1.0 - err / base if base > 0 else 0.0


def discover(df: pd.DataFrame,
             group_by: Optional[str] = None,
             targets: Optional[List[str]] = None,
             holdout_frac: float = 0.3,
             seed: int = 20260731,
             min_skill: float = 0.02,
             screen_top: int = 15,
             n_repeats: int = 4,
             max_iter: int = 60,
             max_train_rows: int = 12000,
             min_coverage: float = 0.05,
             max_classes: int = 50,
             min_importance: float = 0.002,
             deterministic_at: float = 0.97,
             with_shapes: bool = True,
             shape_top: int = 3,
             interaction_ratio: float = 1.5,
             progress=None,
             ordinals: Optional[Dict[str, Any]] = None,
             expand_by: Optional[str] = None
             ) -> Dict[str, Any]:
    """A catalogue of what explains each column, confirmed out of
    sample. Returns claims, not edges - no direction is implied beyond
    'these predict that'."""
    rng = np.random.RandomState(seed)
    X_all, identifiers, dates, _q, _sets_found = prepare(
        df, group_by, ordinals=ordinals, expand_by=expand_by)

    # Split BY PATIENT. Visits from one person are not independent, so
    # a row-wise holdout leaks the same patient into both halves and
    # confirms nearly anything - the single most expensive lesson the
    # old engine learned.
    if group_by is not None and group_by in df.columns:
        groups = df[group_by].astype(str).to_numpy()
        uniq = np.unique(groups)
        rng.shuffle(uniq)
        n_hold = max(1, int(len(uniq) * holdout_frac))
        held = set(uniq[:n_hold].tolist())
        is_hold = np.array([g in held for g in groups])
        n_tr_g, n_te_g = len(uniq) - n_hold, n_hold
    else:
        is_hold = rng.rand(len(df)) < holdout_frac
        n_tr_g, n_te_g = int((~is_hold).sum()), int(is_hold.sum())

    cols = list(X_all.columns)

    # THE RAW COMBINATION STRING IS DROPPED FROM THE SEARCH, both as a
    # feature and as a target, and its indicators stand in for it.
    #
    # As a FEATURE it determines every one of its own indicators
    # exactly, so permuting `conditions__has__t03` leaves the model
    # free to read t03 straight back out of `conditions` and the
    # importance comes back near zero. That is the lag leak again -
    # `planted_simpson_x` correlating 0.924 with its own `__prev`,
    # recall 0% while skill read 0.994 - and the answer there was to
    # stop the two carrying each other.
    #
    # As a TARGET it is a 201-level category that is 49.5% sentinel on
    # a fixture and 69% on the real extract; "what explains this
    # combination string" is not a question anyone asked. The
    # indicators are the targets now, which is also how a token comes
    # to be explained by its neighbours.
    #
    # It stays in the frame: the blueprint builds the list marginal
    # from it, and generation still emits the column itself.
    set_sources = set(_sets_found)

    # An engineered feature is scaffolding, never a target. Modelling
    # one asks "what explains last visit's age" and answers "this
    # visit's age" - arithmetic dressed as a finding, and it topped
    # the catalogue by skill on the first run that worked.
    want = [c for c in (targets or cols)
            if c in cols and (targets is not None or _source(c) == c)
            and c not in set_sources]

    # AN IDENTITY PARTNER IS A TWIN, AND A TWIN BLINDS THE SEARCH.
    # `procedure_count == procedure_quantity` holds on every source
    # row, so quantity explains count at skill ~1.0 and NO external
    # column can ever earn conditional importance in count's claims -
    # permuting visit_type loses nothing while the twin still carries
    # the signal, which is the lag-twin failure this module already
    # documents, produced by the data instead of by engineering. The
    # identity cluster becomes an island: perfectly connected inside,
    # connected to the rest of the table by nothing, and every
    # mediated association through it dies in generation
    # (span_days 0.27 -> -0.003, visit_type -> sizes 0.58 -> 0.01).
    #
    # So exact partners are excluded from each other's FEATURES, the
    # way a set and its own indicators already are - arithmetic, not
    # a discovery. The identity itself is not lost: the `==`
    # constraint records it, and generation copies one column from
    # the other outright. Exact means exact - a 96% near-identity
    # still carries real conditional information and stays.
    ident_partners: Dict[str, set] = {}
    _num_src = [c for c in cols
                if _source(c) == c
                and pd.api.types.is_numeric_dtype(X_all[c])]
    for _i in range(len(_num_src)):
        a = X_all[_num_src[_i]]
        for _j in range(_i + 1, len(_num_src)):
            b = X_all[_num_src[_j]]
            both = a.notna() & b.notna()
            if int(both.sum()) < 30:
                continue
            if float((a[both] == b[both]).mean()) >= 0.995:
                ident_partners.setdefault(
                    _num_src[_i], set()).add(_num_src[_j])
                ident_partners.setdefault(
                    _num_src[_j], set()).add(_num_src[_i])

    claims, unexplained, skipped = [], [], []

    for idx, target in enumerate(want):
        if progress:
            progress(idx, len(want), target)
        y_full = X_all[target]
        observed = y_full.notna().to_numpy()
        if observed.mean() < min_coverage:
            skipped.append({"column": target,
                            "why": "coverage {:.1%} is below the "
                                   "floor".format(observed.mean())})
            continue
        kind = ("regression" if pd.api.types.is_numeric_dtype(y_full)
                else "classification")
        drop = ([target] + _self_lags(target, cols)
                + _set_family(target, cols) + sorted(set_sources))
        # ...and the identity partner's whole FAMILY: the partner's
        # own lag features are twins of the twin.
        for _p in ident_partners.get(_source(target), ()):
            drop += [c for c in cols if _source(c) == _p]
        feats = [c for c in cols if c not in drop]
        if not feats:
            continue

        tr = observed & ~is_hold
        te = observed & is_hold
        if tr.sum() < 50 or te.sum() < 25:
            skipped.append({"column": target,
                            "why": "too few observed rows to train "
                                   "and hold out"})
            continue
        Xtr, Xte = X_all.loc[tr, feats], X_all.loc[te, feats]
        ytr, yte = y_full[tr], y_full[te]
        # A cap on TRAINING rows only. Measured on this fixture: fit
        # cost is roughly linear in rows and is half the per-column
        # bill, while ranking which columns predict a target does not
        # need every row - the holdout, which is what gets reported,
        # is never touched. Rows are sampled at random rather than
        # taken in order, because the file is sorted by patient and a
        # head would be a few hundred people.
        if len(Xtr) > max_train_rows:
            pick = np.random.RandomState(seed).choice(
                len(Xtr), max_train_rows, replace=False)
            Xtr, ytr = Xtr.iloc[pick], ytr.iloc[pick]
        # early_stopping OFF, deliberately. On by default above
        # 10,000 rows, it carves its own validation split out of the
        # training half - stratified for a classifier, which CRASHES
        # the moment a class has one member, and a wide clinical
        # extract has many. It also makes the fit non-deterministic
        # with respect to the holdout the catalogue reports against.
        if kind == "regression":
            if float(ytr.std() or 0.0) == 0.0:
                unexplained.append(target)
                continue
            model = HistGradientBoostingRegressor(
                max_iter=max_iter, random_state=seed,
                early_stopping=False,
                categorical_features="from_dtype")
            base_pred = np.full(int(te.sum()), float(ytr.mean()))
        else:
            n_cls = int(ytr.nunique(dropna=True))
            if n_cls < 2:
                unexplained.append(target)
                continue
            if n_cls > max_classes:
                # A near-free-text column. Predicting one of hundreds
                # of labels is a different problem from finding a
                # pattern, and it costs a model per class to answer
                # badly.
                skipped.append({
                    "column": target,
                    "why": "{} distinct levels - too many to model "
                           "as a class".format(n_cls)})
                continue
            model = HistGradientBoostingClassifier(
                max_iter=max_iter, random_state=seed,
                early_stopping=False,
                categorical_features="from_dtype")
            base_pred = np.full(int(te.sum()),
                                ytr.mode(dropna=True).iloc[0])
        ytr = ytr.astype(float) if kind == "regression" \
            else ytr.astype(str)
        yte_v = (yte.astype(float).to_numpy() if kind == "regression"
                 else yte.astype(str).to_numpy())
        model.fit(Xtr, ytr)
        skill = _skill(kind, yte_v, model.predict(Xte), base_pred)
        if skill < min_skill:
            # Nothing available explains it. This IS the unexplained
            # set, measured on held-out patients rather than inferred
            # from an absence of edges.
            unexplained.append(target)
            continue

        # Permutation importance is the honest ranking - it asks what
        # the model LOSES without a column, on data it never saw -
        # but it costs a pass per feature per repeat. Screening on the
        # model's own split gains first keeps that affordable; the
        # screen only chooses who gets measured, never what the
        # measurement says.
        # IMPORTANCE IS MEASURED PER SOURCE COLUMN, NOT PER FEATURE.
        #
        # Permuting one feature at a time asks "what is lost without
        # this column, given every other column is still there" - and
        # a column with a near-copy in the frame loses nothing,
        # because its twin still carries the signal. Both then score
        # zero and the relationship is invisible.
        #
        # This is not a hypothetical: adding lag features CREATES the
        # twins. Measured on this fixture, planted_simpson_x
        # correlates 0.924 with its own planted_simpson_x__prev. The
        # first run of this module scored recall 0% while reporting
        # skill 0.994 - the model had found the structure and the
        # attribution could not see it.
        #
        # So a column and its lag and its delta are permuted TOGETHER,
        # with the SAME row order, which destroys their joint link to
        # the target while preserving their relationship to each
        # other. The answer comes back in the user's own column names,
        # which is what a person reading the catalogue needs anyway -
        # `x` moved this, not `x__prev`.
        groups: Dict[str, List[str]] = {}
        for c in feats:
            groups.setdefault(_source(c), []).append(c)
        Xp = Xte if len(Xte) <= 4000 else Xte.sample(4000,
                                                     random_state=seed)
        yp = (yte_v if len(Xte) <= 4000
              else yte_v[Xte.index.get_indexer(Xp.index)])
        base_s = _skill(kind, yp, model.predict(Xp),
                        np.full(len(yp), base_pred[0]))
        # ONE copy, reused. Copying the frame per repeat cost more
        # than every model fit put together.
        Xs = Xp.copy()
        flat = np.full(len(yp), base_pred[0])

        # SCREEN BY THE SAME MEASURE, CHEAPLY - one permutation on a
        # small slice - then measure the survivors properly.
        #
        # There is no native importance to screen on: a
        # HistGradientBoosting estimator has no `feature_importances_`
        # at all. Asking for one with getattr(..., None) silently
        # returned None on every column, the fallback took the first
        # fifteen groups in COLUMN ORDER, and the screen stopped being
        # a screen. It reported visit_start_date, year_of_birth and
        # gender as the drivers of every planted relationship - those
        # are simply the leading columns of the schema - and recall
        # was 0% while the model itself scored 0.93 on the same
        # target. A silent no-op is worse than a crash; this version
        # cannot no-op, because the screen and the measurement are the
        # same computation at different budgets.
        ns = min(1500, len(Xs))
        Xq, yq = Xs.iloc[:ns], yp[:ns]
        flat_q = np.full(ns, base_pred[0])
        base_q = _skill(kind, yq, model.predict(Xq), flat_q)
        rough = []
        order_q = np.random.RandomState(seed).permutation(ns)
        for src, members in groups.items():
            keep = dict((c, Xq[c].to_numpy(copy=True)) for c in members)
            for c in members:
                Xq[c] = keep[c][order_q]
            rough.append((base_q - _skill(kind, yq,
                                          model.predict(Xq), flat_q),
                          src))
            for c in members:
                Xq[c] = keep[c]
        rough.sort(key=lambda t: -t[0])
        pool = [s for _d, s in rough[:max(screen_top, 1)]]

        preds = []
        for src in pool:
            members = groups[src]
            keep = dict((c, Xs[c].to_numpy(copy=True)) for c in members)
            drops = []
            for rep in range(n_repeats):
                order_r = np.random.RandomState(
                    seed + rep).permutation(len(Xs))
                for c in members:
                    Xs[c] = keep[c][order_r]
                drops.append(base_s - _skill(
                    kind, yp, model.predict(Xs), flat))
            for c in members:
                Xs[c] = keep[c]
            m, s = float(np.mean(drops)), float(np.std(drops))
            # Two standard deviations clear of zero AND clear of an
            # absolute floor. The sd rule alone admits pure noise the
            # moment every repeat agrees: the first run reported an
            # importance of 1e-05 with sd exactly 0, which passed.
            if m >= min_importance and m > 2.0 * s:
                preds.append({"column": src, "importance": round(m, 5),
                              "sd": round(s, 5),
                              "via": [c for c in members
                                      if c != src] or None})
        preds.sort(key=lambda d: -d["importance"])
        if not preds:
            unexplained.append(target)
            continue

        # THE SHAPE, extracted here because this is the only place the
        # fitted model exists. Importance says a parent matters; the
        # curve says what it does, which is the half a reader can act
        # on and the half a sampler needs.
        interaction = None
        if with_shapes:
            spread = (float(np.std(yp)) if kind == "regression"
                      else 1.0)
            for p in preds[:shape_top]:
                cur = effect_curve(model, Xp, groups[p["column"]],
                                   X_all[p["column"]], kind)
                if cur is None:
                    continue
                p["effect"] = dict(cur)
                p["effect"]["centre"] = round(
                    curve_centre(cur, X_all[p["column"]]), 6)
                p["effect"].update(describe(
                    cur, p["column"], target, spread,
                    fmt_parent=labeller(dates.get(p["column"])),
                    fmt_child=labeller(dates.get(target))))

                # AND THE SAME PARENT ON ITS OWN.
                #
                # A partial-dependence curve is conditional on every
                # other column the model can see, and with correlated
                # parents that is not the relationship a reader means
                # - nor one a sampler can use alone. Measured:
                # mean arterial pressure is (S + 2D)/3, so holding it
                # fixed, diastolic FALLS as systolic rises. The
                # conditional curve is genuinely negative, and using
                # it to generate diastolic from systolic by itself
                # produced a correlation of -0.720 where the source
                # had +0.888.
                #
                # So each parent also gets a curve from a model that
                # sees ONLY that parent. It is what generation uses
                # whenever the other parents are not there, and the
                # two disagreeing in SIGN is itself a finding.
                mcols = [c for c in groups[p["column"]]
                         if c in Xtr.columns]
                if mcols:
                    try:
                        mm = (HistGradientBoostingRegressor
                              if kind == "regression"
                              else HistGradientBoostingClassifier)(
                            max_iter=40, random_state=seed,
                            early_stopping=False,
                            categorical_features="from_dtype")
                        mm.fit(Xtr[mcols], ytr)
                        cm = effect_curve(mm, Xp[mcols], mcols,
                                          X_all[p["column"]], kind)
                    except Exception:
                        cm = None
                    if cm:
                        # AND HOW MUCH THAT PARENT EXPLAINS ALONE.
                        # The shrink term is sqrt(1 - skill), so
                        # pairing a single-parent curve with the whole
                        # claim's skill removes far too much noise and
                        # the output comes out OVER-correlated: 0.978
                        # generated against 0.888 in the source.
                        try:
                            cm["skill"] = round(float(_skill(
                                kind, yte_v, mm.predict(Xte[mcols]),
                                base_pred)), 4)
                        except Exception:
                            cm["skill"] = None
                        cm["centre"] = round(curve_centre(
                            cm, X_all[p["column"]]), 6)
                        d0 = describe(
                            cm, p["column"], target, spread,
                            fmt_parent=labeller(
                                dates.get(p["column"])),
                            fmt_child=labeller(dates.get(target)))
                        cm["shape"] = d0["shape"]
                        cm["description"] = d0["description"]
                        p["effect"]["alone"] = cm
                        a_r = np.asarray(cm["response"], dtype=float)
                        c_r = np.asarray(cur["response"], dtype=float)
                        if (a_r[-1] - a_r[0]) * (c_r[-1] - c_r[0]) < 0:
                            p["effect"]["reverses_when_controlled"] = (
                                "on its own {} moves the other way; "
                                "holding the other parents fixed "
                                "reverses it".format(p["column"]))

            # DOES THE COMBINATION MOVE THE CHILD FURTHER THAN EITHER
            # PARENT ALONE? That is a measurement, and it is the only
            # trustworthy trigger.
            #
            # The first version asked whether both one-parent curves
            # came back "flat", reasoning that a pure interaction has
            # no main effect. True in principle and fragile in fact:
            # on a clean exclusive-or the curves picked up small
            # asymmetries and were named inverted-u and u-shaped, so
            # the interaction - the pattern this system exists to
            # catch - was never reported at all.
            #
            # Comparing surface travel against the best single curve
            # does not depend on a shape name surviving noise. A
            # genuine interaction moves the child much further when
            # both parents move than when either does.
            # EVERY PAIR AMONG THE TOP THREE, not just the top two.
            #
            # `body_mass_index_measured <- height,
            # peripheral_pulse_rate, temperature_oral` could never
            # have its pulse-by-temperature pair examined, because
            # only the leading two parents were ever put on a surface
            # and height took one of those slots permanently. Screened
            # coarsely, then the best pair measured properly - the
            # same two-budget shape the importance screen uses.
            cands = preds[:max(shape_top, 2)]
            best_pair, best_dep = None, 0.0
            for ia in range(len(cands)):
                for ib in range(ia + 1, len(cands)):
                    pa, pb = cands[ia], cands[ib]
                    rough = joint_surface(
                        model, Xp, groups[pa["column"]],
                        groups[pb["column"]], X_all[pa["column"]],
                        X_all[pb["column"]], kind, n=5)
                    if rough is None:
                        continue
                    d = additive_departure(rough, pa.get("effect"),
                                           pb.get("effect"))
                    if d is not None and d > best_dep:
                        best_dep, best_pair = d, (pa, pb)
            top = list(best_pair) if best_pair else []
            if len(top) == 2:
                surf = joint_surface(
                    model, Xp, groups[top[0]["column"]],
                    groups[top[1]["column"]], X_all[top[0]["column"]],
                    X_all[top[1]["column"]], kind)
                if surf is not None:
                    joint = dict(surf)
                    joint.update(describe_joint(
                        surf, top[0]["column"], top[1]["column"],
                        target))
                    alone = max(
                        (p.get("effect") or {}).get("effect_size") or 0.0
                        for p in top)
                    joint["single_parent_best"] = round(alone, 6)
                    joint["beyond_single"] = (
                        round(joint["effect_size"] / alone, 3)
                        if alone > 0 else None)
                    dep = additive_departure(
                        surf, top[0].get("effect"),
                        top[1].get("effect"))
                    joint["departure_from_additive"] = (
                        None if dep is None else round(dep, 6))
                    # An interaction is a departure from ADDITIVITY,
                    # not a longer journey. Judged against the larger
                    # of a quarter of the best single effect and the
                    # floor below which any curve is called flat, so
                    # it can fire beside a dominant parent and cannot
                    # fire on noise.
                    bar = max(0.25 * alone,
                              FLAT_SHARE * max(spread, 1e-9))
                    joint["pair"] = [top[0]["column"],
                                     top[1]["column"]]
                    joint["centre"] = round(surface_centre(
                        surf, X_all[top[0]["column"]],
                        X_all[top[1]["column"]]), 6)
                    if dep is not None and dep >= bar:
                        interaction = joint

            # A PARENT THAT EARNS IMPORTANCE AND EXPLAINS NOTHING is
            # said to be unresolved, rather than described in words
            # that sound like an explanation. "carried by something
            # other than this parent alone" reads as a finding and is
            # not one.
            named = set((interaction or {}).get("pair") or [])
            for p in preds[:shape_top]:
                e = p.get("effect") or {}
                if e.get("shape") == "flat" and \
                        p["column"] not in named:
                    e["unresolved"] = True
                    e["description"] = (
                        "{} shows no readable response to {} on its "
                        "own, and no pair this report tested explains "
                        "it either - treat {} as unresolved rather "
                        "than unimportant".format(
                            target, p["column"], p["column"]))
        # HOW THE LEFTOVER SPREAD MOVES WITH THE PREDICTION. Only
        # meaningful for a numeric child; a classifier's residual is
        # not a spread.
        spread_profile = None
        if kind == "regression":
            # NOT `groups` - that name is rebound to the
            # permutation-grouping dict earlier in this loop, so
            # indexing it with a boolean mask raises `unhashable
            # type` from a line that looks correct. Read from the
            # frame, which cannot be shadowed.
            _pids = (df[group_by].astype(str).to_numpy()[te]
                     if group_by is not None and group_by in df.columns
                     else None)
            spread_profile = _residual_spread(
                yte_v, model.predict(Xte), _pids)

        claims.append({
            "child": target,
            "kind": kind,
            "skill": round(float(skill), 4),
            "residual_spread": spread_profile,
            "predictors": preds,
            "n_train_rows": int(tr.sum()),
            "n_holdout_rows": int(te.sum()),
            "train_patients": n_tr_g,
            "holdout_patients": n_te_g,
            "interaction": interaction,
            # NEAR-DETERMINISTIC: arithmetic, not a discovery.
            #
            # On the real extract the top of the report was
            # `returned_within_30d <- days_to_next_visit` at 1.00 and
            # `age_at_visit <- year_of_birth` at 0.98 - one is a
            # threshold on the other, the second is a subtraction, and
            # both are columns the wrangler computed rather than
            # anything a clinic recorded. They buried the finding that
            # mattered: haematocrit from haemoglobin at 0.96, which is
            # physiology and which nobody put there on purpose.
            #
            # The boundary is a HEURISTIC and is stated as one. On
            # that extract arithmetic measured 0.98-1.00 and the
            # strongest genuine relationship measured 0.96, so 0.97
            # separates them there. It is not a law, which is why
            # nothing is hidden - only sorted.
            "near_deterministic": bool(skill >= deterministic_at),
        })

    return {
        "claims": claims,
        "unexplained": sorted(unexplained),
        "skipped": skipped,
        "identifiers_dropped": identifiers,
        "columns_considered": len(want),
        "holdout": {"patients": n_te_g, "train_patients": n_tr_g,
                    "by": "patient" if group_by else "row"},
        "note": "skill and importance are both measured on held-out "
                "patients, so a pattern that lived only in the "
                "patients the search looked at scores zero by "
                "construction. No multiple-comparison correction is "
                "applied or needed: nothing here is a p-value from "
                "the data that chose it.",
    }


def claims_as_edges(cat: Dict[str, Any],
                    max_predictors: int = 3) -> List[Dict[str, Any]]:
    """The catalogue in the shape the existing scorer reads.

    Capped at `max_predictors` ON PURPOSE. The scorer counts a planted
    relationship as recovered when its columns are a SUBSET of an
    edge's, so a claim listing twenty predictors would swallow planted
    pairs by breadth and read as recall. Three is what the old engine
    could report, which is what makes the two comparable."""
    out = []
    for cl in cat.get("claims", []):
        ps = [p["column"] for p in cl["predictors"][:max_predictors]]
        if ps:
            out.append({"child": cl["child"], "parents": ps})
    return out
