"""A column holding a SET, and the indicators that let anything learn
from it.

WHY THIS EXISTS. `_list_marginal` already models a set column AS a set
- which tokens appear, how often, and how many a row carries - so
generation stopped emitting 69% `__other__`. What it did not fix is
that DISCOVERY still sees the raw text.

Measured on a fixture shaped after the real extract (2,064 distinct
combinations, 49.5% of rows falling to the sentinel against 69% there):

    severity <- conditions, as a capped category   r2 +0.324
    severity <- the one token that drives it       r2 +0.733

Less than half the signal. A combination string is a bad feature for
the same reason it was a bad level: `t01;t03;t09` and `t03;t14` share
the thing that matters and share no level. That is why five of ten
categorical associations were lost on the real run, three of them on
these columns, and why `condition_count <- conditions` fell from 0.54
to 0.03.

WHAT AN INDICATOR IS HERE. One binary column per published token, plus
the set size. `conditions` becomes `conditions__has__t03`,
`conditions__has__t07`, ... and `conditions__n`. They are ordinary
columns to everything downstream: a model can split on them, an effect
curve over two levels is exact rather than a 201-level grid, and
generation can apply that curve because it knows the value.

THE VOCABULARY IS DEFINED ONCE, HERE, AND BOTH HALVES CALL IT. The
whole defect being fixed is discovery and generation disagreeing about
what a set column contains. Screening tokens twice, with the code
written out twice, is how that disagreement comes back.

WHAT THIS STILL DOES NOT DO, said here rather than found later:

  co-occurrence   tokens are drawn independently given the set size,
                  so two drugs always prescribed together appear
                  together only by chance. Indicators are DERIVED from
                  the drawn set, not drawn themselves, so nothing here
                  changes that.
  informative     a token cannot yet depend on another column. The
  selection       set is drawn from its marginal and only then
                  observed. Downstream columns can depend on the
                  token, which is the direction the measurement above
                  says is being lost; the reverse is not built.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

LIST_SEPARATORS = (";", "|", ",")

# The marker that makes an indicator recognisable as scaffolding.
# Split on the LEFTMOST occurrence, so a token that itself contains
# the marker still resolves back to the right column.
HAS = "__has__"
SIZE = "__n"
# THE LEVEL A PRESENT-BUT-EMPTY SET ENCODES TO.
#
# The typed frame folds "", "nan", "none" and "null" together into
# NaN, which is right for a category - they are three spellings of
# missing - and wrong for a SET, where an empty list is a fact about
# the visit. On the real extract 80.5% of `procedures` rows and 36.0%
# of `drug_routes` rows are empty, so generation was inventing
# procedures for four fifths of the visits that had none.
#
# It needs a LEVEL rather than a bare "" because the frame's
# categorical path drops blanks by construction, and because a
# reader of the encoded frame should be able to see the difference.
# The raw output file still carries "" - this marker never reaches
# it.
EMPTY = "__empty__"

# A row must carry more than one token on average, and the separator
# must appear on a real share of rows, before this is a set at all.
SEP_SHARE = 0.30
MIN_MEAN_SIZE = 1.2
MIN_ROWS = 50

# HOW MANY TOKENS BECOME COLUMNS. Discovery fits a model per column,
# so expanding every published token of every set column multiplies
# the run - the 800-patient extract took 274s at 73 columns, and two
# set columns at 60 tokens each would have added 120 more. The cap is
# on the EXPANSION only: the marginal still publishes every token that
# clears k, so generation emits them all.
#
# Never silently. Anything past the cap is reported.
EXPAND_CAP = 24
# HOW THE EXPANDED TOKENS ARE CHOSEN: "frequency" or "signal".
#
# `frequency` keeps the most common, which is what this has always
# done. `signal` keeps the ones that MEASURE as related to a numeric
# column - see `rank_by_signal`.
#
# WHY THIS IS A CHOICE AND NOT A FIX. Signal selection demonstrably
# finds relationships frequency misses: on a fixture whose driver
# sits on 8% of rows against thirty tokens at 30%, frequency ranks it
# last of 31 and no cap keeps it, while signal ranks it first and
# discovery recovers it at skill 0.598. But the budget is fixed, so
# every slot spent on signal is taken from frequency, and on the one
# paired seed where the two could be compared with the cap actually
# binding, signal related 14 pairs against frequency's 15 - zero
# inversions and 100% sign kept BOTH ways.
#
# One paired seed cannot say which bet is better, and this file is
# emphatic that a single-seed measurement of a discovery change is
# worth nothing. It also cannot be settled here: the fixture's set
# columns are noise apart from one planted driver, whereas on the
# real extract the EXCLUDED tokens sit on 829 and 1,516 rows and
# carry 72.4% and 69.0% of their columns' mass. That is an argument
# for signal, not a measurement of it.
#
# So the default does not move, the alternative is one flag away, and
# the run says which rule it used.
EXPAND_BY = "frequency"
# WHY THE CAP IS NOT A BUDGETING PROBLEM. Reallocating these slots
# across columns was built and MEASURED against the extract's own
# token distribution, and rejected:
#
#   greedy on raw share      active_drugs 24->54, but procedures
#                            24->8 and its mass 67% -> 52%; summed
#                            per-column coverage 2.247 -> 2.192, WORSE
#   greedy on column share   sum 2.247 -> 2.266, but three of four
#                            columns lose ground and the worst goes
#                            29.4% -> 27.0%
#
# The budget is the constraint, not its distribution. Covering 80% of
# each column's mass needs 521 slots against the 96 available -
# `conditions` alone needs 308 - which would take the search from 138
# columns to about 563. Ninety-six slots cannot be arranged into five
# hundred.
#
# So the question is not WHICH COLUMN gets slots, it is WHICH TOKENS:
# with a budget five times too small, spend it on the tokens most
# likely to carry signal rather than the most common ones. Frequency
# is a poor proxy - the top `conditions` token is on 6,840 rows and
# may predict nothing.



def source_of(col: str) -> Optional[str]:
    """The set column an indicator belongs to, or None."""
    if HAS in col:
        return col.split(HAS, 1)[0]
    if col.endswith(SIZE):
        return col[:-len(SIZE)]
    return None


def indicator_name(col: str, token: str) -> str:
    return "{}{}{}".format(col, HAS, token)


def size_name(col: str) -> str:
    return "{}{}".format(col, SIZE)


def _separator(txt: pd.Series) -> Optional[str]:
    for cand in LIST_SEPARATORS:
        if float(txt.str.contains(cand, regex=False).mean()) >= SEP_SHARE:
            return cand
    return None


def vocabulary(raw: pd.Series, groups, k: int = 10,
               cap: int = 0) -> Optional[Dict[str, Any]]:
    """Which tokens this column holds, screened by PATIENTS.

    `cap` limits how many are returned, most common first; 0 means
    every token that clears k. A token held by fewer than k patients
    is not published, for the same reason a level is not - and it is
    patients, never rows, because one person seen two hundred times
    would otherwise carry a token over the bound alone.

    Returns None when the column is not a set, so a caller can fall
    through to the ordinary categorical path."""
    # PRESENT-AND-EMPTY IS A ROW THIS COLUMN HAS, and dropping it
    # here is why generation invents content for visits that had
    # none. A `drug_routes` list is empty when the visit had no
    # drugs; that is a fact about the visit, not an absence of
    # information, and `coverage` already counts such a row as
    # present because it tests `notna`.
    #
    # Measured on the real extract: excluding them made the published
    # `p` a share of NON-EMPTY rows while the fidelity comparison
    # measured a share of ALL present rows, and the five `drug_routes`
    # tokens outside tolerance were all high by the same 1.57x - a
    # constant multiplier across five tokens is a denominator, not a
    # sampler. 1/1.57 = 0.637 is the share of rows that had routes.
    #
    # WHETHER THE COLUMN IS A SET is still judged on the rows with
    # content - a separator cannot be detected in an empty string,
    # and a column should not stop being a set because many of its
    # rows are legitimately empty.
    present = raw.dropna().astype(str).str.strip()
    filled = present[present.str.len() > 0]
    if len(filled) < MIN_ROWS:
        return None
    sep = _separator(filled)
    if sep is None:
        return None
    if float(filled.str.split(sep).map(len).mean()) < MIN_MEAN_SIZE:
        return None
    txt = present
    parts = txt.str.split(sep)
    sizes = parts.map(
        lambda ts: sum(1 for x in ts if x.strip()))

    g = pd.Series(np.asarray(groups)[txt.index], index=txt.index)
    holders: Dict[str, set] = {}
    counts: Dict[str, int] = {}
    for idx, toks in parts.items():
        who = g.loc[idx]
        for tok in set(x.strip() for x in toks if x.strip()):
            holders.setdefault(tok, set()).add(who)
            counts[tok] = counts.get(tok, 0) + 1
    kept = [(tok, counts[tok]) for tok, who in holders.items()
            if len(who) >= k]
    if len(kept) < 2:
        return None
    kept.sort(key=lambda x: (-x[1], x[0]))
    n_found = len(holders)
    n_kept = len(kept)
    if cap and len(kept) > cap:
        kept = kept[:cap]

    n_rows = float(len(txt))
    # SIZE IS MEASURED OVER THE TOKENS THAT CAN BE PUBLISHED, not
    # over every token the source row held.
    #
    # A set drawn at the source's own size has to fill that size from
    # a vocabulary the k rule has thinned, so every surviving token
    # runs proportionally hot - on the real extract `conditions`
    # found 7,974 tokens, 1,050 cleared a k=10 PATIENT floor, and the
    # 6,924 rare ones carried 1.02 of the 4.27-per-row budget. Drawn
    # at 4.27 from the 1,050, each published share came out 1.32x its
    # true value. That is not the sampler failing; it is the size and
    # the vocabulary describing different populations.
    #
    # So the published size is the size of the PUBLISHABLE subset.
    # Generated sets are visibly smaller than source sets and that is
    # the k rule's cost, stated rather than hidden: `mean_set_size`
    # is what generation draws, `mean_set_size_source` is what the
    # rows actually held, and the caller reports the difference.
    keep_set = set(tok for tok, _ in kept)
    _held = parts.map(lambda ts: sum(1 for x in ts if x.strip()))
    _pub = parts.map(
        lambda ts: sum(1 for x in ts if x.strip() in keep_set))
    # AN EMPTY SET MEANS "HELD NOTHING", NEVER "HELD SOMETHING WE
    # CANNOT PUBLISH", and conflating those two asserts something
    # false about a patient.
    #
    # A visit with no drugs genuinely has an empty routes list - 36%
    # of them on the real extract - and generation should say so. A
    # visit whose conditions were ALL below the k floor is a
    # different thing entirely: that patient had conditions, and
    # emitting an empty list claims they had none. Publishing one
    # token instead is also not what they had, but it preserves the
    # fact that they had something, which is the property every
    # relationship involving the column depends on.
    #
    # Measured: emitting empties for BOTH cases put a sign inversion
    # into the pair sweep where the previous code had none, on a
    # fixture whose source is 0% empty and whose zero-size mass is
    # entirely sub-k. An inverted relationship reads as a finding,
    # which is the failure these checks exist to catch.
    pub_sizes = _pub.where(~((_held > 0) & (_pub == 0)), 1)
    pub_sizes = pub_sizes.where(_held > 0, 0)
    size_counts = pub_sizes.value_counts(normalize=True).sort_index()
    return {
        "separator": sep,
        "tokens": [{"value": tok, "p": round(c / n_rows, 6)}
                   for tok, c in kept],
        "set_size": {"v": [int(x) for x in size_counts.index],
                     "p": [round(float(x), 6) for x in size_counts]},
        "mean_set_size": round(float(pub_sizes.mean()), 6),
        "mean_set_size_source": round(float(sizes.mean()), 6),
        # REPORTED APART, because they mean different things to
        # whoever opens the file: one is a visit that had nothing,
        # the other is a visit whose content the k rule removed.
        "empty_in_source_share": round(float((_held == 0).mean()), 6),
        "unpublishable_row_share": round(
            float(((_held > 0) & (_pub == 0)).mean()), 6),
        "empty_after_k_share": round(
            float((pub_sizes == 0).mean()), 6),
        "distinct_combinations": int(txt.nunique()),
        "tokens_found": int(n_found),
        "tokens_above_k": int(n_kept),
        "tokens_returned": len(kept),
        "tokens_are_k_anonymous": k,
    }


def rank_by_signal(raw: pd.Series, tokens, frame, groups=None):
    """Order tokens by how much signal each one carries, best first.

    THE CAP KEEPS THE MOST COMMON TOKENS, AND COMMON IS NOT
    INFORMATIVE. Only an expanded token can carry a relationship, and
    the budget is about five times too small to expand what a real
    extract holds - 96 slots against the 521 needed to cover 80% of
    each column's mass. When you cannot afford them all, frequency is
    a poor way to choose: the top `conditions` token is on 6,840 rows
    and may predict nothing, while one on 829 rows may drive a lab
    value.

    The score is the largest absolute point-biserial correlation
    between the token's indicator and any numeric column in `frame`.
    That is the same shape of quantity discovery goes on to measure,
    which is the point - a screen computed a different way can
    silently rank on something the measurement does not use, and this
    file records what that cost when importances came back from a
    `getattr` that returned None.

    Cheap by construction: the indicators are built in ONE pass over
    the rows, so the work is rows x tokens-per-row rather than rows x
    vocabulary - 238,000 operations on the real extract rather than
    58 million.

    Ties, and tokens with no measurable association, fall back to
    frequency order, so this can only reorder what the old rule would
    have taken - never drop a token the cap would have kept for a
    reason nothing measured."""
    import numpy as _np

    vals = [t["value"] if isinstance(t, dict) else t for t in tokens]
    if not vals:
        return []
    idx = dict((v, i) for i, v in enumerate(vals))
    txt = raw.astype(str).str.strip()
    sep = _separator(txt[txt.str.len() > 0])
    if sep is None:
        return list(vals)

    n = len(txt)
    rows_i, cols_i = [], []
    for r, cell in enumerate(txt.to_numpy()):
        if not cell:
            continue
        for tok in cell.split(sep):
            j = idx.get(tok.strip())
            if j is not None:
                rows_i.append(r)
                cols_i.append(j)
    if not rows_i:
        return list(vals)
    M = _np.zeros((n, len(vals)), dtype=_np.float32)
    M[_np.asarray(rows_i), _np.asarray(cols_i)] = 1.0

    best = _np.zeros(len(vals), dtype=float)
    for c in frame.columns:
        col = frame[c]
        if not pd.api.types.is_numeric_dtype(col):
            continue
        y = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)
        ok = _np.isfinite(y)
        if int(ok.sum()) < MIN_ROWS:
            continue
        yv = y[ok]
        sd = float(yv.std())
        if not sd:
            continue
        Mv = M[ok]
        cnt = Mv.sum(axis=0)
        m = float(len(yv))
        # A token present on almost no rows, or on all of them,
        # cannot be measured against anything.
        live = (cnt >= 5) & (cnt <= m - 5)
        if not bool(live.any()):
            continue
        s1 = Mv.T.dot(yv)
        p_ = cnt / m
        mean1 = _np.where(cnt > 0, s1 / _np.maximum(cnt, 1), 0.0)
        mean0 = _np.where(cnt < m,
                          (yv.sum() - s1) / _np.maximum(m - cnt, 1),
                          0.0)
        r = ((mean1 - mean0) * _np.sqrt(_np.clip(p_ * (1 - p_), 0, 1))
             / sd)
        r = _np.abs(_np.where(live, r, 0.0))
        best = _np.maximum(best, _np.nan_to_num(r))

    order = sorted(range(len(vals)),
                   key=lambda i: (-best[i], i))
    return [vals[i] for i in order]


def is_scaffolding(name: str) -> bool:
    """True for a column this module BUILT rather than one the
    operator supplied.

    `conditions__has__E03.9` and `procedures__n` exist so a set can
    carry a relationship through the search, and they are dropped
    before the file is written. Anything reported about them is
    reported about a column the operator never receives - a real run
    said 843/933 orderings held and then listed page after page of
    `procedures__has__Oxygen Therapy <= procedure_count broken on
    52,197 rows`, burying the one ordering that was about their data.

    It lives here because BOTH halves need the same answer, and two
    files deciding separately what counts as scaffolding is how the
    two sides come to disagree."""
    n = str(name)
    return HAS in n or n.endswith(SIZE)


def has_token(raw: pd.Series, sep: str, token: str) -> pd.Series:
    """1.0 where the row's set contains the token, 0.0 where it does
    not, NaN where the column itself is missing.

    MISSING IS NOT ABSENT, and conflating them is the mistake this
    codebase keeps finding. A row with no value for `conditions` did
    not have zero conditions recorded; nothing is known about it, and
    coverage is the concept that describes that."""
    txt = raw.astype(str).str.strip()
    # PRESENT-AND-EMPTY IS NOT MISSING, and this disagreed with the
    # coverage measured beside it. `blueprint` calls a row present
    # when the cell is `notna`, so an empty set counts as COVERED;
    # these two read the same cell as unknown and returned NaN. Two
    # numbers describing one row disagreeing is the signal.
    #
    # It began to matter when set size started being drawn from the
    # PUBLISHABLE vocabulary: a row whose tokens all sit below the k
    # floor now legitimately draws an empty set, and that row holds
    # zero of every published token - a known zero. Reading it as
    # unknown would bury the k rule's cost inside the missingness
    # model, which is the last place anyone would look for it.
    present = raw.notna()
    hit = txt.str.split(sep).map(
        lambda ts: token in set(x.strip() for x in ts)
        if isinstance(ts, list) else False)
    return pd.Series(np.where(present, hit.astype(float), np.nan),
                     index=raw.index, dtype=float)


def sizes_of(raw: pd.Series, sep: str) -> pd.Series:
    """How many tokens each row carries; NaN where the value is
    missing."""
    txt = raw.astype(str).str.strip()
    # PRESENT-AND-EMPTY IS NOT MISSING, and this disagreed with the
    # coverage measured beside it. `blueprint` calls a row present
    # when the cell is `notna`, so an empty set counts as COVERED;
    # these two read the same cell as unknown and returned NaN. Two
    # numbers describing one row disagreeing is the signal.
    #
    # It began to matter when set size started being drawn from the
    # PUBLISHABLE vocabulary: a row whose tokens all sit below the k
    # floor now legitimately draws an empty set, and that row holds
    # zero of every published token - a known zero. Reading it as
    # unknown would bury the k rule's cost inside the missingness
    # model, which is the last place anyone would look for it.
    present = raw.notna()
    n = txt.str.split(sep).map(
        lambda ts: float(len([x for x in ts if x.strip()]))
        if isinstance(ts, list) else np.nan)
    return pd.Series(np.where(present, n, np.nan), index=raw.index,
                     dtype=float)


def expand(raw: pd.Series, spec: Dict[str, Any]) -> Dict[str, pd.Series]:
    """`{column_name: series}` for the size and every token in spec."""
    name = str(raw.name)
    sep = spec["separator"]
    out: Dict[str, pd.Series] = {size_name(name): sizes_of(raw, sep)}
    for t in spec["tokens"]:
        out[indicator_name(name, t["value"])] = has_token(
            raw, sep, t["value"])
    return out


def collisions(columns) -> List[str]:
    """Real column names that would be mistaken for scaffolding.

    A source column genuinely called `foo__has__bar` would be folded
    into `foo` by `source_of` and then skipped as scaffolding, which
    silently drops a real column - the failure mode this repository
    exists to refuse. Detected and reported rather than risked."""
    return [str(c) for c in columns if source_of(str(c)) is not None]
