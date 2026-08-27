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
    txt = raw.dropna().astype(str).str.strip()
    txt = txt[txt.str.len() > 0]
    if len(txt) < MIN_ROWS:
        return None
    sep = _separator(txt)
    if sep is None:
        return None
    parts = txt.str.split(sep)
    sizes = parts.map(len)
    if float(sizes.mean()) < MIN_MEAN_SIZE:
        return None

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
    pub_sizes = parts.map(
        lambda ts: sum(1 for x in ts if x.strip() in keep_set))
    size_counts = pub_sizes.value_counts(normalize=True).sort_index()
    return {
        "separator": sep,
        "tokens": [{"value": tok, "p": round(c / n_rows, 6)}
                   for tok, c in kept],
        "set_size": {"v": [int(x) for x in size_counts.index],
                     "p": [round(float(x), 6) for x in size_counts]},
        "mean_set_size": round(float(pub_sizes.mean()), 6),
        "mean_set_size_source": round(float(sizes.mean()), 6),
        "empty_after_k_share": round(
            float((pub_sizes == 0).mean()), 6),
        "distinct_combinations": int(txt.nunique()),
        "tokens_found": int(n_found),
        "tokens_above_k": int(n_kept),
        "tokens_returned": len(kept),
        "tokens_are_k_anonymous": k,
    }


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
