"""Numbers wearing punctuation, and how to get them back.

WHY THIS EXISTS. `$1,234.56`, `45%` and `14:32` are all numbers, and
none of them survive `pd.to_numeric`. Each therefore fell to the
categorical branch, where a column with more distinct values than the
level cap becomes the `__other__` sentinel. Measured on a plainly
tabular file: a currency column came out 85% sentinel and a
time-of-day column 62%, both generated as `__other__`, both passing
every per-column check - the same fault dates had at 88%.

The lesson from dates was not "add a date parser". It was that a
column read as the wrong type fails silently and survives every check
that exists. So `sentinel_share` now reports the failure whatever
causes it, and this module removes three common causes.

Deliberately NOT here:

  booleans      TRUE/FALSE and Y/N make a two-level categorical, and
                a two-level categorical is modelled correctly. There
                is nothing to fix.
  ordinals      mild/moderate/severe has an order and north/south/
                east/west does not, and no inspection of the strings
                tells them apart. Guessing invents structure that was
                never in the data, which is worse than missing it -
                that has to be declared, not detected.

Every parser here follows the shape `dates.py` set: detect on a
seeded sample, convert to a number, record what it was so generation
can put the decoration back, and leave the column alone when the
evidence is thin.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# The share of present values that must parse before a column is
# called one of these. Matches the date threshold, for the same
# reason: a column that is 80% money and 20% free text is not a
# quantity column, it is a mess, and silently blanking a fifth of it
# would be the coverage loss dates already taught us to report.
QUANTITY_SHARE = 0.9

_CURRENCY = re.compile(
    r"^\s*(?P<neg1>-)?\s*(?P<sym>[$£€¥])\s*(?P<neg2>-)?"
    r"(?P<num>\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*$")
_PERCENT = re.compile(
    r"^\s*(?P<num>-?\d+(?:\.\d+)?)\s*%\s*$")
_CLOCK = re.compile(
    r"^\s*(?P<h>\d{1,2}):(?P<m>\d{2})(?::(?P<s>\d{2}))?\s*$")


def _sample(s: pd.Series) -> pd.Series:
    """A seeded sample, not the head and not a stride.

    The same reasoning as the date detector: a file sorted by anything
    is typed from its first rows alone if you take the head, and a
    stride aliases against any period in the data - which was measured
    there, not assumed."""
    txt = s.dropna().astype(str).str.strip()
    txt = txt[txt.str.len() > 0]
    if len(txt) > 5000:
        txt = txt.sample(n=5000, random_state=0)
    return txt


def quantity_kind(s: pd.Series) -> Optional[Dict[str, Any]]:
    """Which decorated-number shape this column is, or None.

    Only called on a column that already failed the numeric test and
    the date test, so nothing that parses as a plain number or a date
    can be pulled in here."""
    txt = _sample(s)
    if len(txt) == 0:
        return None

    hits = txt.str.match(_CURRENCY)
    share = float(hits.mean())
    if share >= QUANTITY_SHARE:
        m = txt[hits].str.extract(_CURRENCY)
        sym = m["sym"].mode()
        decimals = m["num"].str.contains(".", regex=False)
        return {
            "kind": "currency",
            "symbol": (str(sym.iloc[0]) if len(sym) else "$"),
            "thousands": bool(m["num"].str.contains(",",
                                                    regex=False).any()),
            "decimals": 2 if bool(decimals.any()) else 0,
            "unparsed_share": round(1.0 - share, 6),
        }

    hits = txt.str.match(_PERCENT)
    share = float(hits.mean())
    if share >= QUANTITY_SHARE:
        m = txt[hits].str.extract(_PERCENT)
        dec = m["num"].str.split(".").str[1].dropna()
        return {
            "kind": "percent",
            "decimals": int(dec.str.len().max()) if len(dec) else 0,
            "unparsed_share": round(1.0 - share, 6),
        }

    hits = txt.str.match(_CLOCK)
    share = float(hits.mean())
    if share >= QUANTITY_SHARE:
        m = txt[hits].str.extract(_CLOCK)
        # A clock is CYCLICAL and this treats it as a line from
        # midnight, so 23:59 and 00:01 land at opposite ends when they
        # are two minutes apart. Said here rather than discovered
        # later: it is right for durations-of-day and wrong for
        # anything where the wrap matters.
        return {
            "kind": "clock",
            "seconds": bool(m["s"].notna().any()),
            "unparsed_share": round(1.0 - share, 6),
            "note": "minutes since midnight; the wrap at midnight is "
                    "not modelled",
        }
    return None


def to_number(s: pd.Series, spec: Dict[str, Any]) -> pd.Series:
    """The column as a float, with NaN for whatever did not parse."""
    txt = s.astype(str).str.strip()
    kind = spec["kind"]
    if kind == "currency":
        num = txt.str.extract(_CURRENCY)["num"].str.replace(
            ",", "", regex=False)
        neg = (txt.str.extract(_CURRENCY)["neg1"].notna()
               | txt.str.extract(_CURRENCY)["neg2"].notna())
        out = pd.to_numeric(num, errors="coerce")
        return out.where(~neg, -out)
    if kind == "percent":
        return pd.to_numeric(
            txt.str.extract(_PERCENT)["num"], errors="coerce")
    m = txt.str.extract(_CLOCK)
    h = pd.to_numeric(m["h"], errors="coerce")
    mi = pd.to_numeric(m["m"], errors="coerce")
    sec = pd.to_numeric(m["s"], errors="coerce").fillna(0)
    return h * 60.0 + mi + sec / 60.0


def from_number(v, spec: Dict[str, Any]) -> pd.Series:
    """Back to text, wearing what it was wearing."""
    x = pd.Series(np.asarray(v, dtype=float))
    kind = spec["kind"]
    if kind == "currency":
        dec = int(spec.get("decimals", 2))
        sym = spec.get("symbol", "$")
        fmt = "{:,." + str(dec) + "f}" if spec.get("thousands") \
            else "{:." + str(dec) + "f}"

        def one(z):
            if z != z:
                return None
            return ("-" + sym + fmt.format(abs(z)) if z < 0
                    else sym + fmt.format(z))
        return x.map(one)
    if kind == "percent":
        dec = int(spec.get("decimals", 0))
        fmt = "{:." + str(dec) + "f}%"
        return x.map(lambda z: None if z != z else fmt.format(z))
    with_sec = bool(spec.get("seconds"))

    def clock(z):
        if z != z:
            return None
        total = int(round(max(z, 0.0) * 60.0)) % (24 * 3600)
        h, rem = divmod(total, 3600)
        mi, sec = divmod(rem, 60)
        return ("{:02d}:{:02d}:{:02d}".format(h, mi, sec) if with_sec
                else "{:02d}:{:02d}".format(h, mi))
    return x.map(clock)
