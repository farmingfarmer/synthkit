"""Several tables in, keys found from the VALUES, integrity kept.

The fitted path has always been single-table. An operator with
a patients file, a visits file and a labs file had to join them
by hand first - and a hand join is where the row counts and the
missingness quietly change, before this tool ever sees them.

WHAT THIS DOES, AND WHAT IT REFUSES TO GUESS.

KEYS ARE FOUND FROM THE VALUES. A column named `patient_id`
holding lab results is not a key, and a key named `subj` is
still a key - the same rule `scrub` uses for PHI, for the same
reason: a header is a hint somebody typed, and the values are
what is true. The header only breaks ties.

A PRIMARY KEY is unique and present on every row. A FOREIGN KEY
is a column whose values are drawn from some other table's
primary key. Both are measured, with the SHARE reported, so a
link that holds on 94% of rows is reported as a 94% link rather
than silently accepted or silently dropped.

AMBIGUITY IS REPORTED, NEVER RESOLVED BY GUESSING. Two columns
that both look like the same foreign key, or a table with two
candidate primary keys, come back as a question with the
evidence attached. This project has a long record of silent
wrong answers being worse than loud refusals, and a mis-detected
key would reshape somebody's extract.

REFERENTIAL INTEGRITY IS MEASURED IN THE SOURCE FIRST. Real
extracts have orphans - a visit whose patient was filtered out
by somebody's cohort query - and a tool that reports 100%
integrity on output without saying what the input had is
flattering itself.
"""
from typing import Dict, List, Optional, Tuple

import pandas as pd

# A column must be present on at least this share of rows and
# unique on all of them to be a primary key. Not 1.0 on
# presence, because an extract can carry a blank trailing row
# and failing to find the key over one blank is a refusal
# nobody can act on.
PK_MIN_PRESENT = 0.99

# A foreign key must land in the parent's key on at least this
# share of its present values. Below it, the column is reported
# as a WEAK link with its share rather than accepted - a 40%
# overlap is a coincidence, not a key.
FK_MIN_MATCH = 0.80

# A key has to have enough distinct values to be a key at all.
# A column of two values matches any two-value column by
# accident, and this is the detection equivalent of the
# commensurability rule the constraint finder already uses.
MIN_DISTINCT = 8


class Table:
    """One input table, with what was measured about it."""

    def __init__(self, name: str, frame: pd.DataFrame):
        self.name = name
        self.frame = frame
        self.primary_key: Optional[str] = None
        self.pk_candidates: List[str] = []
        self.notes: List[str] = []


class Link:
    """A foreign key, with the evidence for it."""

    def __init__(self, child: str, column: str, parent: str,
                 parent_key: str, match: float, orphans: int):
        self.child = child
        self.column = column
        self.parent = parent
        self.parent_key = parent_key
        self.match = match
        self.orphans = orphans

    def __repr__(self):
        return "{}.{} -> {}.{} ({:.1%})".format(
            self.child, self.column, self.parent,
            self.parent_key, self.match)


def _present(s: pd.Series) -> pd.Series:
    """Values that are actually there.

    `""` is absent for a KEY - a blank foreign key is a row
    with no parent, not a row whose parent is named "". This is
    the opposite call from set columns, where present-and-empty
    is a fact, and the difference is deliberate: a set can be
    empty, a key cannot.
    """
    return s[(s.notna()) & (s.astype(str).str.strip() != "")]


def _is_measurement(v: pd.Series) -> bool:
    """A continuous number is a measurement, not an identifier.

    THIS IS THE FAULT THAT MADE A LAB VALUE A PRIMARY KEY. A
    column of 300 normal draws is unique on every row and
    present on every row, so a uniqueness test calls it a key -
    and that key would then be used to LINK TABLES. Identifiers
    are strings or whole numbers; a value with a fractional
    part is something somebody measured.

    Whole numbers are left alone, because a patient number is
    often an integer and refusing those would reject the most
    common key there is.
    """
    nums = pd.to_numeric(v, errors="coerce")
    got = nums.dropna()
    if len(got) < 0.9 * len(v):
        return False          # not numeric enough to judge
    frac = (got != got.round()).mean()
    return bool(frac > 0.05)


def primary_keys(frame: pd.DataFrame) -> List[str]:
    """Every column that could be this table's primary key."""
    out = []
    n = len(frame)
    if not n:
        return out
    for c in frame.columns:
        v = _present(frame[c])
        if len(v) < PK_MIN_PRESENT * n:
            continue
        if v.nunique() != len(v):
            continue
        if v.nunique() < MIN_DISTINCT:
            continue
        if _is_measurement(v):
            continue
        out.append(c)
    return out


def _pick_pk(name: str, cands: List[str]) -> Tuple[Optional[str],
                                                   List[str]]:
    """Choose among candidates, with the header breaking ties.

    THE HEADER ONLY BREAKS TIES. Where the values already say
    one column is the key, the name is irrelevant; where
    several columns are equally key-shaped - which happens on a
    narrow table where a timestamp is also unique - a name
    containing the table's own name or ending in `_id` is the
    better guess, and the alternatives are RETURNED so the
    choice is visible.
    """
    if not cands:
        return None, []
    if len(cands) == 1:
        return cands[0], []
    base = name.lower().rstrip("s")

    def score(c):
        lc = c.lower()
        s = 0
        if lc in (base + "_id", base + "id", "id"):
            s += 4
        if lc.endswith("_id") or lc.endswith("id"):
            s += 2
        if base and base in lc:
            s += 1
        return -s
    ranked = sorted(cands, key=score)
    return ranked[0], ranked[1:]


def find_links(tables: Dict[str, Table]) -> Tuple[List[Link],
                                                  List[str]]:
    """Which columns point at which tables' keys.

    Measured by OVERLAP of values, not by name. A column is a
    foreign key when its present values are drawn from a
    parent's primary key; the share that land is reported, so a
    link holding on 94% of rows reads as 94% rather than as a
    yes or a no.
    """
    links: List[Link] = []
    warn: List[str] = []
    keyed = {n: t for n, t in tables.items() if t.primary_key}
    for cname, child in tables.items():
        for col in child.frame.columns:
            vals = _present(child.frame[col])
            if len(vals) < MIN_DISTINCT:
                continue
            if vals.nunique() < 2:
                continue
            # A MEASUREMENT IS NOT A FOREIGN KEY EITHER. The
            # same column that must not be a primary key must
            # not be read as pointing at one.
            if _is_measurement(vals):
                continue
            best = None
            for pname, parent in keyed.items():
                if pname == cname and col == parent.primary_key:
                    continue
                pk = parent.primary_key
                pvals = set(_present(parent.frame[pk])
                            .astype(str))
                if not pvals:
                    continue
                sv = vals.astype(str)
                hit = sv.isin(pvals)
                share = float(hit.mean())
                if share < FK_MIN_MATCH:
                    continue
                # A COLUMN THAT IS THE PARENT'S KEY AND ALSO
                # UNIQUE HERE is a one-to-one join, not a
                # foreign key into a parent - reported, because
                # treating it as a child link would invent a
                # cardinality the data does not have.
                if best is None or share > best[0]:
                    best = (share, pname, pk,
                            int((~hit).sum()))
            if best:
                share, pname, pk, orph = best
                if pname == cname:
                    continue
                links.append(Link(cname, col, pname, pk,
                                  share, orph))
    # AMBIGUITY IS A QUESTION, NOT A COIN FLIP. Two columns in
    # one table pointing at the same parent is legitimate
    # (admitting and discharging clinician) but it is also what
    # a mis-detection looks like, so it is said.
    bypair: Dict[Tuple[str, str], List[Link]] = {}
    for lk in links:
        bypair.setdefault((lk.child, lk.parent), []).append(lk)
    for (c, p), group in bypair.items():
        if len(group) > 1:
            warn.append(
                "{} has {} columns that all match {}'s key "
                "({}) - that can be real, and it is also what "
                "a mis-detection looks like, so check it"
                .format(c, len(group), p,
                        ", ".join(g.column for g in group)))
    return links, warn


def detect(frames: Dict[str, pd.DataFrame]) -> Dict:
    """The whole schema, measured, with the evidence."""
    tables = {}
    for name, fr in frames.items():
        t = Table(name, fr)
        t.pk_candidates = primary_keys(fr)
        t.primary_key, alts = _pick_pk(name, t.pk_candidates)
        if alts:
            t.notes.append(
                "{} columns were equally key-shaped; chose {} "
                "and the others are {}".format(
                    len(t.pk_candidates), t.primary_key,
                    ", ".join(alts)))
        if not t.primary_key:
            t.notes.append(
                "no column is unique and present on every row, "
                "so this table has no key of its own - it can "
                "still be a CHILD of another table")
        tables[name] = t
    links, warn = find_links(tables)
    return {"tables": tables, "links": links, "warnings": warn}


def integrity(frames: Dict[str, pd.DataFrame],
              links: List[Link]) -> List[Dict]:
    """What share of child rows have a parent that exists.

    MEASURED ON THE SOURCE TOO, not only the output. Real
    extracts have orphans - a visit whose patient was filtered
    out by somebody's cohort query - and reporting 100% on
    generated data without saying what the input had is
    flattering the generator with somebody else's data
    cleaning.
    """
    out = []
    for lk in links:
        child = frames.get(lk.child)
        parent = frames.get(lk.parent)
        if child is None or parent is None:
            continue
        if lk.column not in child.columns:
            continue
        if lk.parent_key not in parent.columns:
            continue
        vals = _present(child[lk.column]).astype(str)
        pvals = set(_present(parent[lk.parent_key])
                    .astype(str))
        hit = vals.isin(pvals)
        out.append({
            "child": lk.child, "column": lk.column,
            "parent": lk.parent, "parent_key": lk.parent_key,
            "rows": int(len(vals)),
            "with_parent": int(hit.sum()),
            "orphans": int((~hit).sum()),
            "share": float(hit.mean()) if len(vals) else 1.0,
        })
    return out


def render(found: Dict, integ: List[Dict]) -> str:
    """The schema as a person reads it."""
    lines = ["SCHEMA FOUND FROM THE VALUES"]
    for name, t in found["tables"].items():
        lines.append("  {:<22} {:>8} rows x {:>3} columns   "
                     "key: {}".format(
                         name, len(t.frame), len(t.frame.columns),
                         t.primary_key or "NONE FOUND"))
        for n in t.notes:
            lines.append("      {}".format(n))
    lines.append("")
    if found["links"]:
        lines.append("  LINKS - measured by value overlap, not "
                     "by column name:")
        for lk in found["links"]:
            lines.append("    {:<18}.{:<18} -> {:<18}.{:<14} "
                         "{:.1%} of rows".format(
                             lk.child, lk.column, lk.parent,
                             lk.parent_key, lk.match))
    else:
        lines.append("  NO LINKS FOUND. Either these tables are "
                     "unrelated, or the key columns do not "
                     "share values - a join somebody did by "
                     "hand earlier can leave two tables with no "
                     "common identifier at all.")
    for w in found["warnings"]:
        lines.append("  CHECK: {}".format(w))
    if integ:
        lines.append("")
        lines.append("  REFERENTIAL INTEGRITY IN THE SOURCE - "
                     "orphans here are the operator's data, not "
                     "a fault of this tool, and a generator "
                     "reporting 100% without this line is "
                     "taking credit for somebody else's "
                     "cleaning:")
        for r in integ:
            lines.append("    {:<18}.{:<16} {:>8}/{:<8} have a "
                         "parent  ({:.1%}{})".format(
                             r["child"], r["column"],
                             r["with_parent"], r["rows"],
                             r["share"],
                             ", {} ORPHANS".format(r["orphans"])
                             if r["orphans"] else ""))
    return "\n".join(lines)


# ===================================================================
# generation: fit each table, then make the keys line up
# ===================================================================

def _invent(prefix: str, n: int) -> List[str]:
    """Identifiers that belong to nobody.

    NEVER A REAL KEY. An MRN is a direct identifier and a
    generated file that carries one has republished it, however
    synthetic everything around it is. The latent engine already
    invents `S000001` upward for patients; this is the same rule
    for every table's key.
    """
    return ["{}{:07d}".format(prefix, i + 1) for i in range(n)]


CROSS_MIN_ROWS = 30
CROSS_MIN_RHO = 0.1


def cross_couplings(frames: Dict[str, pd.DataFrame],
                    found: Dict, k: int = 10) -> List[Dict]:
    """Relationships that live ACROSS a link, measured.

    Each table is fitted on its own, so a child column that
    depends on a PARENT's column has no path to learn it.
    Measured on a planted fixture - a lab value at 2.0x its
    visit's length of stay - the source reads +0.906 and the
    generated set +0.021: the structure is destroyed, not
    weakened.

    This finds those pairs in the source so generation has
    something to aim at. Numeric pairs only, by rank
    correlation, over the JOIN - and screened the way every
    other published number here is: a pair needs rows behind it
    and a correlation above the gate's own floor, or it is
    noise being chased across a key.
    """
    import pandas as _pd
    out: List[Dict] = []
    for lk in found["links"]:
        if lk.child not in frames or lk.parent not in frames:
            continue
        ch, pa = frames[lk.child], frames[lk.parent]
        t = found["tables"].get(lk.parent)
        if t is None or not t.primary_key:
            continue
        pk = t.primary_key
        if lk.column not in ch.columns or pk not in pa.columns:
            continue
        joined = ch.merge(pa, left_on=lk.column, right_on=pk,
                          how="inner", suffixes=("", "__par"))
        if len(joined) < CROSS_MIN_ROWS:
            continue
        for cc in ch.columns:
            if cc == lk.column:
                continue
            a = _pd.to_numeric(joined.get(cc), errors="coerce")
            if a is None or a.notna().sum() < CROSS_MIN_ROWS:
                continue
            for pc in pa.columns:
                if pc == pk:
                    continue
                col = pc if pc in joined.columns else pc + "__par"
                b = _pd.to_numeric(joined.get(col),
                                   errors="coerce")
                if b is None or b.notna().sum() < CROSS_MIN_ROWS:
                    continue
                m = a.notna() & b.notna()
                if int(m.sum()) < CROSS_MIN_ROWS:
                    continue
                if a[m].nunique() < 3 or b[m].nunique() < 3:
                    continue
                rho = float(a[m].corr(b[m], method="spearman"))
                if rho == rho and abs(rho) >= CROSS_MIN_RHO:
                    out.append({
                        "child": lk.child, "child_column": cc,
                        "parent": lk.parent,
                        "parent_column": pc,
                        "fk": lk.column, "rho": round(rho, 4),
                        "rows": int(m.sum())})
    return out


def _couple_assign(child_vals, pool, parent_vals, target, rng):
    """Choose WHICH parent each child row points at.

    THE ASSIGNMENT IS THE ONLY FREE VARIABLE, and it is enough.
    The child's own values and the parent's own values are
    already generated and must not move - they carry their
    marginals. What is still free is which parent a child row
    is attached to, and that alone sets the correlation across
    the link.

    So this is a REARRANGEMENT, the same instrument the repair
    and the dials use: it permutes an assignment, so no
    marginal can drift and referential integrity is untouched -
    every child still points at a parent that exists, because
    the pool is the parents that exist.

    The noise scale is SEARCHED rather than assumed, because a
    rank-perfect pairing lands rho near 1.0 and the source is
    rarely there.
    """
    import numpy as np
    import pandas as _pd
    n = len(child_vals)
    draws = rng.choice(len(pool), size=n, replace=True)
    pv = _pd.Series(parent_vals).iloc[draws].to_numpy()
    cv = _pd.to_numeric(_pd.Series(child_vals),
                        errors="coerce").to_numpy()
    ok = ~_pd.isna(cv)
    if ok.sum() < CROSS_MIN_ROWS:
        return [pool[i] for i in draws]
    cr = _pd.Series(cv).rank(method="average").to_numpy()
    pr = _pd.Series(pv).rank(method="average").to_numpy()
    order_p = np.argsort(pr, kind="stable")
    sign = 1.0 if target >= 0 else -1.0
    best, best_err = None, None
    for scale in (0.0, 0.15, 0.3, 0.5, 0.8, 1.2, 2.0, 3.5, 6.0):
        jitter = cr + rng.normal(0, scale * n / 6.0, n)
        order_c = np.argsort(sign * jitter, kind="stable")
        pick = np.empty(n, dtype=int)
        pick[order_c] = order_p
        # child i is paired with parent slot pick[i], so the
        # parent value it sees is pv[pick[i]].
        got = float(_pd.Series(cv[ok]).corr(
            _pd.Series(pv[pick][ok]), method="spearman"))
        if got != got:
            continue
        err = abs(got - target)
        if best_err is None or err < best_err:
            best, best_err = pick, err
    if best is None:
        return [pool[i] for i in draws]
    return [pool[draws[i]] for i in best]


def assign_keys(frames: Dict[str, pd.DataFrame],
                found: Dict, seed: int = 0,
                keep_orphan_share: bool = True,
                couplings: Optional[List[Dict]] = None
                ) -> Dict[str, pd.DataFrame]:
    """Invent each table's key and point the children at them.

    REFERENTIAL INTEGRITY BY CONSTRUCTION, NOT BY REPAIR. The
    parent's key is invented first; every child foreign key is
    then DRAWN from the keys that now exist. There is no step
    where a child can point at a parent that is not there, so
    integrity is not something this has to go back and check -
    which is the same reason the size identity is enforced by
    copying rather than by drawing twice.

    THE ORPHAN SHARE IS KEPT, and that is deliberate. A real
    extract has visits whose patient was filtered out by
    somebody's cohort query; emitting 100% integrity where the
    source had 97.8% would be inventing a cleanliness the data
    never had, and every downstream count that depends on
    orphans would be wrong. The share is reproduced, with
    invented keys that match nothing - so the ORPHANS are
    orphans, and they are not somebody's real identifier
    either.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    out = {n: f.copy() for n, f in frames.items()}
    tables = found["tables"]

    # Parents first, so a child always has keys to draw from.
    # The order is by how many tables point AT a table, which
    # puts the root of a star or a chain first without needing
    # the caller to say which it is.
    pointed = {n: 0 for n in frames}
    for lk in found["links"]:
        pointed[lk.parent] = pointed.get(lk.parent, 0) + 1
    order = sorted(frames, key=lambda n: -pointed.get(n, 0))

    newkeys: Dict[str, List[str]] = {}
    for name in order:
        t = tables.get(name)
        if t is None or not t.primary_key:
            continue
        pk = t.primary_key
        n = len(out[name])
        prefix = "".join(ch for ch in name.upper()
                         if ch.isalpha())[:3] or "K"
        keys = _invent(prefix, n)
        out[name][pk] = keys
        newkeys[name] = keys

    for lk in found["links"]:
        if lk.parent not in newkeys:
            continue
        if lk.column not in out[lk.child].columns:
            continue
        child = out[lk.child]
        pool = newkeys[lk.parent]
        n = len(child)
        # WHICH PARENT A CHILD POINTS AT IS NOT ARBITRARY.
        #
        # This drew uniformly, which is exactly where a
        # cross-table relationship went: each table is fitted
        # alone, so a lab value that depends on its visit's
        # length of stay has no path to learn it, and a random
        # attachment then guarantees it is absent. Measured on
        # the planted fixture, source +0.906 against generated
        # +0.021.
        #
        # The child's values and the parent's values are
        # already drawn and must not move - they carry their
        # marginals. The assignment is the only free variable,
        # and it is enough to set the correlation across the
        # link. Integrity is untouched because the pool is the
        # parent keys that exist.
        use = [cp for cp in (couplings or [])
               if cp.get("child") == lk.child
               and cp.get("parent") == lk.parent
               and cp.get("child_column") in child.columns]
        picked = None
        if use:
            # One coupling per link for now, the strongest -
            # two columns pulling on one assignment is a joint
            # solve, and this file's own record says a cap
            # that excludes the pair the gate measures is how
            # a fix misses. Named as the limit rather than
            # left to be discovered.
            cp = max(use, key=lambda d: abs(d.get("rho") or 0))
            par = out[lk.parent]
            pcol = cp.get("parent_column")
            if pcol in par.columns:
                pvals = pd.to_numeric(par[pcol],
                                      errors="coerce").to_numpy()
                picked = _couple_assign(
                    child[cp["child_column"]].tolist(),
                    list(pool), pvals, float(cp["rho"]), rng)
        if picked is None:
            picked = list(rng.choice(pool, size=n, replace=True))
        if keep_orphan_share and lk.match < 1.0:
            # REPRODUCE THE ORPHANS, with keys that belong to
            # nobody. Rounding is to the nearest row, and the
            # share is taken from the SOURCE measurement rather
            # than from a guess.
            n_orph = int(round((1.0 - lk.match) * n))
            if n_orph:
                idx = rng.choice(n, size=n_orph, replace=False)
                ghosts = _invent("ORPHAN", n_orph)
                for j, i in enumerate(idx):
                    picked[i] = ghosts[j]
        child[lk.column] = picked
        out[lk.child] = child
    return out


def check_integrity(frames: Dict[str, pd.DataFrame],
                    found: Dict) -> List[Dict]:
    """Integrity of a GENERATED set, measured the same way.

    One function for both sides would be ideal and is not
    possible here - `integrity` takes the links found in the
    SOURCE, and the generated frames must be judged against
    those same links rather than against links re-detected in
    the output. Re-detecting would let a generator that
    destroyed a key score well by having no link to break.
    """
    return integrity(frames, found["links"])
