"""How a parent moves a child, in a form a person can read and a
generator can sample from.

WHY BOTH AT ONCE. "systolic explains creatinine, skill 0.94" tells a
reader that something is there and nothing about what. Somebody who
has never seen the data - which is the whole premise of this system -
cannot act on a skill score. What they can act on is "creatinine
climbs slowly to a systolic of about 150 and then sharply", which is
the same object a sampler needs to draw the child from the parent.

So the curve is extracted once and serves as the explanation AND the
generative rule. That also settles a privacy question the alternative
would have opened: the other way to generate is to ship the fitted
boosted tree back from the machine holding the extract, and a tree
fitted on real patients sits far closer to those records than an
averaged response curve does. The project's rule is to learn the
patterns and never copy the records, and a curve keeps it.

WHAT IS AVERAGED, AND OVER WHAT. This is a partial-dependence curve:
the parent is set to each grid value across every held-out row in
turn, and the model's prediction is averaged. It answers "if this
quantity were v, what would the child be on average, holding the rest
of the population as it is" - not "what is the child among rows where
the parent happens to be v", which would be confounded by everything
that travels with the parent.

A column is moved TOGETHER WITH ITS OWN LAG, for the same reason
importance is measured that way: they are the same quantity at two
times, and moving one while pinning the other asks a question no
patient could answer.

THE GRID NEVER LEAVES THE OBSERVED RANGE. Points are quantiles of the
values actually seen, so a curve is never a claim about territory the
data did not cover.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

N_GRID = 11
# A response that moves less than this share of the child's own spread
# is called flat. Below it the curve is noise dressed as a finding.
FLAT_SHARE = 0.05
# One step carrying this much of the total rise is a threshold rather
# than a slope.
STEP_DOMINANCE = 0.5
# An interior turning point must clear this share of the range at both
# ends before the curve is called U-shaped.
TURN_CLEARANCE = 0.25
# Monotone with the last third's slope this far below the first
# third's is saturating rather than merely rising.
SATURATION_RATIO = 0.25


def _grid_for(s: pd.Series, n_grid: int = N_GRID):
    """Quantile grid for a numeric column, levels for a categorical.

    Quantiles, not an even split of min to max: a skewed column would
    otherwise spend most of its grid in a tail holding a handful of
    rows, and the curve would be mostly noise where the data is
    thinnest."""
    if pd.api.types.is_numeric_dtype(s):
        v = s.dropna().to_numpy(dtype=float)
        if v.size == 0:
            return [], "numeric"
        qs = np.linspace(0.02, 0.98, n_grid)
        g = np.unique(np.quantile(v, qs))
        return [float(x) for x in g], "numeric"
    counts = s.dropna().astype(str).value_counts()
    return [str(x) for x in counts.index[:12]], "categorical"


def effect_curve(model, Xp, members: List[str], source: pd.Series,
                 child_kind: str, n_grid: int = N_GRID
                 ) -> Optional[Dict[str, Any]]:
    """The child's average response as the parent sweeps its range."""
    grid, gkind = _grid_for(source, n_grid)
    if len(grid) < 2:
        return None
    present = [c for c in members if c in Xp.columns]
    if not present:
        return None

    X = Xp.copy()
    keep = dict((c, X[c].to_numpy(copy=True)) for c in present)
    rows = []
    for v in grid:
        for c in present:
            if gkind == "categorical" and \
                    isinstance(X[c].dtype, pd.CategoricalDtype):
                if v not in set(X[c].cat.categories):
                    rows.append(None)
                    break
            X[c] = v
        else:
            if child_kind == "regression":
                rows.append(float(np.mean(model.predict(X))))
            else:
                p = model.predict_proba(X)
                rows.append(np.asarray(p).mean(axis=0))
            continue
    # WHAT HAPPENS WHEN THE PARENT IS NOT MEASURED AT ALL.
    #
    # The grid sweeps observed values only, so a relationship carried
    # by the parent's ABSENCE reads as a flat curve under a skill of
    # 1.00 - an internal contradiction, and the real extract produced
    # one immediately: `is_last_visit <- days_to_next_visit` explains
    # 100% of it, and days_to_next_visit is missing on exactly the
    # last visit. The value never mattered; being measured did.
    miss = None
    for c in present:
        X[c] = np.nan
    try:
        if child_kind == "regression":
            miss = float(np.mean(model.predict(X)))
        else:
            miss = np.asarray(model.predict_proba(X)).mean(axis=0)
    except Exception:
        miss = None
    for c in present:
        X[c] = keep[c]

    ok = [(g, r) for g, r in zip(grid, rows) if r is not None]
    if len(ok) < 2:
        return None
    grid = [g for g, _ in ok]

    if child_kind == "regression":
        resp = [float(r) for _, r in ok]
        label = None
        miss_v = None if miss is None else float(miss)
    else:
        # One curve, and it must be the informative one: the class
        # whose probability MOVES most across the grid. Picking the
        # modal class instead would report a flat line whenever the
        # parent shifts the rare outcomes, which is usually the
        # clinically interesting case.
        mat = np.vstack([r for _, r in ok])
        j = int(np.argmax(mat.max(axis=0) - mat.min(axis=0)))
        resp = [float(x) for x in mat[:, j]]
        label = str(model.classes_[j])
        miss_v = (None if miss is None
                  else float(np.asarray(miss).ravel()[j]))
    return {"grid": grid, "response": resp, "grid_kind": gkind,
            "class_of_interest": label,
            "response_when_missing": miss_v}


def describe(curve: Dict[str, Any], parent: str, child: str,
             child_spread: float) -> Dict[str, Any]:
    """Name the shape and say it in a sentence.

    The names are the vocabulary a reader thinks in - rising,
    U-shaped, threshold, saturating - not coefficients."""
    g, r = curve["grid"], np.asarray(curve["response"], dtype=float)
    lo, hi = float(r.min()), float(r.max())
    rng = hi - lo
    cls = curve.get("class_of_interest")
    what = ("P({} = {})".format(child, cls) if cls else child)

    if curve["grid_kind"] == "categorical":
        order = np.argsort(r)
        shape = "flat" if rng < FLAT_SHARE * max(child_spread, 1e-9) \
            else "varies-by-level"
        text = ("{} barely differs across {}".format(what, parent)
                if shape == "flat" else
                "{} is lowest at {}={!r} ({:.4g}) and highest at "
                "{}={!r} ({:.4g})".format(
                    what, parent, g[order[0]], r[order[0]],
                    parent, g[order[-1]], r[order[-1]]))
        return {"shape": shape, "description": text,
                "effect_size": round(rng, 6),
                "monotone": False}

    if rng < FLAT_SHARE * max(child_spread, 1e-9):
        mv = curve.get("response_when_missing")
        if mv is not None and abs(mv - float(r.mean())) > \
                max(2.0 * rng, 0.2 * max(child_spread, 1e-9)):
            # The value does nothing; being measured does everything.
            return {"shape": "presence-only", "monotone": False,
                    "effect_size": round(
                        abs(mv - float(r.mean())), 6),
                    "response_when_missing": round(mv, 6),
                    "description": (
                        "{} does not depend on the VALUE of {} at all "
                        "- it depends on whether {} was measured. "
                        "Observed it sits near {:.4g}; when {} is "
                        "absent it is {:.4g}".format(
                            what, parent, parent, float(r.mean()),
                            parent, mv))}
        return {"shape": "flat", "monotone": False,
                "effect_size": round(rng, 6),
                "description": "{} hardly moves as {} varies - the "
                               "relationship is carried by something "
                               "other than this parent alone".format(
                                   what, parent)}

    d = np.diff(r)
    up, down = bool(np.all(d >= -rng * 0.02)), bool(
        np.all(d <= rng * 0.02))
    span = "{} from {:.4g} to {:.4g} as {} goes {:.4g} to {:.4g}"

    # an interior turning point beats every other reading
    imin, imax = int(np.argmin(r)), int(np.argmax(r))
    inner = range(1, len(r) - 1)
    if imin in inner and (r[0] - lo) > TURN_CLEARANCE * rng and \
            (r[-1] - lo) > TURN_CLEARANCE * rng:
        return {"shape": "u-shaped", "monotone": False,
                "effect_size": round(rng, 6),
                "turning_point": round(float(g[imin]), 6),
                "description": "{} falls then rises as {} increases - "
                               "U-shaped, lowest near {}={:.4g}".format(
                                   what, parent, parent, g[imin])}
    if imax in inner and (hi - r[0]) > TURN_CLEARANCE * rng and \
            (hi - r[-1]) > TURN_CLEARANCE * rng:
        return {"shape": "inverted-u", "monotone": False,
                "effect_size": round(rng, 6),
                "turning_point": round(float(g[imax]), 6),
                "description": "{} rises then falls as {} increases - "
                               "peaks near {}={:.4g}".format(
                                   what, parent, parent, g[imax])}

    # A THRESHOLD IS A BIG STEP WITH FLAT GROUND ON BOTH SIDES.
    #
    # Dominance alone is not enough to tell one from a saturating
    # curve. Measured on a planted 10*(1-exp(-x/1.5)): the first grid
    # step carried 73% of the whole range, so the dominance rule fired
    # and called it a threshold - while its last-third slope was 0.3%
    # of its first-third slope, which is saturation by any reading.
    # The difference is not the size of the step but what surrounds
    # it: a threshold is quiet BEFORE as well as after, and a
    # saturating curve is steep from the start.
    k = int(np.argmax(np.abs(d)))
    interior = 1 <= k <= len(d) - 2
    calm_before = interior and float(np.mean(np.abs(d[:k]))) < \
        0.25 * abs(d[k])
    calm_after = interior and float(np.mean(np.abs(d[k + 1:]))) < \
        0.25 * abs(d[k])
    if abs(d[k]) > STEP_DOMINANCE * rng and calm_before and calm_after:
        at = (g[k] + g[k + 1]) / 2.0
        return {"shape": "threshold", "monotone": up or down,
                "effect_size": round(rng, 6),
                "turning_point": round(float(at), 6),
                "description": "{} steps {} sharply around {}={:.4g} - "
                               "most of the change happens at that "
                               "one point rather than gradually"
                               .format(what,
                                       "up" if d[k] > 0 else "down",
                                       parent, at)}

    if up or down:
        third = max(1, len(d) // 3)
        first = float(np.mean(np.abs(d[:third])))
        last = float(np.mean(np.abs(d[-third:])))
        if first > 0 and last / first < SATURATION_RATIO:
            return {"shape": "saturating", "monotone": True,
                    "effect_size": round(rng, 6),
                    "description": ("{} rises steeply and then "
                                    "flattens as {} increases - past "
                                    "about {:.4g} more {} buys little"
                                    ).format(what, parent,
                                             g[max(third, 1)], parent)}
        return {"shape": "increasing" if up else "decreasing",
                "monotone": True, "effect_size": round(rng, 6),
                "description": ("{} " + ("rises " if up else "falls ")
                                + span.format("", r[0], r[-1], parent,
                                              g[0], g[-1])).replace(
                                    "  ", " ").format(what)}

    return {"shape": "non-monotone", "monotone": False,
            "effect_size": round(rng, 6),
            "description": "{} moves with {} but not in one direction "
                           "- it ranges {:.4g} to {:.4g} without a "
                           "simple trend".format(what, parent, lo, hi)}


def joint_surface(model, Xp, members_a: List[str],
                  members_b: List[str], source_a: pd.Series,
                  source_b: pd.Series, child_kind: str,
                  n: int = 11) -> Optional[Dict[str, Any]]:
    """The child's response across a grid of TWO parents at once.

    A one-parent curve cannot show an interaction, and reporting two
    flat curves for a relationship the model scores highly on is
    technically true and useless. This is the case the whole system
    was built to catch - a pure exclusive-or, where neither factor
    carries an effect alone - so it deserves the one picture that can
    actually show it.

    GRID RESOLUTION IS A FIDELITY DIAL, measured rather than picked.
    A value within half a step of a sharp boundary snaps to the grid
    line sitting ON it, which is a blur of both regimes. Regenerating
    a planted exclusive-or whose true corner contrast is 1.00:

        7x7     contrast 0.575
        11x11   contrast 0.740
        15x15   contrast 0.810

    with no measurable difference in discovery time, because the cost
    is dominated by fitting rather than by surface lookups. 11 takes
    most of the gain; n-squared predictions per interaction still
    matter on a wide extract carrying many claims."""
    ga, ka = _grid_for(source_a, n)
    gb, kb = _grid_for(source_b, n)
    if ka != "numeric" or kb != "numeric" or len(ga) < 2 \
            or len(gb) < 2:
        return None
    pa = [c for c in members_a if c in Xp.columns]
    pb = [c for c in members_b if c in Xp.columns]
    if not pa or not pb:
        return None

    X = Xp.copy()
    keep = dict((c, X[c].to_numpy(copy=True)) for c in pa + pb)
    cls_j = None
    if child_kind != "regression":
        # the class that moves most across the whole surface
        probe = []
        for va in (ga[0], ga[-1]):
            for vb in (gb[0], gb[-1]):
                for c in pa:
                    X[c] = va
                for c in pb:
                    X[c] = vb
                probe.append(np.asarray(
                    model.predict_proba(X)).mean(axis=0))
        pm = np.vstack(probe)
        cls_j = int(np.argmax(pm.max(axis=0) - pm.min(axis=0)))

    grid = []
    for va in ga:
        row = []
        for vb in gb:
            for c in pa:
                X[c] = va
            for c in pb:
                X[c] = vb
            if child_kind == "regression":
                row.append(float(np.mean(model.predict(X))))
            else:
                row.append(float(np.asarray(
                    model.predict_proba(X)).mean(axis=0)[cls_j]))
        grid.append(row)
    for c in pa + pb:
        X[c] = keep[c]
    return {"grid_a": ga, "grid_b": gb, "response": grid,
            "class_of_interest": (None if cls_j is None
                                  else str(model.classes_[cls_j]))}


def describe_joint(surf: Dict[str, Any], a: str, b: str,
                   child: str) -> Dict[str, Any]:
    """Name what the combination does, in corners a reader can check."""
    r = np.asarray(surf["response"], dtype=float)
    ll, lh = float(r[0, 0]), float(r[0, -1])
    hl, hh = float(r[-1, 0]), float(r[-1, -1])
    rng = float(r.max() - r.min())
    cls = surf.get("class_of_interest")
    what = "P({} = {})".format(child, cls) if cls else child
    same, cross = (ll + hh) / 2.0, (lh + hl) / 2.0

    if rng <= 0:
        return {"pattern": "none", "description":
                "{} does not respond to {} and {} together "
                "either".format(what, a, b)}
    if (cross - same) > 0.25 * rng:
        pat, sent = "opposing", (
            "{} is HIGH when exactly one of {} and {} is high, and low "
            "when they agree - neither column moves it alone, so the "
            "effect lives entirely in the combination".format(
                what, a, b))
    elif (same - cross) > 0.25 * rng:
        pat, sent = "reinforcing", (
            "{} is HIGH when {} and {} are both high or both low - "
            "they act together rather than separately".format(
                what, a, b))
    else:
        pat, sent = "conditional", (
            "{} depends on {} and {} jointly, but not as a simple "
            "agree/disagree rule - read the surface".format(
                what, a, b))
    return {"pattern": pat, "description": sent,
            "corners": {"low_low": round(ll, 6),
                        "low_high": round(lh, 6),
                        "high_low": round(hl, 6),
                        "high_high": round(hh, 6)},
            "effect_size": round(rng, 6)}
