"""One machine-readable metrics record per run, for a dashboard.

WHY THIS EXISTS. Every number this project measures already lands
somewhere - `fidelity.json`, `blueprint.json`, the deck, the
sign-off page - and a reader who wanted to chart a run against the
last one had to open four files and know which key in each. Asked
in a sync whether the metrics were "tracked in like a log", the
honest answer was no: they were tracked in a UI, one run at a
time.

WHAT IT IS NOT. It computes NOTHING. Every figure is read from the
run's own artifacts, and the gate comes from `gate.assess`, which
is the one implementation - a second copy is how a dashboard and a
release verdict come to disagree, and this project has that fault
on record for token screening, the scaffolding predicate, a whole
engine and a renamed key that cost an hour.

THE THREE SECTIONS ARE THE ASK, kept apart on purpose:
    patterns    what the fit found in the SOURCE
    fidelity    how much of it the synthetic file carries
    settings    what was tweaked, and what answers the
                "how did that change the output" question
and `privacy` is a FOURTH, separate from `series`, because a
privacy reading on a fidelity trend line invites somebody to
optimise it.

`series` IS THE DASHBOARD SURFACE and it is deliberately FLAT
scalars with stable names. A dashboard stacks N runs and charts a
name over time; a nested blob cannot be stacked, and renaming a
key silently breaks a chart that then reads as a flat line rather
than as an error. Everything richer lives in the sections below
it, which a reader opens rather than charts.
"""
from __future__ import annotations

import io
import json
import os
from typing import Any, Dict, List, Optional

from . import gate as _gate

SCHEMA = 1

# WHAT THIS RECORD MUST NEVER CARRY. Data, not prose: a renderer
# cannot drop what it is handed, and a reader of the dashboard is
# entitled to know what was deliberately left out rather than
# inferring it from what happens to be present.
WITHHELD = (
    "no row of the source and no row of the generated file",
    "no bin count, cell count or per-group count - a bin holding "
    "three patients describes those three, which is why the deck "
    "suppresses them too",
    "no published value of any column - distances, counts of "
    "COLUMNS and column NAMES only",
    "no patient identifier, real or invented",
)

# A LEAF ARRAY LONGER THAN THE COLUMN COUNT IS A DATA DUMP. A
# histogram, a row slice and a per-patient series all look like
# that, and a name-based ban would only catch the ones somebody
# thought to name. Structural, so it catches the class.
ARRAY_SLACK = 8


# THE CHARTED METRICS: name, the summary field it reads, and
# what it means in one line. ONE table, so `series` and
# `catalog` cannot drift - a second list is the hand-kept
# fault this project records for a dependency table and a
# dial list, both of which went stale the day somebody
# added an entry.
_DIRECT = (
        ("pairs_related", "pairs",
         "relationships the SOURCE relates at 0.1 or more - the denominator for the three pair metrics below"),
        ("pairs_direction_kept", "pairs_sign_ok",
         "of those, how many keep their direction in the output"),
        ("pairs_close", "pairs_close",
         "of those, how many land within 0.2 of the source correlation"),
        ("pairs_inverted", "pairs_inverted",
         "relationships pointing the OPPOSITE way with magnitude - the absolute stop, and zero is the only acceptable value"),
        ("pairs_own", "pairs_own",
         "the same denominator restricted to columns the operator receives, excluding search scaffolding"),
        ("pairs_own_direction_kept", "pairs_own_sign_ok",
         "direction kept on the operator's own columns"),
        ("pairs_own_close", "pairs_own_close",
         "close on the operator's own columns"),
        ("pairs_own_inverted", "pairs_own_inverted",
         "inversions on the operator's own columns - this is the one that governs release"),
        ("pairs_scaffold", "pairs_scaffold",
         "how many of the related pairs are set-token search machinery dropped before the file is written"),
        ("columns_measured", "columns",
         "numeric columns the per-column criteria could grade"),
        ("coverage_ok", "coverage_ok",
         "columns whose presence rate lands within 0.05 of the source"),
        ("centre_ok", "centre_ok",
         "columns whose centre lands within 10% of their own spread"),
        ("spread_ok", "spread_ok",
         "columns whose spread lands in band"),
        ("cluster_ok", "cluster_ok",
         "columns whose missingness clustering lands in band"),
        ("constraints_checked", "constraints_checked",
         "row-level orderings the source never violates"),
        ("constraints_held", "constraints_held",
         "of those, how many the output never violates"),
        ("shapes_published", "shapes_published",
         "effect curves the contract publishes - the denominator"),
        ("shapes_compared", "shapes_compared",
         "of those, how many could be graded at all"),
        ("shapes_tracked", "shapes_ok",
         "of those, how many the output follows within 0.35 sd"),
        ("shapes_not_readable", "shapes_not_readable",
         "curves whose own sampling noise reaches the bar, so nobody can judge them - a statement about cohort size, not about the data"),
        ("surfaces_compared", "surfaces_compared",
         "two-variable interaction surfaces that could be graded"),
        ("surfaces_tracked", "surfaces_ok",
         "of those, how many survive generation"),
        ("surfaces_own_compared", "surfaces_own_compared",
         "the same, restricted to the operator's own columns"),
        ("surfaces_own_tracked", "surfaces_own_ok",
         "surfaces surviving on the operator's own columns"),
        ("surfaces_not_readable", "surfaces_not_readable",
         "surfaces whose cells are too thin to judge"),
        ("dynamics_compared", "dynamics_compared",
         "within-patient properties graded - a patient's course over time"),
        ("dynamics_tracked", "dynamics_ok",
         "of those, how many land in band"),
        ("dynamics_own_compared", "dynamics_own_compared",
         "the same, on the operator's own columns"),
        ("dynamics_own_tracked", "dynamics_own_ok",
         "within-patient properties landing on the operator's own columns"),
        ("set_tokens_compared", "set_tokens_compared",
         "published set tokens whose share could be compared"),
        ("set_tokens_ok", "set_tokens_ok",
         "of those, how many land within 0.05"),
        ("set_tokens_never_emitted", "set_tokens_never_emitted",
         "tokens the contract published that appear in no generated row - a published promise the output does not keep"),
        ("set_empty_compared", "set_empty_compared",
         "set columns whose empty rate could be compared"),
        ("set_empty_ok", "set_empty_ok",
         "of those, how many land in band"),
        ("categorical_compared", "categorical_compared",
         "categorical pairs graded, which rank correlation cannot see"),
        ("categorical_kept", "categorical_kept",
         "of those, how many keep their association"),
)


def _load(run: str, name: str) -> Optional[Dict[str, Any]]:
    """One artifact, or None where the run does not have it.

    ABSENT IS NOT EMPTY. A run made before a field existed cannot
    answer a question about it, and returning `{}` would let a
    dashboard chart a zero for a measurement nobody took.
    """
    p = os.path.join(run, name)
    if not os.path.exists(p):
        return None
    try:
        return json.load(io.open(p, encoding="utf-8"))
    except (ValueError, OSError):
        return None


def _n(d: Optional[Dict[str, Any]], *path: str) -> Any:
    """A value down a path, or None - never a guessed default."""
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def patterns(bp: Optional[Dict[str, Any]],
             cat: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """What the fit found in the SOURCE.

    This is the half a reader asks "was the original data worth
    modelling" of, and it is about the SOURCE rather than about
    the output - so it belongs apart from the fidelity block even
    though both come from the same run.
    """
    rels = (bp or {}).get("relationships") or []
    curves = 0
    surfaces = 0
    for rel in rels:
        ev = (rel.get("evidence") or {})
        curves += len((ev.get("effect") or {}))
        if (ev.get("interaction") or {}).get("pair"):
            surfaces += 1
    cols = (bp or {}).get("columns") or {}
    # `tokens`, NOT `values`. The thirteenth wrong-key fault on
    # this project, and the first one caught by a check before it
    # reached anything: every token count read ABSENT, which is
    # indistinguishable from a cohort with no set column.
    tokens = found = above_k = 0
    set_cols = 0
    unpub = []
    for c in cols.values():
        m = (c or {}).get("marginal") or {}
        if m.get("type") != "list":
            continue
        set_cols += 1
        tokens += len(m.get("tokens") or [])
        found += int(m.get("tokens_found") or 0)
        above_k += int(m.get("tokens_above_k") or 0)
        if m.get("unpublishable_row_share") is not None:
            unpub.append(float(m["unpublishable_row_share"]))
    claims = (cat or {}).get("claims") or []
    skill = [c.get("skill") for c in claims
             if isinstance(c.get("skill"), (int, float))]
    return {
        "columns_published": len(cols) or None,
        "relationships": len(rels) or None,
        "effect_curves": curves or None,
        "interaction_surfaces": surfaces or None,
        "arithmetic_identities": len(
            (bp or {}).get("arithmetic_identities") or []),
        "constraints": len((bp or {}).get("constraints") or []),
        "set_columns": set_cols or None,
        "set_tokens_published": tokens or None,
        # WHAT THE PRIVACY RULE COSTS, which is the question a
        # governance reader asks of this section and which no
        # single number answered before: of the tokens the SOURCE
        # holds, how many cleared the patient floor and how many
        # were published. The review deck already shows this as
        # 9.6% for the extract's conditions; a dashboard wants it
        # as a metric rather than a slide.
        "set_tokens_in_source": found or None,
        "set_tokens_above_k": above_k or None,
        "set_tokens_published_share": (
            round(tokens / float(found), 4) if found else None),
        "set_rows_unpublishable_worst": (
            max(unpub) if unpub else None),
        "token_cooccurrence_pairs": len(
            (bp or {}).get("token_cooccurrence") or []) or None,
        "claims": len(claims) or None,
        # THE SKILL SPREAD, NOT A MEAN. One number over claims of
        # different strengths is the pooled-correlation fault this
        # project withdrew a conclusion for.
        "claim_skill_min": min(skill) if skill else None,
        "claim_skill_max": max(skill) if skill else None,
        "columns_considered": (cat or {}).get("columns_considered"),
        "identifiers_dropped": len(
            (cat or {}).get("identifiers_dropped") or []),
    }


def privacy(bp: Optional[Dict[str, Any]],
            prov: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The posture, in the contract's own words where it has them.

    SEPARATE FROM `series` ON PURPOSE. These are not metrics to
    improve run over run - k is a policy choice, and a chart of it
    beside a fidelity line invites somebody to read a lower k as
    progress.
    """
    pv = (bp or {}).get("privacy") or {}
    return {
        "k": pv.get("k", _n(prov, "privacy", "k")),
        "counted_in": pv.get("counted_in"),
        "numeric_bounds": pv.get("numeric_bounds"),
        "knots_are_k_anonymous": pv.get("knots_are_k_anonymous"),
        "NOT_a_formal_guarantee": pv.get("NOT_a_formal_guarantee"),
    }


def settings(prov: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """What was tweaked.

    AND WHAT THIS CANNOT ANSWER, said rather than implied. "How
    did that change the output" is a question about TWO runs, and
    one run cannot answer it however many settings it records -
    `scripts/arm_compare.py` is the tool, and it refuses a pair
    whose source, seed or engine differs because those make the
    two columns answers to different questions.
    """
    s = (prov or {}).get("settings") or {}
    return {
        "seed": s.get("seed"),
        "group_by": s.get("group_by"),
        "time_col_requested": s.get("time_col_requested"),
        "time_col_used": s.get("time_col_used"),
        "lags": s.get("lags"),
        "holdout": s.get("holdout"),
        "excluded": s.get("excluded"),
        "generate_args": s.get("generate_args"),
        "dials": (prov or {}).get("dials"),
        "comparison_needs_two_runs":
            "a setting's EFFECT is a question about two runs; "
            "run scripts/arm_compare.py RUN_A RUN_B --testing "
            "'what this pair measures'",
    }


def series(fid: Optional[Dict[str, Any]],
           verdict: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The flat scalars a dashboard charts, with stable names.

    EVERY VALUE IS A NUMBER, A BOOL OR None. A nested value here
    cannot be stacked across runs, and the check asserts it rather
    than the convention being remembered.

    THE COUNTS AND THEIR DENOMINATORS BOTH, never a percentage
    alone: this project has three recorded cases of a conclusion
    drawn from a share whose denominator had moved.
    """
    s = (fid or {}).get("summary") or {}
    out: Dict[str, Any] = {
        "gate_met": (None if verdict is None
                     else bool(verdict.get("met"))),
        "gate_criteria_passed": (
            None if verdict is None
            else sum(1 for c in verdict["criteria"] if c["ok"])),
        "gate_criteria_total": (
            None if verdict is None else len(verdict["criteria"])),
    }
    # NAME -> the summary field it reads. A table rather than a
    # block of lines, so a new metric cannot be added without a
    # name, and the check can walk it.
    for name, key, _why in _DIRECT:
        out[name] = s.get(key)
    if verdict is not None:
        out["tvd_accuracy"] = _n(verdict, "tvd", "accuracy")
        out["tvd_univariate"] = _n(verdict, "tvd", "univariate")
        out["tvd_bivariate"] = _n(verdict, "tvd", "bivariate")
        out["marginals_measured"] = _n(
            verdict, "marginals", "columns_measured")
        # COUNTS HERE, THE COLUMN LISTS IN THE SECTION BELOW.
        # `off_source`, `sampler_faults` and `no_power` are LISTS
        # of columns, and putting a list on the chart surface is
        # the exact thing `series` exists to prevent - caught by
        # this suite's own scalar check rather than by a reader.
        for _k in ("off_source", "sampler_faults", "no_power"):
            _v = _n(verdict, "marginals", _k)
            out["marginals_" + _k] = (
                None if _v is None else len(_v))
        out["marginals_worst"] = _n(
            verdict, "marginals", "worst_source_vs_output")
        out["bar_direction"] = _n(verdict, "bars", "direction")
        out["bar_close"] = _n(verdict, "bars", "close")
        out["bars_custom"] = _n(verdict, "bars", "custom")
    return out


def catalog() -> List[Dict[str, str]]:
    """Every charted metric with what it means, in one line each.

    THE LIST A DASHBOARD IS BUILT AGAINST. It is DERIVED from the
    same table `series` reads, so there is no second list to
    drift - this project has the hand-kept-list fault on record
    for a dependency table and for a dial list, both of which
    went stale the day somebody added an entry.

    The three gate scalars and the blocks read off the verdict
    are described here too, because a name a chart carries and a
    name only the code knows are different things to a person
    building a tab.
    """
    out = [
        {"name": "gate_met",
         "means": "did every gate criterion pass - the release "
                  "verdict, and it is full marks or fail by "
                  "design"},
        {"name": "gate_criteria_passed",
         "means": "how many of the criteria passed"},
        {"name": "gate_criteria_total",
         "means": "how many criteria were judged - the "
                  "denominator, which moves when a criterion "
                  "has nothing to grade"},
    ]
    for name, _key, why in _DIRECT:
        out.append({"name": name, "means": why})
    out += [
        {"name": "tvd_accuracy",
         "means": "the distance measure the commercial tools "
                  "publish - reported, never gated"},
        {"name": "tvd_univariate",
         "means": "mean total variation distance per column"},
        {"name": "tvd_bivariate",
         "means": "mean total variation distance over pairs - "
                  "this is what a shuffled column shows up in"},
        {"name": "marginals_measured",
         "means": "numeric columns compared against their own "
                  "published distribution"},
        {"name": "marginals_off_source",
         "means": "how many sit further than 0.05 from the "
                  "source - the count; the column NAMES are in "
                  "the fidelity section"},
        {"name": "marginals_sampler_faults",
         "means": "of those, how many are the sampler's rather "
                  "than the contract's - they take opposite "
                  "fixes"},
        {"name": "marginals_no_power",
         "means": "columns too thinly recorded to judge, which "
                  "is a statement about the cohort"},
        {"name": "marginals_worst",
         "means": "the worst column's distance from its own "
                  "source distribution"},
        {"name": "bar_direction",
         "means": "the direction bar this run was judged "
                  "against - policy, so a MET must name it"},
        {"name": "bar_close",
         "means": "the close bar this run was judged against"},
        {"name": "bars_custom",
         "means": "were either bar overridden - without this a "
                  "MET cannot be compared across runs"},
    ]
    return out


def collect(run: str) -> Dict[str, Any]:
    """The whole record for one run directory."""
    prov = _load(run, "provenance.json")
    fid = _load(run, "fidelity.json")
    bp = _load(run, "blueprint.json")
    cat = _load(run, "catalogue.json")
    verdict = None
    if fid is not None:
        try:
            verdict = _gate.assess(fid)
        except (KeyError, TypeError, ValueError):
            verdict = None
    return {
        "schema": SCHEMA,
        "run": {
            # THE DIRECTORY NAME IS THE ID OPERATORS USE, and
            # provenance does not carry one. Stamped with the
            # build and the time so a dashboard cannot show a
            # stale row as current.
            "id": os.path.basename(os.path.normpath(run)),
            "made_at": (prov or {}).get("made_at"),
            "build": (prov or {}).get("build"),
            # THE BUILD ID AS A STRING, beside the dict. `build`
            # is a DICT - {source, id, note} - and this project
            # has already nearly thrown mid-demo by slicing it as
            # a string. A dashboard wants one stable scalar to
            # label a row with, so it gets one.
            "build_id": _n(prov, "build", "id"),
            "code_fingerprint": (prov or {}).get("code_fingerprint"),
            "blueprint_version": (prov or {}).get(
                "blueprint_version"),
            "stack": (prov or {}).get("stack"),
            "source_name": _n(prov, "source", "name"),
            "source_digest": _n(prov, "source", "digest"),
            "rows_read": _n(prov, "source", "rows_read"),
            "columns_read": _n(prov, "source", "columns_read"),
            "patients": _n(prov, "source", "patients"),
            "generated_rows": _n(bp, "patients", "rows"),
            # A RUN WITH NO FIDELITY HAS NOT GENERATED. Said as a
            # field rather than left for a reader to infer from a
            # column of nulls.
            "generated": fid is not None,
        },
        "series": series(fid, verdict),
        "patterns": patterns(bp, cat),
        "fidelity": {
            "criteria": (None if verdict is None
                         else verdict["criteria"]),
            "met": (None if verdict is None
                    else bool(verdict.get("met"))),
            "marginals": (None if verdict is None
                          else verdict.get("marginals")),
            "surfaces": (None if verdict is None
                         else verdict.get("surfaces")),
            "dynamics_misses": _n(fid, "summary", "dynamics_misses"),
        },
        "settings": settings(prov),
        "privacy": privacy(bp, prov),
        "withheld": list(WITHHELD),
    }


def long_arrays(obj: Any, limit: int,
                path: str = "") -> List[str]:
    """Every array in the record longer than `limit`, by path.

    THE STRUCTURAL PRIVACY CHECK. A histogram, a row slice and a
    per-patient series all look like a long array of numbers, and
    banning them BY NAME would only catch the ones somebody
    thought to name. The limit is the column count plus slack,
    because a per-COLUMN list is legitimate and a per-ROW one is
    not.
    """
    bad: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            bad += long_arrays(v, limit, path + "/" + str(k))
    elif isinstance(obj, (list, tuple)):
        if len(obj) > limit:
            bad.append("{} ({} entries)".format(path, len(obj)))
        for i, v in enumerate(obj):
            bad += long_arrays(v, limit, path + "/" + str(i))
    return bad


def check_record(rec: Dict[str, Any]) -> None:
    """Raise if the record carries anything row-shaped.

    A PURE FUNCTION, so the refusal can be exercised on a planted
    record rather than waiting for a real one to go wrong - this
    project has four recorded cases of a decision asserted only
    as the sentence it printed.

    IT RAISES RATHER THAN WARNING. A record that grew to carry
    rows is a privacy problem, and writing it with a warning
    beside it puts it on the dashboard anyway.
    """
    ncols = (rec.get("patterns") or {}).get(
        "columns_published") or 0
    bad = long_arrays(rec, int(ncols) + ARRAY_SLACK)
    if bad:
        raise ValueError(
            "metrics record carries {} array(s) longer than the "
            "column count, which is what a row dump or a "
            "histogram looks like: {}".format(
                len(bad), ", ".join(bad[:4])))


def write(run: str, dest: Optional[str] = None) -> str:
    """Write `metrics.json` into the run directory."""
    rec = collect(run)
    check_record(rec)
    dest = dest or os.path.join(run, "metrics.json")
    io.open(dest, "w", encoding="utf-8").write(
        json.dumps(rec, indent=2, sort_keys=True))
    return dest
