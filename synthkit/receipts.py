"""One receipt per published claim, from artifacts a run already wrote.

WHY THIS IS NOT JUST `findings.txt` REARRANGED. The evidence for a
claim is spread across three files and two of them are prose. A
reader who wants to know whether ONE pattern is trustworthy has to
hold `catalogue.json` (skill, support, fragility), `blueprint.json`
(the effect shape, the importances, the k screening) and
`fidelity.json` (did it survive generation) in their head at once,
and then find the trim note for it in a page of text. A receipt is
that assembly done per claim, so a pattern can be audited one at a
time.

NOTHING HERE IS RECOMPUTED. Every number is read from the run's own
artifacts, because a second implementation is how the shipped
artifact and the report come to disagree - this project has recorded
that about token screening, about scaffolding, about the two
categorical draw paths and about a whole engine. If a number is not
in the artifacts, the receipt says it is not measured rather than
deriving it here.

AND THE LIMITS SECTION IS THE POINT. A receipt that only carried the
evidence would read as a certificate. Each one states what it does
NOT establish, per claim and from the measured record: association
rather than causation, one cohort rather than a population, a weak
skill called weak, a seed-fragile attribution named as fragile, a
sign that reverses when its co-parents are held fixed, and a claim
on a search column that cannot be checked in the output at all.
"""

import io
import json
import os
from typing import Any, Dict, List, Optional

from . import sets as _sets

# A claim whose out-of-sample skill is under this is reported as
# WEAK in its own limits. Not a filter - a weak claim is still a
# claim, and suppressing it would hide the fact that discovery
# found little. The bar is the same 0.30 the dashboard uses to
# decide a column is predictable at all, so the two cannot
# disagree about what "weak" means.
WEAK_SKILL = 0.30

# How far a generated rank correlation may sit from the source's
# before the survival verdict stops saying `kept`. This is the
# gate's own `close` bar, read from one place for the same reason.
CLOSE_DELTA = 0.20


def _load(rundir: str, name: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(rundir, name)
    if not os.path.exists(path):
        return None
    with io.open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _pairs_index(fid: Optional[Dict[str, Any]]) -> Dict:
    """Survival readings, keyed by (child, parent)."""
    out: Dict = {}
    if not fid:
        return out
    rel = fid.get("relationships") or {}
    for p in (rel.get("pairs") or []):
        out[(p.get("child"), p.get("parent"))] = p
    return out


def _verdict(pair: Optional[Dict[str, Any]]) -> str:
    if pair is None:
        return "not measured"
    src, gen = pair.get("source"), pair.get("generated")
    if src is None or gen is None:
        return "not measured"
    # A SIGN FLIP IS ITS OWN VERDICT, and the worst one: an
    # inverted relationship reads as a finding, where a missing
    # one reads as nothing. That is why the gate keeps INVERTED
    # absolute at every bar setting.
    if (src > 0) != (gen > 0) and min(abs(src), abs(gen)) > 0.05:
        return "INVERTED"
    if abs(gen - src) <= CLOSE_DELTA:
        return "kept"
    return "faded" if abs(gen) < abs(src) else "overshot"


def _effects(bp: Optional[Dict[str, Any]]) -> Dict:
    """The effect shape per (child, parent), in the run's words."""
    out: Dict = {}
    if not bp:
        return out
    for r in (bp.get("relationships") or []):
        child = r.get("child")
        ev = r.get("evidence") or {}
        for parent, eff in (ev.get("effect") or {}).items():
            if not isinstance(eff, dict):
                continue
            out[(child, parent)] = {
                "shape": eff.get("shape"),
                "description": eff.get("description"),
            }
    return out


def _importances(bp: Optional[Dict[str, Any]]) -> Dict:
    out: Dict = {}
    if not bp:
        return out
    for r in (bp.get("relationships") or []):
        ev = r.get("evidence") or {}
        out[r.get("child")] = (ev.get("importance") or {},
                               ev.get("importance_sd") or {})
    return out


def _limits(claim: Dict[str, Any], survival: List[Dict[str, Any]],
            k: Optional[int]) -> List[str]:
    """What this claim does NOT establish, from the record."""
    out = []
    # THE TWO THAT ALWAYS APPLY. The arrow is chosen by
    # out-of-sample skill, which says which direction PREDICTS
    # better - not which direction causes. And the holdout is
    # held-out PATIENTS from the same extract, so it tests
    # generalisation to new people in this cohort and nothing
    # about a different hospital, a different era or a
    # different inclusion rule.
    out.append(
        "ASSOCIATION, NOT CAUSATION. The direction was chosen "
        "because it predicts better out of sample, not because "
        "anything here knows which way the causation runs.")
    out.append(
        "ONE COHORT. The skill is measured on held-out PATIENTS "
        "from this same extract, so it speaks to new people in "
        "this cohort - not to another site, another era or "
        "another inclusion rule.")
    skill = claim.get("skill")
    if isinstance(skill, (int, float)) and skill < WEAK_SKILL:
        out.append(
            "WEAK CLAIM: out-of-sample skill {:.3f}, under the "
            "{:.2f} this project calls predictable. It is "
            "reported rather than suppressed, because a thin "
            "finding and no finding are different facts."
            .format(skill, WEAK_SKILL))
    frag = claim.get("attribution_fragile_parents") or []
    if frag:
        names = ", ".join(
            str(f.get("column", f)) if isinstance(f, dict) else str(f)
            for f in frag)
        out.append(
            "ATTRIBUTION IS SEED-FRAGILE for: {}. These parents "
            "sit near discovery's inclusion threshold, and which "
            "one leads has been measured to flip between seeds. "
            "Do not read the ordering of drivers here as "
            "stable.".format(names))
    if claim.get("near_deterministic"):
        out.append(
            "NEAR-DETERMINISTIC: the parents explain almost all "
            "of the child, so this is a DERIVATION somebody "
            "already knew rather than a discovery.")
    if any(s.get("reverses_when_controlled") for s in survival):
        out.append(
            "THE SIGN REVERSES when the other parents are held "
            "fixed. Both readings are true of the data and they "
            "answer different questions; a curve conditioned on "
            "co-parents has produced an inverted output here "
            "before.")
    scaff = [s["parent"] for s in survival
             if s.get("is_scaffolding")]
    if _sets.is_scaffolding(str(claim.get("child") or "")):
        out.append(
            "THE CHILD IS A SEARCH COLUMN. `{}` is built so the "
            "search can see a set's tokens and is DROPPED before "
            "the output file is written, so this claim cannot be "
            "checked in the generated data at all - nobody can "
            "read a column that is not there."
            .format(claim.get("child")))
    elif scaff:
        out.append(
            "SEARCH COLUMNS AMONG THE PARENTS: {}. They are "
            "dropped before the output is written, so their half "
            "of this claim is unverifiable in the generated "
            "file.".format(", ".join(scaff)))
    unmeasured = [s["parent"] for s in survival
                  if s["verdict"] == "not measured"
                  and not s.get("is_scaffolding")]
    if unmeasured:
        out.append(
            "SURVIVAL NOT MEASURED for: {}. The run related no "
            "pair for them, so whether generation kept this part "
            "is unknown rather than fine."
            .format(", ".join(unmeasured)))
    out.append(
        "TRIMS ARE NOT IN THIS RECEIPT. Whether a parent was cut "
        "to order a cycle and then re-applied by the refinement "
        "sweeps is recorded in `findings.txt` as prose and not "
        "as a field, so it is not asserted here. Trimmed is not "
        "the same as lost.")
    if k:
        out.append(
            "EVERY PUBLISHED NUMBER IS k-SCREENED AT k={}, "
            "counted in PATIENTS. A magnitude can fall short of "
            "its source because the privacy bound removed the "
            "tail, which is the rule working rather than a "
            "sampler fault.".format(k))
    return out


def build(rundir: str) -> Dict[str, Any]:
    """Assemble one receipt per claim from a run directory."""
    cat = _load(rundir, "catalogue.json")
    if cat is None:
        raise ValueError(
            "no catalogue.json in {} - receipts are assembled "
            "from a run's own artifacts and there is nothing to "
            "assemble".format(rundir))
    bp = _load(rundir, "blueprint.json")
    fid = _load(rundir, "fidelity.json")
    pairs = _pairs_index(fid)
    effects = _effects(bp)
    imps = _importances(bp)
    k = ((bp or {}).get("privacy") or {}).get("k")
    holdout = cat.get("holdout") or {}

    receipts = []
    for claim in (cat.get("claims") or []):
        child = claim.get("child")
        parents = [p.get("column") for p in
                   (claim.get("predictors") or [])
                   if p.get("column")]
        imp, imp_sd = imps.get(child, ({}, {}))
        survival = []
        for parent in parents:
            pair = pairs.get((child, parent))
            survival.append({
                "parent": parent,
                "source": (pair or {}).get("source"),
                "generated": (pair or {}).get("generated"),
                "delta": (pair or {}).get("delta"),
                "measure": (pair or {}).get("measure"),
                "verdict": _verdict(pair),
                "reverses_when_controlled":
                    bool((pair or {}).get(
                        "reverses_when_controlled")),
                "is_scaffolding":
                    _sets.is_scaffolding(str(parent)),
            })
        receipts.append({
            "claim": "{} <- {}".format(child, ", ".join(parents)),
            "child": child,
            "parents": parents,
            "kind": claim.get("kind"),
            "strength": {
                "skill_out_of_sample": claim.get("skill"),
                "importance": dict(imp),
                "importance_sd": dict(imp_sd),
            },
            "shape": dict(
                (p, effects.get((child, p), {})) for p in parents),
            "support": {
                "train_rows": claim.get("n_train_rows"),
                "holdout_rows": claim.get("n_holdout_rows"),
                "train_patients": claim.get("train_patients"),
                "holdout_patients": claim.get("holdout_patients"),
                "split_by": holdout.get("by"),
            },
            "survival": survival,
            "privacy": {
                "k": k,
                "counted_in": ((bp or {}).get("privacy")
                               or {}).get("counted_in"),
                "residual_bins_are_k_anonymous":
                    (claim.get("residual_spread") or {}).get(
                        "bins_are_k_anonymous"),
            },
            "limits": _limits(claim, survival, k),
        })

    verdicts: Dict[str, int] = {}
    for r in receipts:
        for sv in r["survival"]:
            verdicts[sv["verdict"]] = verdicts.get(
                sv["verdict"], 0) + 1
    return {
        "receipts": receipts,
        "claims": len(receipts),
        "survival_verdicts": verdicts,
        "scaffolding_claims": sum(
            1 for r in receipts
            if _sets.is_scaffolding(str(r["child"]))),
        "generated_compared": fid is not None,
        "note": (
            "One receipt per published claim, assembled from this "
            "run's own artifacts and recomputing nothing. Each "
            "states what it does NOT establish. A claim whose "
            "child is a search column cannot be checked in the "
            "output file at all, and says so."),
    }


def render(doc: Dict[str, Any]) -> str:
    """Plain text, one block per claim."""
    out = []
    out.append("RECEIPTS: {} claim(s)".format(doc["claims"]))
    if doc["survival_verdicts"]:
        out.append("survival across all parent pairs: " + ", ".join(
            "{} {}".format(v, kk) for kk, v
            in sorted(doc["survival_verdicts"].items())))
    if not doc["generated_compared"]:
        out.append(
            "NO fidelity.json in this run, so no survival was "
            "measured for any claim - absent, not clean.")
    if doc["scaffolding_claims"]:
        out.append(
            "{} claim(s) are on SEARCH columns, dropped before "
            "the output is written and so unverifiable there"
            .format(doc["scaffolding_claims"]))
    out.append("")
    for r in doc["receipts"]:
        out.append("=" * 68)
        out.append(r["claim"])
        sk = r["strength"]["skill_out_of_sample"]
        out.append("  skill out of sample: {}".format(
            "{:.4f}".format(sk)
            if isinstance(sk, (int, float)) else "not measured"))
        sup = r["support"]
        out.append(
            "  measured on {} train row(s) / {} patient(s), held "
            "out {} row(s) / {} patient(s), split by {}".format(
                sup["train_rows"], sup["train_patients"],
                sup["holdout_rows"], sup["holdout_patients"],
                sup["split_by"]))
        for p in r["parents"]:
            sh = r["shape"].get(p) or {}
            imp = r["strength"]["importance"].get(p)
            sd = r["strength"]["importance_sd"].get(p)
            line = "  {}: ".format(p)
            if imp is not None:
                line += "importance {:.4f}".format(imp)
                if sd is not None:
                    line += " (sd {:.4f})".format(sd)
            if sh.get("shape"):
                line += " shape {}".format(sh["shape"])
            out.append(line)
            if sh.get("description"):
                out.append("      {}".format(sh["description"]))
        for sv in r["survival"]:
            if sv["source"] is None:
                out.append("  survival {}: {}".format(
                    sv["parent"], sv["verdict"]))
            else:
                out.append(
                    "  survival {}: source {:+.4f} -> generated "
                    "{:+.4f}  {}".format(
                        sv["parent"], sv["source"],
                        sv["generated"], sv["verdict"]))
        out.append("  WHAT THIS DOES NOT ESTABLISH")
        for lim in r["limits"]:
            out.append("    - {}".format(lim))
    return "\n".join(out)
