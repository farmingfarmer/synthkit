"""Compile a PROFILE into a draft synthkit spec — the bridge from
"what the real data looks like" to "a recipe a human can edit".

    python scripts/profile_to_spec.py --profile profile.json
                                      -o draft_spec.json
                                      [--rows N] [--title T]

What crosses the bridge: fitted distributions, categorical weights
(rare levels already suppressed), missing rates, date ranges, and
CONFIRMED correlations. What does NOT cross: any record, any value,
any unconfirmed relationship.

What is deliberately left for the human: the OUTCOME. Planted truth
is an authoring decision, not something to be inferred from data —
it is what keeps the answer key exact and the ceiling knowable. The
draft therefore lands with an empty `outcomes` array and a note
saying so; it goes through the same review gate as any other spec.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def numeric_dist(p):
    fam, pr = p["family"], p["params"]
    if fam == "lognormal":
        base = {"kind": "lognormal", "mu": pr["mu"],
                "sigma": pr["sigma"]}
    else:
        base = {"kind": "normal", "mean": pr["mean"],
                "std": pr["sd"]}
    zi = p.get("zero_inflation")
    if zi:
        # a zero-inflated column is a mixture of "exactly zero" and
        # the fitted body — modelled explicitly rather than smeared
        return {"kind": "mixture",
                "weights": [round(zi, 4), round(1 - zi, 4)],
                "components": [
                    {"kind": "normal", "mean": 0.0, "std": 0.0001},
                    base]}
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("-o", "--out", default="draft_spec.json")
    ap.add_argument("--rows", type=int, default=0)
    ap.add_argument("--title", default="")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    P = json.loads(Path(a.profile).read_text(encoding="utf-8"))
    rows = a.rows or P["source"]["rows"]
    cols, deferred, notes = [], [], []

    for name, p in P["columns"].items():
        kind = p["kind"]
        mess = {}
        mr = p.get("missing_rate", 0.0)
        if mr > 0.001:
            mess["missing_rate"] = round(mr, 4)
        if p.get("outlier_rate"):
            mess["outlier_rate"] = p["outlier_rate"]
            if p.get("outlier_factor"):
                mess["outlier_factor"] = p["outlier_factor"]
        if kind == "numeric":
            col = {"name": name,
                   "ctype": "int" if p.get("integer_valued")
                   else "float",
                   "distribution": numeric_dist(p)}
            if p.get("warning"):
                notes.append("{}: {}".format(name, p["warning"]))
        elif kind == "categorical":
            levels = list(p["weights"].keys())
            col = {"name": name, "ctype": "category",
                   "distribution": {
                       "kind": "categorical",
                       "choices": levels,
                       "weights": [p["weights"][l]
                                   for l in levels]}}
            if p.get("suppressed_levels"):
                notes.append(
                    "{}: {} rare level(s) were suppressed into "
                    "OTHER_SUPPRESSED before this draft — rename or "
                    "redistribute that weight as you see fit"
                    .format(name, p["suppressed_levels"]))
        elif kind == "binary":
            col = {"name": name, "ctype": "bool",
                   "distribution": {"kind": "bernoulli",
                                    "p": p["rate"]}}
        elif kind == "date":
            dist = {"kind": "date_range", "start": p["start"],
                    "end": p["end"]}
            if p.get("bucket_weights"):
                dist["weights"] = p["bucket_weights"]
            col = {"name": name, "ctype": "date",
                   "distribution": dist}
        elif kind == "list":
            deferred.append({
                "name": name, "reason": "list-valued column",
                "measured": {
                    "mean_length": p["mean_length"],
                    "empty_rate": p["empty_rate"],
                    "distinct_items": p["distinct_items"],
                    "item_rates": p["item_rates"],
                    "cooccurrence": p.get("cooccurrence", [])[:10]},
                "how_to_use": "author this as a `note` column: each "
                              "frequent item becomes an element with "
                              "phrasings and a density equal to its "
                              "measured rate; co-occurring pairs can "
                              "share an element or become a "
                              "distractor pair"})
            continue
        elif kind == "constant":
            notes.append("{}: single value in the source — omitted "
                         "as uninformative".format(name))
            continue
        else:
            continue
        if mess:
            col["mess"] = mess
        cols.append(col)

    corr = [{"a": c["a"], "b": c["b"], "spearman": c["spearman"]}
            for c in P["joint"]["numeric_correlations"]
            if c.get("confirmed") and not c.get("redundant")]
    dropped = [c for c in P["joint"]["numeric_correlations"]
               if c.get("confirmed") and c.get("redundant")]
    colnames = {c["name"] for c in cols}
    rules = []
    for c in dropped:
        prop = c.get("proportional")
        if prop and prop["target"] in colnames \
                and prop["source"] in colnames:
            rules.append({"kind": "derived",
                          "target": prop["target"],
                          "source": prop["source"],
                          "factor": prop["factor"],
                          "noise_sigma": prop["noise_sigma"]})
            notes.append(
                "{} <-> {}: near-perfect ({:+.3f}) because one is a "
                "multiple of the other — restored as a DERIVED rule "
                "({} = {} x {}), not as a correlation"
                .format(c["a"], c["b"], c["spearman"],
                        prop["target"], prop["source"],
                        prop["factor"]))
        elif c.get("proportional_declined"):
            notes.append(
                "{} <-> {}: near-perfect ({:+.3f}) but NOT restored "
                "as a rule — {}"
                .format(c["a"], c["b"], c["spearman"],
                        c["proportional_declined"]))
        else:
            notes.append(
                "{} <-> {}: correlation {:+.3f} is near-perfect but "
                "not a simple multiple — these encode the same "
                "quantity by some other formula; author a rule if "
                "the relationship matters"
                .format(c["a"], c["b"], c["spearman"]))

    spec = {
        "title": a.title or "profiled from {}".format(
            P["source"]["file"]),
        "rows": rows,
        "master_seed": 20260730,
        "columns": cols,
        "rules": rules,
        "correlations": corr,
        "outcomes": [],
        "_provenance": {
            "profiled_from": P["source"]["file"],
            "source_rows": P["source"]["rows"],
            "k_threshold": P["privacy"]["k_threshold"],
            "correlations_imposed": len(corr),
            "derived_rules_emitted": len(rules),
            "correlations_available_unconfirmed": sum(
                1 for c in P["joint"]["numeric_correlations"]
                if not c.get("confirmed")),
            "privacy": P["privacy"]["contract"],
            "outcomes": "EMPTY BY DESIGN — planted truth is "
                        "authored, not inferred. Add an outcome "
                        "with known coefficients over these "
                        "columns; that is what makes the answer key "
                        "exact and the performance ceiling "
                        "knowable.",
            "review": "This is a DRAFT. Every number here is a "
                      "dial: change any distribution, weight, mess "
                      "rate or correlation before compiling.",
            "editing_notes": notes,
            "deferred_columns": deferred,
        },
    }
    Path(a.out).write_text(json.dumps(spec, indent=1),
                           encoding="utf-8")
    print("WROTE {} : {} columns, {} correlations imposed, "
          "{} deferred".format(a.out, len(cols), len(corr),
                               len(deferred)))
    if a.report:
        print("\n--- DRAFT SPEC (every value is editable) ---")
        for c in cols:
            d = c["distribution"]
            k = d["kind"]
            if k == "categorical":
                desc = "categorical over {} levels".format(
                    len(d["choices"]))
            elif k == "mixture":
                desc = "mixture ({} components, zero-inflated)" \
                    .format(len(d["components"]))
            elif k == "bernoulli":
                desc = "bernoulli p={}".format(d["p"])
            elif k == "date_range":
                desc = "dates {} .. {}".format(d["start"], d["end"])
            elif k == "lognormal":
                desc = "lognormal mu={} sigma={}".format(
                    d["mu"], d["sigma"])
            else:
                desc = "normal mean={} std={}".format(
                    d["mean"], d["std"])
            m = c.get("mess", {})
            print("  {:26s} {}{}".format(
                c["name"], desc,
                "   missing {:.1%}".format(m["missing_rate"])
                if m.get("missing_rate") else ""))
        print("\n  correlations imposed: {}".format(
            ", ".join("{}~{} {:+.2f}".format(
                c["a"], c["b"], c["spearman"]) for c in corr)
            or "none confirmed in the source"))
        if deferred:
            print("  deferred for authoring: {}".format(
                ", ".join(d["name"] for d in deferred)))
        for n in notes[:8]:
            print("  NOTE  {}".format(n))
        print("\n  outcomes: EMPTY — author the planted truth, then "
              "compile.")


if __name__ == "__main__":
    main()
