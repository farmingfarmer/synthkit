"""What the blueprint declares, the generated frame must obey.

WHY THIS EXISTS. On 2026-08-19 a real extract's generated file held
patients whose year of birth changed between their own visits. The
blueprint DECLARED that column patient-level; generation drew it once
per patient, correctly; and a later pass put visit-to-visit variation
back into it. Every fidelity number stayed green, because fidelity
measures RESEMBLANCE - coverage, center, spread, steadiness - and a
column can resemble its source in every aggregate while violating a
property the source holds on every single row. The defect was caught
by a person reading two numbers off a photograph, one round trip too
late.

So this pass checks OBEDIENCE rather than resemblance: every property
the blueprint states about a column is asserted against the frame that
came out. No source data is needed - the blueprint is the contract,
and that is what makes the check cheap enough to run on every
generation, everywhere, including machines that hold no extract.

WHAT A RULE HERE MAY BE: a property the blueprint DECLARES. Not a
property inferred from the output, and not a resemblance judgement -
those belong to fidelity. The declarations checked:

  level: patient   one value per patient, on the rows where the
                   column is present at all
  integral         whole numbers come out whole
  quantile bounds  the published k-anonymous bound is the k rule's
                   PROMISE - a value beyond it publishes something
                   the rule said nobody's data would

A column whose dials MOVED it (shift, scale) is exempt from the
bounds check and says so: the operator asked for the move, and the
dial report already carries requested-against-achieved."""
from __future__ import annotations

from typing import Any, Dict, List

# Rounding at the bound itself must not fire the check: the sampler
# may emit the bound exactly, and an integral column rounds onto it.
TOL = 1e-6


def find(frame, bp: Dict[str, Any], group_by: str
         ) -> List[Dict[str, Any]]:
    """Every declared property the frame violates."""
    import numpy as np
    import pandas as pd

    out: List[Dict[str, Any]] = []
    cols = bp.get("columns") or {}
    for name, spec in cols.items():
        if name not in frame.columns:
            continue
        # Scaffolding is derived and dropped before the file is
        # written; anything still present is the operator's own.
        m = spec.get("marginal") or {}
        numeric = spec.get("kind") == "numeric"
        v = pd.to_numeric(frame[name], errors="coerce") if numeric \
            else None

        if spec.get("level") == "patient" and group_by in frame:
            per = frame[frame[name].notna()].groupby(group_by)[name] \
                .nunique(dropna=True)
            if len(per):
                share = float((per <= 1).mean())
                if share < 0.999:
                    worst = int(per.max())
                    out.append({
                        "column": name, "declared": "level: patient",
                        "violated": "{:.1%} of patients hold more "
                                    "than one value; one patient "
                                    "holds {}".format(1 - share,
                                                      worst),
                        "note": "a fact about the person changed "
                                "between their visits. Every "
                                "aggregate can still pass - this is "
                                "obedience, not resemblance."})

        if numeric and m.get("integral") and v is not None:
            x = v.dropna()
            if len(x):
                off = float((np.abs(x - np.round(x)) > TOL).mean())
                if off > 0:
                    out.append({
                        "column": name, "declared": "integral",
                        "violated": "{:.1%} of values are not whole "
                                    "numbers".format(off),
                        "note": "the marginal declares whole "
                                "numbers; a fraction here reads as a "
                                "measurement precision the source "
                                "never had."})

        if numeric and m.get("type") == "quantiles" and v is not None:
            d = spec.get("dials") or {}
            moved = (d.get("shift") not in (None, "n/a", 0, 0.0)
                     or d.get("scale") not in (None, "n/a", 1, 1.0))
            vv = m.get("v") or []
            if not moved and len(vv) >= 2:
                lo, hi = float(vv[0]), float(vv[-1])
                x = v.dropna()
                if len(x):
                    n_out = int(((x < lo - TOL)
                                 | (x > hi + TOL)).sum())
                    if n_out:
                        out.append({
                            "column": name,
                            "declared": "published bound [{:g}, "
                                        "{:g}]".format(lo, hi),
                            "violated": "{} value(s) outside it, "
                                        "worst {:g}".format(
                                            n_out,
                                            float(x.max()) if
                                            float(x.max()) > hi
                                            else float(x.min())),
                            "note": "the bound is the k rule's "
                                    "PROMISE: the mean of the k most "
                                    "extreme patients' own extremes. "
                                    "A value beyond it publishes "
                                    "what the rule said nobody's "
                                    "data would."})
    return out


def render(hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return ""
    L = ["", "", "THE GENERATED DATA DISOBEYS ITS OWN BLUEPRINT",
         "-" * 60,
         "Each line is a property the blueprint DECLARES that the "
         "frame violates.",
         "These are not resemblance findings - a column can pass "
         "every fidelity",
         "check and still break a property the source holds on every "
         "row.", ""]
    for h in hits:
        L.append("    {}".format(h["column"]))
        L.append("        declared {}".format(h["declared"]))
        L.append("        violated: {}".format(h["violated"]))
        L.append("        {}".format(h["note"]))
        L.append("")
    return "\n".join(L)
