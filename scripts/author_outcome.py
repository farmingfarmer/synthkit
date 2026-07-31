"""Author planted truth onto a profiled draft spec.

    python scripts/author_outcome.py --draft draft_spec.json
                                     --profile profile.json
                                     -o final_spec.json
                                     [--note-from COL --note-elements N]
                                     [--outcome NAME --intercept X]
                                     [--coef "col=weight" ...]
                                     [--note-weight "element=w" ...]
                                     [--prevalence lo,hi] [--report]

The division of labour matters here and is deliberate.

  MEASURED (this tool does it for you): which items appear in a
  list-valued column and how often — turning `active_drugs` into a
  proper note column whose elements carry their real densities.
  That is arithmetic on the source, so automating it is safe.

  AUTHORED (you must state it): which of those elements CAUSE the
  outcome and how strongly, plus the intercept and the prevalence
  band. Nothing infers this from the data — inferring it would
  make the answer key a guess, and the whole point of a planted
  outcome is that the ceiling is exact and known before any model
  is tested.

Weights left unstated default to 0.0, i.e. "present in the text
but carries no signal" — a decoy, not a cause.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:                       # console-safe on Windows terminals
    from _console import console_safe
    console_safe()
except Exception:
    pass


def snake(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") \
        or "item"


def phrasings_for(item):
    """A few natural ways a clinician might write the same fact."""
    return ["patient on {}".format(item),
            "continues {} as prescribed".format(item),
            "{} — dose unchanged".format(item),
            "started on {} this admission".format(item)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("-o", "--out", default="final_spec.json")
    ap.add_argument("--note-from", action="append", default=[],
                    help="deferred list column to become a note")
    ap.add_argument("--note-elements", type=int, default=8)
    ap.add_argument("--outcome", default="")
    ap.add_argument("--kind", default="logistic")
    ap.add_argument("--intercept", type=float, default=0.0)
    ap.add_argument("--coef", action="append", default=[],
                    help='"column=weight", repeatable')
    ap.add_argument("--note-weight", action="append", default=[],
                    help='"element_id=weight", repeatable')
    ap.add_argument("--prevalence", default="")
    ap.add_argument("--drop", action="append", default=[],
                    help="column to remove (e.g. a leaky one)")
    ap.add_argument("--calibrate", action="store_true",
                    help="solve the intercept so realized "
                         "prevalence lands in the declared band")
    ap.add_argument("--calibrate-rows", type=int, default=1500)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    spec = json.loads(Path(a.draft).read_text(encoding="utf-8"))
    P = json.loads(Path(a.profile).read_text(encoding="utf-8"))
    prov = spec.get("_provenance", {})
    deferred = {d["name"]: d for d in prov.get(
        "deferred_columns", [])}
    note_weights = {}
    for nw in a.note_weight:
        k, v = nw.split("=", 1)
        note_weights[k.strip()] = float(v)

    planted, authored_notes = [], []
    for colname in a.note_from:
        if colname not in deferred:
            sys.exit("`{}` is not a deferred list column in this "
                     "draft (available: {})".format(
                         colname, ", ".join(deferred) or "none"))
        meas = deferred[colname]["measured"]
        rates = meas["item_rates"]
        top = sorted(rates.items(), key=lambda kv: -kv[1])[
            :a.note_elements]
        elements = []
        for item, rate in top:
            eid = snake(item)
            elements.append({
                "id": eid,
                "density": round(float(rate), 4),
                "weight": note_weights.get(eid, 0.0),
                "phrasings": phrasings_for(item)})
            if note_weights.get(eid):
                planted.append(("{}.{}".format(colname, eid),
                                note_weights[eid], "note element"))
        # a denial trap on the strongest planted element: text that
        # looks extractable and is wrong to extract
        strongest = max(
            (e for e in elements), key=lambda e: abs(e["weight"]),
            default=None)
        distractors = []
        if strongest and strongest["weight"]:
            distractors.append({
                "id": "denies_" + strongest["id"],
                "density": 0.5,
                "excludes": strongest["id"],
                "phrasings": [
                    "denies taking any {}".format(
                        strongest["id"].replace("_", " ")),
                    "no current {} on the list".format(
                        strongest["id"].replace("_", " ")),
                    "{} discontinued prior to admission".format(
                        strongest["id"].replace("_", " "))]})
        spec["columns"].append({
            "name": colname, "ctype": "note",
            "note": {"elements": elements,
                     "distractors": distractors,
                     "fillers": [
                         "Vital signs reviewed at discharge",
                         "Medication list reconciled with pharmacy",
                         "Follow-up arranged with primary care",
                         "Patient education completed",
                         "No acute distress on examination"]}})
        authored_notes.append((colname, len(elements),
                               len(distractors)))

    for d in a.drop:
        spec["columns"] = [c for c in spec["columns"]
                           if c["name"] != d]
        spec["rules"] = [r for r in spec.get("rules", [])
                         if r.get("target") != d
                         and r.get("source") != d]
        spec["correlations"] = [c for c in
                                spec.get("correlations", [])
                                if d not in (c["a"], c["b"])]

    if a.outcome:
        names = {c["name"] for c in spec["columns"]}
        coefs = {}
        for c in a.coef:
            k, v = c.split("=", 1)
            k = k.strip()
            base = k.split("=")[0].split(".")[0]
            if base not in names:
                sys.exit("coefficient `{}` names no column in the "
                         "spec".format(k))
            coefs[k] = float(v)
            planted.append((k, float(v), "column"))
        for colname, _, _ in authored_notes:
            for e in next(c for c in spec["columns"]
                          if c["name"] == colname)["note"][
                              "elements"]:
                if e["weight"]:
                    coefs["{}.{}".format(colname, e["id"])] = \
                        e["weight"]
        oc = {"name": a.outcome, "kind": a.kind,
              "intercept": a.intercept, "coefficients": coefs}
        if a.prevalence:
            lo, hi = [float(x) for x in a.prevalence.split(",")]
            oc["target_prevalence"] = [lo, hi]
        spec["outcomes"] = [oc]

    calib = None
    if a.calibrate and not (a.outcome and a.prevalence):
        print("  --calibrate needs both --outcome and "
              "--prevalence; skipping calibration")
    if a.outcome and a.calibrate and a.prevalence:
        # The COEFFICIENTS are causal claims and stay exactly as
        # authored. The INTERCEPT is only a base-rate knob, so
        # solving it to hit the declared prevalence band is what
        # the declaration already means. Bisection on a sample.
        sys.path.insert(0, str(Path(__file__).resolve().parent
                               .parent))
        from synthkit.tablespec import TableSpec
        from synthkit.tableplan import plan_table
        lo_t, hi_t = spec["outcomes"][0]["target_prevalence"]
        want = (lo_t + hi_t) / 2.0

        def realized(b):
            probe = {k: v for k, v in spec.items()
                     if not k.startswith("_")}
            probe = json.loads(json.dumps(probe))
            probe["rows"] = a.calibrate_rows
            probe["outcomes"][0]["intercept"] = b
            probe["outcomes"][0].pop("target_prevalence", None)
            tr = plan_table(TableSpec.from_json(json.dumps(probe)))
            n = len(tr.clean_rows)
            return sum(1 for r in tr.clean_rows
                       if r[a.outcome] == "True") / n

        # 16 bisection steps resolve the intercept to ~0.0008,
        # far finer than the +/-0.8% sampling noise a 1,500-row
        # probe carries at 10% prevalence. More steps would buy
        # precision the measurement cannot see.
        steps = 16
        lo_b, hi_b = -25.0, 25.0
        print("  calibrating intercept to hit {:.0%}-{:.0%} "
              "({} probes of {} rows)...".format(
                  lo_t, hi_t, steps, a.calibrate_rows))
        for i in range(steps):
            mid = (lo_b + hi_b) / 2.0
            got_i = realized(mid)
            if i % 4 == 3 or i == steps - 1:
                print("    step {:2d}/{}  intercept {:+7.3f} -> "
                      "{:5.1%}".format(i + 1, steps, mid, got_i),
                      flush=True)
            if got_i < want:
                lo_b = mid
            else:
                hi_b = mid
        best = round((lo_b + hi_b) / 2.0, 4)
        spec["outcomes"][0]["intercept"] = best
        got = realized(best)
        calib = {"requested_band": [lo_t, hi_t],
                 "solved_intercept": best,
                 "realized_on_probe": round(got, 4),
                 "in_band": lo_t <= got <= hi_t,
                 "note": "coefficients unchanged — only the base "
                         "rate was solved"}
        print("  calibrated intercept {:+.4f} -> prevalence "
              "{:.1%} (band {:.0%}-{:.0%}) {}".format(
                  best, got, lo_t, hi_t,
                  "OK" if calib["in_band"] else "STILL OUT"))

    prov["outcomes"] = ("AUTHORED by author_outcome.py — the "
                        "coefficients below are planted truth, "
                        "stated by a human, not inferred from the "
                        "source data. That is what makes the "
                        "answer key exact and the ceiling known.")
    prov["authored"] = {
        "note_columns": [{"column": c, "elements": n,
                          "distractors": d}
                         for c, n, d in authored_notes],
        "planted_terms": [{"term": t, "weight": w, "kind": k}
                          for t, w, k in planted],
        "measured_vs_authored": "densities and item rates were "
                                "MEASURED from the source; weights "
                                "and the prevalence band were "
                                "AUTHORED; the intercept may be "
                                "SOLVED to hit that band",
        "calibration": calib,
    }
    spec["_provenance"] = prov
    Path(a.out).write_text(json.dumps(spec, indent=1),
                           encoding="utf-8")
    print("WROTE {} : {} columns, {} note column(s), outcome `{}`"
          .format(a.out, len(spec["columns"]),
                  len(authored_notes), a.outcome or "NONE"))

    if a.report:
        for colname, n, d in authored_notes:
            print("\n  note column `{}` — {} elements from measured "
                  "rates, {} denial trap(s)".format(colname, n, d))
            for e in next(c for c in spec["columns"]
                          if c["name"] == colname)["note"][
                              "elements"]:
                mark = ("SIGNAL w={:+.2f}".format(e["weight"])
                        if e["weight"] else "decoy  (no signal)")
                print("      {:28s} appears {:>5.1%}   {}".format(
                    e["id"], e["density"], mark))
        if a.outcome:
            final_b = spec["outcomes"][0]["intercept"]
            print("\n  PLANTED TRUTH for `{}` (intercept {:+.3f}{})"
                  .format(a.outcome, final_b,
                          " — solved to hit the declared "
                          "prevalence" if calib else
                          " — as stated"))
            for t, w, k in planted:
                print("      {:+.3f}   {:36s} [{}]".format(
                    w, t, k))
            print("\n  Everything above was stated by a human. The "
                  "densities were measured; the CAUSES were not.")


if __name__ == "__main__":
    main()
