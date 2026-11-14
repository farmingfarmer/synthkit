"""Smoke: rank a synthetic column by a model fitted on real data.

TWO PROPERTIES CARRY THIS TOOL and both are checks below.

THE MODEL MAY NEVER LEAVE. A boosted tree fitted on real rows has
leaves defined by a handful of real patients, which is what the k
rule exists to prevent - and this project has measured that
surface at 0.776 for a weights-leak adversary against 0.514-0.551
output-only. So nothing the tool writes may contain a parameter of
it.

AND A MODEL WITH NO SKILL ORDERS BY NOISE. Arranging a column by
noise destroys whatever the engine had right while reporting that
something was done, which is worse than doing nothing - so the
refusal is load-bearing and gets its own fixture.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                              # noqa: E402
import pandas as pd                             # noqa: E402

PY_ = sys.executable
MR = str(ROOT / "scripts" / "model_rank.py")
PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def run(*a, timeout=900):
    r = subprocess.run([PY_, "-u"] + [str(x) for x in a],
                       capture_output=True, text=True,
                       timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def fixture(path, seed=7, noisy=False):
    """A count driven by the other columns - or, with `noisy`, a
    column driven by NOTHING, which is the shape the refusal
    exists for and which a predictable fixture cannot reach."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(220):
        base = float(rng.normal(0, 1))
        for v in range(int(rng.integers(3, 9))):
            cc = int(max(0, rng.poisson(max(0.2, 3 + 1.2 * base))))
            if noisy:
                pq = int(rng.integers(1, 9))
            else:
                # the signal has to clear the skill FLOOR with
                # room, or the tool refuses and every check
                # below it measures the refusal instead
                lam = float(np.exp(0.55 * base
                                   + 0.30 * (cc - 3)))
                pq = int(rng.poisson(
                    max(0.05, min(40.0, lam)))) + 1
            # THE TARGET IS SPARSE, like the real one at 19%
            # coverage. A fully populated fixture cannot reach
            # the coverage guard at all - a mutation that
            # arranges absent rows too then changes nothing.
            recorded = rng.random() < 0.45
            rows.append(dict(
                person_id="P%03d" % p, visit_number=v + 1,
                age_at_visit=int(60 + 12 * base
                                 + rng.normal(0, 4)),
                condition_count=cc,
                spo2=round(float(rng.normal(96, 2)), 1),
                procedure_quantity=(pq if recorded else "")))
    pd.DataFrame(rows).to_csv(path, index=False)


def fit(td, src, out):
    # `run` already prefixes the interpreter - passing it again
    # ran python ON ITSELF and the fit silently produced nothing.
    return run("-m", "synthkit.cli", "fit", "--src", src,
               "--out", out, "--group-by", "person_id",
               "--generate")


def unit_checks():
    """Two properties the end-to-end run cannot reach."""
    import tempfile as _tf
    from model_rank import fit_ranker, model_leak

    # THE PARAMETER TRIPWIRE. The dict the tool assembles is
    # built from scalars, so the guard cannot fire on it - which
    # makes an inline loop unreachable by any mutation. Tested on
    # a PLANTED dict instead, so it is load-bearing.
    check("a written record carrying a model parameter is "
          "CAUGHT by name",
          model_leak({"ok": 1, "m": {"tree_": [1, 2]}})
          == "tree_")
    check("...and an ordinary record is not flagged",
          model_leak({"column": "procedure_quantity",
                      "skill": 0.42}) is None)

    # THE SPLIT IS BY PATIENT. Splitting ROWS puts the same
    # person on both sides, so the skill measured is the same
    # records twice and reads HIGH - this project records that
    # fault for the close ceiling. The fixture's signal is
    # patient-level, so the inflation is measurable.
    with _tf.TemporaryDirectory() as td:
        src = Path(td) / "s.csv"
        fixture(src)
        real = pd.read_csv(src, dtype=str, keep_default_na=False)
        _, _, by_pat, _ = fit_ranker(real, "procedure_quantity",
                                     group_by="person_id", seed=0)
        _, _, by_row, _ = fit_ranker(real, "procedure_quantity",
                                     group_by=None, seed=0)
        check("a ROW split reports a higher skill than a PATIENT "
              "split on patient-level signal, which is why the "
              "split is by patient: row {:+.4f} against patient "
              "{:+.4f}".format(by_row, by_pat),
              by_row > by_pat)


def main():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "src.csv"
        fixture(src)
        rd = td / "run"
        rc, log = fit(td, src, rd)
        okfit = rc == 0 and (rd / "generated.csv").exists()
        check("a fitted run exists to rank against{}".format(
            "" if okfit else " -- LOG: " + log[-500:]), okfit)
        if not okfit:
            print()
            print("1 of 1 checks failed.")
            sys.exit(1)

        rc, out = run(MR, rd, "--src", src, "--column",
                      "procedure_quantity", "--group-by",
                      "person_id")
        check("the pass runs and reports its held-out skill on "
              "real PATIENTS held back - splitting rows would "
              "put the same person on both sides",
              rc == 0 and "held-out skill on real PATIENTS" in out)
        check("it says the model is discarded, in the run's own "
              "output rather than only in a docstring",
              "discarded" in out and "column of numbers" in out)
        check("it re-measures the WHOLE gate, not only the "
              "column it aims at",
              "THE WHOLE GATE" in out and "direction kept" in out)
        out_model = out
        check("nothing is written without --apply",
              "nothing written" in out
              and not (Path(str(rd) + "_modelrank")).exists())

        # --- the values come from the PUBLISHED grid, and
        # coverage cannot move: the pass touches only rows the
        # generated file already marks present
        rc, out = run(MR, rd, "--src", src, "--column",
                      "procedure_quantity", "--group-by",
                      "person_id", "--apply")
        od = Path(str(rd) + "_modelrank")
        check("--apply writes a drop-in run directory",
              rc == 0 and (od / "generated.csv").exists()
              and (od / "blueprint.json").exists())
        g0 = pd.read_csv(rd / "generated.csv", dtype=str,
                         keep_default_na=False)
        g1 = pd.read_csv(od / "generated.csv", dtype=str,
                         keep_default_na=False)
        check("the ORIGINAL run is untouched",
              (g0["procedure_quantity"]
               != g1["procedure_quantity"]).any()
              and len(g0) == len(g1))
        pres = lambda s: (~s.astype(str).str.strip().isin(
            ("", "nan", "None"))).mean()
        check("coverage is IDENTICAL - the pass arranges, it does "
              "not decide which rows are present: {:.6f} against "
              "{:.6f}".format(pres(g0["procedure_quantity"]),
                              pres(g1["procedure_quantity"])),
              abs(pres(g0["procedure_quantity"])
                  - pres(g1["procedure_quantity"])) < 1e-12)
        others = [c for c in g0.columns
                  if c != "procedure_quantity"]
        check("every OTHER column is byte-identical, so the pass "
              "acts only where its mechanism lives",
              all(g0[c].equals(g1[c]) for c in others))

        # --- THE MODEL MUST NOT LEAVE
        blob = (od / "fidelity.json").read_text(encoding="utf-8")
        leaked = [w for w in ("tree_", "coef_", "_predictor",
                              "nodes", "estimators_")
                  if w in blob]
        check("the written fidelity carries NO model parameter: "
              "{}".format(leaked or "none"),
              not leaked)
        fid = json.loads(blob)
        rem = fid.get("remeasured") or {}
        check("...and it SAYS the model was not published, so a "
              "reader does not have to infer it",
              rem.get("model_published") is False
              and "discarded" in (rem.get("note") or ""))
        check("...and records the skill the arrangement rested on",
              isinstance(rem.get(
                  "model_skill_on_held_out_real_patients"),
                  float))

        # --- --order-only is a TRUE permutation
        rc, out = run(MR, rd, "--src", src, "--column",
                      "procedure_quantity", "--group-by",
                      "person_id", "--order-only", "--apply",
                      "--out", td / "oo")
        oo = pd.read_csv(td / "oo" / "generated.csv", dtype=str,
                         keep_default_na=False)
        m0 = sorted(pd.to_numeric(g0["procedure_quantity"],
                                  errors="coerce").dropna())
        m1 = sorted(pd.to_numeric(oo["procedure_quantity"],
                                  errors="coerce").dropna())
        check("--order-only holds the IDENTICAL multiset, so the "
              "marginal cannot move and the privacy posture is "
              "the repair tool's",
              rc == 0 and m0 == m1)

        # --- THE REFUSAL, on a column driven by nothing
        nsrc = td / "noise.csv"
        fixture(nsrc, seed=3, noisy=True)
        nd = td / "nrun"
        rc, _ = fit(td, nsrc, nd)
        check("a run exists whose target is driven by nothing",
              rc == 0)
        rc, out = run(MR, nd, "--src", nsrc, "--column",
                      "procedure_quantity", "--group-by",
                      "person_id", "--apply")
        check("a model with NO SKILL is REFUSED - arranging by "
              "noise destroys what the engine had right while "
              "reporting that something was done",
              rc != 0 and "orders the rows by NOISE" in out)
        check("...and it wrote nothing",
              not (Path(str(nd) + "_modelrank")).exists())

        # --- the provenance guard
        half = pd.read_csv(src).head(200)
        hp = td / "half.csv"
        half.to_csv(hp, index=False)
        rc, out = run(MR, rd, "--src", hp, "--column",
                      "procedure_quantity", "--group-by",
                      "person_id")
        check("a source that never produced this run is refused "
              "by row count", rc != 0 and "not the extract" in out)

        # THE CONTROL ARM. The same values from the same
        # published grid, assigned in RANDOM order: the marginal
        # cannot care who got which value, so it must land
        # IDENTICALLY - and whatever the gate loses is what the
        # model was buying. Without this the whole marginal gain
        # could be the re-valuing with the model contributing
        # nothing, and nobody could tell.
        rc, ctl = run(MR, rd, "--src", src, "--column",
                      "procedure_quantity", "--group-by",
                      "person_id", "--random-order")
        check("the control arm runs and SAYS it is a control",
              rc == 0 and "CONTROL ARM" in ctl)
        grab = lambda t: ([l.strip() for l in t.splitlines()
                           if "source marginal" in l] or [""])[0]
        l_m, l_c = grab(out_model), grab(ctl)
        # ON A THIN COLUMN THE MARGINAL IS NOT MEASURABLE AT ALL
        # and both arms print None, which is still the property:
        # the marginal does not care who got which value. Reading
        # the LINE rather than parsing a number survives that,
        # and an absent measurement is not a mismatch.
        check("the marginal lands the SAME in both arms, which is "
              "what makes the GATE the place to read the model's "
              "worth rather than the marginal: {!r}".format(l_m),
              l_m and l_m == l_c)

        rc, out = run(MR, rd, "--src", src, "--column",
                      "nope", "--group-by", "person_id")
        check("an unknown column is refused by name",
              rc != 0 and "nope" in out)

    unit_checks()

    print()
    if FAIL:
        print("{} of {} checks failed.".format(FAIL, PASS + FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
