"""ONE real run, every consumer driven against it (AC-1).

THE WRONG-KEY FAULT HAS HAPPENED EIGHT TIMES IN THIS REPO, and
every fix was applied case by case AFTER the failure:

  jointcheck's skill_real / skill_syn / driver_agreement, read
  as r2_real / r2_synth / drivers_match - twelve rows of
  em-dashes on a page about to go to a stakeholder;
  verify_frame's effects / requested_in_sds / recovered_in_sds;
  integrity's `share`, read as `match`; a list marginal's
  `value`, read as `token`, which nearly became a report that
  the blueprint declares relationships on unpublished tokens;
  the dial matrix's marginal `type`, read as `kind`, which cost
  six round trips to the machine that holds the extract;
  membership_new_path's runs / nearest_neighbor / likelihood,
  invented as one `attack.json` with `arms`; and
  determination's fidelity.reconstruction.ratio and
  blueprint.privacy, read as nn_ratio / reconstruction_gap /
  bounds_are_k_anonymous.

EVERY ONE IS THE SAME SHAPE: module A reads module B's dict with
names that SOUND right, and the checks cannot see it because
they build B's dict BY HAND with the same invented names. The
fixture agrees with the reader and both disagree with the only
thing that matters.

SO THIS SUITE WRITES ONE REAL RUN AND DRIVES EVERY CONSUMER
AGAINST IT. Nothing here is hand-assembled: the artifacts come
from `write_run_artifacts`, the reconstruction measurement from
the engine's own method, and each consumer must produce
NON-EMPTY output - because a consumer reading the wrong key does
not crash, it returns an empty table and a reassuring page.

The test is the DIRECTION the fault always travels: a reader
whose keys are wrong finds nothing. So every assertion here is
"it found something", named.
"""

import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def guarded(label, fn):
    """Run a consumer, turning a CRASH into a red check.

    A reader with the wrong key usually returns EMPTY rather
    than raising - but when it does raise, a crash emits a
    traceback and no `FAIL:` line, which this project records as
    indistinguishable from a check that never ran.
    """
    try:
        return fn()
    except Exception as e:                      # noqa: BLE001
    # (printed, so a crash is visible in the log beside the
    #  red check it causes)
        print("    ({} raised {}: {})".format(
            label, type(e).__name__, str(e)[:90]))
        return None


def build_run(td):
    """A real run directory, written by the real writer."""
    from latent_challenger import LatentGen, write_run_artifacts
    rng = np.random.default_rng(11)
    rows = []
    for p in range(120):
        lvl = float(rng.normal(0, 1))
        for v in range(4):
            rows.append({
                "person_id": "P{:03d}".format(p),
                "visit_number": v + 1,
                "sys": round(126 + 12 * lvl + rng.normal(0, 4), 1),
                "dia": round(78 + 7 * lvl + rng.normal(0, 3), 1),
                "severity": round(3.1 + 0.8 * lvl
                                  + rng.normal(0, 0.4), 2),
                "site": "a" if lvl > 0 else "b",
                "conds": ";".join(sorted(set(
                    "c{:02d}".format(int(abs(rng.normal(0, 3))) % 8)
                    for _ in range(3)))),
            })
    df = pd.DataFrame(rows).astype(str)
    ids = sorted(df["person_id"].unique())
    hold = df[df["person_id"].isin(ids[-20:])]
    train = df[df["person_id"].isin(ids[:-20])]
    g = LatentGen(seed=0, k=5)
    g.fit(train, "person_id")
    gen = g.generate(len(df), seed=0)
    recon = g.reconstruction_gap(hold)
    run = os.path.join(td, "run")
    fid = write_run_artifacts(df, gen, "person_id", run,
                              source_name="contract.csv",
                              reconstruction=recon,
                              gen_engine=g, k=5)
    src = os.path.join(td, "src.csv")
    df.to_csv(src, index=False)
    return run, src, fid


def main():
    with tempfile.TemporaryDirectory() as td:
        run, src, _fid = build_run(td)
        files = sorted(os.listdir(run))
        read = {}
        for name in ("fidelity.json", "blueprint.json",
                     "catalogue.json", "provenance.json"):
            path = os.path.join(run, name)
            read[name] = (json.load(io.open(path))
                          if os.path.exists(path) else None)

        check("the real writer produced every artifact the "
              "consumers read ({}) - everything below is driven "
              "from THESE files, never from a dict assembled in "
              "this suite, because a hand-built fixture carrying "
              "the reader's own invented names is how eight "
              "wrong-key faults survived their checks"
              .format(", ".join(f for f in files
                                if f.endswith(".json")
                                or f.endswith(".csv"))),
              all(read[k] is not None for k in read))

        # ---- THE GATE
        from synthkit import gate as GT
        gres = guarded("gate.assess",
                       lambda: GT.assess(read["fidelity.json"]))
        named = ([c.get("name") for c in gres["criteria"]]
                 if gres else [])
        check("gate.assess finds its criteria in this run's "
              "fidelity.json - {} of them, including {} - and a "
              "gate that read the wrong keys would return an "
              "empty criteria list and a page of nothing"
              .format(len(named),
                      ", ".join(str(n) for n in named[:3])),
              gres is not None and len(named) >= 4)
        # AND THIS CHECK'S OWN FIRST VERSION GUESSED A FIELD
        # NAME - `have` - inside the suite written to stop
        # people guessing field names. A criterion carries
        # `name`, `ok` and `detail`; the measured pair lives in
        # `detail` as text a reader can check the arithmetic
        # against, which is the actual contract.
        _detailed = [c for c in (gres or {}).get("criteria", [])
                     if c.get("detail") and "/" in c["detail"]]
        check("...and each criterion carries its MEASURED pair "
              "in `detail` ({} of {} do) rather than a bare "
              "verdict, so a reader can check the arithmetic. "
              "The first version of this check asked for a "
              "`have` field, which does not exist - a guessed "
              "field name inside the suite written to stop "
              "guessed field names"
              .format(len(_detailed),
                      len((gres or {}).get("criteria", []))),
              gres is not None and len(_detailed) >= 3)

        # ---- THE DETERMINATION PACKAGE
        from synthkit import determination as DET
        dres = guarded("determination.assess", lambda: DET.assess(
            fidelity=read["fidelity.json"],
            blueprint=read["blueprint.json"],
            provenance=read["provenance.json"]))
        check("determination.assess reads MEMORIZATION and "
              "AGGREGATES out of this run - the two elements "
              "whose paths were invented, and which read OWED on "
              "a real extract until the tool was pointed at real "
              "artifacts",
              dres is not None
              and "memorization" not in dres["elements_owed"]
              and "aggregates" not in dres["elements_owed"])
        check("...and the three a RUN genuinely cannot supply "
              "stay OWED, so the package is not passing "
              "everything by accident: {}"
              .format(", ".join(dres["elements_owed"])
                      if dres else "-"),
              dres is not None
              and {"membership", "attribute", "linkage",
                   "residual"} <= set(dres["elements_owed"]))

        # ---- THE RECEIPTS
        from synthkit import receipts as RC
        rres = guarded("receipts.build",
                       lambda: RC.build(run))
        claims = (rres.get("receipts") or rres.get("claims") or []
                  ) if isinstance(rres, dict) else []
        check("receipts.build assembles a receipt per published "
              "claim from this run - {} of them - where a reader "
              "with the wrong keys would assemble zero and "
              "report a run with no patterns"
              .format(len(claims)),
              rres is not None and len(claims) >= 1)
        check("...and every receipt carries its LIMITS, which is "
              "the half that stops it reading as a certificate",
              bool(claims) and all(
                  c.get("limits") for c in claims))

        # ---- THE DECK
        import fidelity_deck as FD
        # ...and `build_deck` takes (SRC, RUN), which the first
        # version passed the other way round and got
        # NotADirectoryError. Read the signature; do not
        # remember it.
        # ...and it returns `(html, stats)`, not html. Its
        # docstring said "the deck HTML as a string" and had
        # been returning a tuple for as long as the stats
        # existed - so this check read `len(result)` as 2 BYTES
        # and failed for the right reason with the wrong
        # explanation. A stale return contract is the same fault
        # as a wrong field name. Docstring corrected.
        deck = guarded("fidelity_deck.build_deck",
                       lambda: FD.build_deck(src, run,
                                             group_by="person_id"))
        html, dstats = (deck if isinstance(deck, tuple)
                        else (deck, {}))
        check("fidelity_deck.build_deck renders a page from this "
              "run ({:,} bytes over {} columns) - a deck whose "
              "reader missed a key renders the section headings "
              "and no numbers, which is exactly how twelve rows "
              "of em-dashes once shipped behind forty green "
              "checks"
              .format(len(html or ""), dstats.get("columns")),
              html is not None and len(html) > 20000
              and "fidelity" in html.lower())
        check("...and it reports how many source bins it "
              "SUPPRESSED ({}), because a fidelity report on "
              "real data is itself a release - a histogram bin "
              "holding three patients describes a group small "
              "enough to gossip about"
              .format(dstats.get("suppressed")),
              "suppressed" in dstats
              and dstats.get("suppressed") is not None)

        # ---- THE SCRIPTS, THROUGH THE PROCESS BOUNDARY
        def cli(*args):
            r = subprocess.run(
                [sys.executable] + list(args),
                capture_output=True, text=True, cwd=str(ROOT))
            return r
        m0 = cli(str(ROOT / "scripts" / "m0_gate.py"), run)
        check("m0_gate.py reads this run through the process "
              "boundary and prints its criteria, exit {} - it "
              "crosses a process boundary in real use and a "
              "path or key fault there is invisible to an "
              "in-process check".format(m0.returncode),
              ("criteri" in m0.stdout.lower()
               or "MET" in m0.stdout)
              and "Traceback" not in m0.stderr)
        sign = cli(str(ROOT / "scripts" / "signoff.py"), run,
                   "-o", os.path.join(td, "signoff.html"))
        wrote = os.path.exists(os.path.join(td, "signoff.html"))
        check("signoff.py writes its page from this run, exit "
              "{} - and it is the page a decision-maker signs, "
              "so a silent empty section there is the worst "
              "place for one".format(sign.returncode),
              wrote and "Traceback" not in sign.stderr)
        # AND THE DISTANCE REACHES THE PAGE A DECISION-MAKER
        # SIGNS. It is the form the commercial tools publish, so a
        # reader comparing vendors needs the same number on both
        # sides - and it must say it is not a criterion, because a
        # figure printed beside a verdict table reads as one.
        _sp = os.path.join(td, "signoff.html")
        _html = ""
        if wrote:
            with io.open(_sp, encoding="utf-8") as _fh:
                _html = _fh.read()
        check("...and that page carries the DISTANCE beside the "
              "verdicts, marked as not a criterion - a measurement "
              "that never reaches a page might as well not exist "
              "for the person deciding",
              "Distance (not a criterion)" in _html
              and "variation distance" in _html)

        # AND THE SHAPE, which no criterion on that page asks
        # about. Presence, centre and spread can all pass with
        # most of a column's rows on one value - which is what
        # procedure_quantity did for months on the real extract -
        # and this is the page somebody signs.
        check("...and the MARGINAL SHAPE reaches it too, marked "
              "as not a criterion, because the verdict table "
              "above it cannot see a column that passed coverage "
              "with the wrong distribution",
              "Marginal shape (not a criterion)" in _html)
        # WHICH CASE THIS RUN IS IN, said out loud. The measure
        # refuses a column whose own sampling noise reaches the
        # share bar, and on a small cohort that is every column -
        # so the page reports NOT READABLE, where there is no
        # good number to be misread as privacy and the warning
        # would be about nothing.
        if "NOT READABLE on this run" in _html:
            check("...and on a cohort too thin to read it says so "
                  "on the page, because a vanished section on the "
                  "page somebody signs reads as a pass",
                  "size of the cohort" in _html)
        else:
            check("...carrying the warning that a verbatim copy "
                  "would score perfectly on it, so a good line is "
                  "never read as privacy on the one page that is "
                  "signed",
                  "not at all to privacy" in _html
                  or "track both their own published summary"
                  in _html)

        det = cli("-m", "synthkit.cli", "determination", run)
        check("the determination CLI runs on this run and exits "
              "3 while elements are owed ({}), rather than "
              "reporting a readiness it has not got"
              .format(det.returncode),
              det.returncode == 3 and "OWED" in det.stdout
              and "Traceback" not in det.stderr)
        rcp = cli("-m", "synthkit.cli", "receipts", run)
        check("the receipts CLI runs on this run, exit {} with "
              "an EMPTY stderr - a receipts command once printed "
              "a perfect report and died at its write step, "
              "invisible because stderr is unbuffered while "
              "piped stdout is not".format(rcp.returncode),
              rcp.returncode == 0 and not rcp.stderr.strip())

    print()
    if FAIL:
        print("{} of {} checks failed.".format(FAIL, PASS + FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
