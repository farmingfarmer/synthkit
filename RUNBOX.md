# RUNBOX — the full-extract duel, entirely inside the bench

Updated 2026-09-25. The Duel station exists: both engines against
the FULL real dataset, one page - every shared column drawn three
ways (gray original, cardinal blueprint, gold latent), THE GAP as
its own heatmap saying which engine drifts where, the mined
interactions per engine, and both privacy postures on the page.
Verified end to end on fixtures and by headless-Chrome eyes
before shipping.

## 1. Pull once

Zipball per WINDOWS.md. Expect **69 suites, 1940 checks, ALL
GREEN**. Then:

```bat
pip install -e .
```

Relaunch the bench, hard-refresh, wordmark matches
`python -m synthkit.cli version`.

## 2. The full-extract duel, in the bench

Open the new **DUEL** station (map rail, above Roadmap). Fill
the five fields - BROWSER FIELDS DO NOT EXPAND %USERPROFILE%, so
type the expanded form of each path (your Desktop RUNBOX.html
carries them literally, ready to copy):

- source CSV path: your `%USERPROFILE%\dev\tidy_visits.csv`
- blueprint run directory: your `%USERPROFILE%\dev\run_seed11b`
  (the full-extract fit; its generated.csv is the blueprint side)
- patient / entity column: `person_id`
- latent output directory: your `%USERPROFILE%\dev\latent_full`
- latent seed: `0`

Press **Generate with the latent engine** - it runs as a bench
job with the loader up. MEASURED on an extract-shaped stand-in
at full scale (61,121 rows x 58 columns, 800 patients):
**3-7 minutes**. When it says the draws are written, press
**Draw the duel** - also a polled job, MEASURED at **7-9
minutes** at that scale (the interaction mining is real work
over every numeric child). Both show elapsed seconds while they
run; neither blocks the browser.

**Expect on the page:** the four headline cards (per-engine
sign/close and INVERTED counts), three-body histograms for every
shared column (press and hold to fan them apart), the gap
heatmap (cardinal cells = blueprint drifts further, gold =
latent does), the mined-interaction bars (on the 300-sample the
blueprint read 0.000 on all eight - this is the full-data
version of that reading), and the posture footer.

For calibration, the latent engine on that full-scale stand-in
read sign 625/630, close 618/630, **INVERTED 1**, nn-ratio 1.03
- notably better than the 2-4 inversions it showed at
300-patient width. Whether your real extract agrees is exactly
what this run answers.

**Send back:** screenshots of the headline cards, the gap
heatmap, and the mined-interactions section. Real-derived files
(the latent output directory) STAY on the machine.

## 3. The same duel from the terminal (fallback / artifact)

The bench and the CLI are one code path; to write the page as a
shareable-on-this-machine file instead:

```bat
python scripts\fidelity_deck.py --src %USERPROFILE%\dev\tidy_visits.csv --run %USERPROFILE%\dev\run_seed11b -o %USERPROFILE%\dev\duel_full.html --group-by person_id --latent %USERPROFILE%\dev\latent_full\latent_gen_seed0.csv
```
