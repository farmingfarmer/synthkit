# Marketing assets

Shareable visuals for decks and demos. Everything here is built from
FULLY INVENTED demo data — no real patient records, so these travel
freely.

## The 3D fidelity loops (drop straight into PowerPoint — they loop)

- `SynthKit_pullapart_loop.gif` — real and synthetic fused, parting
  into two clouds that each keep the same shape, then merging back.
  Each cloud carries its own labeled axes. **The hero.**
- `SynthKit_spin_realvsynthetic.gif` — blue real + red synthetic in one
  rotating cloud on labeled axes. The ten-second hook.
- `SynthKit_generation_reveal.gif` — real data, then its synthetic twin
  fades in, then the real fades out leaving the synthetic alone.
- `SynthKit_spin_fidelity.gif` — colored green (common, reproduced) to
  red (rare, protected by the k rule).
- `SynthKit_feature_gradient.gif` — real vs synthetic side by side,
  colored by one feature at a time: the same structure in both.
- `SynthKit_distributions.gif` — each feature's distribution, original
  against synthetic, one after another.

All 3D views are plotted on three REAL labeled axes (age, blood
pressure, glucose). The axes are labeled because they are real
measurements — the autoencoder's own latent axes are abstract and
cannot be honestly named, so they are not used for position.

## Slide stills

- `SynthKit_slide4_user_input.png` — the free-text box: describe a
  dataset in plain English.
- `SynthKit_slide4_describe_context.png` — the bench with its workflow
  rail and the ready-made templates.
- `SynthKit_slide4_recipe.png` — that English compiled into a
  structured, reviewable recipe.
- `SynthKit_slide4_synthetic_output.png` — the generated table, the
  record count, and the planted rate landing inside its promised band.

## Logo

- `SynthKit_logo_alpha.webm` — transparent animated logo (web / the
  bench). True alpha.
- `SynthKit_logo_still_alpha.png` — transparent still (PowerPoint
  supports transparent PNG, but NOT alpha video).
- `SynthKit_bench_logo_header.png` — the logo in the bench header.

A ProRes 4444 `.mov` of the logo exists for video editors but is NOT
in the repo: at 178 MB it exceeds GitHub's 100 MB file limit. Re-render
it from the source with the recipe in the lab notebook, or ask for it
directly.

## The slide

- `SynthKit_showcase_slide.html` — the self-contained showcase page
  (spinning cloud, pull-apart loop, performance plots, gate scorecard).
  Regenerate from any run with `python scripts/showcase.py RUNDIR --src SRC.csv`.
