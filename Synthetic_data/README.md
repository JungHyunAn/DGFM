# Synthetic Data Experiments

This directory contains synthetic experiments for evaluating Dimension-Guided Flow Matching (DGFM) on controlled low-dimensional manifolds embedded in higher-dimensional spaces.

## Supported methods

The implementations now live in `Synthetic_data.models`:

- **UniformFM** - standard flow matching; pass `time_sampling="shifted"` for shifted-time sampling.
- **DGFMv2** - condition-free DGFM with full-dimensional cluster intermediates.
- **OT_CFM** - minibatch optimal-transport conditional flow matching.

The evaluation entry point compares UniformFM and DGFMv2.

## Supported distributions

The evaluation code supports Normal, three quadratic variants, Linear Branched, SwissRoll, TwoMoon, and PinWheel distributions.

## Running evaluation

Run scripts from the project root. One configuration can be evaluated with:

```bash
python -m Synthetic_data.run_eval \
  --sample_size 320 \
  --cluster_num 8 \
  --target_distribution 6
```

Distribution keys 6, 7, and 8 select SwissRoll, TwoMoon, and PinWheel.

Run the fixed sweep with seed 1000 using:

```bash
python -m Synthetic_data.run_eval_sweep 1000
```

Pass `--resume` to skip completed matching runs. A sweep writes three aggregate
files directly under `Synthetic_data/eval_results/sweep_<seed>/`:
`SwissRoll.json`, `TwoMoon.json`, and `PinWheel.json`. Each file contains all
per-trial results and per-sample-size metric averages for its distribution.

## Visualizing results

After evaluation, generate plots with:

```bash
python -m Synthetic_data.visualize_results
```
