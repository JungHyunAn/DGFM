# Synthetic Data Experiments

This directory contains synthetic experiments used to evaluate **Dimension-Guided Flow Matching (DGFM)** on controlled low-dimensional manifolds embedded in higher-dimensional spaces. These experiments are designed to isolate geometric and statistical properties of the learned flows and to provide clear, reproducible comparisons against baseline methods.

## Supported methods

The following flow-matching variants are implemented:

- **UniformFM** – standard flow matching with an isotropic prior  
- **ShiftedFM** – flow matching with a shifted (biased) prior  
- **DGFM** – dimension-guided flow matching

## Supported distributions

Experiments can be conducted on the following synthetic target distributions:

- **Quadratic unimodal**
- **Quadratic multimodal**
- **Linear branched**
- **Swiss roll**

## Running evaluation

All scripts should be executed from the **project root** directory.

To evaluate a selected method on a given synthetic distribution, run:

```bash
python -m Synthetic_data.run_eval
```

Evaluation outputs, including quantitative metrics and intermediate results, are saved to:

```bash
Synthetic_data/eval_results/
```

## Visualizing results

After completing the evaluation, generate plots and summary figures by running:

```bash
python -m Synthetic_data.visualize_results
```

The resulting graphs and visualizations are saved to:

```bash
Synthetic_data/eval_graphs/
```

Please refer to the script headers or inline configuration options to specify the desired method and target distribution.

These synthetic experiments serve as a controlled testbed for validating DGFM’s geometric inductive bias and dimensional guidance before transitioning to robot simulation benchmarks.
