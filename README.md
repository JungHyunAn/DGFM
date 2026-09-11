# Dimension-Guided Flow Matching (DGFM)

This repository documents **Dimension-Guided Flow Matching (DGFM)**, a research project on using the local geometry of robot demonstrations to guide flow-matching policies.

Robot action chunks are not arbitrary points in their ambient space. Kinematic, contact, and task-success constraints often concentrate successful trajectories near lower-dimensional local structure. DGFM estimates this structure from neighboring demonstrations and uses it to construct geometry-guided training samples or probability paths.

> **Project status:** This is a completed research prototype. The repository preserves the progression from DGFMv1 to DGFMv3, including the approaches that worked in selected settings and the limitations encountered when moving to vision-conditioned policies.

## Main finding

The clearest improvement was obtained on the **vision-conditioned simulated Door task with 80 demonstrations**, averaged over three seeds:

| Method | Success rate |
| --- | ---: |
| DGFMv2 / NGFM | **72.1%** |
| Vanilla Flow Matching | 60.3% |
| Diffusion Policy | 56.9% |

These numbers match the result summarized on [Junghyun An's homepage](https://junghyunan.github.io/). The gain was meaningful in this setting, but it did **not** remain consistent across all demonstration budgets, tasks, and random seeds. The broader hypothesis that local geometry always improves scarce-data visuomotor learning was therefore not supported.

The name **Neighborhood-Guided Flow Matching (NGFM)** on the homepage refers to the neighborhood-based DGFMv2 robot-policy formulation described below.

## Research progression

Let $x_1$ denote a demonstrated action chunk, $c$ its condition, $z \sim \mathcal{N}(0,I)$ the source noise, and $\tilde{x}$ a sample from an estimated local demonstration model.

### DGFMv1: split-time joint geometry

[DGFMv1](Robot_simulation/models/DGFM_class.py) explicitly separates early- and late-timestep training. It clusters demonstrations in joint action-condition space, fits local PCA models, and samples paired augmentations $(\tilde{x}_i,\tilde{c}_i)$.

During the **early timestep**, the vector field is trained using the jointly sampled pseudo-pair: source noise is transported toward $\tilde{x}_i$ while the policy is conditioned on $\tilde{c}_i$. During the **late timestep**, training returns to the demonstrated pair and transports the intermediate action toward $x_i$ under the original condition $c_i$.

```math
t \in [0,\tau]: \quad (z,\tilde{c}_i) \rightarrow (\tilde{x}_i,\tilde{c}_i),
t \in (\tau,1]: \quad (\tilde{x}_i,c_i) \rightarrow (x_i,c_i).
```

The split is intentional: local joint geometry supplies additional supervision early in the probability path, while the late segment is anchored to the real demonstration. DGFMv1 therefore differs from a single condition-fixed path through an intermediate action. It was developed primarily for compact state-vector conditions, where joint action-condition neighborhoods can be estimated directly.
### DGFMv2: action-only intermediate geometry

[DGFMv2](Robot_simulation/models/DGFMv2_class.py) removes condition variables from the local PCA distribution. It estimates local trajectory geometry in action space, samples $\tilde{x}$ from a cluster associated with each demonstration, and keeps the observed condition $c$ fixed throughout the guided path

```math
z \rightarrow \tilde{x} \rightarrow x_1.
```

This separation made the method compatible with high-dimensional vision conditions without treating learned image embeddings as a fixed condition manifold. The implementation supports several path parameterizations, including piecewise-linear, cosine, and quadratic Bézier paths. DGFMv2 is the variant associated with the strongest Door result above and is referred to as NGFM on the project homepage.

### DGFMv3: joint action-pixel condition augmentation

[DGFMv3](Robot_simulation/models/DGFMv3_class.py) explores condition augmentation for fine-tuned vision policies. It constructs correlated pseudo-pairs $(\tilde{x},\tilde{c})$ from local joint action-condition geometry:

- action chunks are represented with cluster-local PCA coordinates;
- camera images are compressed with camera-wise global pixel PCA followed by cluster-local low-rank image-score PCA;
- proprioception is included directly in the joint local covariance;
- a shared latent perturbation produces correlated changes in action, pixels, and proprioception.

Instead of introducing a moving condition inside one flow path, the final implementation mixes ordinary conditional-FM supervision from real and pseudo endpoint pairs:

```math
\mathcal{L}_{\mathrm{DGFMv3}}
=
(1-\rho)\,\mathcal{L}_{\mathrm{FM}}(x,c)
+
\rho\,\mathcal{L}_{\mathrm{FM}}(\tilde{x},\tilde{c}).
```

This retains standard conditional-FM training and inference while testing whether locally coupled action-condition augmentation improves generalization. In the evaluated 20- and 40-demonstration regimes, DGFMv3 did not show a clear, consistent advantage over Vanilla Flow Matching.

## What the project established

- Robot action chunks exhibited strongly concentrated local PCA spectra: a small number of directions explained most observed variation despite a much larger ambient trajectory dimension.
- Local linear models can produce useful geometry-guided interpolants, as demonstrated by the Door 80-demo result.
- Low-dimensional action structure alone does not guarantee better closed-loop policy performance.
- Moving from state conditions to pixels changes the central problem: generalization must occur across visual conditions, and locally plausible action perturbations need not correspond to valid unseen observations.
- Joint pixel-action augmentation is possible while fine-tuning the vision encoder, but its estimated pseudo-pairs did not provide a robust improvement in the tested low-data settings.

## Repository structure

- [Synthetic_data/](Synthetic_data/) — controlled experiments on low-dimensional manifolds embedded in higher-dimensional spaces.
- [Robot_simulation/](Robot_simulation/) — robosuite dataset generation, training, sweeps, and closed-loop evaluation for Door, Nut Assembly, and Two-Arm Lift.
- [Robot_real/](Robot_real/) — real-robot data collection and policy experiments.
- [Robot_simulation/models/](Robot_simulation/models/) — VanillaFM, Diffusion Policy, DGFMv1, DGFMv2, DGFMv3, and vision-policy components.

The subdirectories contain task-specific instructions and experiment commands.

## Environment setup

Create and activate a Conda environment with Python 3.13.5, then install the dependencies:

```bash
conda create -n DGFM python=3.13.5
conda activate DGFM
pip install -r requirements.txt
```

See [Robot_simulation/README.md](Robot_simulation/README.md) and [Synthetic_data/README.md](Synthetic_data/README.md) for dataset generation, training, evaluation, and analysis instructions.
