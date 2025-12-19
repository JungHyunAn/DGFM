# Dimension-Guided Flow Matching (DGFM)

This repository contains the official implementation of **Dimension-Guided Flow Matching (DGFM)**, a generative modeling framework for learning low-dimensional manifold-supported distributions embedded in high-dimensional spaces. DGFM is motivated by robotic motion generation problems in which valid trajectories lie on constrained manifolds induced by task-specific degrees of freedom (DOF).

The core idea of DGFM is to incorporate **explicit dimensional guidance through manifold-aligned approximate intermediate distribution**, enabling efficient learning of flows that decrease dimensionality. This can improve sample efficiency and overall performance in low-data settings (e.g., dexterous manipulation with limited demonstrations).

## Repository structure

- **`Synthetic_data/`**  
  Controlled synthetic experiments on low-dimensional manifolds embedded in higher-dimensional spaces. These experiments are used to analyze DGFM’s geometric/statistical behavior and compare against baseline flow matching approaches.

- **`Robot_simulation/`**  
  Robot simulation experiments evaluating DGFM on manipulation tasks in physics-based environments, including trajectory generation and quantitative rollouts.

Each subdirectory includes its own README with detailed instructions for running experiments and reproducing results.
