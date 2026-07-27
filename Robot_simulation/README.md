# Robot Simulation Experiments

This directory contains robot simulation experiments for evaluating **Dimension-Guided Flow Matching (DGFM)** on manipulation tasks in physics-based environments. All experiments are built on **ROBOSUITE** and focus on trajectory generation under task-specific kinematic and dynamic constraints.

## Supported environments

Three ROBOSUITE manipulation tasks are provided:

- **Door opening**
- **Nut assembly**
- **Two-arm lift**

## Dataset generation

For each task, a dataset of arbitrary-size demonstrations can be generated using heuristic trajectory planners. These heuristics encode task-specific planning logic and are implemented in:

```bash
heuristic_{task_name}.py
```

To generate a dataset, run ```bash Robot_simulation.environments.generate_data``` as the following command from the **project root**:

```bash
python -m Robot_simulation.environments.generate_data --n 1000 --task_name door --render --num_workers 10 --verbose
```

For a vision dataset, enable both default cameras (`frontview` and `robot0_eye_in_hand`):

```bash
python -m Robot_simulation.environments.generate_data --n 1000 --task_name door --vision --num_workers 10
```

Images are written as compact JPEG files beside the HDF5 file. The HDF5 stores relative per-step paths under each episode's `image_paths` group and uses the `_vision.hdf5` suffix. Camera names, image size, and JPEG quality can be changed with `--camera_names`, `--image_height`, `--image_width`, and `--jpeg_quality`.

Each call automatically invokes the corresponding heuristic trajectory generator. Generated datasets are saved to:

```bash
Robot_simulation/heuristic_dataset/
```

with filenames indicating the task and number of demonstrations.

## Running evaluation

To evaluate a flow-matching method (**UniformFM**, **ShiftedFM**, or **DGFM**) on a given task using a specified number of demonstrations, run ```bash Robot_simulation.run_eval``` as:

```bash
python -m Robot_simulation.run_eval --config Robot_simulation/sim_config/door_uniform.json \
  --dataset_path Robot_simulation/heuristic_dataset/door_dataset_10000.hdf5 \
```

For vision-conditioned training and live rendered evaluation, add:

```bash
  --observation_type vision
```

The policy concatenates state history with features from one shared, frozen ImageNet ResNet-18 across both camera views.

The JSON config bundles simulation and training settings for reproducibility. Keep
`dataset_path` on the command line so the same config can be reused across
machines and dataset locations. Command-line arguments override config values.

For **DGFM**, additional parameters such as the multiplication factor (mf) must be specified:

```bash
python -m Robot_simulation.run_eval --config Robot_simulation/sim_config/door_dgfm_mf4.json \
  --dataset_path Robot_simulation/heuristic_dataset/door_dataset_10000.hdf5 \
```

For a quick wiring check from heuristic windows through FM evaluation:

```bash
python -m Robot_simulation.run_eval --config Robot_simulation/sim_config/door_sanity.json \
  --dataset_path Robot_simulation/heuristic_dataset/door_dataset_10000.hdf5
```

## Running full task evaluations

To run all evaluation configurations for a single task, execute:

```bash
bash Robot_simulation/run_all.sh <task_name> <seed>
```

## Primary simulation metric

We report the mean success rate over the final ten validation checkpoints, where every checkpoint is evaluated on the same fixed set of 50 task instances. This is the primary run-level simulation metric. Maximum validation success is retained as a secondary diagnostic. Paper-level tables aggregate the run-level metric across training seeds using mean and standard deviation; it is not labeled as held-out test performance.

## Full-rank local PCA in NGFM

NGFM summarizes local trajectory neighborhoods with PCA while retaining the full ambient trajectory basis. Principal variances encode locally dominant directions, while orthogonal or low-variance directions receive a positive variance floor. The result is a full-rank anisotropic Gaussian concentrated near locally observed trajectory geometry. This avoids singular covariance operations while preserving local geometric bias. The robot implementation does not explicitly infer or truncate to an intrinsic manifold dimension. We refer to this design as a **full-rank local PCA covariance with regularized low-variance directions**, or a **regularized near-manifold intermediate distribution**.

## Results

For each experiment, the following outputs are saved under the specified ```bash results_path ```, organized by experiment timestamp:

- Success rates
- Trajectory execution videos
- Training logs and configuration details

These robot simulation experiments complement the synthetic benchmarks by validating DGFM on realistic manipulation tasks with constrained degrees of freedom.
