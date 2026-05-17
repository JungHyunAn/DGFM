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

## Results

For each experiment, the following outputs are saved under the specified ```bash results_path ```, organized by experiment timestamp:

- Success rates
- Trajectory execution videos
- Training logs and configuration details

These robot simulation experiments complement the synthetic benchmarks by validating DGFM on realistic manipulation tasks with constrained degrees of freedom.
