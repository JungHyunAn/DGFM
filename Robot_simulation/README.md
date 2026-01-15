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

To generate a dataset, run ```bash Robot_simulation.generate_data``` as the following command from the **project root**:

```bash
python -m Robot_simulation.generate_data --n 20000 --task_name two_arm --render --num_workers 10 --verbose
```

Each call automatically invokes the corresponding heuristic trajectory generator. Generated datasets are saved to:

```bash
Robot_simulation/heuristic_dataset/
```

with filenames indicating the task and number of demonstrations.

## Running evaluation

To evaluate a flow-matching method (**UniformFM**, **ShiftedFM**, or **DGFM**) on a given task using a specified number of demonstrations, run ```bash Robot_simulation.run_eval``` as:

```bash
python -m Robot_simulation.run_eval --FM_type UniformFM --N 1000 \
  --dataset_path Robot_simulation/heuristic_dataset/door_dataset_80000.hdf5 \
  --task_name door --results_path Robot_simulation/eval_results/door \
  --device cuda --val_period 30 --batch_size 500 --max_epochs 3000 \
  --warmup_steps 600 --seed 1000
```

For **DGFM**, additional parameters such as the multiplication factor (mf) must be specified:

```bash
python -m Robot_simulation.run_eval --FM_type DGFM --mf 4 --N 1000 \
  --dataset_path Robot_simulation/heuristic_dataset/door_dataset_80000.hdf5 \
  --task_name door --results_path Robot_simulation/eval_results/door \
  --device cuda --val_period 6 --batch_size 250 --max_epochs 600 \
  --warmup_steps 120
```

## Running full task evaluations

To run all evaluation configurations for a single task, execute:

```bash
bash run_all.sh <task_name> <seed>
```

## Results

For each experiment, the following outputs are saved under the specified ```bash results_path ```, organized by experiment timestamp:

- Success rates
- Trajectory execution videos
- Training logs and configuration details

These robot simulation experiments complement the synthetic benchmarks by validating DGFM on realistic manipulation tasks with constrained degrees of freedom.
