# Real-Robot Training and Inference

This directory trains vision-conditioned joint-space policies from synchronized
real-camera and teleoperation recordings. The supported policy methods are:

- `UniformFM`: vanilla flow matching with uniformly sampled flow time.
- `DGFMv2`: flow matching through an X-only clustered PCA intermediate distribution.
- `DiffusionPolicy`: diffusion training with DDIM-style sampling.

All methods use the shared `VectorField` and ResNet-18 implementation from
`Robot_simulation/models`.

## Download the data

From the repository root:

```bash
python -m Robot_real.download_data
```

Datasets are expected under `Robot_real/real_dataset`:

```text
{task}_camera/<rollout timestamp>/
    frames.csv
    realsense/realsense_1/    # front view
    realsense/realsense_2/    # wrist view

{task}_trajectory/<rollout timestamp>/
    teleop_action_joint.csv    # legacy/sweep/peg-in-hole format
    right_arm_joints.csv       # pick-and-place arm positions
    right_arm_gripper.csv      # pick-and-place finger positions
```

Camera rollouts and trajectory rollouts are paired in timestamp-folder order.
Each 10 Hz camera timestamp is matched to the closest arm-joint sample. Camera
frames outside the trajectory recording's time span are discarded, and matching
fails if the nearest arm-joint sample is more than 50 ms away. The pick-and-place
gripper position is linearly interpolated at the camera timestamps because its
CSV can have a different sampling phase.

Each training example contains:

- Current front and wrist RGB images, resized to `224 x 224`.
- Current joint angles.
- The next 16 absolute joint-position targets at the camera rate (10 Hz).

### Gripper selection

The `use_gripper` setting controls which coordinates become part of the current
joint state and every timestep of the target action chunk:

- With `use_gripper: false`, gripper/finger coordinates are excluded. For
  pick-and-place, only the seven positions in `right_arm_joints.csv` are loaded,
  so `joint_state` has shape `(7,)` and each action chunk has shape `(16, 7)`.
  `right_arm_gripper.csv` is not required or read.
- With `use_gripper: true`, pick-and-place still gets its seven arm positions
  from `right_arm_joints.csv`, then appends one `gripper_position`. This value is
  the mean of `finger_joint1_position` and `finger_joint2_position` from
  `right_arm_gripper.csv`. Arm positions are matched to the nearest samples and
  the gripper position is linearly interpolated at each camera timestamp, so
  `joint_state` has shape `(8,)` and each action chunk has shape `(16, 8)`.

For the legacy sweep and peg-in-hole trajectory format, `use_gripper: true`
retains all coordinates recorded in `teleop_action_joint.csv`, while `false`
removes coordinates whose joint name contains `finger`.

The selected coordinates are used consistently for normalization, model input,
training targets, checkpoint metadata, and rollout output. At inference time,
the current joint vector must match `policy.joint_names` exactly in both length
and order.

Joint angles are normalized independently to `[-1, 1]` using the minimum and
maximum values from the selected training demonstrations. The checkpoint stores
the per-joint `min`, `max`, and `range` needed for inference.

## Train a model

Run training from the repository root. Configs are provided for `peg_in_hole`, `sweep`,
and `pick_and_place`; swap the task prefix in the config filename to train a
different real dataset.

### UniformFM

```bash
python -m Robot_real.train_model \
  --config Robot_real/real_config/peg_in_hole_uniformfm_50.json
```

### DGFMv2

```bash
python -m Robot_real.train_model \
  --config Robot_real/real_config/peg_in_hole_dgfmv2_50.json
```

### DiffusionPolicy

```bash
python -m Robot_real.train_model \
  --config Robot_real/real_config/peg_in_hole_diffusion_50.json
```
    
To train pick-and-place, use the matching task config:

```bash
python -m Robot_real.train_model \
  --config Robot_real/real_config/pick_and_place_uniformfm_50.json
```

To train the corresponding 25-demo variant:

```bash
python -m Robot_real.train_model \
  --config Robot_real/real_config/peg_in_hole_uniformfm_25.json \
  --device cuda
```

`dataset_size` selects demonstrations as uniformly as possible across the
timestamp-ordered dataset. Validation first uses demonstrations not selected
for training. If more validation demonstrations are requested, the remainder
are selected uniformly from the training demonstrations. The split depends
only on the total demonstration count, `dataset_size`, and `val_samples`; the
model type and random seed do not affect it.

Checkpoints are written under `Robot_real/checkpoints`. The saved weights are
always from the final training epoch:

- With `use_ema: true`, the final EMA policy weights are saved.
- With `use_ema: false`, the final regular policy weights are saved.
- The final vision encoder and normalization metadata are saved with the policy.

Validation MSE is diagnostic only and does not select checkpoint weights.
Epoch metrics are written to a neighboring `*.history.json` file.

### Resized-image cache

The supplied configurations set `cache_images: true`. Before the epoch loop,
every selected front/wrist image pair is decoded and resized to `224 x 224`
once, then retained as a uint8 tensor in CPU RAM. Training epochs read these
cached tensors instead of repeatedly reading and decoding the full-resolution
PNGs from storage. `cache_workers` controls parallel cache construction, and
the training DataLoader keeps its workers alive between epochs.

The cache uses about 37 MiB for a 129-frame demonstration, or roughly
1.4--1.7 GiB for 40 typical demonstrations. It is process-local and is rebuilt
when training is restarted. Set `cache_images: false` if CPU RAM is limited.

## Generate an action chunk

`RealRobotPolicy` loads the checkpoint once and should be reused inside a
control loop:

```python
import cv2
import numpy as np

from Robot_real.rollout_model import RealRobotPolicy


policy = RealRobotPolicy(
    "Robot_real/checkpoints/peg_in_hole_uniformfm.pt",
    device="cuda",
)

# NumPy inputs must be RGB. OpenCV camera frames are normally BGR.
front_bgr = front_camera.read()
wrist_bgr = wrist_camera.read()
front_rgb = cv2.cvtColor(front_bgr, cv2.COLOR_BGR2RGB)
wrist_rgb = cv2.cvtColor(wrist_bgr, cv2.COLOR_BGR2RGB)

current_joints = robot.get_joint_positions()
action_chunk = policy.predict_action_chunk(
    images=(front_rgb, wrist_rgb),
    joint_angles=current_joints,
)

assert action_chunk.shape == (16, policy.dof)
```

The returned values are denormalized absolute joint positions, not deltas. For
the default non-gripper model, the expected joint vector and output dimension
are seven. A pick-and-place checkpoint trained with `use_gripper: true` expects
the seven arm joints followed by the averaged `gripper_position`, for eight
values total. `policy.joint_names` gives the required order.

Images can be NumPy arrays, PyTorch tensors, or file paths. Array and tensor
inputs are assumed to already use RGB channel order. Image paths are loaded and
converted from OpenCV BGR to RGB automatically.

## Command-line inference

For a smoke run without camera images or joint readings, only pass the checkpoint.
Missing images are replaced with zero tensors and missing joints are replaced
with a zero vector sized from the checkpoint metadata:

```bash
python -m Robot_real.rollout_model \
  --checkpoint Robot_real/checkpoints/peg_in_hole_uniformfm_50.pt
```

For a saved pair of images:

```bash
python -m Robot_real.rollout_model \
  --checkpoint Robot_real/checkpoints/peg_in_hole_uniformfm.pt \
  --front-image front.png \
  --wrist-image wrist.png \
  --joint-angles q1 q2 q3 q4 q5 q6 q7
```

The command prints a JSON object containing `joint_names`, output shape, and the
denormalized action chunk. Use `--output result.json` to save it.
