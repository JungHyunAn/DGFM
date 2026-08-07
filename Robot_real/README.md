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
    teleop_state_joint.csv     # measured arm joints and gripper_width
    teleop_action_joint.csv    # commanded arm joints and gripper_target_width

# Legacy pick-and-place format:
    right_arm_joints.csv       # measured arm positions
    right_arm_gripper.csv      # measured finger positions
```

Camera rollouts and trajectory rollouts are paired in timestamp-folder order.
For the preferred teleoperation format, each 10 Hz camera timestamp is matched
to the nearest state and action samples. Camera frames outside their shared time
span are discarded, and matching fails if either nearest sample is more than
50 ms away. In the legacy pick-and-place format, arm joints use nearest-sample
matching and the measured gripper position is linearly interpolated because its
CSV can have a different sampling phase.

Each training example contains:

- Current front and wrist RGB images, resized to `224 x 224`.
- A history of current proprioceptive states.
- The configured action horizon of absolute joint-position targets at 10 Hz.

### Gripper selection

The `use_gripper` setting controls which coordinates become part of the current
joint state and every timestep of the target action chunk:

- With `use_gripper: false`, gripper/finger coordinates are excluded, leaving
  seven arm coordinates in both state and action.
- With `use_gripper: true` and the preferred teleoperation format, the state is
  the seven measured arm positions from `teleop_state_joint.csv` followed by
  its continuous `gripper_width`. The action is the seven commanded arm targets
  from `teleop_action_joint.csv` followed by its continuous
  `gripper_target_width`. State and action therefore both have eight coordinates.
- With `use_gripper: true` and the legacy pick-and-place format, one
  `gripper_position` is appended to the seven arm positions. It is the mean of
  `finger_joint1_position` and `finger_joint2_position` from
  `right_arm_gripper.csv`. Because this format has no separate state/action
  recordings, its aligned measured trajectory is also copied as the action
  target.

For the legacy sweep and peg-in-hole trajectory format, `use_gripper: true`
retains all coordinates recorded in `teleop_action_joint.csv`, while `false`
removes coordinates whose joint name contains `finger`.

The selected coordinates are used consistently for model input, training
targets, checkpoint metadata, and rollout output. At inference time, the
current joint vector must match `policy.joint_names` exactly in both length and
order.

### State and action normalization

A policy trained with `use_gripper: false` has seven-dimensional state and action
vectors. With the existing no-gripper configs, all seven model-state arm angles
and all seven action targets are independently min-max normalized to `[-1, 1]`
using training data statistics:

```text
model state (7D)
  normalize_to_-1_1([measured_q1, ..., measured_q7])

action target at each chunk step (7D)
  normalize_to_-1_1([target_q1, ..., target_q7])
```

The pick-and-place configs set
`proprioception_normalization: "gripper_0_1"`. Their model tensors have different
state and action transforms by design:

```text
proprioceptive state (8D)
  [q1, ..., q7, clip((measured_gripper_width - grip_min) /
                       (grip_max - grip_min), 0, 1)]
   raw radians       normalized with training-state gripper bounds

action target at each chunk step (8D)
  normalize_to_-1_1([target_q1, ..., target_q7, gripper_target_width])
```

Thus all output-action coordinates, including the continuous gripper target,
are min-max normalized to `[-1, 1]` using action statistics from the selected
training demonstrations. Arm proprioception is not normalized in this mode;
the only normalized state coordinate is the current measured gripper width. Its
minimum and maximum are fitted from the selected training demonstrations, and
values outside that fitted interval are clipped to `[0, 1]`. The state contains
no previous gripper measurement or command.

Training computes and stores separate state and action statistics. At rollout,
`RealRobotPolicy` applies the state transform above, samples a normalized action
chunk, and denormalizes every action coordinate with the stored action
statistics before returning it. Configs without `proprioception_normalization:
"gripper_0_1"` retain the legacy behavior of min-max normalizing every state
coordinate to `[-1, 1]`.

## Train a model

Run training from the repository root. The supplied peg-in-hole examples below
disable the gripper and produce seven-dimensional policies. The
pick-and-place configs enable the gripper and produce
eight-dimensional policies.

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

For the strict 8D pick-and-place contract documented above, use one of the
25-demo pick-and-place configs:

```bash
python -m Robot_real.train_model \
  --config Robot_real/real_config/pick_and_place_uniformfm_25.json

python -m Robot_real.train_model \
  --config Robot_real/real_config/pick_and_place_dgfmv2_25.json

python -m Robot_real.train_model \
  --config Robot_real/real_config/pick_and_place_diffusion_25.json
```

These settings apply when the models are retrained. Existing `.pt` files retain
the normalization metadata with which they were originally trained and do not
change merely because their JSON config was edited.

To train the corresponding 25-demo peg-in-hole variant:

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


# NumPy inputs must be RGB. OpenCV camera frames are normally BGR.
front_bgr = front_camera.read()
wrist_bgr = wrist_camera.read()
front_rgb = cv2.cvtColor(front_bgr, cv2.COLOR_BGR2RGB)
wrist_rgb = cv2.cvtColor(wrist_bgr, cv2.COLOR_BGR2RGB)


# Policy without gripper: seven physical arm angles in radians.
arm_policy = RealRobotPolicy(
    "Robot_real/checkpoints/peg_in_hole_uniformfm_50.pt",
    device="cuda",
)
arm_state = robot.get_arm_joint_positions()
arm_action_chunk = arm_policy.predict_action_chunk(
    images=(front_rgb, wrist_rgb),
    joint_angles=arm_state,
)
assert arm_action_chunk.shape == (arm_policy.horizon, 7)

# Policy with gripper: append the current physical gripper width in meters.
gripper_policy = RealRobotPolicy(
    "Robot_real/checkpoints/pick_and_place_uniformfm_25.pt",
    device="cuda",
)
gripper_state = np.concatenate(
    [robot.get_arm_joint_positions(), [robot.get_gripper_width()]]
)
gripper_action_chunk = gripper_policy.predict_action_chunk(
    images=(front_rgb, wrist_rgb),
    joint_angles=gripper_state,
)
assert gripper_action_chunk.shape == (gripper_policy.horizon, 8)
```

Pass physical values to both calls; do not pre-normalize them.

- Without gripper, pass the seven current arm-joint angles in radians.
  `arm_policy` uses its checkpoint state statistics internally and returns seven
  denormalized absolute arm-joint targets.
- With gripper, pass the seven current arm-joint angles followed by the current
  measured total gripper width in meters. `gripper_policy` leaves the arm angles
  raw for the model and normalizes only the gripper coordinate using the fitted
  training-state bounds. It returns seven absolute arm-joint targets followed
  by the continuous gripper target width.

Both returned chunks contain physical absolute targets, not normalized values
or delta commands. Use each policy’s `joint_names`, `dof`, and `horizon`
properties rather than hard-coding the interface in a controller.

For compatibility with checkpoints trained from `teleop_action_joint.csv`
with two trailing Franka finger coordinates, rollout exposes the same
eight-value interface. The input `gripper_position` is duplicated internally
for the checkpoint's two finger inputs, and the two generated finger targets
are averaged back into one `gripper_position`.

Images can be NumPy arrays, PyTorch tensors, or file paths. Array and tensor
inputs are assumed to already use RGB channel order. Image paths are loaded and
converted from OpenCV BGR to RGB automatically.

## Command-line inference

For a no-gripper smoke run without camera images or joint readings, only pass
the checkpoint. Missing images are replaced with zero tensors, and missing
joints are replaced with a zero vector sized for the policy rollout interface:

```bash
python -m Robot_real.rollout_model \
  --checkpoint Robot_real/checkpoints/peg_in_hole_uniformfm_50.pt
```

For a with-gripper run using a saved image pair and physical 8D state:

```bash
python -m Robot_real.rollout_model \
  --checkpoint Robot_real/checkpoints/pick_and_place_uniformfm_25.pt \
  --front-image front.png \
  --wrist-image wrist.png \
  --joint-angles q1 q2 q3 q4 q5 q6 q7 gripper_width
```

The command prints a JSON object containing `joint_names`, output shape, and the
denormalized action chunk. Use `--output result.json` to save it.
