from unittest import mock

import numpy as np

from Robot_simulation import env_util


def test_rollout_history_indices_match_training_frequency() -> None:
    # Eight actions at 10 Hz are upsampled to 15 controls at 20 Hz. The final
    # two query-aligned observations must therefore be two controls apart.
    assert env_util._rollout_history_sample_indices(15, 2, 2) == (12, 14)
    assert env_util._rollout_history_sample_indices(15, 1, 2) == (14,)


def test_rollout_worker_sends_consecutive_state_and_image_history() -> None:
    class FakeConnection:
        def __init__(self) -> None:
            self.messages = [
                {"type": "act", "q_low": np.zeros((3, 1), dtype=np.float32)},
                {"type": "close"},
            ]
            self.sent = []

        def recv(self):
            return self.messages.pop(0)

        def send(self, message) -> None:
            self.sent.append(message)

        def close(self) -> None:
            pass

    class FakeEnv:
        def __init__(self) -> None:
            self.control_step = 0

        def step(self, action):
            del action
            self.control_step += 1
            return None, 0.0, False, {}

        def close(self) -> None:
            pass

    connection = FakeConnection()
    fake_env = FakeEnv()

    def current_q(env, task_name, action_representation):
        del task_name, action_representation
        return np.asarray([env.control_step], dtype=np.float32)

    def camera_views(env, camera_names, image_width, image_height):
        del camera_names, image_width, image_height
        return np.asarray([env.control_step], dtype=np.float32)

    with mock.patch.object(env_util, "make_env", return_value=fake_env), \
         mock.patch.object(env_util, "_current_robot_q", side_effect=current_q), \
         mock.patch.object(env_util, "_to_action_from_q", side_effect=lambda q, *args: q), \
         mock.patch.object(env_util, "_state_policy_success", return_value=False), \
         mock.patch.object(
             env_util,
             "_upsample_policy_trajectory",
             return_value=np.zeros((5, 1), dtype=np.float32),
         ), \
         mock.patch.object(env_util, "capture_camera_views", side_effect=camera_views):
        env_util._state_policy_env_worker(
            connection,
            idx=0,
            task_name="door",
            setting={},
            static_c=np.zeros((0,), dtype=np.float32),
            seq_len=3,
            param_len=2,
            max_policy_steps=2,
            executed_horizon=3,
            observation_horizon=2,
            recorded_control_freq=20,
            trajectory_control_freq=10,
            observation_type="vision",
            camera_names=["frontview"],
            image_height=4,
            image_width=4,
        )

    np.testing.assert_array_equal(connection.sent[0]["cond"], [0.0, 0.0])
    np.testing.assert_array_equal(connection.sent[1]["cond"], [3.0, 5.0])
    np.testing.assert_array_equal(connection.sent[1]["images"][0], [3.0])
    np.testing.assert_array_equal(connection.sent[1]["images"][1], [5.0])
