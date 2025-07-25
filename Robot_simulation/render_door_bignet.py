# Robot_simulation/render_door_bignet.py

import os
import imageio
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC

def make_env():
    controller_config = suite.controllers.composite.composite_controller_factory.load_composite_controller_config(
        robot="Panda"
    )
    env = suite.make(
        "Door",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,     # offscreen render via sim.render()
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        control_freq=20,
        horizon=200,
    )
    return GymWrapper(env)

if __name__ == "__main__":
    # 1) Load your trained SAC (no env passed here)
    model_path = "Robot_simulation/trained_RL/sac_door_parallel_bignet"
    model = SAC.load(model_path, device="cuda")

    # 2) Create the single env
    env = make_env()

    # 3) Prepare video writer
    os.makedirs("Robot_simulation/videos", exist_ok=True)
    video_path = os.path.join("Robot_simulation/videos", "door_eval_bignet.mp4")
    writer = imageio.get_writer(video_path, fps=20)

    # 4) Run one deterministic episode
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    done = False

    while not done:
        # grab offscreen frame from MuJoCo
        frame = env.env.sim.render(width=640, height=480, camera_name="frontview")
        writer.append_data(frame)

        # predict action
        action, _ = model.predict(obs, deterministic=True)

        # step env
        step_out = env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            obs, reward, done, info = step_out

        if isinstance(obs, tuple):
            obs = obs[0]

    # 5) Finalize
    writer.close()
    print(f"Video saved to {video_path}")
