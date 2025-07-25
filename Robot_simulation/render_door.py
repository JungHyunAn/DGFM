# Robot_simulation/render_door.py
import os
import imageio
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC

# 1) Environment factory (no Monitor, no DummyVecEnv)
def make_env():
    controller_config = suite.controllers.composite.composite_controller_factory.load_composite_controller_config(
        robot="Panda"
    )
    env = suite.make(
        "Door",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,     # we'll use offscreen render()
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        control_freq=20,
        horizon=200,
    )
    return GymWrapper(env)

if __name__ == "__main__":
    # 2) Load your trained SAC (no env passed here!)
    model_path = "Robot_simulation/trained_RL/sac_door"
    model = SAC.load(model_path, device="cuda")

    # 3) Create one env instance and attach it
    env = make_env()
    model.set_env(env)

    # 4) Prepare video writer (20 FPS to match control_freq=20)
    os.makedirs("Robot_simulation/videos", exist_ok=True)
    video_path = os.path.join("Robot_simulation/videos", "door_eval.mp4")
    writer = imageio.get_writer(video_path, fps=20)

    # 5) Run a single deterministic rollout, collecting frames
    #    Note: GymWrapper.reset() returns just obs
    obs, _ = env.reset()
    done = False
    while not done:
        # 5a) Grab the current frame from MuJoCo
        #     env.env is the underlying robosuite.Environment
        frame = env.env.sim.render(width=640, height=480, camera_name="agentview")
        #    MuJoCo returns an H×W×3 uint8 array
        writer.append_data(frame)

        # 5b) Model step
        action, _ = model.predict(obs, deterministic=True)

        # 5c) Step the env, unpack correctly
        step_out = env.step(action)
        # GymWrapper.step returns either
        #   (obs, reward, terminated, truncated, info)  [Gymnasium]
        # or
        #   (obs, reward, done, info)                  [classic Gym]
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            obs, reward, done, info = step_out

    # 6) Finish up
    writer.close()
    print(f"Video saved to {video_path}")
