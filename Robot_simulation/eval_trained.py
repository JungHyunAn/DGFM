import os
import torch
import numpy as np
from Robot_simulation.FM_util import VectorField, run_flow, compute_smooth_trajectory, _get_environment_params
from Robot_simulation.heuristics_util import make_env, write_grid_video, render_trajectory


def run_trained(
    model_pt_path: str,
    model_class,
    task_name: str,
    seq_len: int,
    dof: int,
    param_len: int,
    device: torch.device,
    render_dir: str,
    video_name: str = "run",
    seed: int = 42,
):
    """
    Load a trained flow model and render a single trajectory rollout to video.

    Args:
        model_pt_path: Path to the saved model state dict (.pt file).
        model_class: The class used to construct the model (e.g., VectorField).
        task_name: One of ['door', 'wipe', 'two_arm', 'nut'].
        seq_len: Length of the flow input sequence.
        dof: Degrees of freedom of the robot.
        param_len: Length of environment parameter vector.
        device: torch device ('cpu' or 'cuda').
        render_dir: Directory where to save the output video.
        video_name: Filename (without extension) for the rendered video.
        seed: Random seed for reproducibility.
    """
    # 1) Load model
    model = model_class(seq_len, dof, param_len).to(device)
    state = torch.load(model_pt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 2) Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 3) Create environment with renderer
    env = make_env(
        task_name,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        use_joint_control=True
    )

    # 4) Capture environment parameters
    env_param = _get_environment_params(env, task_name)
    env_param = torch.tensor(env_param, dtype=torch.float32, device=device).unsqueeze(0)

    # 5) Sample and run flow to generate low-frequency trajectory
    x0 = torch.randn(1, seq_len, dof, device=device)
    with torch.no_grad():
        q_low = run_flow(model, x0, env_param, device)
    q_low = q_low.cpu().numpy()[0]

    # 6) Upsample to high-frequency trajectory
    q_high = compute_smooth_trajectory(env, task_name, q_low, env.control_freq)
    print(q_high)

    # 7) Render trajectory frames
    # First argument: environment, second: task name, third: trajectory, fourth: initial pose, fifth: camera
    frames = render_trajectory(
        env,
        task_name,
        q_high,
        q_high[0, :],
        camera_name="frontview"
    )

    # 8) Write video (single cell grid)
    os.makedirs(render_dir, exist_ok=True)
    out_path = os.path.join(render_dir, f"{task_name}_{video_name}_rendered.mp4")
    write_grid_video([frames], out_path, grid_shape=(1, 1))
    print(f"[Saved rendered video to {out_path}]")

    env.close()
    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("run_trained")
    parser.add_argument("--model_pt", type=str, required=True,
                        help="Path to the trained model (.pt file)")
    parser.add_argument("--task_name", type=str, required=True,
                        choices=["door","wipe","two_arm","nut"])
    parser.add_argument("--seq_len", type=int, required=True,
                        help="Sequence length used during training")
    parser.add_argument("--dof", type=int, required=True,
                        help="Degrees of freedom of the robot")
    parser.add_argument("--param_len", type=int, required=True,
                        help="Length of environment parameter vector")
    parser.add_argument("--render_dir", type=str, default="./renders",
                        help="Directory to save rendered videos")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run the model on, e.g., 'cpu' or 'cuda'")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    # device setup
    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    
    run_trained(
        model_pt_path=args.model_pt,
        model_class=VectorField,
        task_name=args.task_name,
        seq_len=args.seq_len,
        dof=args.dof,
        param_len=args.param_len,
        device=device,
        render_dir=args.render_dir,
        video_name="single",
        seed=args.seed
    )
