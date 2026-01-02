import os
import torch
import numpy as np
from Robot_simulation.FM_util import VectorField, run_flow, compute_smooth_trajectory, _get_environment_params
from Robot_simulation.heuristics_util import make_env, write_grid_video, render_trajectory
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from Robot_simulation.FM_util import eval_model
from Robot_simulation.run_eval import _spawn_env_once


def run_trained(
    model_pt_path: str,
    model_class,
    task_name: str,
    seq_len: int,
    dof: int,
    param_len: int,
    device: torch.device,
    render_dir: str,
    evaluation_samples: int = 100,
    use_vision: bool = False,
    vision_encoder_path: str = None,
    vision_encoder_class = None,
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
    if (use_vision):
        print("evaluating model ", model_pt_path, f" with {evaluation_samples} samples, vision encoder is used\n")
    else:
        print("evaluating model ", model_pt_path, f" with {evaluation_samples} samples, vision encoder isn't used\n")
    
    # 1) Load model
    model = model_class(seq_len, dof, param_len).to(device)
    state = torch.load(model_pt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    if use_vision:
        vision_encoder = vision_encoder_class.to(device)
        vision_encoder_state = torch.load(vision_encoder_path, map_location=device)
        vision_encoder.load_state_dict(vision_encoder_state)
        vision_encoder.eval()

    # 2) Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 3) Parallel ENV generation 
    # obtain gripper indexes
    gripper_idx = None
    if task_name in ["door", "nut"]:
        gripper_idx = [7, 8]
    elif task_name == "two_arm":
        gripper_idx = [7, 8, 16, 17]      

    env_params_list: list[np.ndarray] = [None] * evaluation_samples
    env_settings_all: list[dict]      = [None] * evaluation_samples
    env_vision_list: list[np.ndarray] = [None] * evaluation_samples

    # choose worker count (env creation is CPU-bound)
    ENV_WORKERS = min(
        evaluation_samples,
        max(1, (os.cpu_count() or 4) - 2),
        int(os.getenv("EVAL_ENV_WORKERS", "10"))
    )

    ctx = get_context("spawn")  # safe with MuJoCo/OpenGL
    with ProcessPoolExecutor(max_workers=ENV_WORKERS, mp_context=ctx) as ex:
        futs = [ex.submit(_spawn_env_once, task_name, seed + i, i) for i in range(evaluation_samples)]
        for fut in as_completed(futs):
            idx, setting, params, vision = fut.result()
            env_settings_all[idx] = setting
            env_params_list[idx]  = params
            env_vision_list[idx]  = vision

    # stack to (N, Dc) float32 (order matches idx)
    eval_params = np.asarray(env_params_list, dtype=np.float32)

    if (use_vision):
        eval_visions = np.asarray(env_vision_list, dtype=np.float32)
        v = torch.from_numpy(eval_visions)

        # If data is NHWC, convert to NCHW (common for PyTorch encoders).
        if v.ndim == 4 and v.shape[-1] in (1, 3, 4):  # NHWC
            v = v.permute(0, 3, 1, 2).contiguous()
        v = v.to(device, non_blocking=True)

        VISION_BS = int(os.getenv("EVAL_VISION_BS", "64"))
        outs = []
        with torch.no_grad():
            for s in range(0, v.shape[0], VISION_BS):
                vb = v[s : s + VISION_BS]
                z = vision_encoder(vb)   # z: (B, D) or similar
                # If encoder returns a tuple/dict, adapt here.
                if isinstance(z, (tuple, list)):
                    z = z[0]
                outs.append(z.detach().cpu())

        eval_params = torch.cat(outs, dim=0).numpy().astype(np.float32) # overwrite eval_params to vision-based conditions

    # 4) Run evaluation 
    render_dir = render_dir + '/' + model_pt_path[-20:-1]
    success_rate_best, avg_reward_best = eval_model(model=model,
                                                model_class=model_class,
                                                task_name=task_name,
                                                seq_len=seq_len,
                                                dof=dof,
                                                param_len=param_len,
                                                gripper_idx=gripper_idx,
                                                render_dir=render_dir,
                                                video_name=video_name,
                                                val_params=eval_params,
                                                env_settings_all=env_settings_all,
                                                device=device,
                                                trials=evaluation_samples,
                                                render_width=4,
                                                render_num=8,
                                                base_seed=seed)
    print(f"Success rate : {success_rate_best:.3f}, Average reward : {avg_reward_best:.3f}")

    return


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
    parser.add_argument("--render_dir", type=str, default="./render_trained",
                        help="Directory to save rendered videos")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run the model on, e.g., 'cpu' or 'cuda'")
    parser.add_argument("--eval_samples", type=int, default=100)
    parser.add_argument("--use_vision", action="store_true")
    parser.add_argument("--vision_encoder_path", type=str, default=None)
    parser.add_argument("--vision_encoder_class", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1000,
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
        evaluation_samples=args.eval_samples,
        use_vision=args.use_vision,
        vision_encoder_path=args.vision_encoder_path,
        vision_encoder_class=args.vision_encoder_class,
        seed=args.seed
    )
