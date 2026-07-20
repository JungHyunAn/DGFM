"""Train and evaluate UniformFM and condition-free DGFMv2 on synthetic data."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from Synthetic_data.distributions import (
    Linear_Branched,
    NormalDistribution,
    PinWheel,
    Quadratic_Multimodal,
    Quadratic_Unimodal,
    Quadratic_Uniform,
    SwissRoll,
    TwoMoon,
)
from Synthetic_data.models import DGFMv2, UniformFM, VectorField, run_flow


DISTRIBUTIONS = {
    "1": "Normal",
    "2": "Quadratic_Uniform",
    "3": "Quadratic_Unimodal",
    "4": "Quadratic_Multimodal",
    "5": "Linear_Branched",
    "6": "SwissRoll",
    "7": "TwoMoon",
    "8": "PinWheel",
}


def get_cosine_schedule_with_warmup(
    optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_scale: float = 0.05,
):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * max(0.0, min(1.0, progress))))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return LambdaLR(optimizer, lr_lambda)


def make_distribution(name: str, ambient_dim: int, latent_dim: int, device, noise_std: float):
    constructors = {
        "Normal": lambda: NormalDistribution(ambient_dim, device),
        "Quadratic_Uniform": lambda: Quadratic_Uniform(
            ambient_dim, device, latent_dim, noise_std=noise_std
        ),
        "Quadratic_Unimodal": lambda: Quadratic_Unimodal(
            ambient_dim, device, latent_dim, noise_std=noise_std
        ),
        "Quadratic_Multimodal": lambda: Quadratic_Multimodal(
            ambient_dim, device, latent_dim, noise_std=noise_std
        ),
        "Linear_Branched": lambda: Linear_Branched(
            ambient_dim, device, latent_dim, noise_std=noise_std
        ),
        "SwissRoll": lambda: SwissRoll(ambient_dim, device, latent_dim, noise_std=noise_std),
        "TwoMoon": lambda: TwoMoon(ambient_dim, device, latent_dim, noise_std=noise_std),
        "PinWheel": lambda: PinWheel(ambient_dim, device, latent_dim, noise_std=noise_std),
    }
    try:
        return constructors[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown distribution {name!r}") from exc


def _optimizer_and_scheduler(model, total_steps: int):
    optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=int(0.2 * total_steps),
        total_steps=total_steps,
    )
    return optimizer, scheduler


def _evaluate(model, distribution, ambient_dim: int, test_size: int, device) -> dict:
    generated = run_flow(model, np.random.randn(test_size, ambient_dim), device)
    alignment = distribution.geometric_alignment(generated)
    if torch.is_tensor(alignment):
        alignment = alignment.float().mean().item()
    else:
        alignment = float(np.asarray(alignment).mean())
    return {
        "eval_wasserstein2": float(
            distribution.wasserstein2_distance(generated.cpu().numpy(), test_size)
        ),
        "eval_geometric_alignment": alignment,
    }


def run_one_trial(
    *,
    sample_size: int,
    seed: int,
    trial_idx: int,
    distribution_name: str,
    ambient_dim: int,
    latent_dim: int,
    total_steps: int,
    batch_size: int,
    batch_num: int,
    total_n_t: int,
    global_n_t: int,
    local_n_t: int,
    noise_std: float,
    truncation: float,
    cluster_num: int,
    injection_time: float,
    early_stopping: bool,
    test_size: int,
    device,
    progress: bool = True,
) -> dict:
    """Train the two supported evaluation methods for one seeded dataset."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    distribution = make_distribution(
        distribution_name, ambient_dim, latent_dim, device, noise_std
    )
    target = distribution.sample(sample_size)
    train_size = sample_size - min(max(1, int(0.1 * sample_size)), 2000)
    effective_batch_size = max(batch_size, math.ceil(train_size / batch_num))
    batches_per_epoch = math.ceil(train_size / effective_batch_size)
    max_epochs = max(1, math.ceil(total_steps / batches_per_epoch))
    cluster_size = max(2, math.ceil(train_size / cluster_num))

    results = {}

    model = VectorField(ambient_dim).to(device)
    optimizer, scheduler = _optimizer_and_scheduler(model, total_steps)
    trainer = UniformFM(model, optimizer, scheduler, device=device, time_sampling="uniform")
    started = time.perf_counter()
    best_model, _, records = trainer.train(
        target,
        n_t=total_n_t,
        max_epochs=max_epochs,
        batch_size=effective_batch_size,
        early_stopping=early_stopping,
        progress=progress,
    )
    uniform_time = time.perf_counter() - started
    results["UniformFM"] = {
        "records": records,
        "final_epoch": records[-1]["epoch"],
        "train_time": uniform_time,
        **_evaluate(best_model, distribution, ambient_dim, test_size, device),
    }

    model = VectorField(ambient_dim).to(device)
    optimizer, scheduler = _optimizer_and_scheduler(model, total_steps)
    trainer = DGFMv2(model, optimizer, scheduler, device=device, time_sampling="uniform")
    started = time.perf_counter()
    best_model, _, records, sampler = trainer.train(
        target,
        n_t=total_n_t,
        cluster_size=cluster_size,
        max_epochs=max_epochs,
        batch_size=effective_batch_size,
        injection_time=injection_time,
        truncation=truncation,
        early_stopping=early_stopping,
        progress=progress,
    )
    dgfm_time = time.perf_counter() - started
    results["DGFMv2"] = {
        "records": records,
        "final_epoch": records[-1]["epoch"],
        "train_time": dgfm_time,
        "cluster_count": len(sampler.clusters),
        **_evaluate(best_model, distribution, ambient_dim, test_size, device),
    }

    return {
        "datetime": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "seed": seed,
        "trial_idx": trial_idx,
        "distribution": distribution_name,
        "sample_size": sample_size,
        "ambient_dim": ambient_dim,
        "latent_dim": latent_dim,
        "total_steps": total_steps,
        "batch_size": batch_size,
        "effective_batch_size": effective_batch_size,
        "batch_num": batch_num,
        "max_epochs": max_epochs,
        "total_n_t": total_n_t,
        "global_n_t": global_n_t,
        "local_n_t": local_n_t,
        "noise_std": noise_std,
        "truncation": truncation,
        "cluster_num": cluster_num,
        "cluster_size": cluster_size,
        "int_inject": injection_time,
        "early_stopping": early_stopping,
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample_size", type=int, required=True)
    parser.add_argument("--trial_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--target_distribution",
        choices=tuple(DISTRIBUTIONS),
        default="6",
        help="Distribution number: 6=SwissRoll, 7=TwoMoon, 8=PinWheel.",
    )
    parser.add_argument("--ambient_dim", type=int, default=40)
    parser.add_argument("--latent_dim", type=int, default=10)
    parser.add_argument("--total_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--batch_num", type=int, default=8)
    parser.add_argument("--total_n_t", type=int, default=1)
    parser.add_argument("--global_n_t", type=int, default=1)
    parser.add_argument("--local_n_t", type=int, default=1)
    parser.add_argument("--noise_std", type=float, default=1e-4)
    parser.add_argument("--truncation", type=float, default=1.5)
    parser.add_argument("--cluster_num", type=int, required=True)
    parser.add_argument("--int_inject", type=float, default=0.5)
    parser.add_argument("--early_stopping", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--test_size", type=int, default=2000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--results_path", type=Path, default=Path("Synthetic_data/eval_results"))
    parser.add_argument("--no_progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.sample_size, args.total_steps, args.batch_size, args.batch_num, args.cluster_num) <= 0:
        raise ValueError(
            "sample_size, total_steps, batch_size, batch_num, and cluster_num must be positive"
        )

    result = run_one_trial(
        sample_size=args.sample_size,
        seed=args.seed,
        trial_idx=args.trial_idx,
        distribution_name=DISTRIBUTIONS[args.target_distribution],
        ambient_dim=args.ambient_dim,
        latent_dim=args.latent_dim,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        batch_num=args.batch_num,
        total_n_t=args.total_n_t,
        global_n_t=args.global_n_t,
        local_n_t=args.local_n_t,
        noise_std=args.noise_std,
        truncation=args.truncation,
        cluster_num=args.cluster_num,
        injection_time=args.int_inject,
        early_stopping=args.early_stopping,
        test_size=args.test_size,
        device=torch.device(args.device),
        progress=not args.no_progress,
    )
    args.results_path.mkdir(parents=True, exist_ok=True)
    output_path = args.results_path / "results.json"
    with output_path.open("w") as output:
        json.dump(result, output, indent=2)
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
