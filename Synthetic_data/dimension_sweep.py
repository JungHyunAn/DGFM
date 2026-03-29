import argparse
import json
import math
import os
import random
import time
from datetime import datetime

import numpy as np
import ot
import torch
import torch.nn as nn
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
from Synthetic_data.FM_utils import train_dgfm, train_uniform_FM, run_flow


class VectorField(nn.Module):
    def __init__(self, dim):
        super().__init__()
        time_hidden = max(100, min(2 * dim, 1000))
        hidden_small = max(200, min(4 * dim, 2000))
        hidden_large = max(400, min(8 * dim, 4000))

        self.time_encoder = nn.Sequential(
            nn.Linear(1, time_hidden),
            nn.SiLU(),
            nn.Linear(time_hidden, time_hidden),
            nn.SiLU(),
        )

        self.net = nn.Sequential(
            nn.Linear(dim + time_hidden, hidden_small),
            nn.SiLU(),
            nn.Linear(hidden_small, hidden_large),
            nn.SiLU(),
            nn.Linear(hidden_large, hidden_large),
            nn.SiLU(),
            nn.Linear(hidden_large, hidden_small),
            nn.SiLU(),
            nn.Linear(hidden_small, dim),
        )

    def forward(self, x, t):
        if t.dim() == 1:
            t = t.unsqueeze(1)
        t_encoded = self.time_encoder(t)
        return self.net(torch.cat([x, t_encoded], dim=1))


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_scale=0.05, last_step=-1):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = max(0.0, min(1.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return LambdaLR(optimizer, lr_lambda, last_step)


def instantiate_distribution(dist_name, ambient_dim, latent_dim, device, noise_std):
    if dist_name == "Normal":
        return NormalDistribution(ambient_dim, device, noise_std=noise_std)
    if dist_name == "Quadratic_Uniform":
        return Quadratic_Uniform(ambient_dim, device, latent_dim, noise_std=noise_std)
    if dist_name == "Quadratic_Unimodal":
        return Quadratic_Unimodal(ambient_dim, device, latent_dim, noise_std=noise_std)
    if dist_name == "Quadratic_Multimodal":
        return Quadratic_Multimodal(ambient_dim, device, latent_dim, noise_std=noise_std)
    if dist_name == "Linear_Branched":
        return Linear_Branched(ambient_dim, device, latent_dim, noise_std=noise_std)
    if dist_name == "SwissRoll":
        return SwissRoll(ambient_dim, device, latent_dim, noise_std=noise_std)
    if dist_name == "TwoMoon":
        return TwoMoon(ambient_dim, device, latent_dim, noise_std=noise_std)
    if dist_name == "PinWheel":
        return PinWheel(ambient_dim, device, latent_dim, noise_std=noise_std)
    raise ValueError(f"Unknown distribution: {dist_name}")


def evaluate_model(best_model, dist, ambient_dim, test_size, device):
    x0 = np.random.randn(test_size, ambient_dim)
    xgen = run_flow(best_model, x0, device)
    x_eval = dist.sample(test_size)
    return {
        "eval_wasserstein2": float(
            np.sqrt(
                ot.emd2(
                    np.ones(test_size) / test_size,
                    np.ones(test_size) / test_size,
                    ot.dist(x_eval.cpu().numpy(), xgen.cpu().numpy()) ** 2,
                )
            )
        ),
        "eval_geometric_alignment": float(dist.geometric_alignment(xgen)),
    }


def print_final_metrics(method_name, metrics):
    print(
        f"  {method_name}: "
        f"W2={metrics['eval_wasserstein2']:.6f}, "
        f"Geo={metrics['eval_geometric_alignment']:.6f}, "
        f"epoch={metrics['final_epoch']}, "
        f"train_time={metrics['train_time']:.2f}s"
    )


def run_uniform_fm(X_train, dist, ambient_dim, total_n_t, max_steps, batch_size, early_stopping, test_size, device):
    epochs = int(max_steps * batch_size * 10 / X_train.shape[0] / 9)
    model = VectorField(ambient_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.2 * max_steps), max_steps)

    t0 = time.thread_time()
    epoch, _, records, best_model = train_uniform_FM(
        model,
        optimizer,
        scheduler,
        X_train,
        ambient_dim,
        device,
        n_t=total_n_t,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping=early_stopping,
    )
    train_time = time.thread_time() - t0

    metrics = evaluate_model(best_model, dist, ambient_dim, test_size, device)
    records.append({
        "final_epoch": int(epoch),
        "train_time": float(train_time),
        **metrics,
    })
    return records


def run_dgfm(
    X_train,
    dist,
    ambient_dim,
    latent_dim,
    mf,
    global_n_t,
    local_n_t,
    max_steps,
    batch_size,
    truncation,
    cluster_num,
    early_stopping,
    test_size,
    device,
    intermediate_injection,
):
    epochs = int(max_steps * batch_size * 10 / X_train.shape[0] / 9 / (1 + mf))
    cluster_size = max(int(0.9 * X_train.shape[0] / cluster_num), latent_dim + 5)

    model = VectorField(ambient_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.2 * max_steps), max_steps)

    t0 = time.thread_time()
    _, epoch, _, records, best_model = train_dgfm(
        model,
        optimizer,
        scheduler,
        X_train,
        ambient_dim,
        mf,
        device,
        mixture_sampler=None,
        n_t_global=global_n_t,
        n_t_local=local_n_t,
        epochs=epochs,
        batch_size=batch_size,
        cluster_size=cluster_size,
        cluster_d=latent_dim,
        truncation=truncation,
        early_stopping=early_stopping,
        intermediate_injection=intermediate_injection,
    )
    train_time = time.thread_time() - t0

    metrics = evaluate_model(best_model, dist, ambient_dim, test_size, device)
    records.append({
        "final_epoch": int(epoch * (1 + mf) + mf),
        "train_time": float(train_time),
        **metrics,
    })
    return records


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep synthetic experiments over ambient dimension at fixed sample size.")
    parser.add_argument("--distribution", default="Quadratic_Unimodal")
    parser.add_argument("--sample-size", type=int, default=1280)
    parser.add_argument("--ambient-dims", type=int, nargs="+", default=[80, 200, 500])
    parser.add_argument("--latent-dim", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-steps-list", type=int, nargs="+", default=[1000, 2000, 4000])
    parser.add_argument("--min-batch-size", type=int, default=72)
    parser.add_argument("--batch-num", type=int, default=4)
    parser.add_argument("--total-n-t", type=int, default=1)
    parser.add_argument("--global-n-t", type=int, default=1)
    parser.add_argument("--local-n-t", type=int, default=1)
    parser.add_argument("--mf", type=int, default=4)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--noise-std", type=float, default=1e-4)
    parser.add_argument("--truncation", type=float, default=1.5)
    parser.add_argument("--cluster-num", type=int, default=8)
    parser.add_argument("--intermediate-injection", type=float, default=0.5)
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    results = {}
    summary = {}
    n = args.sample_size
    max_steps_list = args.max_steps_list if args.max_steps_list is not None else [args.max_steps]
    if len(args.ambient_dims) != len(max_steps_list):
        raise ValueError(
            "ambient_dims and max_steps_list must have the same length. "
            f"Got {len(args.ambient_dims)} ambient dims and {len(max_steps_list)} max-step values."
        )

    for ambient_dim, max_steps in zip(args.ambient_dims, max_steps_list):
        print(
            f"Running n={n}, ambient_dim={ambient_dim}, "
            f"latent_dim={args.latent_dim}, max_steps={max_steps}"
        )
        dist = instantiate_distribution(
            args.distribution,
            ambient_dim,
            args.latent_dim,
            device,
            args.noise_std,
        )
        x_train = dist.sample(n)
        batch_size = max(args.min_batch_size, int(0.9 * n / args.batch_num))

        uniform_records = run_uniform_fm(
            x_train,
            dist,
            ambient_dim,
            args.total_n_t,
            max_steps,
            batch_size,
            args.early_stopping,
            args.test_size,
            device,
        )
        dgfm_records = run_dgfm(
            x_train,
            dist,
            ambient_dim,
            args.latent_dim,
            args.mf,
            args.global_n_t,
            args.local_n_t,
            max_steps,
            batch_size,
            args.truncation,
            args.cluster_num,
            args.early_stopping,
            args.test_size,
            device,
            args.intermediate_injection,
        )

        results[ambient_dim] = {
            "max_steps": max_steps,
            "UniformFM": uniform_records,
            f"DGFM-{args.mf}": dgfm_records,
        }
        summary[ambient_dim] = {
            "max_steps": max_steps,
            "UniformFM": uniform_records[-1],
            f"DGFM-{args.mf}": dgfm_records[-1],
        }

        print_final_metrics("UniformFM", uniform_records[-1])
        print_final_metrics(f"DGFM-{args.mf}", dgfm_records[-1])

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = args.output or f"Synthetic_data/eval_results/dimension_sweep_{args.distribution}_{timestamp}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "datetime": timestamp,
                "seed": args.seed,
                "distribution": args.distribution,
                "sample_size": args.sample_size,
                "ambient_dims": args.ambient_dims,
                "latent_dim": args.latent_dim,
                "max_steps": args.max_steps,
                "max_steps_list": max_steps_list,
                "min_batch_size": args.min_batch_size,
                "batch_num": args.batch_num,
                "vanilla_n_t": args.total_n_t,
                "global_n_t": args.global_n_t,
                "local_n_t": args.local_n_t,
                "mf": args.mf,
                "noise_std": args.noise_std,
                "truncation": args.truncation,
                "cluster_num": args.cluster_num,
                "intermediate_injection": args.intermediate_injection,
                "early_stopping": args.early_stopping,
                "results": results,
                "summary": summary,
            },
            f,
            indent=2,
        )

    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
