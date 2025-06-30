import time
import numpy as np
import torch
import torch.optim as optim
from datetime import datetime
import json
import concurrent.futures
import multiprocessing
from tqdm import tqdm
from zoneinfo import ZoneInfo
from Synthetic_data.distributions import *
from Synthetic_data.FM_utils import *

NUM_WORKERS = 5  # Number of parallel workers for training DGFM


def run_one_trial(n, trial_idx, dist_name, ambient_dim, latent_dim,
                  total_n_t, global_n_t, local_n_t, max_epochs, batch_size, early_stopping,
                  mf_list, test_size, device):
    """
    Runs one trial of both Vanilla FM and DGFM (for each mf in mf_list)
    on sample size n, returns a dict mapping method names to their
    per-epoch records.
    """
    # 1) instantiate distribution
    if dist_name == "Normal":
        dist = NormalDistribution(ambient_dim, device)
    else:
        dist = QuadraticManifoldDistribution(ambient_dim, device, latent_dim)

    # 2) sample training data
    X_train = dist.sample(n)

    trial_results = {}

    # ----- VanillaFM -----
    model_van = VectorField(ambient_dim).to(device)
    opt_van   = optim.Adam(model_van.parameters(), lr=1e-3)
    time_van  = 0.0

    # 3) train Vanilla FM
    t0 = time.thread_time()
    epoch, _, recs_van = train_vanilla_FM(
        model_van, opt_van, X_train,
        ambient_dim, device,
        n_t=total_n_t,
        epochs=max_epochs,  # you can pass eval_interval directly if you prefer
        batch_size=batch_size,
        early_stopping=early_stopping,
    )
    time_van += time.thread_time() - t0

    # eval
    X0   = np.random.randn(test_size, ambient_dim)
    Xgen = run_flow(model_van, X0, device)
    w2   = dist.wasserstein2_distance(Xgen.cpu().numpy(), test_size)
    geo  = dist.geometric_alignment(Xgen, device)

    recs_van.append({
        "final_epoch": epoch,
        "train_time": time_van,
        "eval_wasserstein2": w2,
        "eval_geometric_alignment": geo
    })

    trial_results["VanillaFM"] = recs_van

    # ----- DGFM (one entry per mf) -----
    for mf in mf_list:
        key_dg    = f"DGFM_mf{mf}"
        model_dg  = VectorField(ambient_dim).to(device)
        opt_dg    = optim.Adam(model_dg.parameters(), lr=1e-3)
        time_dg   = 0.0
        mixture_sampler = None

        t0 = time.thread_time()
        mixture_sampler, epoch, _, recs_dg = train_dgfm(
            model_dg, opt_dg,
            X_train, ambient_dim,
            mf, device,
            mixture_sampler=mixture_sampler,
            n_t_global=global_n_t,
            n_t_local=local_n_t,
            epochs=max_epochs,
            batch_size=batch_size,
            cluster_size=50,
            cluster_d=latent_dim,
            early_stopping=early_stopping
        )
        time_dg += time.thread_time() - t0

        # eval
        X0   = np.random.randn(test_size, ambient_dim)
        Xgen = run_flow(model_dg, X0, device)
        w2   = dist.wasserstein2_distance(Xgen.cpu().numpy(), test_size)
        geo  = dist.geometric_alignment(Xgen, device)

        recs_dg.append({
            "final_epoch": epoch,
            "train_time": time_dg,
            "eval_wasserstein2": w2,
            "eval_geometric_alignment": geo
        })

        trial_results[key_dg] = recs_dg

    return trial_idx, trial_results


if (__name__ == "__main__"):
    multiprocessing.set_start_method("spawn", force=True)

    print("Welcome to the Flow-Matching evaluation script!")
    print("You'll be prompted to select a distribution and experiment parameters.")
    print("This program trains Vanilla FM and DGFM on your chosen distribution, then measures:")
    print("  • Training time")
    print("  • Wasserstein-2 distance (W₂)")
    print("  • Geometric alignment to the support manifold")
    print("Results are saved in a nested JSON file with full metadata.\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}\n")

    # 1) pick sample size & trials
    sample_sizes = list(map(int,
                        (input("Enter sample sizes (comma separated, default 1000,2000,4000,8000): ")
                        .strip() or "1000,2000,4000,8000").split(",")))
    repeats  = int(input("Number of trials per config (default 5): ") or 5)

    # 2) experiment setup
    ambient_dim   = int(input("Ambient dimension (default 40): ") or 40)
    latent_dim    = int(input("Latent dimension (default 8): ") or 8)
    total_epochs  = int(input("Maximum number of epochs (default 100): ") or 100)
    batch_size    = int(input("Batch size (default 200): ") or 200)
    mf_list       = list(map(int,
                        (input("Multiplier for global FM (comma separated, default 2,4): ")
                        .strip() or "2,4").split(",")))
    total_n_t     = int(input("Number of timesteps per sample for vanilla FM (default 4): ") or 4)
    global_n_t     = int(input("Number of timesteps per sample for global FM (default 2): ") or 2)
    local_n_t     = int(input("Number of timesteps per sample for local FM (default 2): ") or 2)
    early_stopping = input("Use early stopping? (y/n, default y): ").strip().lower() != 'n'

    # 3) pick distribution
    print("Choose target distribution:")
    print("  [1] Normal")
    print("  [2] Quadratic")
    dkey = input(">>> ").strip()
    if dkey == "1":
        dist_name = "Normal"
    elif dkey == "2":
        dist_name = "Quadratic"
    else:
        print("Invalid choice."); exit(1)

    # 4) prepare nested results dict & run experiments
    data = {}
    test_size = 2000

    for n in sample_sizes:
        data[n] = {}
        # launch repeats trials in parallel
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=NUM_WORKERS,
                mp_context=ctx
            ) as executor:

            futures = {
                executor.submit(
                    run_one_trial, n, trial, dist_name,
                    ambient_dim, latent_dim,
                    total_n_t, global_n_t, local_n_t,
                    total_epochs, batch_size, early_stopping,
                    mf_list, test_size, device
                ): trial
                for trial in range(repeats)
            }

            for future in tqdm(concurrent.futures.as_completed(futures),
                               total=repeats,
                               desc=f"Trials @ n={n}"):
                trial_idx, trial_results = future.result()
                for method, recs in trial_results.items():
                    data[n].setdefault(method, {})[trial_idx] = recs

    # 5) write nested JSON
    """
    JSON format:
    {
    "date": "YYYY-MM-DD",
    "distribution": "<Normal|Quadratic|...>",
    "ambient_dim": <int>,
    "latent_dim": <int>,
    "sample_sizes": [<int>, ...],
    "vanilla_n_t": <int>,
    "global_n_t": <int>,
    "local_n_t": <int>,
    "early_stopping": <bool>,
    "mf_list": [<int>, ...],
    "results": {
        "<sample_size>": {
            "VanillaFM": {
                "<trial_index>": [
                    { "epoch": <int>, "validation_w2": <float>, "train_loss": <float> },
                    ...
                ],
                ...
            },
            "DGFM_mf<multiplier>": {
                "<trial_index>": [ ... ],
                ...
            },
            ...
        },
        ...
    }
    "summary": {
        "<sample_size>": {
            "<method>": [
                { "final_epoch_mean": <float>, "final_epoch_std": <float>,
                  "train_time_per_epoch_mean": <float>, "train_time_per_epoch_std": <float>,
                  "eval_wasserstein2_mean": <float>, "eval_wasserstein2_std": <float>,
                  "eval_geometric_alignment_mean": <float>, "eval_geometric_alignment_std": <float> },
                ...
            ],
            ...
        },
        ...
    }
    }   
    """
    print("Evaluation complete! Writing results...")

    summary = {}
    for n in sample_sizes:
        summary[n] = {}
        for method, trials in data[n].items():
            final_epochs = [r[-1]['final_epoch'] for r in trials.values()]
            times_per_epoch = [r[-1]['train_time']/r[-1]['final_epoch'] for r in trials.values()]
            w2s = [r[-1]['eval_wasserstein2'] for r in trials.values()]
            geos= [r[-1]['eval_geometric_alignment'] for r in trials.values()]
            summary[n][method] = {
                "final_epoch_mean": float(np.mean(final_epochs)),
                "final_epoch_std":  float(np.std(final_epochs)),
                "train_time_per_epoch_mean": float(np.mean(times_per_epoch)),
                "train_time_per_epoch_std":  float(np.std(times_per_epoch)),
                "eval_wasserstein2_mean":    float(np.mean(w2s)),
                "eval_wasserstein2_std":     float(np.std(w2s)),
                "eval_geometric_alignment_mean": float(np.mean(geos)),
                "eval_geometric_alignment_std":  float(np.std(geos))
            }

    fname = f"Synthetic_data/eval_results/eval_{dist_name}_{datetime.now(ZoneInfo("Asia/Seoul")).isoformat()}.json"
    with open(fname, 'w') as f:
        json.dump({
            "datetime" : datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
            "distribution": dist_name,
            "ambient_dim": ambient_dim,
            "latent_dim": latent_dim,
            "sample_sizes": sample_sizes,
            "vanilla_n_t": total_n_t,
            "global_n_t": global_n_t,
            "local_n_t": local_n_t,
            "early_stopping": early_stopping,
            "mf_list": mf_list,
            "results": data,
            "summary": summary
        }, f, indent=2)
    print(f"Results written to {fname}")
