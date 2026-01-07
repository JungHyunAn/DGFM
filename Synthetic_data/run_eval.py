import os
import time
import random
import numpy as np
import torch
import torch.optim as optim
from datetime import datetime
import json
import concurrent.futures
import multiprocessing
from tqdm import tqdm
from zoneinfo import ZoneInfo
from Synthetic_data.distributions import NormalDistribution, \
                                         Quadratic_Uniform, \
                                         Quadratic_Unimodal, \
                                         Quadratic_Multimodal, \
                                         Linear_Branched, \
                                         SwissRoll
from Synthetic_data.FM_utils import VectorField, \
                                    train_uniform_FM, \
                                    train_shifted_FM, \
                                    train_dgfm, \
                                    train_gfm,  \
                                    train_lfm,  \
                                    run_flow

NUM_WORKERS = 5  # Number of parallel workers for training DGFM


def run_one_trial(n, seed, trial_idx, dist_name, ambient_dim, latent_dim, beta_a, beta_b,
                  total_n_t, global_n_t, local_n_t, max_steps, batch_size, truncation, cluster_num,
                  early_stopping, mf_list, test_size, device, noise_std=1e-4):
    """
    Runs one trial of both Vanilla FM and DGFM (for each mf in mf_list)
    on sample size n, returns a dict mapping method names to their
    per-epoch records.
    """
    # 1) instantiate distribution & set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if dist_name == "Normal":
        dist = NormalDistribution(ambient_dim, device, noise_std=noise_std)
    elif dist_name == "Quadratic_Uniform":
        dist = Quadratic_Uniform(ambient_dim, device, latent_dim, noise_std=noise_std)
    elif dist_name == "Quadratic_Unimodal":
        dist = Quadratic_Unimodal(ambient_dim, device, latent_dim, noise_std=noise_std)
    elif dist_name == "Quadratic_Multimodal":
        dist = Quadratic_Multimodal(ambient_dim, device, latent_dim, noise_std=noise_std)
    elif dist_name == "Linear_Branched":
        dist = Linear_Branched(ambient_dim, device, latent_dim, noise_std=noise_std)
    elif dist_name == "SwissRoll":
        dist = SwissRoll(ambient_dim, device, latent_dim, noise_std=noise_std)
    else:
        raise ValueError(f"Unknown distribution: {dist_name}")
    
    
    # 2) sample training data
    X_train = dist.sample(n)

    trial_results = {}

    # ----- VanillaFM -----
    model_van = VectorField(ambient_dim).to(device)
    opt_van   = optim.Adam(model_van.parameters(), lr=1e-3)
    time_van  = 0.0

    # 3) train FM with uniform t sampling
    t0 = time.thread_time()
    epoch, _, recs_uniformFM, best_model_van = train_uniform_FM(
        model_van, opt_van, X_train,
        ambient_dim, device,
        n_t=total_n_t,
        epochs=int(max_steps * batch_size * 10 / n / 9),
        batch_size=batch_size,
        early_stopping=early_stopping,
    )
    time_van += time.thread_time() - t0

    # eval
    X0   = np.random.randn(test_size, ambient_dim)
    Xgen = run_flow(best_model_van, X0, device)
    w2   = dist.wasserstein2_distance(Xgen.cpu().numpy(), test_size)
    geo  = dist.geometric_alignment(Xgen)

    recs_uniformFM.append({
        "final_epoch": epoch,
        "train_time": time_van,
        "eval_wasserstein2": w2,
        "eval_geometric_alignment": geo
    })

    trial_results["UniformFM"] = recs_uniformFM

    # 4) train FM with shifted t sampling
    model_shifted = VectorField(ambient_dim).to(device)
    opt_shifted   = optim.Adam(model_shifted.parameters(), lr=1e-3)
    time_shifted  = 0.0

    t0 = time.thread_time()
    epoch, _, recs_shiftedFM, best_model_shifted = train_shifted_FM(
        model_shifted, opt_shifted, X_train,
        ambient_dim, device,
        n_t=total_n_t,
        epochs=int(max_steps * batch_size * 10 / n / 9),  
        batch_size=batch_size,
        early_stopping=early_stopping,
        beta_a=beta_a, beta_b=beta_b
    )
    time_shifted += time.thread_time() - t0

    # eval
    X0   = np.random.randn(test_size, ambient_dim)
    Xgen = run_flow(best_model_shifted, X0, device)
    w2   = dist.wasserstein2_distance(Xgen.cpu().numpy(), test_size)
    geo  = dist.geometric_alignment(Xgen)

    recs_shiftedFM.append({
        "final_epoch": epoch,
        "train_time": time_shifted,
        "eval_wasserstein2": w2,
        "eval_geometric_alignment": geo
    })

    trial_results["ShiftedFM"] = recs_shiftedFM


    cluster_size = min(int(n/cluster_num), latent_dim+5)

    # 5) train DGFM (one entry per mf)
    for mf in mf_list:
        key_dg    = f"DGFM-{mf}"
        model_dg  = VectorField(ambient_dim).to(device)
        opt_dg    = optim.Adam(model_dg.parameters(), lr=1e-3)
        time_dg   = 0.0
        mixture_sampler = None

        t0 = time.thread_time()
        mixture_sampler, epoch, _, recs_dg, best_model_dg = train_dgfm(
            model_dg, opt_dg,
            X_train, ambient_dim,
            mf, device,
            mixture_sampler=mixture_sampler,
            n_t_global=global_n_t,
            n_t_local=local_n_t,
            epochs=int(max_steps * batch_size * 10 / n / 9 / (1+mf)),
            batch_size=batch_size,
            cluster_size=cluster_size,
            cluster_d=latent_dim,
            truncation=truncation,
            early_stopping=early_stopping
        )
        time_dg += time.thread_time() - t0

        # eval
        X0   = np.random.randn(test_size, ambient_dim)
        Xgen = run_flow(best_model_dg, X0, device)
        w2   = dist.wasserstein2_distance(Xgen.cpu().numpy(), test_size)
        geo  = dist.geometric_alignment(Xgen)

        recs_dg.append({
            "final_epoch": epoch,
            "train_time": time_dg,
            "eval_wasserstein2": w2,
            "eval_geometric_alignment": geo
        })

        trial_results[key_dg] = recs_dg

    # 6) train GFM only
    model_GFM       = VectorField(ambient_dim).to(device)
    opt_GFM         = optim.Adam(model_GFM.parameters(), lr=1e-3)
    time_GFM        = 0.0
    mixture_sampler = None

    t0 = time.thread_time()
    mixture_sampler, epoch, _, recs_GFM, best_model_GFM = train_gfm(
        model_GFM, opt_GFM, X_train,
        ambient_dim, device,
        mixture_sampler=mixture_sampler,
        n_t_global=global_n_t,
        epochs=int(max_steps * batch_size * 10 / n / 9),
        batch_size=batch_size,
        cluster_size=cluster_size,
        cluster_d=latent_dim,
        truncation=truncation,
        early_stopping=early_stopping
    )
    time_GFM += time.thread_time() - t0

    # eval
    X0   = np.random.randn(test_size, ambient_dim)
    Xgen = run_flow(best_model_GFM, X0, device)
    w2   = dist.wasserstein2_distance(Xgen.cpu().numpy(), test_size)
    geo  = dist.geometric_alignment(Xgen)

    recs_GFM.append({
        "final_epoch": epoch,
        "train_time": time_GFM,
        "eval_wasserstein2": w2,
        "eval_geometric_alignment": geo
    })

    trial_results["GFM"] = recs_GFM

    # 7) train LFM only
    model_LFM       = VectorField(ambient_dim).to(device)
    opt_LFM         = optim.Adam(model_LFM.parameters(), lr=1e-3)
    time_LFM        = 0.0
    mixture_sampler = None

    t0 = time.thread_time()
    mixture_sampler, epoch, _, recs_LFM, best_model_LFM = train_lfm(
        model_LFM, opt_LFM, X_train,
        ambient_dim, device,
        mixture_sampler=mixture_sampler,
        n_t_local=local_n_t,
        epochs=int(max_steps * batch_size * 10 / n / 9),
        batch_size=batch_size,
        cluster_size=cluster_size,
        cluster_d=latent_dim,
        truncation=truncation,
        early_stopping=early_stopping
    )
    time_LFM += time.thread_time() - t0

    # eval
    X0, _ = mixture_sampler.truncated_sample(test_size)
    Xgen  = run_flow(best_model_LFM, X0, device)
    w2    = dist.wasserstein2_distance(Xgen.cpu().numpy(), test_size)
    geo   = dist.geometric_alignment(Xgen)

    recs_LFM.append({
        "final_epoch": epoch,
        "train_time": time_LFM,
        "eval_wasserstein2": w2,
        "eval_geometric_alignment": geo
    })

    trial_results["LFM"] = recs_LFM

    # 8) GMM baseline
    Xgen, _ = mixture_sampler.truncated_sample(test_size)
    w2    = dist.wasserstein2_distance(Xgen.cpu().numpy(), test_size)
    geo   = dist.geometric_alignment(Xgen)

    recs_GMM = [{"final_epoch": 1,
                 "train_time": 0,
                 "eval_wasserstein2": w2,
                 "eval_geometric_alignment": geo}]

    trial_results["GMM"] = recs_GMM


    return trial_idx, trial_results


if (__name__ == "__main__"):
    multiprocessing.set_start_method("spawn", force=True)

    print("Welcome to the Flow-Matching evaluation script!")
    print("You'll be prompted to select a distribution and experiment parameters.")
    print("This program trains Vanilla FM with uniform/shifted t sampling and DGFM on your chosen distribution, then measures:")
    print("  • Training time")
    print("  • Wasserstein-2 distance (W₂)")
    print("  • Geometric alignment to the support manifold")
    print("Results are saved in a nested JSON file with full metadata.\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}\n")

    # 1) pick sample size & trials & seed
    sample_sizes = list(map(int,
                        (input("Enter sample sizes (comma separated, default 250,500,1000,2000,4000): ")
                        .strip() or "250,500,1000,2000,4000").split(",")))
    repeats      = int(input("Number of trials per config (default 5): ") or 5)
    seed         = int(input("Input the seed (default 1000): ")           or 1000)

    # 2) experiment setup
    ambient_dim    = int(input("Ambient dimension (default 80): ")              or 80)
    latent_dim     = int(input("Latent dimension (default 20): ")               or 20)
    total_steps    = int(input("Maximum number of opt. steps (default 2880): ") or 2880)
    batch_size     = int(input("Minimum Batch size (default 64): ")             or 64)
    batch_num      = int(input("Number of batches per epoch (default 8): ")     or 8)
    beta_a, beta_b = list(map(float,
                        (input("Beta parameters for shifted FM (comma separated, default 1.5,1): ")
                        .strip() or "1.5,1").split(",")))
    mf_list        = list(map(int,
                        (input("DGFM multiplier for global FM (comma separated, default 4,8): ")
                        .strip() or "4,8").split(",")))

    total_n_t      = int(input("Number of timesteps per sample for vanilla FM (default 1): ")     or 1)
    global_n_t     = int(input("Number of timesteps per sample for DGFM global FM (default 1): ") or 1)
    local_n_t      = int(input("Number of timesteps per sample for DGFM local FM (default 1): ")  or 1)
    noise_std      = float(input("Noise level of distribution (default 1e-4): ")                  or 1e-4)
    truncation     = float(input("Truncation of GMM (default 1.5): ")                             or 1.5)
    cluster_num    = int(input("Number of expected clusters (default 8): ")                       or 8)

    early_stopping = input("Use early stopping if validation loss doesn't improve for three epochs? (y/n, default n): ").strip().lower() == 'y'

    # 3) pick distribution
    print("Choose target distribution:")
    print("  [1] Normal")
    print("  [2] Quadratic_Uniform")
    print("  [3] Quadratic_Unimodal")
    print("  [4] Quadratic_Multimodal")
    print("  [5] Branched Linear")
    print("  [6] SwissRoll")
    dkey = input(">>> ").strip()
    if dkey == "1":
        dist_name = "Normal"
    elif dkey == "2":
        dist_name = "Quadratic_Uniform"
    elif dkey == "3":
        dist_name = "Quadratic_Unimodal"
    elif dkey == "4":
        dist_name = "Quadratic_Multimodal"
    elif dkey == "5":
        dist_name = "Linear_Branched"
    elif dkey == "6":
        dist_name = "SwissRoll"
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
                    run_one_trial, n, seed + trial, trial, dist_name, ambient_dim, latent_dim, beta_a, beta_b,
                    total_n_t, global_n_t, local_n_t, total_steps, max(batch_size, int(0.9 * n / batch_num)), truncation, cluster_num,
                    early_stopping, mf_list, test_size, device, noise_std
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
            "DGFM-<multiplier>": {
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
                "final_epoch_mean"             : float(np.mean(final_epochs)),
                "final_epoch_std"              : float(np.std(final_epochs)),
                "train_time_per_epoch_mean"    : float(np.mean(times_per_epoch)),
                "train_time_per_epoch_std"     : float(np.std(times_per_epoch)),
                "eval_wasserstein2_mean"       : float(np.mean(w2s)),
                "eval_wasserstein2_std"        : float(np.std(w2s)),
                "eval_geometric_alignment_mean": float(np.mean(geos)),
                "eval_geometric_alignment_std" : float(np.std(geos))
            }

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fname = f"Synthetic_data/eval_results/eval_{dist_name}_{timestamp}.json"

    os.makedirs(os.path.dirname(fname), exist_ok=True)

    with open(fname, 'w') as f:
        json.dump({
            "datetime"      : timestamp,
            "seed"          : seed,
            "distribution"  : dist_name,
            "ambient_dim"   : ambient_dim,
            "latent_dim"    : latent_dim,
            "total_steps"   : total_steps,
            "sample_sizes"  : sample_sizes,
            "beta_a"        : beta_a,
            "beta_b"        : beta_b,
            "vanilla_n_t"   : total_n_t,
            "global_n_t"    : global_n_t,
            "local_n_t"     : local_n_t,
            "noise_std"     : noise_std,
            "truncation"    : truncation,
            "cluster_num"   : cluster_num,
            "early_stopping": early_stopping,
            "mf_list"       : mf_list,
            "results"       : data,
            "summary"       : summary
        }, f, indent=2)
    print(f"Results written to {fname}")
