import os
import time
import random
import numpy as np
import torch
import torch.optim as optim
from datetime import datetime
import json
from tqdm import tqdm
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use("Agg")  # safe for headless servers
import matplotlib.pyplot as plt

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


def _instantiate_dist(dist_name, ambient_dim, latent_dim, device):
    if dist_name == "Normal":
        return NormalDistribution(ambient_dim, device)
    elif dist_name == "Quadratic_Uniform":
        return Quadratic_Uniform(ambient_dim, device, latent_dim)
    elif dist_name == "Quadratic_Unimodal":
        return Quadratic_Unimodal(ambient_dim, device, latent_dim)
    elif dist_name == "Quadratic_Multimodal":
        return Quadratic_Multimodal(ambient_dim, device, latent_dim)
    elif dist_name == "Linear_Branched":
        return Linear_Branched(ambient_dim, device, latent_dim)
    elif dist_name == "SwissRoll":
        return SwissRoll(ambient_dim, device, latent_dim)
    else:
        raise ValueError(f"Unknown distribution: {dist_name}")


def _run_single_trial(seed, dist, X_train, ambient_dim, device,
                      mf, cluster_d, global_n_t, local_n_t, max_epochs, batch_size, cluster_size, early_stopping,
                      test_size):
    # Per-trial seeds
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = VectorField(ambient_dim).to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    mixture_sampler = None

    t0 = time.thread_time()
    mixture_sampler, epoch, _, recs_dg, best_model = train_dgfm(
        model, opt,
        X_train, ambient_dim,
        mf, device,
        mixture_sampler=mixture_sampler,
        n_t_global=global_n_t,
        n_t_local=local_n_t,
        epochs=max_epochs,
        batch_size=batch_size,
        cluster_size=cluster_size,
        cluster_d=int(cluster_d),
        early_stopping=early_stopping
    )
    train_time = time.thread_time() - t0

    X0   = np.random.randn(test_size, ambient_dim)
    Xgen = run_flow(best_model, X0, device)
    w2   = dist.wasserstein2_distance(Xgen.cpu().numpy(), test_size)
    geo  = dist.geometric_alignment(Xgen)

    return {
        "final_epoch": int(epoch),
        "train_time": float(train_time),
        "w2": float(w2),
        "geo": float(geo),
    }


def _run_single_trial_GFM(seed, dist, X_train, ambient_dim, device,
                      cluster_d, global_n_t, max_epochs, batch_size, cluster_size, early_stopping,
                      test_size):
    # Per-trial seeds
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = VectorField(ambient_dim).to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    mixture_sampler = None

    t0 = time.thread_time()
    mixture_sampler, epoch, _, recs_dg, best_model = train_gfm(
        model, opt,
        X_train, ambient_dim,
        device,
        mixture_sampler=mixture_sampler,
        n_t_global=global_n_t,
        epochs=max_epochs,
        batch_size=batch_size,
        cluster_size=cluster_size,
        cluster_d=int(cluster_d),
        early_stopping=early_stopping
    )
    train_time = time.thread_time() - t0

    X0   = np.random.randn(test_size, ambient_dim)
    Xgen = run_flow(best_model, X0, device)
    w2   = dist.wasserstein2_distance(Xgen.cpu().numpy(), test_size)
    geo  = dist.geometric_alignment(Xgen)

    return {
        "final_epoch": int(epoch),
        "train_time": float(train_time),
        "w2": float(w2),
        "geo": float(geo),
    }


def _run_single_trial_LFM(seed, dist, X_train, ambient_dim, device,
                      cluster_d, local_n_t, max_epochs, batch_size, cluster_size, early_stopping,
                      test_size):
    # Per-trial seeds
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = VectorField(ambient_dim).to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    mixture_sampler = None

    t0 = time.thread_time()
    mixture_sampler, epoch, _, recs_dg, best_model = train_lfm(
        model, opt,
        X_train, ambient_dim,
        device,
        mixture_sampler=mixture_sampler,
        n_t_local=local_n_t,
        epochs=max_epochs,
        batch_size=batch_size,
        cluster_size=cluster_size,
        cluster_d=int(cluster_d),
        early_stopping=early_stopping
    )
    train_time = time.thread_time() - t0

    X0, _ = mixture_sampler.truncated_sample(test_size)
    Xgen  = run_flow(best_model, X0, device)
    w2    = dist.wasserstein2_distance(Xgen.cpu().numpy(), test_size)
    geo   = dist.geometric_alignment(Xgen)

    return {
        "final_epoch": int(epoch),
        "train_time": float(train_time),
        "w2": float(w2),
        "geo": float(geo),
    }


def _flow_states_at_times(model, X0_np, device, t_list, steps_per_interval=50):
    """
    Returns dict {t: torch.Tensor(N,D)} by either:
      - calling run_flow(model, X0, device, times=t_list) if available, or
      - doing a simple explicit-Euler integration of the learned vector field.

    Assumes model eval mode and that model(x, t_tensor) or model(x, scalar_t) works.
    """
    model.eval()

    if torch.is_tensor(X0_np):
        x = X0_np.detach().clone().to(device=device, dtype=torch.float32)
    else:
        # numpy/list -> tensor
        x = torch.as_tensor(X0_np, dtype=torch.float32, device=device).clone()

    out = {t_list[0]: x.clone()} # zero
    t_prev = t_list[0]
    for t in t_list[1:]:
        nsteps = max(1, int(np.ceil((t - t_prev) * steps_per_interval)))
        dt = (t - t_prev) / nsteps
        for k in range(nsteps):
            tau = t_prev + (k + 1) * dt
            # IMPORTANT: 1D shape (N,), not (N,1)
            t_1d = torch.full((x.shape[0],), float(tau), device=x.device, dtype=x.dtype)
            try:
                v = model(x, t_1d)          # expected by VectorField.forward (will do t.unsqueeze(1) inside)
            except TypeError:
                v = model(x, float(tau))    # some models accept scalar t
            x = x + dt * v
        out[float(t)] = x.clone()
        t_prev = t
    return out


def run_dimension_analysis(n, repeats, seed, dist_name, ambient_dim, latent_dim, dim_list,
                           global_n_t, local_n_t, max_epochs, batch_size, early_stopping,
                           mf_list, test_size, device):
    """
    Multiple-trial dimensionality misspecification analysis for DGFM.

    Returns:
        results: {
            mf: {
                dim: {
                    "trials": [ {final_epoch, train_time, w2, geo}, ... ],
                    "summary": {
                        "w2_mean", "w2_std", "geo_mean", "geo_std",
                        "final_epoch_mean", "final_epoch_std",
                        "train_time_mean", "train_time_std"
                    }
                }, ...
            }, ...
        }
    """
    # Global seeds
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    cluster_size = int(n/5)

    dist    = _instantiate_dist(dist_name, ambient_dim, latent_dim, device)
    X_train = dist.sample(n)

    results        = {f"DGFM-{mf}": dict() for mf in mf_list}
    results["GFM"] = dict()
    results["LFM"] = dict()

    for mf in mf_list:
        for dim in dim_list:
            trial_metrics = []
            print(f"[DGFM mf={mf}] cluster_d={dim}: running {repeats} trial(s)…")

            for t in tqdm(range(repeats), desc=f"Trials (mf={mf}, d={dim})"):
                trial_seed = seed + t
                m = _run_single_trial(
                    seed=trial_seed,
                    dist=dist,
                    X_train=X_train,
                    ambient_dim=ambient_dim,
                    device=device,
                    mf=mf,
                    cluster_d=dim,
                    global_n_t=global_n_t,
                    local_n_t=local_n_t,
                    max_epochs=max_epochs,
                    batch_size=batch_size,
                    cluster_size=cluster_size,
                    early_stopping=early_stopping,
                    test_size=test_size
                )
                trial_metrics.append(m)

            # Aggregate stats
            w2s  = np.array([m["w2"] for m in trial_metrics], dtype=float)
            geos = np.array([m["geo"] for m in trial_metrics], dtype=float)
            eps  = np.array([m["final_epoch"] for m in trial_metrics], dtype=float)
            tps  = np.array([m["train_time"] for m in trial_metrics], dtype=float)

            results[f"DGFM-{mf}"][int(dim)] = {
                "trials": trial_metrics,
                "summary": {
                    "w2_mean":  float(w2s.mean()),  "w2_std":  float(w2s.std(ddof=0)),
                    "geo_mean": float(geos.mean()), "geo_std": float(geos.std(ddof=0)),
                    "final_epoch_mean": float(eps.mean()), "final_epoch_std": float(eps.std(ddof=0)),
                    "train_time_mean": float(tps.mean()),  "train_time_std":  float(tps.std(ddof=0)),
                }
            }
    
    for dim in dim_list:
        trial_metrics_GFM = []
        trial_metrics_LFM = []

        print(f"[GFM] cluster_d={dim}: running {repeats} trial(s)…")
        for t in tqdm(range(repeats), desc=f"Trials (GFM, d={dim})"):
            trial_seed = seed + t
            m = _run_single_trial_GFM(
                seed=trial_seed,
                dist=dist,
                X_train=X_train,
                ambient_dim=ambient_dim,
                device=device,
                cluster_d=dim,
                global_n_t=global_n_t,
                max_epochs=max_epochs,
                batch_size=batch_size,
                cluster_size=cluster_size,
                early_stopping=early_stopping,
                test_size=test_size
            )
            trial_metrics_GFM.append(m)

        # Aggregate stats
        w2s  = np.array([m["w2"] for m in trial_metrics_GFM], dtype=float)
        geos = np.array([m["geo"] for m in trial_metrics_GFM], dtype=float)
        eps  = np.array([m["final_epoch"] for m in trial_metrics_GFM], dtype=float)
        tps  = np.array([m["train_time"] for m in trial_metrics_GFM], dtype=float)

        results["GFM"][int(dim)] = {
            "trials": trial_metrics_GFM,
            "summary": {
                "w2_mean":  float(w2s.mean()),  "w2_std":  float(w2s.std(ddof=0)),
                "geo_mean": float(geos.mean()), "geo_std": float(geos.std(ddof=0)),
                "final_epoch_mean": float(eps.mean()), "final_epoch_std": float(eps.std(ddof=0)),
                "train_time_mean": float(tps.mean()),  "train_time_std":  float(tps.std(ddof=0)),
            }
        }

        print(f"[LFM] cluster_d={dim}: running {repeats} trial(s)…")
        for t in tqdm(range(repeats), desc=f"Trials (LFM, d={dim})"):
            trial_seed = seed + t
            m = _run_single_trial_LFM(
                seed=trial_seed,
                dist=dist,
                X_train=X_train,
                ambient_dim=ambient_dim,
                device=device,
                cluster_d=dim,
                local_n_t=local_n_t,
                max_epochs=max_epochs,
                batch_size=batch_size,
                cluster_size=cluster_size,
                early_stopping=early_stopping,
                test_size=test_size
            )
            trial_metrics_LFM.append(m)

        # Aggregate stats
        w2s  = np.array([m["w2"] for m in trial_metrics_LFM], dtype=float)
        geos = np.array([m["geo"] for m in trial_metrics_LFM], dtype=float)
        eps  = np.array([m["final_epoch"] for m in trial_metrics_LFM], dtype=float)
        tps  = np.array([m["train_time"] for m in trial_metrics_LFM], dtype=float)

        results["LFM"][int(dim)] = {
            "trials": trial_metrics_LFM,
            "summary": {
                "w2_mean":  float(w2s.mean()),  "w2_std":  float(w2s.std(ddof=0)),
                "geo_mean": float(geos.mean()), "geo_std": float(geos.std(ddof=0)),
                "final_epoch_mean": float(eps.mean()), "final_epoch_std": float(eps.std(ddof=0)),
                "train_time_mean": float(tps.mean()),  "train_time_std":  float(tps.std(ddof=0)),
            }
        }

    return results


def run_convergence_analysis(n, repeats, seed, dist_name, ambient_dim, latent_dim,
                             total_n_t, global_n_t, local_n_t, max_epochs, batch_size,
                             early_stopping, mf_list, beta_a, beta_b, t_list, test_size, device):
    """
    Single-trial convergence analysis:
    Trains UniformFM, ShiftedFM, and DGFM(mf ∈ mf_list) once.
    Evaluates W2 and geometric alignment at each t ∈ t_list.
    Returns:
      {
        "<method>": {
          "per_t": {
            t: {
              "trials": [{"w2":..., "geo":...}],   # single entry
              "summary": {"w2_mean":..., "w2_std":0.0, "geo_mean":..., "geo_std":0.0}
            }, ...
          }
        }, ...
      }
    """
    # normalize t_list to plain floats (avoid np.float64 keys)
    t_list = [float(t) for t in t_list]

    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    dist = _instantiate_dist(dist_name, ambient_dim, latent_dim, device)

    results = {}

    # latent_dim -= 3 # for dimension misspec.

    def _train_uniform(X_train):
        model = VectorField(ambient_dim).to(device)
        opt   = optim.Adam(model.parameters(), lr=1e-3)
        epoch, _, _, best_model = train_uniform_FM(
            model, opt, X_train, ambient_dim, device,
            n_t=total_n_t, epochs=max_epochs, batch_size=batch_size,
            early_stopping=early_stopping,
        )
        return best_model, None

    def _train_shifted(X_train):
        model = VectorField(ambient_dim).to(device)
        opt   = optim.Adam(model.parameters(), lr=1e-3)
        epoch, _, _, best_model = train_shifted_FM(
            model, opt, X_train, ambient_dim, device,
            n_t=total_n_t, epochs=max_epochs, batch_size=batch_size,
            early_stopping=early_stopping, beta_a=beta_a, beta_b=beta_b
        )
        return best_model, None

    def _train_dgfm(X_train, mf, cluster_size):
        model = VectorField(ambient_dim).to(device)
        opt   = optim.Adam(model.parameters(), lr=1e-3)
        ms, epoch, _, _, best_model = train_dgfm(
            model, opt, X_train, ambient_dim, mf, device,
            mixture_sampler=None,
            n_t_global=global_n_t, n_t_local=local_n_t,
            epochs=max_epochs, batch_size=batch_size,
            cluster_size=cluster_size, cluster_d=int(latent_dim),
            early_stopping=early_stopping
        )
        return best_model, None

    def _train_gfm(X_train, cluster_size):
        model = VectorField(ambient_dim).to(device)
        opt   = optim.Adam(model.parameters(), lr=1e-3)
        ms, epoch, _, _, best_model = train_gfm(
            model, opt, X_train, ambient_dim, device,
            mixture_sampler=None,
            n_t_global=global_n_t,
            epochs=max_epochs, batch_size=batch_size,
            cluster_size=cluster_size, cluster_d=int(latent_dim),
            early_stopping=early_stopping
        )
        return best_model, None
    
    def _train_lfm(X_train, cluster_size):
        model = VectorField(ambient_dim).to(device)
        opt   = optim.Adam(model.parameters(), lr=1e-3)
        ms, epoch, _, _, best_model = train_lfm(
            model, opt, X_train, ambient_dim, device,
            mixture_sampler=None,
            n_t_local=local_n_t,
            epochs=max_epochs, batch_size=batch_size,
            cluster_size=cluster_size, cluster_d=int(latent_dim),
            early_stopping=early_stopping
        )
        return best_model, ms

    cluster_size = int(n/5)
    method_specs = [("UniformFM",  _train_uniform),
                    ("ShiftedFM",  _train_shifted)] + \
                   [(f"DGFM-{mf}", lambda Xtr, mf=mf, cluster_size=cluster_size: _train_dgfm(Xtr, mf, cluster_size)) for mf in mf_list] + \
                   [("GFM", lambda Xtr, cluster_size=cluster_size: _train_gfm(Xtr, cluster_size)),
                    ("LFM", lambda Xtr, cluster_size=cluster_size: _train_lfm(Xtr, cluster_size))]

    # Initialize storage
    for mname, _ in method_specs:
        results[mname] = {"per_t": {float(t): {"trials": []} for t in t_list}}

    # ---- Single trial ----
    X_train = dist.sample(n)

    # Train each method once
    trained = {}
    ms      = None
    for mname, trainer in method_specs:
        if mname == "LFM":
            trained[mname], ms = trainer(X_train)
        else:
            trained[mname], _ = trainer(X_train)

    # Evaluate along t_list
    X0 = np.random.randn(test_size, ambient_dim)
    for mname in trained:
        if (mname == "LFM"):
            X0, _ = ms.truncated_sample(test_size)
        states = _flow_states_at_times(trained[mname], X0, device, t_list, steps_per_interval=50)

        for t in t_list:  # make sure t_list = [float(...), ...]
            X_t = states[float(t)]

            # W2: purely numeric, no grad needed
            w2 = dist.wasserstein2_distance(X_t.detach().cpu().numpy(), test_size)

            # Geo: give geometric_alignment a fresh leaf to avoid reusing any graph
            X_geo = X_t.detach().clone().requires_grad_(True)
            geo = dist.geometric_alignment(X_geo)

            results[mname]["per_t"][float(t)]["trials"] = [{"w2": float(w2), "geo": float(geo)}]
            results[mname]["per_t"][float(t)]["summary"] = {
                "w2_mean": float(w2), "w2_std": 0.0,
                "geo_mean": float(geo), "geo_std": 0.0,
            }

    return results


if __name__ == "__main__":
    print("Welcome to the Flow-Matching analysis script!")
    print("This program provides analysis for intrinsic dimensionality misspecification with multiple trials.")
    print("It saves a bar chart (mean ± stddev) and raw JSON under Synthetic_data/analysis_results.\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}\n")

    # ----------- Experiment parameters -----------
    sample_size   = int(input("Enter sample size (default 500): ") or 500)
    repeats       = int(input("Number of trials per (mf,dim) (default 5): ") or 5)
    seed          = int(input("Input the seed (default 1000): ") or 1000)

    ambient_dim   = int(input("Ambient dimension (default 80): ") or 80)
    latent_dim    = int(input("Latent dimension (default 20): ") or 20)
    total_epochs  = int(input("Maximum number of epochs (default 100): ") or 100)
    batch_size_in = int(input("Minimum Batch size (default 100): ") or 100)
    batch_num     = int(input("Number of batches per epoch (default 10): ") or 10)

    mf_list       = list(map(int,
                        (input("DGFM multiplier for global FM (comma separated, default 2,4): ")
                        .strip() or "2,4").split(",")))
    total_n_t   = int(input("Timesteps per sample for Uniform/Shifted FM (default 1): ") or 1)
    beta_a, beta_b = list(map(float,
                    (input("Beta parameters for shifted FM (comma separated, default 1.5,1): ")
                    .strip() or "1.5,1").split(",")))
    global_n_t    = int(input("Number of timesteps per sample for DGFM global FM (default 1): ") or 1)
    local_n_t     = int(input("Number of timesteps per sample for DGFM local FM (default 1): ") or 1)
    early_stopping = input("Use early stopping if validation loss doesn't improve for three epochs? (y/n, default y): ").strip().lower() != 'n'

    print("\nChoose test:")
    print("  [1] Dimension misspecification")
    print("  [2] Convergence along flow")
    test_choice = input(">>> ").strip()

    print("\nChoose target distribution:")
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
        print("Invalid choice.")
        exit(1)

    # ----------- Paths & common setup -----------
    test_size = 2000
    os.makedirs("Synthetic_data/analysis_results", exist_ok=True)
    eff_batch_size = max(batch_size_in, int(sample_size / batch_num))

    if test_choice == "1":

        # ----------- Misspecification setup -----------
        deltas = [-0.1, 0.0, 0.1]
        raw_dims = [max(1, int(round(latent_dim * (1 + d)))) for d in deltas]
        dim_list = sorted(sorted(set(raw_dims)))  # unique + sorted

        # ----------- Run analysis -----------
        print(f"\nRunning dimensionality misspecification with dims = {dim_list} (true latent_dim={latent_dim})")
        results = run_dimension_analysis(
            n=sample_size,
            repeats=repeats,
            seed=seed,
            dist_name=dist_name,
            ambient_dim=ambient_dim,
            latent_dim=latent_dim,
            dim_list=dim_list,
            global_n_t=global_n_t,
            local_n_t=local_n_t,
            max_epochs=total_epochs,
            batch_size=eff_batch_size,
            early_stopping=early_stopping,
            mf_list=mf_list,
            test_size=test_size,
            device=device,
        )

        # ----------- Save JSON -----------
        ts = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
        json_path = f"Synthetic_data/analysis_results/dim_analysis_{dist_name}_n{sample_size}_latent{latent_dim}_trials{repeats}_{ts}.json"
        with open(json_path, "w") as f:
            json.dump({
                "datetime": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
                "seed": seed,
                "distribution": dist_name,
                "ambient_dim": ambient_dim,
                "true_latent_dim": latent_dim,
                "dims_evaluated": dim_list,
                "global_n_t": global_n_t,
                "local_n_t": local_n_t,
                "early_stopping": early_stopping,
                "mf_list": mf_list,
                "sample_size": sample_size,
                "repeats": repeats,
                "batch_size_effective": eff_batch_size,
                "results": results
            }, f, indent=2)

        # ----------- Build bar plots (mean ± std) -----------
        png_path = f"Synthetic_data/analysis_results/dim_analysis_{dist_name}_n{sample_size}_latent{latent_dim}_trials{repeats}_{ts}.png"

        dims = dim_list
        x = np.arange(len(dims), dtype=float)
        G = len(results.keys())
        width = min(0.8 / max(G, 1), 0.28)  # grouped bar width

        plt.figure(figsize=(11, 8))

        # --- Subplot 1: W2 ---
        ax1 = plt.subplot(2, 1, 1)
        w2_means_all = []
        w2_stds_all  = []

        for i, name in enumerate(results.keys()):
            means = [results[name][d]["summary"]["w2_mean"] for d in dims]
            stds  = [results[name][d]["summary"]["w2_std"]  for d in dims]
            xpos  = x + (i - (G-1)/2) * width
            ax1.bar(xpos, means, width=width, yerr=stds, capsize=4, label=name)

            w2_means_all += [results[name][d]["summary"]["w2_mean"] for d in dims]
            w2_stds_all  += [results[name][d]["summary"]["w2_std"]  for d in dims]

        w2_means_all = np.array(w2_means_all, dtype=float)
        w2_stds_all  = np.array(w2_stds_all, dtype=float)

        w2_lo = np.min(w2_means_all - w2_stds_all)
        w2_hi = np.max(w2_means_all + w2_stds_all)
        # W2 can't be negative; clamp and pad a bit
        w2_lo = max(0.0, w2_lo)
        span  = w2_hi - w2_lo
        if span <= 0:
            # degenerate: all same value
            pad = max(1e-6, 0.05 * (w2_hi if w2_hi != 0 else 1.0))
        else:
            pad = 0.08 * span
        ax1.set_ylim(w2_lo - pad, w2_hi + pad)

        ax1.axvline(x=np.where(np.array(dims) == latent_dim)[0][0] if latent_dim in dims else -10,
                    linestyle="--", linewidth=1, label="True latent dim")
        ax1.set_title(f"Dimensionality Misspecification — {dist_name} (n={sample_size}, trials={repeats})")
        ax1.set_ylabel("Wasserstein-2 (↓)")
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(d) for d in dims])
        ax1.set_xlabel("Assumed intrinsic dimension")
        ax1.grid(True, axis="y", alpha=0.3)
        ax1.legend(loc="best")

        # --- Subplot 2: Geometric alignment ---
        ax2 = plt.subplot(2, 1, 2)
        geo_means_all = []
        geo_stds_all  = []

        for i, name in enumerate(results.keys()):
            means = [results[name][d]["summary"]["geo_mean"] for d in dims]
            stds  = [results[name][d]["summary"]["geo_std"]  for d in dims]
            xpos  = x + (i - (G-1)/2) * width
            ax2.bar(xpos, means, width=width, yerr=stds, capsize=4, label=name)

            geo_means_all += [results[name][d]["summary"]["geo_mean"] for d in dims]
            geo_stds_all  += [results[name][d]["summary"]["geo_std"]  for d in dims]

        if latent_dim in dims:
            ax2.axvline(x=np.where(np.array(dims) == latent_dim)[0][0],
                        linestyle="--", linewidth=1, label="True latent dim")
            
        geo_means_all = np.array(geo_means_all, dtype=float)
        geo_stds_all  = np.array(geo_stds_all, dtype=float)

        geo_lo = float(np.min(geo_means_all - geo_stds_all))
        geo_hi = float(np.max(geo_means_all + geo_stds_all))
        span   = geo_hi - geo_lo
        pad    = 0.08 * span if span > 0 else 0.5  # fallback pad if flat
        ax2.set_ylim(geo_lo - pad, geo_hi + pad)

        ax2.set_ylabel("Geometric alignment (↓)")
        ax2.set_xticks(x)
        ax2.set_xticklabels([str(d) for d in dims])
        ax2.set_xlabel("Assumed intrinsic dimension")
        ax2.grid(True, axis="y", alpha=0.3)
        ax2.legend(loc="best")

        plt.tight_layout()
        plt.savefig(png_path, dpi=220)
        plt.close()

        print(f"\nSaved plot to {png_path}")
        print(f"Saved raw JSON to {json_path}")

    elif test_choice == "2":
        # ===================== Convergence-along-flow analysis =====================
        t_list = [round(t, 1) for t in np.linspace(0.0, 1.0, 11, endpoint=True)]
        print(f"\nRunning convergence analysis along t ∈ {t_list} …")

        conv_results = run_convergence_analysis(
            n=sample_size,
            repeats=repeats,
            seed=seed,
            dist_name=dist_name,
            ambient_dim=ambient_dim,
            latent_dim=latent_dim,
            total_n_t=total_n_t,
            global_n_t=global_n_t,
            local_n_t=local_n_t,
            max_epochs=total_epochs,
            batch_size=eff_batch_size,
            early_stopping=early_stopping,
            mf_list=mf_list,
            beta_a=beta_a, beta_b=beta_b,
            t_list=t_list,
            test_size=test_size,
            device=device
        )

        # Save JSON
        ts = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
        conv_json_path = f"Synthetic_data/analysis_results/flow_convergence_{dist_name}_n{sample_size}_latent{latent_dim}_trials{repeats}_{ts}.json"
        with open(conv_json_path, "w") as f:
            json.dump({
                "datetime": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
                "seed": seed,
                "distribution": dist_name,
                "ambient_dim": ambient_dim,
                "true_latent_dim": latent_dim,
                "t_list": t_list,
                "total_n_t": total_n_t,
                "global_n_t": global_n_t,
                "local_n_t": local_n_t,
                "early_stopping": early_stopping,
                "mf_list": mf_list,
                "sample_size": sample_size,
                "repeats": repeats,
                "results": conv_results
            }, f, indent=2)

        # Plot W2(t) and Geo(t) with error bars (mean ± std)
        conv_png_path = f"Synthetic_data/analysis_results/flow_convergence_{dist_name}_n{sample_size}_latent{latent_dim}_trials{repeats}_{ts}.png"
        plt.figure(figsize=(11, 8))

        # Build consistent color/linestyle order
        method_order = ["UniformFM", "ShiftedFM"] + [f"DGFM-{mf}" for mf in mf_list] + ["GFM", "LFM"]

        # --- Subplot 1: W2 vs t
        ax3 = plt.subplot(2, 1, 1)
        all_w2_means, all_w2_stds = [], []
        for mname in method_order:
            means = [conv_results[mname]["per_t"][float(t)]["summary"]["w2_mean"] for t in t_list]
            stds  = [conv_results[mname]["per_t"][float(t)]["summary"]["w2_std"]  for t in t_list]
            ax3.errorbar(t_list, means, yerr=stds, marker="o", capsize=3, label=mname)
            all_w2_means += means; all_w2_stds += stds

        # tighten y-range
        w2_means_all = np.array(all_w2_means, dtype=float)
        w2_stds_all  = np.array(all_w2_stds, dtype=float)
        w2_lo = np.nanmin(w2_means_all - w2_stds_all); w2_hi = np.nanmax(w2_means_all + w2_stds_all)
        w2_lo = max(0.0, float(w2_lo)) if np.isfinite(w2_lo) else 0.0
        span = (w2_hi - w2_lo) if np.isfinite(w2_hi) else 1.0
        pad = 0.08 * span if span > 0 else 0.05
        ax3.set_ylim(w2_lo - pad, (w2_hi if np.isfinite(w2_hi) else w2_lo + 1.0) + pad)

        ax3.set_title(f"Convergence Along Flow — {dist_name})")
        ax3.set_ylabel("Wasserstein-2 (↓)")
        ax3.set_xlabel("time t")
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc="lower left")

        # --- Subplot 2: Geometric alignment vs t
        ax4 = plt.subplot(2, 1, 2)
        all_geo_means, all_geo_stds = [], []
        for mname in method_order:
            means = [conv_results[mname]["per_t"][float(t)]["summary"]["geo_mean"] for t in t_list]
            stds  = [conv_results[mname]["per_t"][float(t)]["summary"]["geo_std"]  for t in t_list]
            ax4.errorbar(t_list, means, yerr=stds, marker="s", capsize=3, label=mname)
            all_geo_means += means; all_geo_stds += stds

        geo_means_all = np.array(all_geo_means, dtype=float)
        geo_stds_all  = np.array(all_geo_stds, dtype=float)
        geo_lo = float(np.nanmin(geo_means_all - geo_stds_all))
        geo_hi = float(np.nanmax(geo_means_all + geo_stds_all))
        span = geo_hi - geo_lo if np.isfinite(geo_hi) else 1.0
        pad  = 0.08 * span if span > 0 else 0.05
        ax4.set_ylim((geo_lo if np.isfinite(geo_lo) else 0.0) - pad,
                    (geo_hi if np.isfinite(geo_hi) else 1.0) + pad)

        ax4.set_ylabel("Geometric alignment (↓)")
        ax4.set_xlabel("time t")
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc="lower left")

        plt.tight_layout()
        plt.savefig(conv_png_path, dpi=220)
        plt.close()

        print(f"Saved convergence plot to {conv_png_path}")
        print(f"Saved convergence JSON to {conv_json_path}")

    else:
        print("Invalid test choice!")