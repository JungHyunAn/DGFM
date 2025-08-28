import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

files = [
    "Synthetic_data/analysis_results/flow_convergence_Quadratic_Unimodal_n500_latent20_trials5_20250827_142635.json",
    "Synthetic_data/analysis_results/flow_convergence_SwissRoll_n500_latent20_trials5_20250827_142908.json",
]

def _to_float_keys(per_t_dict):
    return {float(k): v for k, v in per_t_dict.items()}

for path in files:
    with open(path, "r") as f:
        data = json.load(f)

    dist_name = data.get("distribution", "Unknown")
    t_list = [float(t) for t in data["t_list"]]
    results = data["results"]

    # Gather per-method series
    series = {}
    for method, md in results.items():
        per_t = _to_float_keys(md["per_t"])
        w2m = [per_t[t]["summary"]["w2_mean"] for t in t_list]
        w2s = [per_t[t]["summary"]["w2_std"]  for t in t_list]
        gm  = [per_t[t]["summary"]["geo_mean"] for t in t_list]
        gs  = [per_t[t]["summary"]["geo_std"]  for t in t_list]
        # guard against None/NaN in std
        w2s = [0.0 if (s is None or not np.isfinite(s)) else s for s in w2s]
        gs  = [0.0 if (s is None or not np.isfinite(s)) else s for s in gs]
        series[method] = (w2m, w2s, gm, gs)

    # Plot: half the width (vs the usual ~11)
    plt.figure(figsize=(5.5, 8))

    # W2 vs t
    ax1 = plt.subplot(2, 1, 1)
    for method in series.keys():
        w2m, w2s, _, _ = series[method]
        if method == "DGFM_mf2":
            method = "DGFM-2"
        if method == "DGFM_mf4":
            method = "DGFM-4"
        ax1.errorbar(t_list, w2m, yerr=w2s, marker="o", capsize=3, label=method)
    ax1.set_title(dist_name)
    ax1.set_ylabel("Wasserstein-2 (↓)")
    ax1.set_xlabel("t")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower left")

    # Geometric alignment vs t
    ax2 = plt.subplot(2, 1, 2)
    for method in series.keys():
        _, _, gm, gs = series[method]
        if method == "DGFM_mf2":
            method = "DGFM-2"
        if method == "DGFM_mf4":
            method = "DGFM-4"

        ax2.errorbar(t_list, gm, yerr=gs, marker="s", capsize=3, label=method)
    ax2.set_ylabel("Geometric alignment (↓)")
    ax2.set_xlabel("t")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left")

    plt.tight_layout()
    out = path.replace(".json", "_halfwidth.png")
    plt.savefig(out, dpi=220)
    plt.close()
    print(f"Saved {out}")
