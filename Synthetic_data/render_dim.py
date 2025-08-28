import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

files = [
    "Synthetic_data/analysis_results/dim_analysis_Quadratic_Unimodal_n500_latent20_trials10_20250827_130634.json",
    "Synthetic_data/analysis_results/dim_analysis_SwissRoll_n500_latent20_trials10_20250827_130148.json",
]

def _extract_mf_order(results_dict):
    def _mf_key(k):
        try:
            return int(k.split("mf")[1])
        except Exception:
            return k
    return sorted(results_dict.keys(), key=_mf_key)

def _intify_dim_keys(results):
    """Convert nested dimension keys to int (JSON loads them as strings)."""
    out = {}
    for method, dim_map in results.items():
        new_dim_map = {}
        for k, v in dim_map.items():
            try:
                new_dim_map[int(k)] = v
            except (ValueError, TypeError):
                new_dim_map[k] = v
        out[method] = new_dim_map
    return out

for path in files:
    with open(path, "r") as f:
        data = json.load(f)

    dist_name = data.get("distribution", "Unknown")
    dims = [int(d) for d in data.get("dims_evaluated", [])]
    results = _intify_dim_keys(data["results"])
    methods = _extract_mf_order(results)  # e.g., ["DGFM_mf2", "DGFM_mf4", ...]

    x = np.arange(len(dims), dtype=float)
    G = max(1, len(methods))
    width = min(0.8 / G, 0.28)

    # Collect for tight y-limits
    w2_all_means, w2_all_stds = [], []
    geo_all_means, geo_all_stds = [], []

    plt.figure(figsize=(8.25, 8))  # 75% of ~11 in width

    # --- Subplot 1: W2 ---
    ax1 = plt.subplot(2, 1, 1)
    for i, m in enumerate(methods):
        means = [results[m][d]["summary"]["w2_mean"] for d in dims]
        stds  = [results[m][d]["summary"]["w2_std"]  for d in dims]
        stds  = [0.0 if (s is None or not np.isfinite(s)) else s for s in stds]
        xpos  = x + (i - (G-1)/2) * width
        ax1.bar(xpos, means, width=width, yerr=stds, capsize=4, label=("DGFM-" + m))
        w2_all_means += means; w2_all_stds += stds

    w2_means_all = np.array(w2_all_means, dtype=float)
    w2_stds_all  = np.array(w2_all_stds, dtype=float)
    w2_lo = np.nanmin(w2_means_all - w2_stds_all); w2_hi = np.nanmax(w2_means_all + w2_stds_all)
    w2_lo = max(0.0, float(w2_lo)) if np.isfinite(w2_lo) else 0.0
    span = (w2_hi - w2_lo) if np.isfinite(w2_hi) else 1.0
    pad = 0.08 * span if span > 0 else 0.05
    ax1.set_ylim(w2_lo - pad, (w2_hi if np.isfinite(w2_hi) else w2_lo + 1.0) + pad)

    ax1.set_title(dist_name)
    ax1.set_ylabel("Wasserstein-2 (↓)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(d) for d in dims])
    ax1.set_xlabel("Assumed intrinsic dimension (d')")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(loc=("upper right"))

    # --- Subplot 2: Geometric alignment ---
    ax2 = plt.subplot(2, 1, 2)
    for i, m in enumerate(methods):
        means = [results[m][d]["summary"]["geo_mean"] for d in dims]
        stds  = [results[m][d]["summary"]["geo_std"]  for d in dims]
        stds  = [0.0 if (s is None or not np.isfinite(s)) else s for s in stds]
        xpos  = x + (i - (G-1)/2) * width
        ax2.bar(xpos, means, width=width, yerr=stds, capsize=4, label=("DGFM-" + m))
        geo_all_means += means; geo_all_stds += stds

    geo_means_all = np.array(geo_all_means, dtype=float)
    geo_stds_all  = np.array(geo_all_stds, dtype=float)
    geo_lo = float(np.nanmin(geo_means_all - geo_stds_all))
    geo_hi = float(np.nanmax(geo_means_all + geo_stds_all))
    span = geo_hi - geo_lo if np.isfinite(geo_hi) else 1.0
    pad  = 0.08 * span if span > 0 else 0.05
    ax2.set_ylim((geo_lo if np.isfinite(geo_lo) else 0.0) - pad,
                 (geo_hi if np.isfinite(geo_hi) else 1.0) + pad)

    ax2.set_ylabel("Geometric alignment (↓)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(d) for d in dims])
    ax2.set_xlabel("Assumed intrinsic dimension (d')")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(loc=("upper right"))

    plt.tight_layout()
    out = path.replace(".json", "_75width.png")
    plt.savefig(out, dpi=220)
    plt.close()
    print(f"Saved {out}")
