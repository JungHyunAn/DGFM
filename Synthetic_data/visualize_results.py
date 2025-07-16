import os
import glob
import json
from matplotlib import ticker
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import beta


def list_json_files(input_dir):
    pattern = os.path.join(input_dir, "eval_*.json")
    files = glob.glob(pattern)
    return sorted(files)


def select_file(files):
    print("Available JSON files:")
    for i, f in enumerate(files):
        print(f"[{i}] {os.path.basename(f)}")
    idx = int(input("Select file index: "))
    return files[idx]


def plot_trials(data, out_dir, dist_name):
    results = data.get("results", {})
    for sample_size, methods in results.items():
        for method, trials in methods.items():
            for trial_idx, trial_data in trials.items():
                epochs = [entry.get("epoch") for entry in trial_data]
                val_w2 = [entry.get("validation_w2") for entry in trial_data]
                train_loss = [entry.get("train_loss") for entry in trial_data]

                fig, ax1 = plt.subplots()
                ax1.plot(epochs, val_w2, label="Validation W2")
                ax1.set_xlabel("Epoch")
                ax1.set_ylabel("Validation W2")

                ax2 = ax1.twinx()
                ax2.plot(epochs, train_loss, linestyle="--", label="Train Loss")
                ax2.set_ylabel("Train Loss")

                plt.title(f"{dist_name} - {sample_size} samples - {method} - Trial {trial_idx}")
                lines, labels = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                fig.legend(lines + lines2, labels + labels2, loc="lower center")

                fname = f"{dist_name}_{sample_size}_{method}_trial{trial_idx}.png"
                fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
                plt.close(fig)


def plot_comparison(data, out_dir, dist_name):
    results = data.get("results", {})
    for sample_size, methods in results.items():
        # Determine common epochs across trials for alignment
        # Assuming each trial has entries for all epochs
        # Collect unique epochs
        all_epochs = set()
        for trials in methods.values():
            for trial_data in trials.values():
                for entry in trial_data:
                    ep = entry.get("epoch")
                    if ep is not None:
                        all_epochs.add(ep)
        epochs = sorted(all_epochs)

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        for method, trials in methods.items():
            # Prepare per-epoch lists across trials
            w2_vals = {epoch: [] for epoch in epochs}
            loss_vals = {epoch: [] for epoch in epochs}

            for trial_data in trials.values():
                for entry in trial_data:
                    ep = entry.get("epoch")
                    if ep is not None:
                        w2_vals[ep].append(entry.get("validation_w2"))
                        loss_vals[ep].append(entry.get("train_loss"))

            # Compute means and stds
            w2_means = [np.nanmean(w2_vals[ep]) for ep in epochs]
            w2_stds = [np.nanstd(w2_vals[ep]) for ep in epochs]
            loss_means = [np.nanmean(loss_vals[ep]) for ep in epochs]
            loss_stds = [np.nanstd(loss_vals[ep]) for ep in epochs]

            ax1.plot(epochs, w2_means, marker='o', label=f"{method} W2")
            ax2.plot(epochs, loss_means, marker='x', linestyle='--', label=f"{method} Loss")

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Validation W2")
        ax2.set_ylabel("Train Loss")
        plt.title(f"{dist_name} - Comparison: {sample_size} samples (mean ± std)")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        fig.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        fname = f"{dist_name}_{sample_size}_comparison.png"
        fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
        plt.close(fig)


def plot_summary(data, out_dir, dist_name):
    summary = data.get("summary", {})
    ambient_dim = data.get("ambient_dim")
    latent_dim = data.get("latent_dim")
    if not summary:
        print("No summary section in JSON.")
        return

    # prepare sample sizes and methods
    sample_sizes = sorted(summary.keys(), key=lambda x: int(x))
    methods = list(next(iter(summary.values())).keys())

    # Eval W2 with error bars
    fig1, ax1 = plt.subplots()
    ax1.set_xscale('log')
    x = [int(ss) for ss in sample_sizes]
    for method in methods:
        means = []
        stds = []
        for ss in sample_sizes:
            entries = summary[ss].get(method, [])
            means.append(entries.get("eval_wasserstein2_mean", 0))
            stds.append(entries.get("eval_wasserstein2_std", 0))
        ax1.errorbar(x, means, yerr=stds, marker='o', capsize=5, label=method)

    xmin, xmax = min(x), max(x)
    ax1.set_xlim(0.9*xmin, 1.1*xmax)
    ax1.margins(x=0.1)
    ax1.set_xticks(x)
    ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax1.set_xlabel("Sample Size")
    ax1.set_ylabel("Evaluation W2 Mean ± Std")
    ax1.set_title(f"{dist_name} - (n, d) = ({ambient_dim}, {latent_dim})")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, f"{dist_name}_summary_eval_w2.png"), bbox_inches="tight")
    plt.close(fig1)

    # Eval Geometric Alignment with error bars
    fig2, ax2 = plt.subplots()
    ax2.set_xscale('log')
    #ax2.set_yscale('log') # for big difference in geometric alignment
    for method in methods:
        means = []
        stds = []
        for ss in sample_sizes:
            entries = summary[ss].get(method, [])
            means.append(entries.get("eval_geometric_alignment_mean", 0))
            stds.append(entries.get("eval_geometric_alignment_std", 0))
        ax2.errorbar(x, means, yerr=stds, marker='o', capsize=5, label=method)
    ax2.set_xlim(0.9*xmin, 1.1*xmax)
    ax2.margins(x=0.1)
    ax2.set_xticks(x)
    ax2.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax2.set_xlabel("Sample Size")
    ax2.set_ylabel("Geometric Alignment Mean ± Std")
    ax2.set_title(f"{dist_name} - (n, d) = ({ambient_dim}, {latent_dim})")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, f"{dist_name}_summary_geom_align.png"), bbox_inches="tight")
    plt.close(fig2)


def plot_beta(beta_a, beta_b, out_dir):
    """
    Plot the Beta(a, b) probability density function over [0, 1],
    using scipy.special.beta for normalization, and save the figure.
    """
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    
    # Domain for plotting
    x = np.linspace(0, 1, 500)
    
    # Compute normalization constant B(a, b)
    B = beta(beta_a, beta_b)
    
    # Evaluate PDF: f(x) = x^(a-1) * (1-x)^(b-1) / B(a, b)
    pdf = (1-x)**(beta_a - 1) * x**(beta_b - 1) / B
    
    # Create plot
    plt.figure()
    plt.plot(x, pdf)
    plt.title(f"Beta PDF (a={beta_a}, b={beta_b})")
    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(out_dir, f"beta_{beta_a}_{beta_b}.png")
    plt.savefig(fig_path)
    plt.close()


def main():
    input_dir = os.path.expanduser("~/DGFM/Synthetic_data/eval_results")

    files = list_json_files(input_dir)
    if not files:
        print(f"No JSON files found in {input_dir}")
        return

    selected = select_file(files)
    with open(selected, 'r') as f:
        data = json.load(f)

    output_dir = os.path.expanduser(f"~/DGFM/Synthetic_data/eval_graphs/{os.path.basename(selected)}")
    os.makedirs(output_dir, exist_ok=True)

    dist_name = data.get("distribution", os.path.splitext(os.path.basename(selected))[0]).replace("_", " ")
    # plot_trials(data, output_dir, dist_name)
    plot_summary(data, output_dir, dist_name)
    plot_comparison(data, output_dir, dist_name)
    plot_beta(data["beta_a"], data["beta_b"], output_dir)

    print(f"Graphs saved in {output_dir}")


if __name__ == "__main__":
    main()
