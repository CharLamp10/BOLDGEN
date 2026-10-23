import argparse
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SETTINGS = [
    ("full_original", "10 min, TR = 0.72 s"),
    ("full_subsampled", "10 min, TR = 2.0 s"),
    ("half_original", "7 min, TR = 0.72 s"),
    ("half_subsampled", "7 min, TR = 2.0 s"),
]
TAUS = [("tau_0.05", .05), ("tau_0.1", .1), ("tau_0.15", .15)]
METHODS = ["CRVAE", "rDCM-GEN"]
COLORS = ["#D18A45", "#167B84"]

def get_values(payload, key):
    result = payload["result"]
    if isinstance(result, dict):
        values = np.asarray(result[key], dtype=float).ravel()
    elif isinstance(result, (list, tuple)):
        values = np.concatenate([np.asarray(r[key], dtype=float).ravel() for r in result])
    else:
        raise ValueError("'result' must be a dictionary or list of dictionaries")
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError(f"Invalid {key} values")
    if np.all((values >= 0) & (values <= 1)):
        values *= 100
    return values


def load(parent, patterns):
    data = {}
    for folder, _ in SETTINGS:
        for tau_dir, tau in TAUS:
            for method in METHODS:
                base = Path(os.path.join(parent, "Gender", "SimCLR", folder, tau_dir))
                files = sorted(base.glob(patterns[method]))
                if not files:
                    raise FileNotFoundError(f"No matching files in {base}: {patterns[method]}")
                for metric, key in [("AUC", "best_test_auc"), ("F1", "best_test_f1")]:
                    chunks = []
                    for file in files:
                        with file.open("rb") as f:
                            chunks.append(get_values(pickle.load(f), key))
                    values = np.concatenate(chunks)
                    data[folder, tau, method, metric] = values
                    sd = np.std(values, ddof=1) if len(values) > 1 else 0.
                    print(f"{folder} tau={tau:.2f} {method} {metric}: {values.mean():.2f} +/- {sd:.2f} (n={len(values)})")
    return data


def plot(data, output_dir, prefix="gender_simclr"):
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 28,
        "axes.labelsize": 34,
        "xtick.labelsize": 30,
        "ytick.labelsize": 30,
        "legend.fontsize": 28,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })

    x = np.arange(len(TAUS))
    width = 0.35

    # Create four independent figures
    for folder, label in SETTINGS:

        fig, ax = plt.subplots(figsize=(12, 8))

        for j, method in enumerate(METHODS):

            means, sds = [], []

            for _, tau in TAUS:
                values = data[folder, tau, method, "AUC"]

                means.append(values.mean())
                sds.append(
                    values.std(ddof=1) if len(values) > 1 else 0
                )

            ax.bar(
                x + (j - 0.5) * width,
                means,
                width * 0.94,
                yerr=sds,
                capsize=5,
                color=COLORS[j],
                edgecolor="white",
                linewidth=0.8,
                error_kw={
                    "elinewidth": 1.5,
                    "capthick": 1.5
                },
                label="CR-VAE" if j == 0 else "rDCM-GEN"
            )

        ax.set_xticks(
            x,
            ["0.05", "0.10", "0.15"]
        )

        ax.set_ylim(0, 100)

        ax.set_ylabel("AUC (%)")
        ax.set_xlabel("SimCLR temperature (τ)", labelpad=15)

        # No title: add configuration labels in Overleaf
        # ax.set_title(label, weight="bold", pad=13)

        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)

        ax.spines[["top", "right"]].set_visible(False)

        ax.tick_params(
            axis="x",
            labelsize=30,
            length=0,
            pad=12
        )

        ax.tick_params(
            axis="y",
            labelsize=30,
            length=0,
            pad=10
        )

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=2,
            frameon=False,
            fontsize=28,
            columnspacing=1.2,
            handlelength=1.4
        )

        fig.tight_layout()

        # Save each temporal configuration separately
        filename = f"{prefix}_{folder}"

        fig.savefig(
            os.path.join(output_dir, f"{filename}.png"),
            dpi=600,
            bbox_inches="tight"
        )

        # Vector PDF for Overleaf
        fig.savefig(
            os.path.join(output_dir, f"{filename}.pdf"),
            bbox_inches="tight"
        )

        plt.close(fig)
    
    
def main():
    parser = argparse.ArgumentParser(description="Plot gender AUC/F1 across SimCLR settings")
    parser.add_argument("--parent_path", default=r'/home/student1/Desktop/Charalampos_Lamprou/rDCM-GEN_results_HCP_2026')
    parser.add_argument("--output-dir", default=r'/home/student1/Desktop/Charalampos_Lamprou/rDCM-GEN/plots/Gender_CLS')
    parser.add_argument("--crvae-pattern", default="results_CRVAE*.pkl")
    parser.add_argument("--rdcm-pattern", default="results_rDCM-GEN*.pkl")
    args = parser.parse_args()
    data = load(args.parent_path, {"CRVAE": args.crvae_pattern, "rDCM-GEN": args.rdcm_pattern})
    plot(data, args.output_dir)

if __name__ == "__main__":
    main()