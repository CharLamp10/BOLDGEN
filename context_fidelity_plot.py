import argparse
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

SETTINGS = [
    ("full_original", "10 min\nTR = 0.72 s"),
    ("full_subsampled", "10 min\nTR = 2.0 s"),
    ("half_original", "7 min\nTR = 0.72 s"),
    ("half_subsampled", "7 min\nTR = 2.0 s"),
    ("ABIDEI", "ABIDE I"),
]
METHODS = ["TimeVAE", "TimeGAN", "T2I-Diff", "DSFM", "CR-VAE", "rDCM-GEN"]
COLORS = {
    "TimeVAE": "#718096",
    "TimeGAN": "#8C6BB1",
    "T2I-Diff": "#D59443",
    "DSFM": "#7CA35A",
    "CR-VAE": "#C46763",
    "rDCM-GEN": "#167B84",
}
# Override if actual folder names differ, e.g. {"T2IDIFF": "T2I-Diff"}.
FOLDER_NAMES = {method: method for method in METHODS}


def load_stats(parent):
    means = {m: [] for m in METHODS}
    stds = {m: [] for m in METHODS}
    for folder, _ in SETTINGS:
        for method in METHODS:
            file = os.path.join(parent, "cFID", folder, FOLDER_NAMES[method], "results.pkl")
            if not os.path.isfile(file):
                raise FileNotFoundError(f"Missing file: {file}")
            with open(file, "rb") as f:
                result = pickle.load(f)
            if not isinstance(result, dict) or "context_FID" not in result:
                raise ValueError(f"Expected dict with 'context_FID' in {file}")
            values = np.asarray(result["context_FID"], dtype=float).ravel()
            if values.size == 0 or not np.all(np.isfinite(values)):
                raise ValueError(f"Empty/non-finite context_FID values in {file}")
            means[method].append(float(values.mean()))
            stds[method].append(float(values.std(ddof=1)) if len(values)>1 else 0.)
            print(f"{folder:17s} {method:9s}: {means[method][-1]:8.3f} +/- {stds[method][-1]:.3f} (n={len(values)})")
    return means, stds


def plot(means, stds, output_dir, prefix="context_fid"):
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(11.8, 5.4))
    x = np.arange(len(SETTINGS))
    width = 0.13
    for j, method in enumerate(METHODS):
        positions = x + (j - (len(METHODS)-1)/2) * width
        ax.bar(positions, means[method], width*0.94, yerr=stds[method],
               capsize=2.3, color=COLORS[method], label=method,
               edgecolor="white", linewidth=0.5,
               error_kw={"elinewidth": 0.9, "capthick": 0.9})
    ax.set_yscale("log")
    ax.set_ylim(1, 150)
    ax.set_yticks([1, 2, 5, 10, 20, 50, 100])
    ax.set_yticklabels(["1", "2", "5", "10", "20", "50", "100"])
    ax.set_ylabel("Context-FID")
    ax.set_xticks(x, [lab for _, lab in SETTINGS])
    ax.tick_params(axis="x", length=0, pad=9)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", which="major", alpha=0.20, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14),
              ncol=6, frameon=False, fontsize=9.5)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.15, top=0.82)
    fig.savefig(os.path.join(output_dir, f"{prefix}.png"), dpi=400, bbox_inches="tight")
    plt.close(fig)
    
    
def main():
    parser = argparse.ArgumentParser(description="Plot Context-FID from pickle results")
    parser.add_argument("--parent", default=r'/home/student1/Desktop/Charalampos_Lamprou/rDCM-GEN_results_HCP_2026')
    parser.add_argument("--output-dir", default=r'/home/student1/Desktop/Charalampos_Lamprou/rDCM-GEN/plots/cFID')
    args = parser.parse_args()
    means, stds = load_stats(args.parent)
    plot(means, stds, args.output_dir)


if __name__ == "__main__":
    main()