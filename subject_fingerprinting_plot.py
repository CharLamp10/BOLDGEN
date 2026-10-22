import argparse
import pickle
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np

CONDITIONS = [
    ("full_original", "10 min\nTR = 0.72 s"),
    ("full_subsampled", "10 min\nTR = 2.0 s"),
    ("half_original", "7 min\nTR = 0.72 s"),
    ("half_subsampled", "7 min\nTR = 2.0 s"),
]
METHODS = [
    ("orig", "Original", "#667788"),
    ("CRVAE", "CR-VAE", "#D18A45"),
    ("rDCM-GEN", "rDCM-GEN", "#197C83"),
]

def load_results(parent: Path):
    stats = {}
    for dirname, _ in CONDITIONS:
        path = os.path.join(parent, "SF", dirname, "results.pkl")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing results file: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            raise TypeError(f"Expected dictionary in {path}, got {type(data).__name__}")
        stats[dirname] = {}
        for key, _, _ in METHODS:
            if key not in data:
                raise KeyError(f"Missing key {key!r} in {path}; available: {list(data)}")
            values = np.asarray(data[key], dtype=float).ravel()
            values = values*100
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Non-finite values for {key} in {path}")
            stats[dirname][key] = (float(values.mean()), float(values.std(ddof=1)))
    return stats

def plot(stats, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 15,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    x = np.arange(len(CONDITIONS))
    width = 0.24
    for idx, (key, label, color) in enumerate(METHODS):
        means = np.array([stats[dirname][key][0] for dirname, _ in CONDITIONS])
        stds = np.array([stats[dirname][key][1] for dirname, _ in CONDITIONS])
        bars = ax.bar(
            x + (idx - 1) * width, means, width * 0.92,
            yerr=stds, capsize=3, color=color, label=label,
            edgecolor="white", linewidth=0.6,
            error_kw={"elinewidth": 1.1, "capthick": 1.1},
        )
        for bar, mean, sd in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, mean + sd + 1.0,
                    f"{mean:.1f}", ha="center", va="bottom", fontsize=12)
            
    ax.set_xticks(x, [label for _, label in CONDITIONS])
    ax.set_ylabel("Subject fingerprinting accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.yaxis.grid(True, alpha=0.18)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=10, labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=15)

    fig.savefig(os.path.join(output_dir, "subject_fingerprinting_barplot.png"),
                dpi=600, bbox_inches="tight")
    plt.close(fig)
    

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent_path", default=r'/home/student1/Desktop/Charalampos_Lamprou/rDCM-GEN_results_HCP_2026')
    parser.add_argument("--output-dir", default=r'/home/student1/Desktop/Charalampos_Lamprou/rDCM-GEN/plots/SF')
    args = parser.parse_args()
    stats = load_results(args.parent_path)
    print("Condition                 Method      Mean      SD (sample)")
    for dirname, _ in CONDITIONS:
        for key, _, _ in METHODS:
            mean, sd = stats[dirname][key]
            print(f"{dirname:25s} {key:10s} {mean:7.2f}   {sd:7.2f}")
    plot(stats, args.output_dir)


if __name__ == "__main__":
    main()
