"""Plot subject fingerprinting accuracy and uncorrected paired t-test annotations.

Pairing requires each method's values within a condition to correspond to the
same subjects/repetitions in the same order. This cannot be verified from scores.
"""

import argparse
import itertools
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel

CONDITIONS = [
    ("full_original", "10 min\nTR = 0.72 s"),
    ("full_subsampled", "10 min\nTR = 2.0 s"),
    ("half_original", "7 min\nTR = 0.72 s"),
    ("half_subsampled", "7 min\nTR = 2.0 s"),
]
METHODS = [
    ("orig", "Original", "#667788"),
    ("CR-VAE", "CR-VAE", "#D18A45"),
    ("rDCM-GEN", "rDCM-GEN", "#197C83"),
]


def load_results(parent: Path):
    stats = {}
    for dirname, _ in CONDITIONS:
        path = parent / "SF" / dirname / "results.pkl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing results file: {path}")
        with path.open("rb") as f:
            data = pickle.load(f)  # Load trusted pickle files only.
        if not isinstance(data, dict):
            raise TypeError(f"Expected dictionary in {path}, got {type(data).__name__}")
        stats[dirname] = {}
        for key, _, _ in METHODS:
            if key not in data:
                raise KeyError(f"Missing key {key!r} in {path}; available: {list(data)}")
            values = np.asarray(data[key], dtype=float).ravel() * 100
            if not len(values) or not np.all(np.isfinite(values)):
                raise ValueError(f"Empty/non-finite values for {key} in {path}")
            stats[dirname][key] = {
                "values": values,
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.,
            }
    return stats


def significance_label(p):
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def paired_tests(stats):
    tests = {}
    for dirname, _ in CONDITIONS:
        tests[dirname] = []
        for i, j in itertools.combinations(range(len(METHODS)), 2):
            key_a, key_b = METHODS[i][0], METHODS[j][0]
            a, b = stats[dirname][key_a]["values"], stats[dirname][key_b]["values"]
            if len(a) != len(b):
                raise ValueError(
                    f"{dirname}: {key_a} ({len(a)}) vs {key_b} ({len(b)}) have "
                    "unequal repetition counts. Paired tests require aligned runs."
                )
            if len(a) < 2:
                p, t = np.nan, np.nan
            elif np.allclose(a - b, (a - b)[0], rtol=0, atol=1e-12):
                # No variance in paired differences => t-test not estimable.
                p, t = np.nan, np.nan
            else:
                result = ttest_rel(a, b, alternative="two-sided")
                p, t = float(result.pvalue), float(result.statistic)
            tests[dirname].append({
                "i": i, "j": j, "method_a": key_a, "method_b": key_b,
                "t": t, "p": p, "label": significance_label(p), "n": len(a),
            })
    return tests


def draw_bracket(ax, x1, x2, y, label, rise=1.3):
    ax.plot([x1, x1, x2, x2], [y, y + rise, y + rise, y],
            color="black", lw=1.05, clip_on=False)
    ax.text((x1 + x2) / 2, y + rise + 0.5, label,
            ha="center", va="bottom", fontsize=12, color="black")


def plot(stats, tests, output_dir, dpi=600, show_ns=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 15,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    }):
        fig, ax = plt.subplots(figsize=(12, 6.3), layout="constrained")
        x = np.arange(len(CONDITIONS), dtype=float)
        width = 0.24
        highest = 100.
        for idx, (key, label, color) in enumerate(METHODS):
            means = np.array([stats[d][key]["mean"] for d, _ in CONDITIONS])
            stds = np.array([stats[d][key]["sd"] for d, _ in CONDITIONS])
            bars = ax.bar(
                x + (idx - 1) * width, means, width * 0.92,
                yerr=stds, capsize=3, color=color, label=label,
                edgecolor="white", linewidth=0.6,
                error_kw={"elinewidth": 1.1, "capthick": 1.1},
            )
            for bar, mean, sd in zip(bars, means, stds):
                ax.text(bar.get_x() + bar.get_width() / 2, mean + sd + 0.8,
                        f"{mean:.1f}", ha="center", va="bottom", fontsize=11)

        # Draw three brackets per condition, comparing every pair of methods.
        # Stack brackets above the highest mean+SD so they never cross the bars.
        # Close-by bracket labels are deliberately staggered vertically.
        bracket_order = [(0, 1), (1, 2), (0, 2)]
        for cond_idx, (dirname, _) in enumerate(CONDITIONS):
            top = max(stats[dirname][key]["mean"] + stats[dirname][key]["sd"]
                      for key, _, _ in METHODS)
            base = top + 5.5
            by_pair = {(row["i"], row["j"]): row for row in tests[dirname]}
            level = 0
            for i, j in bracket_order:
                row = by_pair[i, j]
                if row["label"] == "ns" and not show_ns:
                    continue
                y = base + level * 5.4
                x1 = x[cond_idx] + (i - 1) * width
                x2 = x[cond_idx] + (j - 1) * width
                draw_bracket(ax, x1, x2, y, row["label"])
                highest = max(highest, y + 5.0)
                level += 1

        ax.set_xticks(x, [label for _, label in CONDITIONS])
        ax.set_ylabel("Subject fingerprinting accuracy (%)")
        ax.set_ylim(0, highest + 2)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.yaxis.grid(True, alpha=0.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", length=0, pad=10, labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.legend(ncol=3, frameon=False, loc="upper center",
                  bbox_to_anchor=(0.5, 1.17), fontsize=15)
        fig.savefig(output_dir / "subject_fingerprinting_barplot.png",
                    dpi=dpi, bbox_inches="tight")
        fig.savefig(output_dir / "subject_fingerprinting_barplot.pdf",
                    bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent_path", type=Path,
                        default=Path("/home/student1/Desktop/Charalampos_Lamprou/rDCM-GEN_results_HCP_2026"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/home/student1/Desktop/Charalampos_Lamprou/rDCM-GEN/plots/SF"))
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--show-ns", action="store_true",
                        help="Show brackets for nonsignificant comparisons too")
    args = parser.parse_args()
    stats = load_results(args.parent_path)
    tests = paired_tests(stats)
    print("Condition                 Method       Mean      SD (sample)")
    for dirname, _ in CONDITIONS:
        for key, _, _ in METHODS:
            d = stats[dirname][key]
            print(f"{dirname:25s} {key:10s} {d['mean']:7.2f}   {d['sd']:7.2f}")
    print("\nUncorrected two-sided paired t-tests (* p<.05, ** p<.01, *** p<.001):")
    for dirname, _ in CONDITIONS:
        for row in tests[dirname]:
            print(f"{dirname:18s} {row['method_a']:9s} vs {row['method_b']:9s} "
                  f"n={row['n']:3d} t={row['t']:9.3g} p={row['p']:.6g} {row['label']}")
    print("\nNOTE: Assumes same subjects/repetitions in same order within each condition.")
    print("No multiple-testing adjustment has been applied.")
    plot(stats, tests, args.output_dir, dpi=args.dpi, show_ns=args.show_ns)
    print(f"Saved PNG and PDF to {args.output_dir}")


if __name__ == "__main__":
    main()