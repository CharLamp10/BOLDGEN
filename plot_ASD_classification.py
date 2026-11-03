"""Plot repetition-level AUC and F1, and compare methods using paired t-tests.

Dependencies: numpy, matplotlib, scipy, scikit-learn.
Inputs: parent_path/ASD_CLS/<architecture>/<method>/results.pkl (trusted files).

Pairing is BY REPETITION ORDER: run i in one method is paired with run i in
another. All methods in an architecture must use matching splits/seeds in the
same order. Equal numbers of repetitions do NOT prove correct pairing; this
script cannot verify subject IDs from y_true/y_pred alone. No runs are dropped
or truncated to make arrays fit.

Tests are two-sided scipy.stats.ttest_rel on per-repetition metric values,
not on individual predictions. Unadjusted p-values are reported for every method pair, separately for each
architecture and metric. Six additional p-value heatmaps are saved.
The 95% confidence intervals for mean paired differences are unadjusted.
Ordinary paired tests assume independent replicate differences; overlapping
train/test splits can violate this, so such p-values are exploratory.

AUC figures keep the original filenames, e.g. VarCoNet.png/pdf.
F1 figures use the suffix _F1, e.g. VarCoNet_F1.png/pdf.
Pairwise p-value heatmaps use _AUC_pvalues or _F1_pvalues suffixes.
Tables: metric_summary.csv, per_repetition_metrics.csv, paired_t_tests.csv.
F1 uses ASD (label 1) as positive class and the supplied fixed threshold.
"""

import argparse
import csv
import itertools
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel, t as student_t
from sklearn.metrics import roc_auc_score, f1_score


MODELS = {
    "ASD-DiagNet": ["ABIDEI", "TimeVAE", "TimeGAN", "T2I-Diff", "DSFM", "CR-VAE", "rDCM-GEN"],
    "BolT": ["ABIDEI", "TimeVAE", "TimeGAN", "T2I-Diff", "DSFM", "CR-VAE", "rDCM-GEN"],
    "VarCoNet": ["ABIDEI", "CR-VAE", "rDCM-GEN"],
}
COLORS = {
    "ABIDEI": "#64748B", "TimeVAE": "#8D9BAC", "TimeGAN": "#9274B2",
    "T2I-Diff": "#D69443", "DSFM": "#82A45E",
    "CR-VAE": "#C46A65", "rDCM-GEN": "#147C85",
}
LABELS = {"ABIDEI": "ABIDE I", "T2I-Diff": "T2I-Diff", "CR-VAE": "CR-VAE", "rDCM-GEN": "rDCM-GEN"}
FOLDERS = {m: m for methods in MODELS.values() for m in methods}
METRICS = ("AUC", "F1")
AUC_YLIMS = {"ASD-DiagNet": (40, 75), "BolT": (40, 75), "VarCoNet": (65, 80)}
F1_YLIMS = {"ASD-DiagNet": (55, 65), "BolT": (55, 65), "VarCoNet": (60, 70)}


def is_one_hot(y):
    return (y.ndim == 2 and y.shape[1] == 2
            and np.all((y == 0) | (y == 1)) and np.all(y.sum(axis=1) == 1))


def extract_repetitions(payload, method, layout="auto"):
    """Accept flat/column/one-hot runs, repeated arrays, ragged runs or run dicts.

    In auto mode an [n,2] one-hot label matrix is a single run. For the unusual
    case of repetitions with exactly two subjects, pass --layout repetitions.
    """
    obj = payload.get(method, payload) if isinstance(payload, dict) else payload
    if isinstance(obj, dict) and "result" in obj and "y_true" not in obj:
        obj = obj["result"]
    if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], dict):
        return [(r["y_true"], r["y_pred"]) for r in obj]
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a result dictionary or list of run dictionaries for {method}")
    yt, yp = obj["y_true"], obj["y_pred"]
    if layout == "single":
        return [(yt, yp)]
    if layout == "repetitions":
        if len(yt) != len(yp) or not len(yt):
            raise ValueError("Unequal or empty repetition lists")
        return list(zip(yt, yp))
    try:
        y = np.asarray(yt)
    except ValueError:  # Ragged test sets across repetitions.
        y = None
    if y is None or (y.dtype == object and y.ndim == 1 and len(y) and not np.isscalar(y[0])):
        if len(yt) != len(yp) or not len(yt):
            raise ValueError("Unequal or empty ragged repetition lists")
        return list(zip(yt, yp))
    if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1) or is_one_hot(y):
        return [(yt, yp)]
    if y.ndim in (2, 3):
        if len(yt) != len(yp) or not len(yt):
            raise ValueError("Unequal or empty repeated prediction arrays")
        return list(zip(yt, yp))
    raise ValueError(f"Unrecognized label shape for {method}: {y.shape}")


def score_one(y_true, y_pred, threshold):
    y, p = np.asarray(y_true), np.asarray(y_pred)
    if y.ndim == 2 and y.shape[1] == 2:
        if not is_one_hot(y):
            raise ValueError("Two-column labels must be valid one-hot labels")
        y = y[:, 1]
    elif y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if p.ndim == 2 and p.shape[1] in (1, 2):
        p = p[:, -1]
    if y.ndim != 1 or p.ndim != 1 or not len(y) or len(y) != len(p):
        raise ValueError("Expected equally sized, nonempty label and probability vectors per run")
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("Labels must be 0 or 1 (checked before conversion to integers)")
    y, p = y.astype(int), p.astype(float)
    if len(np.unique(y)) < 2:
        raise ValueError("Both classes are required to calculate AUC")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Expected finite probability predictions in [0,1]")
    if np.all(np.isin(p, [0, 1])):
        warnings.warn("y_pred is binary; use saved probabilities for full ROC resolution.", stacklevel=2)
    return (100 * roc_auc_score(y, p),
            100 * f1_score(y, p >= threshold, pos_label=1, zero_division=0))


def load_results(parent, threshold, layout="auto"):
    out = {}
    for model, methods in MODELS.items():
        for method in methods:
            file = Path(parent) / "ASD_CLS" / model / FOLDERS[method] / "results.pkl"
            if not file.is_file():
                raise FileNotFoundError(file)
            with file.open("rb") as stream:
                payload = pickle.load(stream)  # Only load trusted pickle files.
            try:
                runs = extract_repetitions(payload, method, layout)
                arr = np.asarray([score_one(y, p, threshold) for y, p in runs], dtype=float)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{file}: {exc}") from exc
            out[model, method] = arr
            sd = arr.std(axis=0, ddof=1) if len(arr) > 1 else np.zeros(2)
            print(f"{model:12} {method:10} n={len(arr):2} "
                  f"AUC={arr[:,0].mean():.2f} +/- {sd[0]:.2f}, "
                  f"F1={arr[:,1].mean():.2f} +/- {sd[1]:.2f}")
    return out


def paired_comparison(a, b):
    """Two-sided paired t-test; differences are method A minus method B."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or not len(a):
        raise ValueError("Paired tests require equally sized, nonempty repetition vectors")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Paired scores must be finite; no runs will be silently dropped")
    n = len(a)
    difference = a - b
    row = dict(n_pairs=n, df=n-1, mean_a_pct=float(a.mean()), mean_b_pct=float(b.mean()),
               mean_difference_pp=float(difference.mean()), t_stat=np.nan, p_raw=np.nan,
               ci95_low_pp=np.nan, ci95_high_pp=np.nan, status="ok")
    if n < 2:
        row["status"] = "insufficient_pairs"
        return row
    # A zero (or machine-precision) variance gives no estimable t-test standard error.
    tolerance = 16 * np.finfo(float).eps * max(1., float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    if np.ptp(difference) <= tolerance:
        row["status"] = "zero_variance_paired_differences"
        return row
    result = ttest_rel(a, b, alternative="two-sided", nan_policy="raise")
    standard_error = difference.std(ddof=1) / np.sqrt(n)
    margin = student_t.ppf(.975, df=n-1) * standard_error
    row.update(t_stat=float(result.statistic), p_raw=float(result.pvalue),
               ci95_low_pp=float(difference.mean()-margin), ci95_high_pp=float(difference.mean()+margin))
    return row


def statistical_tests(data, alpha=.05):
    """Unadjusted paired tests for all method pairs, for AUC and F1."""
    all_rows = []
    for model, methods in MODELS.items():
        counts = {method: len(data[model, method]) for method in methods}
        if len(set(counts.values())) != 1:
            raise ValueError(f"{model}: unequal repetition counts {counts}. Align matching runs first; "
                             "this script does not truncate or guess pairings.")
        for column, metric in enumerate(METRICS):
            print(f"\n{model} | {metric}: two-sided paired t-tests (uncorrected p-values)")
            for method_a, method_b in itertools.combinations(methods, 2):
                row = dict(model=model, metric=metric, method_a=method_a, method_b=method_b,
                           pairing="repetition_index", alpha=alpha)
                row.update(paired_comparison(data[model, method_a][:, column],
                                             data[model, method_b][:, column]))
                row["significant_raw"] = bool(row["p_raw"] < alpha) if np.isfinite(row["p_raw"]) else ""
                if row["status"] == "ok":
                    mark = " *" if row["significant_raw"] else ""
                    print(f"  {method_a:10} vs {method_b:10}: "
                          f"delta={row['mean_difference_pp']:+.3f} pp, t={row['t_stat']:.3f}, "
                          f"p={row['p_raw']:.4g}{mark}")
                else:
                    print(f"  {method_a:10} vs {method_b:10}: {row['status']} (p=NA)")
                all_rows.append(row)
    return all_rows


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_tables(data, tests, output_dir, threshold, alpha):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary, repetitions = [], []
    for (model, method), arr in data.items():
        for column, metric in enumerate(METRICS):
            values = arr[:, column]
            summary.append(dict(model=model, method=method, metric=metric, n_repetitions=len(values),
                mean_pct=float(values.mean()), sd_pct=float(values.std(ddof=1)) if len(values)>1 else np.nan,
                f1_threshold=threshold))
        for i, (auc, f1) in enumerate(arr, start=1):
            repetitions.append(dict(model=model, method=method, repetition=i, auc_pct=auc,
                                    f1_pct=f1, f1_threshold=threshold))
    write_csv(output / "metric_summary.csv", summary)
    write_csv(output / "per_repetition_metrics.csv", repetitions)
    write_csv(output / "paired_t_tests.csv", tests)
    (output / "analysis_notes.txt").write_text(
        f"F1: positive class ASD=1, probability >= {threshold}. AUC uses probabilities.\n"
        "Plots show means and sample SD (ddof=1), not standard errors.\n"
        "Tests use paired repetition-level scores in the stored run order.\n"
        "Verify matching splits/seeds across methods; equal run counts do not verify pairing.\n"
        f"Two-sided paired t-test; alpha={alpha}. P-values are uncorrected for multiple comparisons.\n"
        "mean_difference_pp = method A minus method B, in percentage points.\n"
        "95% confidence intervals are unadjusted paired-difference intervals.\n"
        "p=NaN means insufficient pairs or zero-variance differences; it is not a valid test.\n"
        "Overlapping test splits can make repetition differences dependent; ordinary paired\n"
        "t-test p-values in that setting should be treated as exploratory.\n",
        encoding="utf-8")


def plot(data, output_dir, dpi=600):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    style = {"font.family": "DejaVu Sans", "font.size": 28, "axes.labelsize": 34,
             "xtick.labelsize": 30, "ytick.labelsize": 30, "pdf.fonttype": 42, "svg.fonttype": "none"}
    with plt.rc_context(style):
        for model, methods in MODELS.items():
            for column, metric in enumerate(METRICS):
                fig, ax = plt.subplots(figsize=(14, 8.5))
                means, deviations = [], []
                for j, method in enumerate(methods):
                    values = data[model, method][:, column]
                    mean = float(values.mean())
                    sd = float(values.std(ddof=1)) if len(values)>1 else 0.
                    means.append(mean)
                    deviations.append(sd)
                    ax.bar(j, mean, width=.76, color=COLORS[method], edgecolor="white", linewidth=.6,
                           yerr=sd if len(values)>1 else None, capsize=5,
                           error_kw={"elinewidth":1.5, "capthick":1.5})
                ax.set_xticks(range(len(methods)), [LABELS.get(m,m) for m in methods], rotation=35, ha="right")
                low, high = (AUC_YLIMS if metric == "AUC" else F1_YLIMS)[model]
                # Keep the requested scales unless they would hide a mean/error bar.
                lower = float(np.min(np.asarray(means)-deviations))
                upper = float(np.max(np.asarray(means)+deviations))
                ax.set_ylim(min(low, lower-2) if lower<low else low,
                            max(high, upper+2) if upper>high else high)
                ax.set_ylabel("AUC (%)" if metric == "AUC" else "F1 score (%)")
                ax.grid(axis="y", alpha=.18)
                ax.set_axisbelow(True)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.tick_params(axis="x", labelsize=30, length=0, pad=14)
                ax.tick_params(axis="y", labelsize=30, length=0, pad=10)
                fig.tight_layout()
                filename = model.replace("-", "_") + ("_F1" if metric == "F1" else "")
                fig.savefig(output / f"{filename}.png", dpi=dpi, bbox_inches="tight")
                fig.savefig(output / f"{filename}.pdf", bbox_inches="tight")
                plt.close(fig)


def plot_pvalue_matrices(tests, output_dir, alpha=.05, dpi=600):
    """Six symmetric raw p-value heatmaps: one for each architecture and metric."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for model, methods in MODELS.items():
        n = len(methods)
        indices = {method: i for i, method in enumerate(methods)}
        for metric in METRICS:
            matrix = np.full((n, n), np.nan, dtype=float)
            for row in tests:
                if row["model"] == model and row["metric"] == metric:
                    i, j = indices[row["method_a"]], indices[row["method_b"]]
                    matrix[i, j] = matrix[j, i] = row["p_raw"]

            # Show p >= alpha in the lightest color; numbers retain exact raw values.
            cmap = plt.get_cmap("Blues_r").copy()
            cmap.set_bad("#e9edf2")
            masked = np.ma.masked_invalid(matrix)
            fig_size = max(7.2, 1.2 * n + 2.5)
            fig, ax = plt.subplots(figsize=(fig_size, fig_size * .86))
            im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=alpha, interpolation="nearest")
            names = [LABELS.get(m, m) for m in methods]
            ax.set_xticks(range(n), names, rotation=45, ha="right", fontsize=24)
            ax.set_yticks(range(n), names, fontsize=24)
            #ax.set_title(f"{model} — {metric}: paired t-test p-values", fontsize=18, pad=20)
            ax.set_xticks(np.arange(-.5, n, 1), minor=True)
            ax.set_yticks(np.arange(-.5, n, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=2)
            ax.tick_params(which="minor", bottom=False, left=False)
            ax.tick_params(which="major", length=0)

            for i in range(n):
                for j in range(n):
                    p = matrix[i, j]
                    if i == j:
                        label = "—"
                    elif not np.isfinite(p):
                        label = "NA"
                    else:
                        numeric = f"{p:.2g}" if p >= .001 else f"{p:.1e}"
                        stars = "***" if p < .001 else "**" if p < .01 else "*" if p < alpha else ""
                        #label = f"{numeric}\n{stars}" if stars else numeric
                        label = stars
                    ax.text(j, i, label, ha="center", va="center", fontsize=24,
                            color="white" if np.isfinite(p) and p < alpha * .30 else "#172536")

            cbar = fig.colorbar(im, ax=ax, fraction=.045, pad=.035)
            cbar.set_label("p-value", fontsize=24)
            cbar.ax.tick_params(labelsize=20)
            fig.tight_layout(rect=(0, .035, 1, 1))
            filename = f"{model.replace('-', '_')}_{metric}_pvalues"
            fig.savefig(output / f"{filename}.png", dpi=dpi, bbox_inches="tight")
            fig.savefig(output / f"{filename}.pdf", bbox_inches="tight")
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent_path", "--parent-path", dest="parent_path",
        required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--layout", choices=["auto", "single", "repetitions"], default="auto",
                        help="Prediction array layout; auto handles normal single/repeated result files")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    if not 0 < args.alpha < 1 or args.dpi < 1:
        parser.error("Require 0 < --alpha < 1 and a positive --dpi")
    print("Pairing assumes matching splits/seeds in the same repetition order across methods.")
    data = load_results(args.parent_path, args.threshold, args.layout)
    tests = statistical_tests(data, args.alpha)
    save_tables(data, tests, args.output_dir, args.threshold, args.alpha)
    plot(data, args.output_dir, args.dpi)
    plot_pvalue_matrices(tests, args.output_dir, args.alpha, args.dpi)
    print(f"\nSaved 12 figures (6 bar charts + 6 p-value matrices, PNG + PDF), "
          f"metric tables and paired_t_tests.csv to {args.output_dir}")


if __name__ == "__main__":
    main()



