import argparse
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score
import os

MODELS = {
    "ASD-DiagNet": ["ABIDEI", "TimeVAE", "TimeGAN", "T2IDIFF", "DSFM", "CRVAE", "rDCM-GEN"],
    "BolT": ["ABIDEI", "TimeVAE", "TimeGAN", "T2IDIFF", "DSFM", "CRVAE", "rDCM-GEN"],
    "VarCoNet": ["ABIDEI", "CRVAE", "rDCM-GEN"],
}
COLORS = {
    "ABIDEI": "#64748B", "TimeVAE": "#8D9BAC", "TimeGAN": "#9274B2",
    "T2IDIFF": "#D69443", "DSFM": "#82A45E",
    "CRVAE": "#C46A65", "rDCM-GEN": "#147C85",
}
LABELS = {"ABIDEI": "ABIDE I", "T2IDIFF": "T2I-Diff",
          "CRVAE": "CR-VAE", "rDCM-GEN": "rDCM-GEN"}
# Modify this mapping if your baseline folder/key is named differently.
FOLDERS = {m: m for methods in MODELS.values() for m in methods}


def extract_repetitions(payload, method):
    obj = payload.get(method, payload) if isinstance(payload, dict) else payload
    if isinstance(obj, dict) and "result" in obj and "y_true" not in obj:
        obj = obj["result"]
    if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], dict):
        return [(r["y_true"], r["y_pred"]) for r in obj]
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict/list for {method}")
    yt, yp = obj["y_true"], obj["y_pred"]
    if isinstance(yt, (list, tuple)) and yt and not np.isscalar(yt[0]):
        if len(yt) == len(yp):
            return list(zip(yt, yp))
    yt, yp = np.asarray(yt), np.asarray(yp)
    if yt.ndim == 2 and yp.ndim == 2 and yt.shape == yp.shape:
        # Common shape [n_repetitions, n_test_subjects]
        if yt.shape[1] != 2 or not np.all((yt == 0) | (yt == 1)) or not np.all(yt.sum(axis=1) == 1):
            return list(zip(yt, yp))
    return [(yt, yp)]


def score_one(y_true, y_pred, threshold):
    y = np.asarray(y_true)
    p = np.asarray(y_pred)
    if y.ndim == 2 and y.shape[-1] == 2:
        y = y[:, 1]
    if p.ndim == 2 and p.shape[-1] == 2:
        p = p[:, 1]
    y = y.ravel().astype(int)
    p = p.ravel().astype(float)
    if len(y) != len(p) or not np.all(np.isin(y, [0, 1])):
        raise ValueError("Labels and predictions must have equal lengths and binary labels")
    if len(np.unique(y)) < 2:
        raise ValueError("Both classes required to calculate AUC")
    if not np.all(np.isfinite(p)):
        raise ValueError("Non-finite predictions")
    if np.min(p) < 0 or np.max(p) > 1:
        raise ValueError("Expected probability predictions in [0,1]")
    if np.all(np.isin(p, [0,1])):
        print("WARNING: y_pred looks binary; ROC AUC requires continuous scores.")
    return 100 * roc_auc_score(y, p), 100 * f1_score(y, p >= threshold, zero_division=0)


def load_results(parent, threshold):
    out = {}
    for model, methods in MODELS.items():
        for method in methods:
            file = os.path.join(parent, "ASD_CLS", model, FOLDERS[method], "results.pkl")
            if not os.path.isfile(file):
                raise FileNotFoundError(file)
            with open(file, "rb") as f:
                payload = pickle.load(f)  # Only load trusted pickle files.
            runs = extract_repetitions(payload, method)
            arr = np.asarray([score_one(y, pred, threshold) for y, pred in runs])
            out[model, method] = arr
            std = arr.std(axis=0, ddof=1) if len(arr)>1 else np.zeros(2)
            print(f"{model:12} {method:10} n={len(arr):2} "
                  f"AUC={arr[:,0].mean():.2f} +/- {std[0]:.2f}, "
                  f"F1={arr[:,1].mean():.2f} +/- {std[1]:.2f}")
    return out


def plot(data, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "pdf.fonttype": 42,
        "svg.fonttype": "none"
    })

    # One row with three panels (AUC only)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharey=False)

    for col, (model, methods) in enumerate(MODELS.items()):
        ax = axes[col]

        for j, method in enumerate(methods):
            # Column 0 corresponds to AUC
            values = data[model, method][:, 0]

            mean = values.mean()
            sd = values.std(ddof=1) if len(values) > 1 else 0

            ax.bar(
                j, mean,
                width=.76,
                color=COLORS[method],
                edgecolor="white",
                linewidth=.6,
                yerr=sd if len(values) > 1 else None,
                capsize=3
            )

        ax.set_xticks(
            range(len(methods)),
            [LABELS.get(m, m) for m in methods],
            rotation=40,
            ha="right"
        )

        ylims = {
            "ASD-DiagNet": (40, 75),
            "BolT": (40, 75),
            "VarCoNet": (65, 80)
        }
        
        ax.set_ylim(*ylims[model])

        ax.set_ylabel("AUC (%)")

        ax.set_title(
            model.replace("_", "-"),
            weight="bold",
            pad=14
        )

        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", length=0)

    fig.tight_layout(w_pad=1.8)

    fig.savefig(
        os.path.join(output_dir, "asd_classification.png"),
        dpi=600,
        bbox_inches="tight"
    )

    plt.close(fig)
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent_path", default=r'/home/student1/Desktop/Charalampos_Lamprou/rDCM-GEN_results_HCP_2026')
    parser.add_argument("--output-dir", default=r'/home/student1/Desktop/Charalampos_Lamprou/rDCM-GEN/plots/ASD_CLS')
    parser.add_argument("--threshold", type=float, default=.5)
    args = parser.parse_args()
    plot(load_results(args.parent_path, args.threshold), args.output_dir)

if __name__ == "__main__":
    main()