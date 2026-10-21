"""GPU-oriented functional-connectivity classifier.

Dependencies: torch, numpy, scikit-learn, tqdm.
Run: python fc_classifier_gpu.py
Edit CONFIG at the bottom for paths, device, and cache placement.

Changes:
- Compute upper-triangle correlations once per dataset in batches on the GPU.
- Reuse ABIDEII correlations across repetitions and training methods.
- Precompute one-hot training labels; aggregate losses/predictions on device.
- Optional full GPU feature caching; otherwise use pinned CPU feature batches.
- Infer input size from the number of regions instead of hard-coding 4950.

The original two-output Softmax + BCELoss, Adam settings, and partial-batch
skipping are preserved. Standardization before Pearson correlation is redundant
and is omitted. Float32 correlations may differ slightly from NumPy float64.
Undefined correlations are set to zero instead of propagating NaNs.
"""

import copy
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def load_pair(asd_path, control_path):
    asd = np.load(asd_path)
    control = np.load(control_path)
    signals = np.concatenate((asd, control), axis=0)
    labels = torch.cat((torch.ones(len(asd), 1), torch.zeros(len(control), 1)))
    return signals, labels


def load_training_data(config, method):
    if method == "ABIDEI":
        root = config["path_data"]
        return load_pair(os.path.join(root, "ABIDEI_ASD_nilearn.npy"),
                         os.path.join(root, "ABIDEI_NC_nilearn.npy"))
    if method in ("CRVAE", "rDCM-GEN"):
        key, tag = (("path_augmented_crvae", "CR-VAE") if method == "CRVAE"
                    else ("path_augmented_rdcmgen", "rDCM-GEN"))
        root = config[key]
        signals = torch.load(os.path.join(root, f"augmented_{tag}_ABIDEI.pt"),
                             map_location="cpu", weights_only=True)
        labels = torch.load(os.path.join(root, f"labels_augmented_{tag}_ABIDEI.pt"),
                            map_location="cpu", weights_only=True)
        if signals.ndim != 4:
            raise ValueError(f"{method}: expected [B, N, T, R], got {signals.shape}")
        B, N, T, R = signals.shape
        labels = torch.as_tensor(labels).reshape(-1)
        if labels.numel() != B:
            raise ValueError(f"{method}: expected one label per B")
        return signals.reshape(B * N, T, R), labels.repeat_interleave(N).reshape(-1, 1)
    if method == "TimeVAE":
        root = config["path_augmented_timevae"]
        return load_pair(
            os.path.join(root, "ABIDEI_ASD", "timeVAE_ABIDEI_ASD_prior_samples.npy"),
            os.path.join(root, "ABIDEI_NC", "timeVAE_ABIDEI_NC_prior_samples.npy"))
    if method == "TimeGAN":
        root = config["path_augmented_timegan"]
        return load_pair(os.path.join(root, "augmented_TimeGAN_ABIDEI_ASD.npy"),
                         os.path.join(root, "augmented_TimeGAN_ABIDEI_NC.npy"))
    if method == "T2IDIFF":
        root = os.path.join(config["path_augmented_t2idiff"], "ABIDEI")
        return (np.load(os.path.join(root, "gen_sig_ABIDEI_fold1.npy")),
                np.load(os.path.join(root, "gen_label_sig_ABIDEI_fold1.npy")))
    if method == "DSFM":
        root = os.path.join(config["path_augmented_dsfm"], "ABIDEI", "samples")
        return load_pair(os.path.join(root, "asd_bold.npy"),
                         os.path.join(root, "control_bold.npy"))
    raise ValueError(f"Unknown method: {method}")


def prepare_labels(labels, count):
    labels = torch.as_tensor(labels, dtype=torch.float32).reshape(-1, 1).cpu()
    if len(labels) != count or not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Expected one binary (0 or 1) label per sample")
    return labels.contiguous()


@torch.no_grad()
def compute_correlations(signals, device, batch_size=128, description="Correlations"):
    """[samples, T, R] -> CPU float32 [samples, R*(R-1)//2].

    Pearson correlation is invariant to per-region standardization, so the
    previous separate full-dataset standardization pass is unnecessary.
    Strict upper-triangle order matches np.triu_indices(R, k=1).
    Undefined correlations (e.g. constant regions) are replaced with zero.
    """
    signals = torch.as_tensor(signals)
    if signals.ndim != 3 or not len(signals) or signals.shape[1] < 2 or signals.shape[2] < 2:
        raise ValueError("Expected nonempty [samples, T, R] with T >= 2 and R >= 2")
    if batch_size < 1:
        raise ValueError("corr_batch_size must be positive")
    R = signals.shape[-1]
    rows, cols = torch.triu_indices(R, R, offset=1, device=device)
    features = torch.empty((len(signals), rows.numel()), dtype=torch.float32)
    for start in tqdm(range(0, len(signals), batch_size), desc=description):
        x = signals[start:start + batch_size].to(device=device, dtype=torch.float32)
        x = x - x.mean(dim=1, keepdim=True)
        x = x / x.square().sum(dim=1, keepdim=True).sqrt()
        corr = x.transpose(1, 2) @ x
        corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-1, 1)
        features[start:start + len(x)].copy_(corr[:, rows, cols].cpu())
    return features


class CachedBatches:
    """CPU cache with pinned transfers, or directly indexed GPU cache.

    GPU caching avoids per-sample GPU operations in DataLoader collation.
    No multiprocessing workers are needed because features are precomputed.
    """
    def __init__(self, x, y, batch_size, shuffle=False, drop_last=False):
        if batch_size < 1 or len(x) != len(y):
            raise ValueError("Invalid batch size or mismatched labels")
        self.x, self.y = x, y
        self.batch_size, self.shuffle, self.drop_last = batch_size, shuffle, drop_last
        self.loader = None
        if x.device.type == "cpu":
            self.loader = DataLoader(
                TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle,
                drop_last=drop_last, num_workers=0,
                pin_memory=torch.cuda.is_available())

    def __len__(self):
        n, b = len(self.x), self.batch_size
        return n // b if self.drop_last else (n + b - 1) // b

    def __iter__(self):
        if self.loader is not None:
            yield from self.loader
            return
        order = torch.randperm(len(self.x), device=self.x.device) if self.shuffle else None
        for start in range(0, len(self.x), self.batch_size):
            end = min(start + self.batch_size, len(self.x))
            if self.drop_last and end - start < self.batch_size:
                break
            if order is None:
                yield self.x[start:end], self.y[start:end]
            else:
                idx = order[start:end]
                yield self.x[idx], self.y[idx]


class Model(nn.Module):
    def __init__(self, num_inputs=4950, num_classes=2):
        super().__init__()
        # Retain the original model architecture and training objective.
        self.classifier = nn.Sequential(
            nn.Linear(num_inputs, 400),
            nn.Linear(400, 200),
            nn.Linear(200, 2), 
            nn.Softmax(dim=-1))

    def forward(self, x):
        return self.classifier(x)


def train_epoch(model, batches, device, optimizer, criterion):
    if len(batches) == 0:
        raise ValueError("No training batches: reduce batch_size or set drop_last=False")
    model.train()
    total_loss = torch.zeros((), device=device)
    for batch_x, batch_y in batches:
        x = batch_x.to(device, non_blocking=True)
        # Targets are already float32 [batch, 2] one-hot vectors.
        target = batch_y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), target)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach()
    # One scalar synchronization per epoch.
    return (total_loss / len(batches)).item()


@torch.inference_mode()
def predict(model, batches, device):
    model.eval()
    targets, probabilities = [], []
    for batch_x, batch_y in batches:
        x = batch_x.to(device, non_blocking=True)
        probabilities.append(model(x)[:, 1])
        targets.append(batch_y.reshape(-1))
    if not probabilities:
        raise ValueError("Cannot evaluate an empty dataset")
    return (torch.cat(targets).cpu().numpy(),
            torch.cat(probabilities).cpu().numpy())


def main(config):
    device = torch.device(config["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: use CUDA-enabled PyTorch or device='cpu'")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"{device} does not exist; visible GPUs: {torch.cuda.device_count()}")
        torch.cuda.set_device(device)
    if config["epochs"] < 1 or config["repeats"] < 1:
        raise ValueError("epochs and repeats must be positive")
    seed = config.get("seed")
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    # Compute evaluation features once, outside both loops.
    eval_signals, eval_labels = load_pair(
        os.path.join(config["path_data"], "ABIDEII_ASD_nilearn.npy"),
        os.path.join(config["path_data"], "ABIDEII_NC_nilearn.npy"))
    eval_R = eval_signals.shape[-1]
    eval_labels = prepare_labels(eval_labels, len(eval_signals)).reshape(-1).long()
    eval_features = compute_correlations(
        eval_signals, device, config["corr_batch_size"], "ABIDEII correlations")
    del eval_signals

    cache_device = device if config["cache_on_gpu"] else torch.device("cpu")
    results_by_method = {}
    for method in config["methods"]:
        print(f"\nMethod: {method}")
        signals, labels = load_training_data(config, method)
        if signals.shape[-1] != eval_R:
            raise ValueError(f"{method}: training and evaluation region counts differ")
        labels = prepare_labels(labels, len(signals)).reshape(-1).long()
        train_features = compute_correlations(
            signals, device, config["corr_batch_size"], f"{method} correlations")
        del signals
        num_inputs = train_features.shape[1]
        feature_gib = (train_features.numel() + eval_features.numel()) * 4 / (1024 ** 3)
        print(f"Input features: {num_inputs}; feature cache approximately {feature_gib:.2f} GiB")
        print(f"Feature cache device: {cache_device}")
        train_features = train_features.to(cache_device)
        # One conversion for the whole dataset, not every batch of every epoch.
        train_targets = F.one_hot(labels.to(cache_device), num_classes=2).float()
        del labels
        train_batches = CachedBatches(
            train_features, train_targets, config["batch_size"], shuffle=True,
            drop_last=config["drop_last"])
        test_aucs, y_trues, y_preds = [], [], []

        for i in range(config["repeats"]):
            # seed=None preserves the original unseeded random splitting.
            test_idx, val_idx = train_test_split(
                np.arange(len(eval_features)), test_size=0.2, shuffle=True,
                stratify=eval_labels.numpy(),
                random_state=None if seed is None else seed + i)
            test_batches = CachedBatches(
                eval_features[test_idx].to(cache_device),
                eval_labels[test_idx].to(cache_device), config["eval_batch_size"])
            val_batches = CachedBatches(
                eval_features[val_idx].to(cache_device),
                eval_labels[val_idx].to(cache_device), config["eval_batch_size"])
            model = Model(num_inputs=num_inputs).to(device)
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.0001)
            best_auc, best_state = -float("inf"), None
            with tqdm(range(config["epochs"]), desc=f"{method} run {i + 1}") as progress:
                for _ in progress:
                    loss = train_epoch(model, train_batches, device, optimizer, criterion)
                    y_true, y_pred = predict(model, val_batches, device)
                    auc = float(roc_auc_score(y_true, y_pred))
                    if auc > best_auc:
                        best_auc = auc
                        best_state = copy.deepcopy(model.state_dict())
                    progress.set_postfix(loss=loss, val_auc=auc, best_auc=best_auc)
            if best_state is None:
                raise RuntimeError("No valid validation AUC was produced")
            model.load_state_dict(best_state)
            y_true, y_pred = predict(model, test_batches, device)
            test_auc = float(roc_auc_score(y_true, y_pred))
            test_aucs.append(test_auc)
            y_trues.append(y_true.tolist())
            y_preds.append(y_pred.tolist())
            print(f"AUC = {test_auc:.3f}")
            del model, optimizer, best_state, test_batches, val_batches

        print(f"mean AUC = {np.mean(test_aucs):.3f} | std AUC = {np.std(test_aucs):.3f}")
        results = dict(auc_scores=test_aucs, y_true=y_trues, y_pred=y_preds)
        results_by_method[method] = results
        if config["save_results"]:
            output_dir = os.path.join(config["path_save"], method)
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "results.pkl"), "wb") as file:
                pickle.dump(results, file)
        del train_batches, train_features, train_targets
    return results_by_method


if __name__ == "__main__":
    CONFIG = {
        "path_data": "/media/student1/ExtremePro/rDCM-GEN/data",
        "path_save": "/home/student1/Desktop/Charalampos_Lamprou/rDCM-GEN_results_HCP_2026/ASD_CLC/FC",
        "path_augmented_rdcmgen": "/media/student1/ExtremePro/rDCM-GEN/rDCM-GEN_results/augmented_rDCM-GEN_noFCloss",
        "path_augmented_crvae": "/media/student1/ExtremePro/CRVAE_augmented",
        "path_augmented_timevae": "/media/student1/ExtremePro/TimeVAE_results/outputs/gen_data",
        "path_augmented_timegan": "/media/student1/ExtremePro/TimeGAN_results",
        "path_augmented_t2idiff": "/media/student1/ExtremePro/T2IDIFF",
        "path_augmented_dsfm": "/media/student1/ExtremePro/DSFM_results",
        # Supported: ABIDEI, TimeVAE, TimeGAN, CRVAE, T2IDIFF, DSFM, rDCM-GEN.
        "methods": ["rDCM-GEN"], #["ABIDEI", "TimeVAE", "TimeGAN", "CRVAE", "T2IDIFF", "DSFM", "rDCM-GEN"],
        "atlas": "SCH",
        "batch_size": 128,
        "eval_batch_size": 128,
        "corr_batch_size": 128,  # Reduce if correlation preprocessing exhausts VRAM.
        "epochs": 25,
        "repeats": 10,
        "device": "cuda:0",  # Use cuda:0 if only one GPU is visible.
        "cache_on_gpu": False,  # True avoids feature transfers each epoch; needs VRAM.
        "drop_last": False,  # Original behavior: skip incomplete training batches.
        "seed": 42,  # Set an integer for seeded splits/initialization/shuffling.
        "save_results": True,
    }
    main(CONFIG)