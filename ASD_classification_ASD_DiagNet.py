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
    if method in ("CR-VAE", "rDCM-GEN"):
        key, tag = (("path_augmented_crvae", "CR-VAE") if method == "CR-VAE"
                    else ("path_augmented_rdcmgen", "rDCM-GEN"))
        root = config[key]
        signals = torch.load(os.path.join(root, f"augmented_{tag}_orig_ABIDEI.pt"),
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
        root = config["path_data"]
        temp1 = load_pair(os.path.join(root, "ABIDEI_ASD_nilearn.npy"),
                         os.path.join(root, "ABIDEI_NC_nilearn.npy"))
        root = config["path_augmented_timevae"]
        temp2 = load_pair(
            os.path.join(root, "ABIDEI_ASD", "timeVAE_ABIDEI_ASD_prior_samples.npy"),
            os.path.join(root, "ABIDEI_NC", "timeVAE_ABIDEI_NC_prior_samples.npy"))
        temp = (np.concatenate((temp1[0], temp2[0]), axis=0), np.concatenate((temp1[1], temp2[1]), axis=0))
        return temp
    if method == "TimeGAN":
        root = config["path_data"]
        temp1 = load_pair(os.path.join(root, "ABIDEI_ASD_nilearn.npy"),
                         os.path.join(root, "ABIDEI_NC_nilearn.npy"))
        root = config["path_augmented_timegan"]
        temp2 = load_pair(os.path.join(root, "augmented_TimeGAN_ABIDEI_ASD.npy"),
                         os.path.join(root, "augmented_TimeGAN_ABIDEI_NC.npy"))
        temp = (np.concatenate((temp1[0], temp2[0]), axis=0), np.concatenate((temp1[1], temp2[1]), axis=0))
        return temp
    if method == "T2I-Diff":
        root = config["path_data"]
        temp1 = load_pair(os.path.join(root, "ABIDEI_ASD_nilearn.npy"),
                         os.path.join(root, "ABIDEI_NC_nilearn.npy"))
        root = os.path.join(config["path_augmented_t2idiff"], "ABIDEI")
        temp2 = (np.load(os.path.join(root, "gen_sig_ABIDEI_fold1.npy")),
                np.load(os.path.join(root, "gen_label_sig_ABIDEI_fold1.npy")))
        temp = (np.concatenate((temp1[0][:,:199,:], temp2[0]), axis=0), np.concatenate((temp1[1].squeeze(), temp2[1]), axis=0))
        return temp
    if method == "DSFM":
        root = config["path_data"]
        temp1 = load_pair(os.path.join(root, "ABIDEI_ASD_nilearn.npy"),
                         os.path.join(root, "ABIDEI_NC_nilearn.npy"))
        root = os.path.join(config["path_augmented_dsfm"], "ABIDEI", "samples")
        temp2 = load_pair(os.path.join(root, "asd_bold.npy"),
                         os.path.join(root, "control_bold.npy"))
        temp = (np.concatenate((temp1[0], temp2[0]), axis=0), np.concatenate((temp1[1], temp2[1]), axis=0))
        return temp
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
    Strict lower-triangle order matches the original masked NumPy function.
    """
    signals = torch.as_tensor(signals)
    if signals.ndim != 3 or not len(signals) or signals.shape[1] < 2:
        raise ValueError("Expected nonempty [samples, T, R] with T >= 2")
    if batch_size < 1:
        raise ValueError("corr_batch_size must be positive")
    R = signals.shape[-1]
    rows, cols = torch.tril_indices(R, R, offset=-1, device=device)
    features = torch.empty((len(signals), rows.numel()), dtype=torch.float32)
    for start in tqdm(range(0, len(signals), batch_size), desc=description):
        x = signals[start:start + batch_size].to(device=device, dtype=torch.float32)
        x = x - x.mean(dim=1, keepdim=True)
        x = x / x.square().sum(dim=1, keepdim=True).sqrt()
        corr = x.transpose(1, 2) @ x
        corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-1, 1)
        features[start:start + len(x)].copy_(corr[:, rows, cols].cpu())
    return features


def get_regs(features, regnum):
    # Same high/low ordering as the original feature selection.
    if regnum < 1:
        raise ValueError("Too few correlation features for quarter-based selection")
    order = np.argsort(features.numpy().mean(axis=0))
    return np.concatenate((order[-regnum:][::-1], order[:regnum][::-1])).copy()


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


class MTAutoEncoder(nn.Module):
    def __init__(self, num_inputs=990, num_latent=200, tied=True,
                 num_classes=2, use_dropout=False):
        super().__init__()
        self.tied = tied
        self.num_latent = num_latent
        self.fc_encoder = nn.Linear(num_inputs, num_latent)
        if not tied:
            self.fc_decoder = nn.Linear(num_latent, num_inputs)
        layers = [nn.Dropout(0.5)] if use_dropout else []
        self.classifier = nn.Sequential(*layers, nn.Linear(num_latent, 1))

    def encode(self, x):
        return torch.tanh(self.fc_encoder(x))

    def classify(self, x):
        return self.classifier(self.encode(x))

    def forward(self, x, eval_classifier=False):
        latent = self.encode(x)
        logits = self.classifier(latent) if eval_classifier else None
        rec = (F.linear(latent, self.fc_encoder.weight.t()) if self.tied
               else self.fc_decoder(latent))
        return rec, logits


def train_epoch(model, batches, device, optimizer, criterion_ae, criterion_clf,
                p_bernoulli=None, mode="both", lam_factor=1.0):
    if mode not in ("both", "ae", "clf"):
        raise ValueError(f"Unknown training mode: {mode}")
    if p_bernoulli is not None and not 0 <= p_bernoulli <= 1:
        raise ValueError("p_bernoulli must be a keep probability between 0 and 1")
    if not len(batches):
        raise ValueError("No training batches: reduce batch_size or disable drop_last")
    model.train()
    total = torch.zeros((), device=device)
    for batch_x, batch_y in batches:
        data = batch_x.to(device, non_blocking=True)
        target = batch_y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if mode == "both" and p_bernoulli is None:
            rec, logits = model(data, eval_classifier=True)
            # Preserve the original loss scale and relative AE/classifier weight.
            loss_ae = criterion_ae(rec, data) / len(data)
            loss_clf = criterion_clf(logits, target)
            loss = loss_ae + lam_factor * loss_clf
        else:
            if mode in ("both", "ae"):
                noisy = data
                if p_bernoulli is not None:
                    mask = torch.empty_like(data).bernoulli_(p_bernoulli)
                    noisy = data * mask
                rec, _ = model(noisy)
                loss_ae = criterion_ae(rec, data) / len(data)
            if mode in ("both", "clf"):
                loss_clf = criterion_clf(model.classify(data), target)
            if mode == "both":
                loss = loss_ae + lam_factor * loss_clf
            elif mode == "ae":
                loss = loss_ae
            else:
                loss = loss_clf
        loss.backward()
        optimizer.step()
        total += loss.detach()
    return (total / len(batches)).item()


@torch.inference_mode()
def predict(model, batches, device, return_loss=False):
    model.eval()
    targets, probabilities = [], []
    total_loss = torch.zeros((), device=device)
    total_targets = 0
    for batch_x, batch_y in batches:
        x = batch_x.to(device, non_blocking=True)
        logits = model.classify(x)
        probabilities.append(torch.sigmoid(logits))
        targets.append(batch_y)
        if return_loss:
            target = batch_y.to(device, non_blocking=True)
            total_loss += F.binary_cross_entropy_with_logits(logits, target, reduction="sum")
            total_targets += target.numel()
    if not probabilities:
        raise ValueError("Cannot evaluate an empty dataset")
    # Transfer once per evaluation, not once per batch; sklearn AUC stays on CPU.
    y_true = torch.cat(targets).cpu().numpy()
    y_pred = torch.cat(probabilities).cpu().numpy()
    if return_loss:
        # Dataset mean BCE, including the correctly weighted final partial batch.
        return y_true, y_pred, (total_loss / total_targets).item()
    return y_true, y_pred


def main(config):
    device = torch.device(config["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; install CUDA-enabled PyTorch or use device='cpu'")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"{device} does not exist; visible GPUs: {torch.cuda.device_count()}")
        torch.cuda.set_device(device)
    if config["epochs"] < 1 or config["repeats"] < 1:
        raise ValueError("epochs and repeats must be positive")

    # Evaluation correlations are shared across methods and random splits.
    test_data_all, test_labels_all = load_pair(
        os.path.join(config["path_data"], "ABIDEII_ASD_nilearn.npy"),
        os.path.join(config["path_data"], "ABIDEII_NC_nilearn.npy"))
    test_R = test_data_all.shape[-1]
    test_labels_all = prepare_labels(test_labels_all, len(test_data_all))
    test_corr_all = compute_correlations(
        test_data_all, device, config["corr_batch_size"], "ABIDEII correlations")
    del test_data_all
    results_by_method = {}

    for method in config["methods"]:
        print(f"\nMethod: {method}")
        train_data, labels = load_training_data(config, method)
        if train_data.shape[-1] != test_R:
            raise ValueError(f"{method}: training and evaluation region counts differ")
        labels = prepare_labels(labels, len(train_data))
        all_corr = compute_correlations(
            train_data, device, config["corr_batch_size"], f"{method} correlations")
        del train_data
        num_corr = all_corr.shape[1]
        regions_inds = get_regs(all_corr, num_corr // 4)
        num_inpp = len(regions_inds)
        n_lat = num_inpp // 2
        train_features = all_corr[:, regions_inds].contiguous()
        eval_features = test_corr_all[:, regions_inds].contiguous()
        del all_corr

        model_dir = os.path.join(config["path_save"], "models", config["atlas"], method)
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "model_info.pkl"), "wb") as file:
            pickle.dump(dict(num_corr=num_corr, regions_inds=regions_inds,
                             num_inpp=num_inpp, n_lat=n_lat), file)

        # Set cache_on_gpu=True if selected features fit alongside the model,
        # gradients, and optimizer state. Default CPU caching is memory safer.
        cache_device = device if config["cache_on_gpu"] else torch.device("cpu")
        train_features = train_features.to(cache_device)
        labels = labels.to(cache_device)
        train_batches = CachedBatches(
            train_features, labels, config["batch_size"], shuffle=True,
            drop_last=config["drop_last"])
        test_aucs, y_trues, y_preds = [], [], []

        for i in range(config["repeats"]):
            test_idx, val_idx = train_test_split(
                np.arange(len(eval_features)), test_size=0.5, shuffle=True,
                stratify=test_labels_all.numpy().ravel(), random_state=i + 30)
            test_batches = CachedBatches(
                eval_features[test_idx].to(cache_device),
                test_labels_all[test_idx].to(cache_device), config["eval_batch_size"])
            val_batches = CachedBatches(
                eval_features[val_idx].to(cache_device),
                test_labels_all[val_idx].to(cache_device), config["eval_batch_size"])
            model = MTAutoEncoder(num_inputs=num_inpp, num_latent=n_lat).to(device)
            criterion_ae = nn.MSELoss(reduction="mean")
            criterion_clf = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.SGD([
                {"params": model.fc_encoder.parameters(), "lr": 0.0001},
                {"params": model.classifier.parameters(), "lr": 0.0001},
            ], momentum=0.9)
            best_val_loss = float("inf")
            best_state = None
            with tqdm(range(config["epochs"]), desc=f"{method} run {i + 1}") as progress:
                for _ in progress:
                    loss = train_epoch(
                        model, train_batches, device, optimizer, criterion_ae,
                        criterion_clf, config["p_bernoulli"], config["mode"],
                        config["lam_factor"])
                    y_true, y_pred, val_loss = predict(model, val_batches, device, return_loss=True)
                    auc = roc_auc_score(y_true.ravel(), y_pred.ravel())
                    if np.isfinite(val_loss) and val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_state = copy.deepcopy(model.state_dict())
                    progress.set_postfix(loss=loss, val_loss=val_loss, val_auc=auc, best_val_loss=best_val_loss)
                
            if best_state is None:
                raise RuntimeError("No valid validation AUC was produced")
            model.load_state_dict(best_state)
            y_true, y_pred = predict(model, test_batches, device)
            test_auc = float(roc_auc_score(y_true.ravel(), y_pred.ravel()))
            test_aucs.append(test_auc)
            y_trues.append(y_true.tolist())
            y_preds.append(y_pred.tolist())
            print(f"AUC = {test_auc:.3f}")
            del model, optimizer, best_state, test_batches, val_batches

        print(f"mean AUC = {np.mean(test_aucs):.3f} | std AUC = {np.std(test_aucs):.3f}")
        results = dict(auc_scores=test_aucs, y_true=y_trues, y_pred=y_preds)
        results_by_method[method] = results
        if config["save_results"]:
            result_dir = os.path.join(config["path_save"], method)
            os.makedirs(result_dir, exist_ok=True)
            with open(os.path.join(result_dir, "results.pkl"), "wb") as file:
                pickle.dump(results, file)
        del train_batches, train_features, labels, eval_features
    return results_by_method


if __name__ == '__main__':
    config = {}
    config['path_data'] = r'/media/student1/ExtremePro/rDCM-GEN/data'
    config['path_save'] = r'/home/student1/Desktop/Charalampos_Lamprou/rDCM-GEN_results_HCP_2026/ASD_CLC/ASD-DiagNet3'
    config['path_augmented_rdcmgen'] = r'/media/student1/ExtremePro/rDCM-GEN/rDCM-GEN_results/augmented_rDCM-GEN_noFCloss'
    config['path_augmented_crvae'] = r'/media/student1/ExtremePro/CR-VAE_augmented'
    config['path_augmented_timevae'] = r'/media/student1/ExtremePro/TimeVAE_results/outputs/gen_data'
    config['path_augmented_timegan'] = r'/media/student1/ExtremePro/TimeGAN_results'
    config['path_augmented_t2idiff'] = r'/media/student1/ExtremePro/T2I-Diff'
    config['path_augmented_dsfm'] = r'/media/student1/ExtremePro/DSFM_results'
    config['methods'] = ['ABIDEI', 'rDCM-GEN', 'DSFM','T2I-Diff','TimeVAE','CR-VAE','TimeGAN']
    config['atlas'] = 'SCH'
    config['batch_size'] = 128
    config['corr_batch_size'] = 128
    config['eval_batch_size'] = 128
    config['cache_on_gpu'] = False
    config['epochs'] = 200
    config['repeats'] = 10
    config['drop_last'] = False
    config['mode'] = "both"
    config['p_bernoulli'] = None
    config['lam_factor'] = 1.0
    config['device'] = "cuda:0"
    config['save_results'] = True
    main(config)