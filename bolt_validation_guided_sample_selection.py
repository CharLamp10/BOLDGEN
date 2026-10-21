"""Validation-guided selection of synthetic fMRI samples for BolT.

For every non-ABIDEI augmentation method, this script compares:
  1. full:        all available training samples;
  2. random:      a class-stratified random subset;
  3. gradient_alignment: a same-size subset ranked by validation-gradient alignment.

ABIDE II is split into three disjoint sets:
  * selection set: selects synthetic training samples (25% by default);
  * validation set: chooses the best checkpoint (25% by default);
  * test set:       final evaluation only (50% by default).

The selector uses per-sample gradients of the final binary Linear classifier
(head-gradient approximation), not gradients of every BolT parameter. This is
substantially cheaper, but depends on BolT exposing a final nn.Linear layer
whose logits produce the probabilities returned by Model.step(..., train=False).
The script checks for that condition and raises an informative error otherwise.
"""

import copy
import math
import os
import pickle
import sys
import warnings
from numbers import Integral, Real

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Update this path, or set the BOLT_BASELINES_PATH environment variable.
BASELINES_PATH = os.environ.get(
    "BOLT_BASELINES_PATH",
    r"/home/student1/Desktop/Charalampos_Lamprou/baselines",
)
if BASELINES_PATH not in sys.path:
    sys.path.insert(0, BASELINES_PATH)

from BolT.Models.BolT.model import Model  # noqa: E402
from BolT.Models.BolT.hyperparams import getHyper_bolT  # noqa: E402
from BolT.utils import Option  # noqa: E402


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------

def load_pair(asd_path, control_path):
    """Load ASD and control arrays, with ASD=1 and control=0 labels."""
    asd = np.load(asd_path)
    control = np.load(control_path)
    signals = np.concatenate((asd, control), axis=0)
    labels = torch.cat(
        (torch.ones((len(asd), 1)), torch.zeros((len(control), 1))), dim=0
    )
    return signals, labels


def load_training_data(config, method):
    """Load a method's ABIDE I training samples and binary labels."""
    if method == "ABIDEI":
        root = config["path_data"]
        return load_pair(
            os.path.join(root, "ABIDEI_ASD_nilearn.npy"),
            os.path.join(root, "ABIDEI_NC_nilearn.npy"),
        )

    if method in ("CRVAE", "CR-VAE", "rDCM-GEN"):
        if method in ("CRVAE", "CR-VAE"):
            key, tag = "path_augmented_crvae", "CR-VAE"
        else:
            key, tag = "path_augmented_rdcmgen", "rDCM-GEN"
        root = config[key]
        signals = torch.load(
            os.path.join(root, f"augmented_{tag}_orig_ABIDEI.pt"),
            map_location="cpu",
            weights_only=True,
        )
        labels = torch.load(
            os.path.join(root, f"labels_augmented_{tag}_ABIDEI.pt"),
            map_location="cpu",
            weights_only=True,
        )
        if signals.ndim != 4:
            raise ValueError(
                f"{method}: expected [N, B, T, R], got {tuple(signals.shape)}"
            )
        n_subjects, n_generated, timepoints, rois = signals.shape
        labels = torch.as_tensor(labels).reshape(-1)
        if labels.numel() != n_subjects:
            raise ValueError(
                f"{method}: expected one source-subject label per first dimension "
                f"({n_subjects}), got {labels.numel()} labels"
            )
        return (
            signals.reshape(n_subjects * n_generated, timepoints, rois),
            labels.repeat_interleave(n_generated).reshape(-1, 1),
        )

    if method == "TimeVAE":
        root = config["path_augmented_timevae"]
        return load_pair(
            os.path.join(root, "ABIDEI_ASD", "timeVAE_ABIDEI_ASD_prior_samples.npy"),
            os.path.join(root, "ABIDEI_NC", "timeVAE_ABIDEI_NC_prior_samples.npy"),
        )

    if method == "TimeGAN":
        root = config["path_augmented_timegan"]
        return load_pair(
            os.path.join(root, "augmented_TimeGAN_ABIDEI_ASD.npy"),
            os.path.join(root, "augmented_TimeGAN_ABIDEI_NC.npy"),
        )

    if method == "T2IDIFF":
        root = os.path.join(config["path_augmented_t2idiff"], "ABIDEI")
        return (
            np.load(os.path.join(root, "gen_sig_ABIDEI_fold1.npy")),
            np.load(os.path.join(root, "gen_label_sig_ABIDEI_fold1.npy")),
        )

    if method == "DSFM":
        root = os.path.join(config["path_augmented_dsfm"], "ABIDEI", "samples")
        return load_pair(
            os.path.join(root, "asd_bold.npy"),
            os.path.join(root, "control_bold.npy"),
        )

    raise ValueError(f"Unknown method: {method}")


def prepare_labels(labels, count):
    labels = torch.as_tensor(labels, dtype=torch.float32).reshape(-1).cpu()
    if len(labels) != count or not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Expected one binary (0 or 1) label per sample")
    return labels.contiguous().long()


@torch.no_grad()
def prepare_signals(signals, device, batch_size, standardize=False):
    """Convert once; retain method-specific standardization from the source."""
    source = torch.as_tensor(signals)
    if (
        source.ndim != 3
        or not len(source)
        or source.shape[1] < 4
        or source.shape[2] < 2
    ):
        raise ValueError(
            "Expected nonempty [samples, T, R] with T >= 4 and R >= 2; "
            f"got {tuple(source.shape)}"
        )

    result = torch.empty(source.shape, dtype=torch.float32, device="cpu")
    for start in range(0, len(source), batch_size):
        x = source[start : start + batch_size].to(device=device, dtype=torch.float32)
        if standardize:
            std = x.std(dim=1, keepdim=True)
            if (std == 0).any().item():
                raise ValueError(
                    "Constant time series encountered during method-specific "
                    "standardization; check the generated signals."
                )
            x = (x - x.mean(dim=1, keepdim=True)) / std
        if not torch.isfinite(x).all().item():
            raise ValueError(
                "Non-finite signals after preprocessing; check NaNs/constant regions."
            )
        result[start : start + len(x)].copy_(x.cpu())
    return result


def normalize_for_bolt(x):
    """Use the same per-ROI temporal normalization as the original pipeline."""
    x = torch.permute(x, (0, 2, 1))  # [B, T, R] -> [B, R, T]
    std = x.std(dim=-1, keepdim=True)
    if (std == 0).any().item():
        raise ValueError(
            "A constant ROI time series produced zero standard deviation during "
            "BolT normalization. Check the source signals."
        )
    x = (x - x.mean(dim=-1, keepdim=True)) / std
    if not torch.isfinite(x).all().item():
        raise ValueError("Non-finite values encountered during BolT normalization")
    return x


# ---------------------------------------------------------------------------
# Batches and BolT training/evaluation
# ---------------------------------------------------------------------------

class Batches:
    """Mini-batches from pinned CPU tensors or an optional GPU cache."""

    def __init__(self, data, labels, batch_size, shuffle=False):
        if batch_size < 1 or len(data) != len(labels) or not len(data):
            raise ValueError("Invalid batch size or mismatched/empty dataset")
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.loader = None
        if data.device.type == "cpu":
            self.loader = DataLoader(
                TensorDataset(data, labels),
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
            )

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.loader is not None:
            yield from self.loader
            return
        order = (
            torch.randperm(len(self.data), device=self.data.device)
            if self.shuffle
            else None
        )
        for start in range(0, len(self.data), self.batch_size):
            idx = (
                slice(start, start + self.batch_size)
                if order is None
                else order[start : start + self.batch_size]
            )
            yield self.data[idx], self.labels[idx]


def train_step(model, x, y, device):
    """One mini-batch and optimizer update, performed inside model.step."""
    model.model.train()
    x = normalize_for_bolt(x.to(device=device, dtype=torch.float32, non_blocking=True))
    y = y.to(device=device, dtype=torch.long, non_blocking=True).reshape(-1)
    train_loss, _, _, _ = model.step(x, y, train=True)
    return train_loss.detach()


@torch.no_grad()
def evaluate(model, batches, device):
    model.model.eval()
    targets, probabilities = [], []
    total_loss = torch.zeros((), device=device)
    count = 0

    for x, y in batches:
        x = normalize_for_bolt(
            x.to(device=device, dtype=torch.float32, non_blocking=True)
        )
        y = y.to(device=device, dtype=torch.long, non_blocking=True).reshape(-1)
        test_loss, _, test_probs, _ = model.step(x, y, train=False)
        total_loss += test_loss * len(x)
        count += len(x)
        probabilities.append(test_probs[:, 1].detach())
        targets.append(y.detach())

    y_true = torch.cat(targets).cpu().numpy()
    y_pred = torch.cat(probabilities).cpu().numpy()
    if len(np.unique(y_true)) != 2:
        raise ValueError("AUC evaluation requires both classes in the evaluated set")
    return float(roc_auc_score(y_true, y_pred)), (total_loss / count).item(), y_true, y_pred


def configure_step_scheduler(model, max_steps):
    """Set BolT's OneCycleLR horizon to an exact number of optimizer updates."""
    scheduler = getattr(model, "scheduler", None)
    if scheduler is None:
        return
    if not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
        raise TypeError(
            "Expected BolT OneCycleLR; adapt configure_step_scheduler for the "
            "installed custom scheduler."
        )

    state = scheduler.state_dict()
    groups = scheduler.optimizer.param_groups
    phases = state["_schedule_phases"]
    if len(phases) not in (2, 3):
        raise ValueError("Unsupported OneCycleLR phase layout")

    pct_start = (phases[0]["end_step"] + 1) / state["total_steps"]
    anneal_strategy = state.get("_anneal_func_type")
    if anneal_strategy is None:
        function_name = getattr(state.get("anneal_func"), "__name__", "")
        anneal_strategy = {
            "_annealing_cos": "cos",
            "_annealing_linear": "linear",
        }.get(function_name)
    if anneal_strategy not in ("cos", "linear"):
        raise ValueError("Cannot determine the OneCycleLR annealing strategy")

    div_factor = groups[0]["max_lr"] / groups[0]["initial_lr"]
    final_div_factor = groups[0]["initial_lr"] / groups[0]["min_lr"]
    momentum = {}
    if state["cycle_momentum"]:
        momentum = {
            "base_momentum": [g["base_momentum"] for g in groups],
            "max_momentum": [g["max_momentum"] for g in groups],
        }

    model.scheduler = torch.optim.lr_scheduler.OneCycleLR(
        scheduler.optimizer,
        max_lr=[g["max_lr"] for g in groups],
        total_steps=max_steps,
        pct_start=pct_start,
        anneal_strategy=anneal_strategy,
        cycle_momentum=state["cycle_momentum"],
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        three_phase=len(phases) == 3,
        **momentum,
    )


def seed_everything(seed):
    if seed is None:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))


def build_model(roi_num, n_train, config, scheduler_steps):
    n_batches = math.ceil(n_train / config["batch_size"])
    details = Option(
        {
            "device": torch.device(config["device"]),
            "nOfTrains": int(n_train),
            "nOfClasses": config["num_classes"],
            "batchSize": config["batch_size"],
            # Compatibility with the installed BolT constructor only.
            "nOfEpochs": max(1, math.ceil(scheduler_steps / n_batches)),
        }
    )
    model = Model(getHyper_bolT(rois=roi_num), details)
    configure_step_scheduler(model, scheduler_steps)
    return model


def train_fixed_steps(model, batches, device, max_steps, description):
    """Warm-up/pilot training without checkpoint selection."""
    iterator = iter(batches)
    total_loss = 0.0
    count = 0
    progress = tqdm(range(1, max_steps + 1), desc=description, unit="step")
    for _ in progress:
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(batches)
            x, y = next(iterator)
        loss = train_step(model, x, y, device)
        total_loss += float(loss.item()) * len(x)
        count += len(x)
        if count:
            progress.set_postfix(mean_loss=f"{total_loss / count:.4f}")
    return model


def train_steps(model, train_batches, val_batches, device,
                max_steps, eval_every_n_steps, description):
    """Train to a fixed update budget and retain the min-validation-loss state."""
    best_loss, best_state, best_step = float("inf"), None, None
    train_iterator = iter(train_batches)
    total_loss = torch.zeros((), device=device)
    count = 0

    with tqdm(range(1, max_steps + 1), desc=description, unit="step") as progress:
        for step in progress:
            try:
                x, y = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_batches)
                x, y = next(train_iterator)

            total_loss += train_step(model, x, y, device) * len(x)
            count += len(x)

            if step % eval_every_n_steps != 0 and step != max_steps:
                continue

            interval_train_loss = (total_loss / count).item()
            val_auc, val_loss, _, _ = evaluate(model, val_batches, device)
            if val_loss < best_loss:
                best_loss = val_loss
                best_step = step
                best_state = copy.deepcopy(model.model.state_dict())

            progress.set_postfix(
                train_loss=interval_train_loss,
                val_loss=val_loss,
                val_auc=val_auc,
                best_loss=best_loss,
            )
            total_loss.zero_()
            count = 0

    if best_state is None:
        raise RuntimeError("No valid validation checkpoint was selected")
    return best_state, best_step, best_loss


# ---------------------------------------------------------------------------
# Validation-gradient-alignment sample selection
# ---------------------------------------------------------------------------

@torch.no_grad()
def discover_final_classifier(model, probe_x, probe_y, device):
    """Find a binary Linear layer whose logits reproduce Model.step probabilities."""
    model.model.eval()
    candidates = [
        (name, module)
        for name, module in model.model.named_modules()
        if isinstance(module, nn.Linear) and module.out_features == 2
    ]
    if not candidates:
        raise RuntimeError(
            "No nn.Linear(out_features=2) layer was found in BolT. The selector "
            "needs the final binary linear classifier; inspect model.model and "
            "adapt discover_final_classifier for its output head."
        )

    captured = {}
    handles = []

    def make_hook(name):
        def hook(_module, inputs, output):
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            if not isinstance(output, torch.Tensor):
                return
            captured[name] = (
                inputs[0].detach().float().cpu(),
                output.detach().float().cpu(),
            )
        return hook

    for name, module in candidates:
        handles.append(module.register_forward_hook(make_hook(name)))

    try:
        x = normalize_for_bolt(
            probe_x.to(device=device, dtype=torch.float32, non_blocking=True)
        )
        y = probe_y.to(device=device, dtype=torch.long, non_blocking=True).reshape(-1)
        result = model.step(x, y, train=False)
        probs = result[2].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()

    matches = []
    for name, module in candidates:
        if name not in captured:
            continue
        features, layer_output = captured[name]
        if layer_output.ndim != 2 or tuple(layer_output.shape) != tuple(probs.shape):
            continue
        if features.ndim != 2 or features.shape[1] != module.in_features:
            continue

        output_is_logits = torch.allclose(
            torch.softmax(layer_output, dim=1), probs, atol=2e-4, rtol=2e-3
        )
        output_is_probabilities = torch.allclose(
            layer_output, probs, atol=2e-4, rtol=2e-3
        )
        if output_is_logits or output_is_probabilities:
            matches.append((name, module))

    if not matches:
        names = [name for name, _ in candidates]
        raise RuntimeError(
            "Could not match an nn.Linear(out_features=2) output to the class "
            "probabilities returned by BolT Model.step. Candidate layers: "
            f"{names}. The selector cannot safely infer classifier-head gradients "
            "for this BolT implementation."
        )

    # The last matching layer in module-registration order is the most likely
    # final output head when a model contains more than one binary Linear layer.
    chosen_name, chosen_module = matches[-1]
    print(f"Using classifier head for sample selection: {chosen_name}")
    return chosen_module


@torch.no_grad()
def collect_head_features_and_probs(model, batches, classifier, device):
    """Collect each sample's final-head input feature and predicted probabilities."""
    model.model.eval()
    feature_chunks, probability_chunks, label_chunks = [], [], []

    def hook(_module, inputs, _output):
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError("Classifier hook did not receive a tensor input")
        features = inputs[0].detach()
        if features.ndim != 2:
            raise RuntimeError(
                "Expected a [batch, features] input to the final classifier, got "
                f"{tuple(features.shape)}"
            )
        feature_chunks.append(features.float().cpu())

    handle = classifier.register_forward_hook(hook)
    try:
        for x, y in batches:
            x = normalize_for_bolt(
                x.to(device=device, dtype=torch.float32, non_blocking=True)
            )
            y_device = y.to(device=device, dtype=torch.long, non_blocking=True).reshape(-1)
            result = model.step(x, y_device, train=False)
            probs = result[2]
            if probs.ndim != 2 or probs.shape[1] != 2:
                raise RuntimeError(
                    "Expected BolT to return [batch, 2] class probabilities; got "
                    f"{tuple(probs.shape)}"
                )
            probability_chunks.append(probs.detach().float().cpu())
            label_chunks.append(y_device.detach().long().cpu())
    finally:
        handle.remove()

    features = torch.cat(feature_chunks, dim=0)
    probs = torch.cat(probability_chunks, dim=0)
    labels = torch.cat(label_chunks, dim=0)
    if not (len(features) == len(probs) == len(labels)):
        raise RuntimeError(
            "Classifier hook invocation count does not match number of samples. "
            "The selected layer may be invoked more than once per forward pass."
        )
    if features.shape[1] != classifier.in_features:
        raise RuntimeError("Captured feature width does not match classifier.in_features")
    return features, probs, labels


def per_sample_head_gradients(features, probs, labels, has_bias):
    """Construct per-sample CE gradients for the final Linear layer."""
    features = features.float()
    probs = probs.float().clamp(min=1e-7, max=1.0)
    labels = labels.reshape(-1).long()
    one_hot = F.one_hot(labels, num_classes=2).float()
    delta = probs - one_hot  # d cross-entropy / d logits
    weight_grad = (delta.unsqueeze(2) * features.unsqueeze(1)).reshape(len(features), -1)
    if has_bias:
        return torch.cat((weight_grad, delta), dim=1)
    return weight_grad


def class_stratified_random_indices(labels, fraction, seed):
    """Random subset with the same per-class sample counts as the selector."""
    labels = labels.reshape(-1).cpu().long()
    rng = np.random.default_rng(seed)
    chosen = []
    for class_id in (0, 1):
        class_indices = torch.where(labels == class_id)[0].numpy()
        if len(class_indices) == 0:
            raise ValueError(f"No training samples available for class {class_id}")
        n_keep = min(len(class_indices), max(1, int(round(fraction * len(class_indices)))))
        chosen.extend(rng.choice(class_indices, size=n_keep, replace=False).tolist())
    return torch.tensor(sorted(chosen), dtype=torch.long)


def select_by_validation_gradient(model, train_batches, selection_batches,
                                  train_labels, fraction, device, random_seed):
    """Select samples aligned with class-conditional validation head gradients.

    Scores are cosine similarities between a candidate sample's final-head CE
    gradient and the mean final-head CE gradient of selection-set samples of the
    same class. The highest-scoring samples are retained separately per class.
    """
    # Find the relevant final binary classifier using one selection-set batch.
    try:
        probe_x, probe_y = next(iter(selection_batches))
    except StopIteration as exc:
        raise ValueError("The sample-selection set is empty") from exc
    classifier = discover_final_classifier(model, probe_x, probe_y, device)

    train_features, train_probs, observed_train_labels = collect_head_features_and_probs(
        model, train_batches, classifier, device
    )
    sel_features, sel_probs, sel_labels = collect_head_features_and_probs(
        model, selection_batches, classifier, device
    )

    train_labels_cpu = train_labels.reshape(-1).cpu().long()
    if not torch.equal(observed_train_labels, train_labels_cpu):
        raise RuntimeError("Training-label order changed while computing selection scores")

    train_grads = per_sample_head_gradients(
        train_features, train_probs, observed_train_labels, classifier.bias is not None
    )
    selection_grads = per_sample_head_gradients(
        sel_features, sel_probs, sel_labels, classifier.bias is not None
    )

    chosen = []
    all_scores = torch.full((len(train_labels_cpu),), float("nan"), dtype=torch.float32)
    rng = np.random.default_rng(random_seed)

    for class_id in (0, 1):
        train_indices = torch.where(train_labels_cpu == class_id)[0]
        selection_mask = sel_labels == class_id
        if len(train_indices) == 0 or not selection_mask.any().item():
            raise ValueError(
                f"Class {class_id} must be present in both training and selection sets"
            )

        target_gradient = selection_grads[selection_mask].mean(dim=0)
        candidate_gradients = train_grads[train_indices]
        target_norm = torch.linalg.vector_norm(target_gradient)
        candidate_norms = torch.linalg.vector_norm(candidate_gradients, dim=1)

        n_keep = min(
            len(train_indices),
            max(1, int(round(fraction * len(train_indices)))),
        )

        if target_norm.item() < 1e-12:
            warnings.warn(
                f"Class {class_id}: selection gradient is approximately zero; "
                "using a random subset for this class.",
                RuntimeWarning,
            )
            local_positions = rng.choice(len(train_indices), size=n_keep, replace=False)
            class_scores = torch.zeros(len(train_indices), dtype=torch.float32)
            picked = train_indices[torch.as_tensor(local_positions, dtype=torch.long)]
        else:
            denom = candidate_norms * target_norm
            scores = (candidate_gradients @ target_gradient) / denom.clamp_min(1e-12)
            scores = torch.nan_to_num(scores, nan=-1.0, posinf=1.0, neginf=-1.0)
            class_scores = scores

            positive_positions = torch.where(scores > 0)[0]
            if len(positive_positions) >= n_keep:
                ranked_positions = positive_positions[
                    torch.argsort(scores[positive_positions], descending=True)
                ]
            else:
                warnings.warn(
                    f"Class {class_id}: only {len(positive_positions)} of the requested "
                    f"{n_keep} samples have positive gradient alignment. The best-scoring "
                    "remaining samples will be used to preserve class representation.",
                    RuntimeWarning,
                )
                ranked_positions = torch.argsort(scores, descending=True)
            picked = train_indices[ranked_positions[:n_keep]]

        all_scores[train_indices] = class_scores
        chosen.append(picked)

    selected_indices = torch.cat(chosen).sort().values.cpu()
    if len(selected_indices.unique()) != len(selected_indices):
        raise RuntimeError("The selector returned duplicate training indices")
    return selected_indices, all_scores


# ---------------------------------------------------------------------------
# Splits, validation, and experiment orchestration
# ---------------------------------------------------------------------------

def validate_config(config):
    positive_integer_keys = (
        "max_steps",
        "repeats",
        "batch_size",
        "eval_batch_size",
        "preprocess_batch_size",
        "eval_every_n_steps",
        "selection_warmup_steps",
    )
    for key in positive_integer_keys:
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
        config[key] = int(value)

    if config["num_classes"] != 2:
        raise ValueError("This evaluation pipeline expects two classes")

    for key in (
        "target_selection_fraction",
        "validation_fraction",
        "test_fraction",
        "sample_selection_fraction",
    ):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, Real) or not 0 < value < 1:
            raise ValueError(f"{key} must be between 0 and 1")

    split_sum = (
        config["target_selection_fraction"]
        + config["validation_fraction"]
        + config["test_fraction"]
    )
    if not math.isclose(split_sum, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(
            "target_selection_fraction + validation_fraction + test_fraction "
            "must sum to 1.0"
        )


def make_target_splits(eval_labels, config, run):
    """Create stratified, mutually exclusive selection/validation/test indices."""
    labels_np = eval_labels.cpu().numpy()
    all_indices = np.arange(len(labels_np))
    test_fraction = config["test_fraction"]
    development_idx, test_idx = train_test_split(
        all_indices,
        test_size=test_fraction,
        shuffle=True,
        stratify=labels_np,
        random_state=None if config["seed1"] is None else int(config["seed1"]) + run,
    )

    selection_share_of_development = config["target_selection_fraction"] / (
        config["target_selection_fraction"] + config["validation_fraction"]
    )
    split_seed = None if config["seed1"] is None else int(config["seed1"]) + run + 10_000
    selection_idx, val_idx = train_test_split(
        development_idx,
        test_size=1.0 - selection_share_of_development,
        shuffle=True,
        stratify=labels_np[development_idx],
        random_state=split_seed,
    )

    for name, indices in (
        ("sample-selection", selection_idx),
        ("checkpoint-validation", val_idx),
        ("test", test_idx),
    ):
        if len(np.unique(labels_np[indices])) != 2:
            raise ValueError(f"The {name} split must contain both classes for AUC")

    return selection_idx, val_idx, test_idx


def make_condition_indices(method, train_labels, selected_indices, config, random_seed):
    all_indices = torch.arange(len(train_labels), dtype=torch.long)
    if method == "ABIDEI":
        return {"full": all_indices}

    fraction = config["sample_selection_fraction"]
    label = f"{int(round(100 * fraction))}pct"
    random_indices = class_stratified_random_indices(train_labels, fraction, random_seed)
    return {
        "full": all_indices,
        f"random_{label}": random_indices,
        f"gradient_alignment_{label}": selected_indices,
    }


def main(config):
    validate_config(config)
    device = torch.device(config["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use CUDA-enabled PyTorch or device='cpu'")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"{device} is not a visible GPU")
        torch.cuda.set_device(device)

    # Set initial seeds. Run-specific seeds are reset before every model creation
    # to make the initialization comparable between full/random/selected variants.
    seed_everything(config["seed2"])
    cache_device = device if config["cache_on_gpu"] else torch.device("cpu")

    eval_data, eval_labels_raw = load_pair(
        os.path.join(config["path_data"], "ABIDEII_ASD_nilearn.npy"),
        os.path.join(config["path_data"], "ABIDEII_NC_nilearn.npy"),
    )
    eval_labels = prepare_labels(eval_labels_raw, len(eval_data))
    eval_data = prepare_signals(
        eval_data, device, config["preprocess_batch_size"], standardize=False
    )
    roi_num = eval_data.shape[-1]

    results_by_condition = {}
    for method_number, method in enumerate(config["methods"]):
        print(f"\n{'=' * 80}\nMethod: {method}\n{'=' * 80}")
        train_signals, labels_raw = load_training_data(config, method)
        labels = prepare_labels(labels_raw, len(train_signals))
        train_signals = prepare_signals(
            train_signals,
            device,
            config["preprocess_batch_size"],
            standardize=method in {"TimeVAE", "TimeGAN", "T2IDIFF", "DSFM"},
        )
        if train_signals.shape[-1] != roi_num:
            raise ValueError(
                f"ROI mismatch for {method}: training has {train_signals.shape[-1]}, "
                f"ABIDE II has {roi_num}. Check atlas/preprocessing consistency."
            )
        train_data_cache = train_signals.to(cache_device)
        train_labels_cache = labels.to(cache_device)
        all_train_indices = torch.arange(len(labels), dtype=torch.long)

        # Each variant stores its own result files, while sharing the same split per run.
        condition_names = ["full"] if method == "ABIDEI" else [
            "full",
            f"random_{int(round(100 * config['sample_selection_fraction']))}pct",
            f"gradient_alignment_{int(round(100 * config['sample_selection_fraction']))}pct",
        ]
        method_results = {
            name: {
                "auc_scores": [],
                "y_true": [],
                "y_pred": [],
                "test_indices": [],
                "validation_indices": [],
                "target_selection_indices": [],
                "training_sample_indices": [],
                "selected_sample_scores": [],
            }
            for name in condition_names
        }

        for run in range(config["repeats"]):
            print(f"\n{method}: repetition {run + 1}/{config['repeats']}")
            selection_idx, val_idx, test_idx = make_target_splits(eval_labels, config, run)

            selection_batches = Batches(
                eval_data[selection_idx].to(cache_device),
                eval_labels[selection_idx].to(cache_device),
                config["eval_batch_size"],
                shuffle=False,
            )
            val_batches = Batches(
                eval_data[val_idx].to(cache_device),
                eval_labels[val_idx].to(cache_device),
                config["eval_batch_size"],
                shuffle=False,
            )
            # The final test set is held aside and is not used by the selector or
            # checkpoint selection. The test loader is created after model fitting.

            selected_indices = all_train_indices
            selection_scores = None
            if method != "ABIDEI":
                pilot_seed = (
                    None
                    if config["seed2"] is None
                    else int(config["seed2"]) + 1_000_000 + run
                )
                seed_everything(pilot_seed)
                pilot_model = build_model(
                    roi_num,
                    len(train_data_cache),
                    config,
                    config["selection_warmup_steps"],
                )
                pilot_batches = Batches(
                    train_data_cache,
                    train_labels_cache,
                    config["batch_size"],
                    shuffle=True,
                )
                pilot_model = train_fixed_steps(
                    pilot_model,
                    pilot_batches,
                    device,
                    config["selection_warmup_steps"],
                    description=f"{method} pilot {run + 1}",
                )

                score_batches = Batches(
                    train_data_cache,
                    train_labels_cache,
                    config["eval_batch_size"],
                    shuffle=False,
                )
                selector_seed = (
                    None
                    if config["seed2"] is None
                    else int(config["seed2"]) + 2_000_000 + method_number * 10_000 + run
                )
                selected_indices, selection_scores = select_by_validation_gradient(
                    pilot_model,
                    score_batches,
                    selection_batches,
                    labels,
                    config["sample_selection_fraction"],
                    device,
                    selector_seed,
                )
                print(
                    f"Selected {len(selected_indices)}/{len(labels)} training samples "
                    f"({100 * len(selected_indices) / len(labels):.1f}%) by gradient alignment"
                )

                del pilot_model, pilot_batches, score_batches
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            random_seed = (
                None
                if config["seed2"] is None
                else int(config["seed2"]) + 3_000_000 + method_number * 10_000 + run
            )
            condition_indices = make_condition_indices(
                method, labels, selected_indices, config, random_seed
            )

            for condition_name, train_indices in condition_indices.items():
                print(
                    f"\nTraining condition: {method}/{condition_name} "
                    f"({len(train_indices)} samples)"
                )
                device_indices = train_indices.to(train_data_cache.device)
                condition_data = train_data_cache[device_indices]
                condition_labels = train_labels_cache[device_indices]
                train_batches = Batches(
                    condition_data,
                    condition_labels,
                    config["batch_size"],
                    shuffle=True,
                )

                # Identical initial random seed across conditions in a repetition
                # supports paired comparisons of the subset-selection strategies.
                model_seed = (
                    None if config["seed2"] is None else int(config["seed2"]) + 100 * run
                )
                seed_everything(model_seed)
                model = build_model(
                    roi_num,
                    len(condition_data),
                    config,
                    config["max_steps"],
                )
                best_state, best_step, best_loss = train_steps(
                    model,
                    train_batches,
                    val_batches,
                    device,
                    config["max_steps"],
                    config["eval_every_n_steps"],
                    description=f"{method}/{condition_name} run {run + 1}",
                )
                model.model.load_state_dict(best_state)

                test_batches = Batches(
                    eval_data[test_idx],
                    eval_labels[test_idx],
                    config["eval_batch_size"],
                    shuffle=False,
                )
                test_auc, _, y_true, y_pred = evaluate(model, test_batches, device)
                print(
                    f"Test AUC = {test_auc:.3f} | best step = {best_step} "
                    f"| best validation loss = {best_loss:.4f}"
                )

                out = method_results[condition_name]
                out["auc_scores"].append(test_auc)
                out["y_true"].append(y_true.tolist())
                out["y_pred"].append(y_pred.tolist())
                out["test_indices"].append(test_idx.tolist())
                out["validation_indices"].append(val_idx.tolist())
                out["target_selection_indices"].append(selection_idx.tolist())
                out["training_sample_indices"].append(train_indices.tolist())
                if condition_name.startswith("gradient_alignment_") and selection_scores is not None:
                    out["selected_sample_scores"].append(
                        selection_scores[train_indices].tolist()
                    )
                else:
                    out["selected_sample_scores"].append([])

                if config["save_models"]:
                    model_dir = os.path.join(
                        config["path_save"], method, condition_name, "models"
                    )
                    os.makedirs(model_dir, exist_ok=True)
                    torch.save(
                        {
                            "model_state_dict": {
                                key: value.detach().cpu()
                                for key, value in best_state.items()
                            },
                            "config": config,
                            "method": method,
                            "condition": condition_name,
                            "roi_num": roi_num,
                            "step": best_step,
                            "val_loss": best_loss,
                            "test_indices": test_idx.tolist(),
                            "validation_indices": val_idx.tolist(),
                            "target_selection_indices": selection_idx.tolist(),
                            "training_sample_indices": train_indices.tolist(),
                        },
                        os.path.join(model_dir, f"run_{run + 1}.pt"),
                    )

                del model, best_state, train_batches, test_batches
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        for condition_name, result in method_results.items():
            aucs = np.asarray(result["auc_scores"], dtype=float)
            print(
                f"\nSummary {method}/{condition_name}: mean AUC = {aucs.mean():.3f} "
                f"| std AUC = {aucs.std():.3f} | n = {len(aucs)}"
            )
            key = f"{method}/{condition_name}"
            results_by_condition[key] = result

            if config["save_results"]:
                result_dir = os.path.join(config["path_save"], method, condition_name)
                os.makedirs(result_dir, exist_ok=True)
                with open(os.path.join(result_dir, "results.pkl"), "wb") as file:
                    pickle.dump(result, file)

        del train_data_cache, train_labels_cache, train_signals, labels
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results_by_condition


if __name__ == "__main__":
    config = {}
    config["path_data"] = r"/media/student1/ExtremePro/rDCM-GEN/data"
    config["path_save"] = (
        r"/home/student1/Desktop/Charalampos_Lamprou/"
        r"rDCM-GEN_results_HCP_2026/ASD_CLC/BolT_sample_selection"
    )
    config["path_augmented_rdcmgen"] = (
        r"/media/student1/ExtremePro/rDCM-GEN/"
        r"rDCM-GEN_results/augmented_rDCM-GEN_noFCloss"
    )
    config["path_augmented_crvae"] = r"/media/student1/ExtremePro/CRVAE_augmented"
    config["path_augmented_timevae"] = (
        r"/media/student1/ExtremePro/TimeVAE_results/outputs/gen_data"
    )
    config["path_augmented_timegan"] = r"/media/student1/ExtremePro/TimeGAN_results"
    config["path_augmented_t2idiff"] = r"/media/student1/ExtremePro/T2IDIFF"
    config["path_augmented_dsfm"] = r"/media/student1/ExtremePro/DSFM_results"

    # Set one or more method names: 'rDCM-GEN', 'CRVAE', 'ABIDEI', 'TimeVAE',
    # 'TimeGAN', 'T2IDIFF', 'DSFM'. ABIDEI is evaluated as the no-selection baseline.
    config["methods"] = ["rDCM-GEN"]

    config["atlas"] = "SCH"
    config["device"] = "cuda:1"
    config["num_classes"] = 2

    # Model optimization settings.
    config["max_steps"] = 240
    config["batch_size"] = 64
    config["eval_batch_size"] = 64
    config["preprocess_batch_size"] = 64
    config["eval_every_n_steps"] = 4
    config["selection_warmup_steps"] = 40

    # ABIDE II split: 25% for selecting training samples, 25% for selecting the
    # best checkpoint, and 50% for final testing.
    config["target_selection_fraction"] = 0.25
    config["validation_fraction"] = 0.25
    config["test_fraction"] = 0.50

    # Fraction of each synthetic training class retained by the selector.
    # Try 0.25, 0.50, and 0.75 in separate runs for a subset-size comparison.
    config["sample_selection_fraction"] = 0.50

    config["repeats"] = 10
    config["cache_on_gpu"] = False
    config["seed1"] = 33
    config["seed2"] = 9
    config["save_results"] = True
    config["save_models"] = False

    main(config)
