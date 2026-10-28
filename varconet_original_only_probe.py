"""VarCoNet: complete SSL, freeze the encoder, fit an ORIGINAL-ABIDEI-only probe.

Run: python varconet_original_only_probe.py
Dependencies: torch, numpy, scikit-learn, tqdm, lightning-bolts (pl_bolts).
Edit the configuration at the bottom before running.

The attached script's VarCoNet architecture, SSL loaders, pair/crop sampling,
contrastive loss, and warmup/cosine scheduler are preserved. Complete all 50
SSL epochs and keep the final encoder. The default SSL method remains CRVAE,
loading augmented_CR-VAE_orig_ABIDEI.pt (originals plus generated variants).

Then freeze every encoder parameter and cache ORIGINAL ABIDEI embeddings once.
Train one linear classifier using only ABIDEI_ASD_nilearn.npy and
ABIDEI_NC_nilearn.npy from path_data. No generated signals are appended to the
classifier dataset. There is no candidate bank, attention selector, or subset
search. Each classifier update uses the entire original-ABIDEI feature matrix.
The source's range(1, epochs_cls) convention is retained: 300 means 299 updates.

Keep the same stratified ABIDEII partitions as the attachment, for comparison:
  unused ~25%: the former sample-selection partition, never encoded;
  checkpoint validation ~25%: choose the lowest-loss classifier update;
  test ~50%: evaluate the chosen classifier once, after checkpointing.
The encoder is never chosen using validation and remains fixed during probing.

Results go to a separate VarCoNet_OriginalOnlyProbe_FrozenEncoder directory:
splits.npz, split_info.json, ssl_history.json, frozen_encoder.pt,
probe_initialization.pt, best_model.pt, checkpoint_before_test.json,
and per-run/aggregate results.pkl files. No sample-selection files are produced.
Removing selector work changes RNG consumption, so bitwise agreement with the
attention script's repeated runs is not implied.
"""


import json
import math
from pathlib import Path
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, Linear
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from tqdm import tqdm


def load_pair(asd_path, control_path):
    """Load exactly the two named files, without appending generated samples."""
    asd = np.load(asd_path)
    control = np.load(control_path)
    signals = np.concatenate((asd, control), axis=0)
    labels = torch.cat((torch.ones(len(asd), 1), torch.zeros(len(control), 1)))
    return signals, labels


def _load_training_data(config, method):
    if method == "ABIDEI":
        root = config["path_data"]
        return load_pair(os.path.join(root, "ABIDEI_ASD_nilearn.npy"),
                         os.path.join(root, "ABIDEI_NC_nilearn.npy"))
    if method in ("CRVAE", "rDCM-GEN"):
        key, tag = (("path_augmented_crvae", "CR-VAE") if method == "CRVAE"
                    else ("path_augmented_rdcmgen", "rDCM-GEN"))
        root = config[key]
        signals = torch.load(os.path.join(root, f"augmented_{tag}_orig_ABIDEI.pt"),
                             map_location="cpu", weights_only=True)
        labels = torch.load(os.path.join(root, f"labels_augmented_{tag}_ABIDEI.pt"),
                            map_location="cpu", weights_only=True)
        if signals.ndim != 4:
            raise ValueError(f"{method}: expected [N, B, T, R], got {signals.shape}")
        N, B, T, R = signals.shape
        labels = torch.as_tensor(labels).reshape(-1)
        if labels.numel() != N:
            raise ValueError(f"{method}: expected one label per subject (N)")
        return (signals.reshape(N * B, T, R),
                labels.repeat_interleave(B).reshape(-1, 1), (N, B))
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


def load_training_data(config, method):
    loaded = _load_training_data(config, method)
    return loaded if len(loaded) == 3 else (*loaded, None)


def prepare_labels(labels, count):
    labels = torch.as_tensor(labels, dtype=torch.float32).reshape(-1, 1).cpu()
    if len(labels) != count or not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Expected one binary (0 or 1) label per sample")
    return labels.contiguous()


class MLP(nn.Module):
    def __init__(self, num_features, num_classes):
        super(MLP, self).__init__()
        # Raw logits: binary softmax/BCE is replaced with stable cross-entropy.
        self.fc = Linear(num_features, num_classes)

    def forward(self, x):
        z = self.fc(x)
        return z


class PositionalEncodingTrainable(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncodingTrainable, self).__init__()
        pe = torch.zeros(1, max_seq_length, d_model)
        self.pe = nn.Parameter(pe)#

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class CNN_Transformer(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, dim_feedforward, max_len):
        super(CNN_Transformer, self).__init__()
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers, enable_nested_tensor=False)
        self.conv1 = Conv1d(in_channels=1, out_channels=16, kernel_size=4, stride=2)
        self.IN = MaskedInstanceNorm1d(d_model)
        new_len = (max_len - 4) // 2 + 1
        self.pos_enc = PositionalEncodingTrainable(d_model,new_len)
        
        
    def forward(self, x, x_mask=None):   
        b, k, d = x.shape
        mask = (x[:,:,0] != 0)
        x = self.IN(x,mask)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous()
        x = x.view((b*d, 1, k))
        x = self.conv1(x)
        x = x.mean(dim=1, keepdim=True)
        x = x.view((b, d, -1))
        x = torch.transpose(x, 1, 2)
        last_vals = x[:, -1:, :]
        matches = (x == last_vals)
        counts = matches.sum(dim=1)
        mask = (counts >= 2).unsqueeze(1).expand_as(matches)
        x = x.masked_fill(matches & mask, 0)
        x_mask = (x[:, :, 0] == 0).bool()    
        x = self.pos_enc(x)
        x = x.masked_fill(x_mask.unsqueeze(-1), 0)
        x = self.transformer_encoder(x, src_key_padding_mask=x_mask)  
        x = x.masked_fill(x_mask.unsqueeze(-1), 0)
        x = torch.transpose(x, 1, 2)
        return x


class MaskedInstanceNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

    def forward(self, x, mask):
        """
        x: [B, L, C]  (batch, length, channels)
        mask: [B, L]  (1 for valid positions, 0 for padding)
        """
        B, L, C = x.shape
        mask = mask.unsqueeze(-1).to(x.dtype)  # [B, L, 1]
        x_masked = x * mask
        valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1, 1]
        mean = x_masked.sum(dim=1, keepdim=True) / valid_counts
        var = ((x_masked - mean) * mask).pow(2).sum(dim=1, keepdim=True) / valid_counts
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        x_norm = x_norm * mask
        if self.affine:
            x_norm = x_norm * self.gamma + self.beta

        return x_norm


@torch.no_grad()
def prepare_signals(signals, device, batch_size, standardize=False, min_crop=None):
    """Convert once and cache first all-zero row; preserve original standardization."""
    source = torch.as_tensor(signals)
    if source.ndim != 3 or not len(source) or source.shape[1] < 4:
        raise ValueError("Expected nonempty [samples, T, R], with T >= 4")
    result = torch.empty(source.shape, dtype=torch.float32)
    lengths = torch.empty(len(source), dtype=torch.long)
    for start in range(0, len(source), batch_size):
        x = source[start:start + batch_size].to(device=device, dtype=torch.float32)
        if standardize:
            # Same full-time-axis, unbiased std as the original loader.
            x = (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True)
        if not torch.isfinite(x).all().item():
            raise ValueError("Non-finite signals after preprocessing; check NaNs/constant regions")
        zero_rows = (x == 0).all(dim=-1)
        first = zero_rows.long().argmax(dim=1)
        valid = torch.where(zero_rows.any(dim=1), first, x.shape[1])
        if min_crop is not None and (valid <= min_crop).any().item():
            raise ValueError("A sample is too short for the exclusive crop upper bound; lower min_length")
        result[start:start + len(x)].copy_(x.cpu())
        lengths[start:start + len(x)].copy_(valid.cpu())
    return result, lengths


class SubjectBatchSampler:
    """One generated sample per subject per batch; all N*B samples per epoch.

    Flattened sample index = subject_id * augmentations_per_subject + variant.
    Each of B rounds uses one unused variant from every subject. Subject order
    and per-subject variant order are reshuffled on every iteration. Balanced
    splits avoid a tiny final batch where possible; rounds are never combined.
    """
    def __init__(self, num_subjects, augmentations_per_subject, batch_size):
        if min(num_subjects, augmentations_per_subject, batch_size) < 1:
            raise ValueError("Subject counts, augmentation counts, and batch size must be positive")
        self.n = int(num_subjects)
        self.b = int(augmentations_per_subject)
        self.batch_size = min(int(batch_size), self.n)
        self.groups = (self.n + self.batch_size - 1) // self.batch_size

    def __len__(self):
        return self.b * self.groups

    def __iter__(self):
        variants = np.stack([np.random.permutation(self.b) for _ in range(self.n)])
        for round_idx in range(self.b):
            subjects = np.random.permutation(self.n)
            for group in np.array_split(subjects, self.groups):
                yield (group * self.b + variants[group, round_idx]).tolist()


class Batches:
    """Preconverted tensors with optional subject-disjoint training batches."""
    def __init__(self, tensors, batch_size, shuffle=False, subject_shape=None):
        if batch_size < 1 or any(len(t) != len(tensors[0]) for t in tensors):
            raise ValueError("Invalid batch size or mismatched tensors")
        self.tensors, self.batch_size, self.shuffle = tensors, batch_size, shuffle
        self.sampler = None
        if subject_shape is not None:
            N, B = subject_shape
            if N * B != len(tensors[0]):
                raise ValueError("Subject metadata does not match flattened samples")
            self.sampler = SubjectBatchSampler(N, B, batch_size)
        self.loader = None
        if tensors[0].device.type == 'cpu':
            options = dict(num_workers=0, pin_memory=torch.cuda.is_available())
            if self.sampler is not None:
                options['batch_sampler'] = self.sampler
            else:
                options.update(batch_size=batch_size, shuffle=shuffle)
            self.loader = DataLoader(TensorDataset(*tensors), **options)

    def __len__(self):
        if self.sampler is not None:
            return len(self.sampler)
        return (len(self.tensors[0]) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.loader is not None:
            yield from self.loader
            return
        if self.sampler is not None:
            for indices in self.sampler:
                idx = torch.as_tensor(indices, dtype=torch.long, device=self.tensors[0].device)
                yield tuple(t[idx] for t in self.tensors)
            return
        n = len(self.tensors[0])
        order = torch.randperm(n, device=self.tensors[0].device) if self.shuffle else None
        for start in range(0, n, self.batch_size):
            idx = slice(start, start + self.batch_size) if order is None else order[start:start + self.batch_size]
            yield tuple(t[idx] for t in self.tensors)


class GeneratedPairSampler:
    """One pair of distinct generated variants per subject per epoch.

    A batch contains indices for all first views, then their matched second
    views. Subjects occur once in each view and never act as their own negatives.
    B is inferred from [N, B, T, R] (15 in the supplied generated datasets).
    """
    def __init__(self, num_subjects, num_variants, batch_size):
        if num_subjects < 2 or num_variants < 2 or batch_size < 2:
            raise ValueError("Need >=2 subjects, >=2 variants per subject, and batch_size >=2")
        self.n, self.b = num_subjects, num_variants
        self.batch_size = min(batch_size, num_subjects)
        self.groups = (self.n + self.batch_size - 1) // self.batch_size
        if self.n // self.groups < 2:
            raise ValueError("Increase batch_size to avoid singleton contrastive batches")

    def __len__(self):
        return self.groups

    def __iter__(self):
        subjects = np.random.permutation(self.n)
        first = np.random.randint(self.b, size=self.n)
        # Draw uniformly from all variants except the first, without rejection.
        second = np.random.randint(self.b - 1, size=self.n) #np.random.randint(self.b-1, size=self.n)+1 
        second += second >= first
        for group in np.array_split(subjects, self.groups):
            a = group * self.b + first[group]
            b = group * self.b + second[group]
            yield np.concatenate((a, b)).tolist()


class GeneratedPairBatches:
    """Full-length generated positive pairs; no temporal crop augmentation."""
    def __init__(self, signals, subject_shape, batch_size):
        N, B = subject_shape
        if len(signals) != N * B:
            raise ValueError("Flattened signals do not match subject metadata")
        self.signals = signals
        self.sampler = GeneratedPairSampler(N, B, batch_size)
        self.loader = None
        if signals.device.type == 'cpu':
            self.loader = DataLoader(
                TensorDataset(signals), batch_sampler=self.sampler,
                num_workers=0, pin_memory=torch.cuda.is_available())

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        if self.loader is not None:
            for (paired,) in self.loader:
                k = len(paired) // 2
                yield paired[:k], paired[k:]
        else:
            for indices in self.sampler:
                idx = torch.as_tensor(indices, dtype=torch.long, device=self.signals.device)
                paired = self.signals[idx]
                k = len(paired) // 2
                yield paired[:k], paired[k:]


@torch.no_grad()
def augment(data, valid_lengths, length_limits):
    """Generate both crops for all samples on device, without .item() or loops.

    Retains exclusive crop-length/center upper bounds, 100 candidate center
    pairs, farthest-pair selection, and right-zero-padding from the original.
    Float-uniform draws replace per-sample randint, so RNG streams differ.
    """
    low, high = length_limits
    B, T, R = data.shape
    upper = valid_lengths.clamp(max=high)
    lengths = (torch.rand(B, 2, device=data.device) * (upper[:, None] - low)).long() + low
    center_low = lengths // 2 + 1
    center_high = torch.maximum(valid_lengths[:, None] - lengths // 2 - 2, center_low + 1)
    candidates = (torch.rand(B, 2, 100, device=data.device) *
                  (center_high - center_low).unsqueeze(-1)).long() + center_low.unsqueeze(-1)
    best = (candidates[:, 0] - candidates[:, 1]).abs().argmax(dim=-1)
    centers = candidates.gather(2, best[:, None, None].expand(-1, 2, 1)).squeeze(-1)
    offsets = torch.arange(high, device=data.device)[None, None, :]
    positions = centers.unsqueeze(-1) - (lengths // 2).unsqueeze(-1) + offsets
    # Clamp only padding locations so gather never reads past the physical array.
    batch_idx = torch.arange(B, device=data.device)[:, None, None]
    views = data[batch_idx, positions.clamp(0, T - 1)]
    views = views.masked_fill((offsets >= lengths.unsqueeze(-1)).unsqueeze(-1), 0)
    return views[:, 0].contiguous(), views[:, 1].contiguous()


class VarCoNet(nn.Module):
    def __init__(self, model_config, roi_num):
        super().__init__()
        if roi_num % model_config['n_heads']:
            raise ValueError("ROI count must be divisible by n_heads")
        self.extract = CNN_Transformer(
            roi_num, model_config['layers'], model_config['n_heads'],
            model_config['dim_feedforward'], model_config['max_length'])
        self.register_buffer('triu', torch.triu_indices(roi_num, roi_num, offset=1), persistent=False)

    def forward(self, x):
        x = F.normalize(self.extract(x), p=2, dim=-1)
        similarity = x @ x.transpose(1, 2)
        return similarity[:, self.triu[0], self.triu[1]]


def contrastive_loss(z1, z2, tau):
    # Equivalent to this script's diagonal-positive, cross-view-only InfoNCE.
    # No masks or repeated normalization/matrix multiplication are needed.
    logits = F.normalize(z1, dim=1) @ F.normalize(z2, dim=1).t() / tau
    targets = torch.arange(len(z1), device=z1.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)) * 0.5


@torch.no_grad()
def encode(model, batches, device):
    # no_grad rather than inference_mode: embeddings feed a trainable probe.
    model.eval()
    outputs, labels = [], []
    for batch in batches:
        x, y = batch[:2]
        outputs.append(model(x.to(device, non_blocking=True)))
        labels.append(y.to(device, non_blocking=True).reshape(-1))
    return torch.cat(outputs), torch.cat(labels)


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def validate_evaluation_splits(checkpoint, test, labels):
    """Check disjoint, complete ABIDEII row partitions containing both classes."""
    labels = np.asarray(labels)
    arrays = [np.asarray(a) for a in (checkpoint, test)]
    if labels.ndim != 1 or not np.isin(labels, [0, 1]).all():
        raise ValueError('Evaluation labels must be a 1D binary array')
    if any(a.ndim != 1 or a.dtype.kind not in 'iu' for a in arrays):
        raise ValueError('Evaluation partitions must contain 1D integer subject-row indices')
    if not np.array_equal(np.sort(np.concatenate(arrays)), np.arange(len(labels))):
        raise ValueError('Unused partition, checkpoint validation and test must partition all subjects without overlap')
    if any(len(a) < 2 or len(np.unique(labels[a])) != 2 for a in arrays):
        raise ValueError('Each of the three ABIDEII subsets must contain ASD and NC subjects')
    return tuple(a.astype(np.int64, copy=False) for a in arrays)


def split_evaluation(labels, seed):
    """Keep the former 50% test set; stratify the old validation half into two."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, [0, 1]).all():
        raise ValueError('Expected 1D binary labels for ABIDEII splitting')
    test, checkpoint = train_test_split(np.arange(len(labels)), test_size=.5, shuffle=True,
                                  stratify=labels, random_state=seed)

    return validate_evaluation_splits(checkpoint, test, labels)


def fit_probe(X_train, y_train, X_val, y_val, config, initial_state=None):
    """Fit on all original ABIDEI embeddings; choose the best validation step."""
    X_train, X_val = X_train.detach(), X_val.detach()
    classifier = MLP(X_train.shape[1], 2).to(X_train.device)
    if initial_state is not None:
        classifier.load_state_dict(initial_state)
    optimizer = Adam(classifier.parameters(), lr=config['lr_cls'])
    best_loss, best_state, best_step = float('inf'), None, None
    last_step = config['epochs_cls'] - 1
    for step in range(1, config['epochs_cls']):
        classifier.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(classifier(X_train), y_train.long())
        loss.backward()
        optimizer.step()
        evaluate = step == last_step or (step % config['probe_eval_interval'] == 0)
        if evaluate:
            classifier.eval()
            with torch.no_grad():
                val_loss = F.cross_entropy(classifier(X_val), y_val.long()).item()
            if math.isfinite(val_loss) and val_loss < best_loss:
                best_loss, best_state, best_step = val_loss, cpu_state(classifier), step
    if best_state is None:
        raise RuntimeError('No finite original-only probe validation loss')
    classifier.load_state_dict(best_state)
    classifier.eval()
    with torch.no_grad():
        probabilities = classifier(X_val).softmax(1)[:, 1].cpu().numpy()
    auc = float(roc_auc_score(y_val.cpu().numpy(), probabilities))
    return auc, best_loss, best_state, best_step


def train_epoch(model, batches, optimizer, config, device):
    if getattr(model, '_ssl_permanently_frozen', False):
        raise RuntimeError('VarCoNet was permanently frozen after SSL; further SSL training is disabled')
    model.train()
    total = torch.zeros((), device=device)
    for batch in batches:
        if isinstance(batches, GeneratedPairBatches):
            view1, view2 = (x.to(device, non_blocking=True) for x in batch)
        else:
            data, _, lengths = batch
            data = data.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            view1, view2 = augment(data, lengths, config['train_length_limits'])
        optimizer.zero_grad(set_to_none=True)
        # Keep separate forwards to avoid doubling peak activation memory.
        loss = contrastive_loss(model(view1), model(view2), config['tau'])
        loss.backward()
        optimizer.step()
        total += loss.detach()
    return (total / len(batches)).item()


def freeze_encoder_permanently(model):
    model.zero_grad(set_to_none=True)
    model.requires_grad_(False)
    model.eval()
    model._ssl_permanently_frozen = True
    return dict(parameters={name: value.detach().cpu().clone() for name, value in model.named_parameters()},
                buffers={name: value.detach().cpu().clone() for name, value in model.named_buffers()})


def verify_frozen_encoder(model, snapshot=None):
    if not getattr(model, '_ssl_permanently_frozen', False) or any(m.training for m in model.modules()):
        raise RuntimeError('The frozen encoder must remain in evaluation mode')
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise RuntimeError('The frozen encoder has trainable parameters or accumulated gradients')
    if snapshot is not None:
        for kind, current in [('parameters', dict(model.named_parameters())),
                              ('buffers', dict(model.named_buffers()))]:
            if current.keys() != snapshot[kind].keys():
                raise RuntimeError(f'Frozen encoder {kind} changed structure')
            for name, value in current.items():
                if not torch.equal(value.detach().cpu(), snapshot[kind][name]):
                    raise RuntimeError(f'Frozen encoder {kind} changed: {name}')


def pretrain_and_freeze(model, batches, config, device, description):
    """Stage 1 has no access to a probe, selector, or evaluation embeddings."""
    if getattr(model, '_ssl_permanently_frozen', False):
        raise RuntimeError('Cannot restart SSL on a permanently frozen encoder')
    optimizer = Adam(model.parameters(), lr=config['lr'])
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_start_lr=1e-5,
        warmup_epochs=config['warm_up_epochs'], max_epochs=config['ssl_epochs'])
    history = []
    with tqdm(range(1, config['ssl_epochs'] + 1), desc=description) as progress:
        for epoch in progress:
            used_lr = optimizer.param_groups[0]['lr']
            loss = train_epoch(model, batches, optimizer, config, device)
            if not math.isfinite(loss):
                raise RuntimeError('Non-finite SSL training loss')
            history.append(dict(ssl_epoch=epoch, ssl_loss=loss, learning_rate=used_lr))
            scheduler.step()
            progress.set_postfix(ssl_loss=loss)
    optimizer.zero_grad(set_to_none=True)
    del optimizer, scheduler
    snapshot = freeze_encoder_permanently(model)
    verify_frozen_encoder(model, snapshot)
    return snapshot, history


def main(config):
    device = torch.device(config['device'])
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use CUDA-enabled PyTorch or device='cpu'")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f'{device} is not a visible GPU')
        torch.cuda.set_device(device)
    if config['num_classes'] != 2:
        raise ValueError('This evaluation pipeline expects two classes')
    if config['epochs_cls'] < 2 or config['repeats'] < 1:
        raise ValueError('Require epochs_cls >= 2 and repeats >= 1')
    for key in ['ssl_epochs']:
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if not 0 <= config['warm_up_epochs'] < config['ssl_epochs']:
        raise ValueError('warm_up_epochs must be >=0 and less than ssl_epochs')
    if min(config['probe_eval_interval'], config['preprocess_batch_size']) < 1:
        raise ValueError('Intervals and preprocessing batch size must be positive')
    low, high = config['train_length_limits']
    if low < 4 or high <= low or high > config['model_config']['max_length'] or config['tau'] <= 0:
        raise ValueError('Invalid crop limits, max_length, or temperature')
    seed1, seed2 = config['seed1'], config['seed2']
    if seed1 is not None:
        np.random.seed(seed1)
    if seed2 is not None:
        torch.manual_seed(seed2)
    cache_device = device if config['cache_on_gpu'] else torch.device('cpu')
    rng_devices = [torch.cuda.current_device()] if device.type == 'cuda' else []
    eval_data, eval_labels = load_pair(
        os.path.join(config['path_data'], 'ABIDEII_ASD_nilearn.npy'),
        os.path.join(config['path_data'], 'ABIDEII_NC_nilearn.npy'))
    eval_labels = prepare_labels(eval_labels, len(eval_data)).reshape(-1).long()
    eval_data, _ = prepare_signals(eval_data, device, config['preprocess_batch_size'])
    if eval_data.shape[1] > config['model_config']['max_length']:
        raise ValueError('Evaluation sequence exceeds model max_length')
    roi_num = eval_data.shape[-1]
    # This is PURE original ABIDEI. load_pair no longer appends generated data.
    probe_data, probe_labels = load_pair(
        os.path.join(config['path_data'], 'ABIDEI_ASD_nilearn.npy'),
        os.path.join(config['path_data'], 'ABIDEI_NC_nilearn.npy'))
    probe_labels = prepare_labels(probe_labels, len(probe_data)).reshape(-1).long()
    probe_data, _ = prepare_signals(probe_data, device, config['preprocess_batch_size'])
    if probe_data.shape[-1] != roi_num or probe_data.shape[1] > config['model_config']['max_length']:
        raise ValueError('Original ABIDEI ROI count or sequence length is incompatible')
    probe_data, probe_labels = probe_data.to(cache_device), probe_labels.to(cache_device)
    probe_batches = Batches((probe_data, probe_labels), config['eval_batch_size'])
    print(f'Classifier: {len(probe_data)} original ABIDEI samples only')
    results_by_method = {}
    for method in config['methods']:
        print(f'\nMethod: {method}')
        train_data, labels, subject_shape = load_training_data(config, method)
        labels = prepare_labels(labels, len(train_data)).reshape(-1).long()
        if train_data.shape[-1] != roi_num or train_data.shape[1] > config['model_config']['max_length']:
            raise ValueError('Training ROI count or sequence length is incompatible')
        train_data, lengths = prepare_signals(
            train_data, device, config['preprocess_batch_size'],
            standardize=method in {'TimeVAE', 'TimeGAN', 'T2IDIFF', 'DSFM'},
            min_crop=low if subject_shape is None else None)
        train_data, labels, lengths = (t.to(cache_device) for t in (train_data, labels, lengths))
        if subject_shape is not None:
            n, b = subject_shape
            train_batches = GeneratedPairBatches(train_data, subject_shape, config['batch_size'])
            print(f'SSL: {b} loaded slices per subject, indices 0..{b-1}; {n} subjects; random distinct pairs')
        else:
            train_batches = Batches((train_data, labels, lengths), config['batch_size'], shuffle=True)
        #if method != 'ABIDEI':
        #    probe_data = torch.cat((probe_data, train_data), dim=0)
        #    probe_labels = torch.cat((probe_labels, labels))
        #    probe_batches = Batches((probe_data, probe_labels), config['eval_batch_size'])
        test_aucs, y_trues, y_preds, probe_steps = [], [], [], []
        for run in range(config['repeats']):
            checkpoint_idx, test_idx = split_evaluation(
                eval_labels.numpy(), None if seed1 is None else seed1 + run)
            checkpoint_batches = Batches((eval_data[checkpoint_idx].to(cache_device),
                eval_labels[checkpoint_idx].to(cache_device)), config['eval_batch_size'])
            print(f'ABIDEII run {run+1}: '
                  f'checkpoint validation={len(checkpoint_idx)} (~50%), test={len(test_idx)} (~50%)')
            output = Path(config['path_save']) / method / f'seeds_{seed1}_{seed2}' / f'run_{run + 1:02d}'
            if config['save_results']:
                output.mkdir(parents=True, exist_ok=True)
                np.savez(output / 'splits.npz',
                         validation_checkpoint=checkpoint_idx, test=test_idx,
                         evaluation_labels=eval_labels.numpy(), cohort=np.array('ABIDEII'))
                details = dict(cohort='ABIDEII', probe_training_data='original_ABIDEI',
                    probe_training_samples=len(probe_data), augmented_probe_samples=0,
                    criteria=dict(unused='not used',
                    checkpoint='minimum probe validation CE; encoder fixed at ssl_epochs', test='final evaluation only'),
                    counts={name: dict(total=len(ids), asd=int(eval_labels[ids].sum()),
                        nc=len(ids)-int(eval_labels[ids].sum())) for name, ids in
                        [('checkpoint', checkpoint_idx), ('test', test_idx)]})
                (output / 'split_info.json').write_text(json.dumps(details, indent=2))
            # Stage 1: finish SSL before creating or fitting any classifier.
            model = VarCoNet(config['model_config'], roi_num).to(device)
            frozen_snapshot, ssl_history = pretrain_and_freeze(
                model, train_batches, config, device, f'{method} run {run+1} | SSL only')
            frozen_encoder = cpu_state(model)
            if config['save_results']:
                (output / 'ssl_history.json').write_text(json.dumps(ssl_history, indent=2))
                torch.save(dict(encoder=frozen_encoder, ssl_epochs=config['ssl_epochs'],
                    model_config=config['model_config'], roi_num=roi_num, permanently_frozen=True),
                    output / 'frozen_encoder.pt')
            print(f"Encoder permanently frozen after {config['ssl_epochs']} SSL epochs. Caching original ABIDEI embeddings...")

            # Stage 2: one classifier fit on original ABIDEI only.
            X_train, y_train = encode(model, probe_batches, device)
            X_checkpoint, y_checkpoint = encode(model, checkpoint_batches, device)
            if len(X_train) != len(probe_data) or any(x.requires_grad for x in (X_train, X_checkpoint)):
                raise RuntimeError('Probe cache must include all originals and be detached')
            verify_frozen_encoder(model, frozen_snapshot)
            with torch.random.fork_rng(devices=rng_devices):
                initial_probe = MLP(X_train.shape[1], 2).to(device)
                initial_state = cpu_state(initial_probe)
                del initial_probe
            if config['save_results']:
                torch.save(initial_state, output / 'probe_initialization.pt')
            with torch.random.fork_rng(devices=rng_devices):
                best_auc, best_loss, best_probe, best_probe_step = fit_probe(
                    X_train, y_train, X_checkpoint, y_checkpoint, config,
                    initial_state=initial_state)
            verify_frozen_encoder(model, frozen_snapshot)
            if config['save_results']:
                torch.save(dict(encoder=frozen_encoder, classifier=best_probe,
                    ssl_epochs=config['ssl_epochs'], best_probe_step=best_probe_step,
                    probe_training_data='original_ABIDEI', probe_training_samples=len(probe_data),
                    augmented_probe_samples=0, checkpoint_validation_indices=checkpoint_idx.tolist(),
                    test_indices=test_idx.tolist(), 
                    config=config, roi_num=roi_num), output / 'best_model.pt')
                (output / 'checkpoint_before_test.json').write_text(json.dumps(dict(
                    ssl_epochs=config['ssl_epochs'], best_probe_step=best_probe_step,
                    checkpoint_loss=best_loss, checkpoint_auc=best_auc,
                    probe_training_data='original_ABIDEI', probe_training_samples=len(probe_data),
                    augmented_probe_samples=0), indent=2))
            del X_train, y_train, X_checkpoint, y_checkpoint, initial_state

            # Test inference occurs once, using the chosen classifier checkpoint.
            test_batches = Batches((eval_data[test_idx], eval_labels[test_idx]), config['eval_batch_size'])
            X_test, y_test = encode(model, test_batches, device)
            classifier = MLP(X_test.shape[1], 2).to(device)
            classifier.load_state_dict(best_probe)
            classifier.eval()
            with torch.no_grad():
                logits = classifier(X_test)
                y_pred = logits.softmax(1)[:, 1].cpu().numpy()
                test_loss = F.cross_entropy(logits, y_test).item()
            y_true = y_test.cpu().numpy()
            test_auc = float(roc_auc_score(y_true, y_pred))
            test_aucs.append(test_auc)
            y_trues.append(y_true.tolist())
            y_preds.append(y_pred.tolist())
            probe_steps.append(best_probe_step)
            print(f'ABIDEII test AUC = {test_auc:.3f}; frozen SSL epochs={config["ssl_epochs"]}; '
                  f'original classifier samples={len(probe_data)}; best probe step={best_probe_step}; '
                  f'checkpoint CE={best_loss:.4f}')
            if config['save_results']:
                with open(output / 'results.pkl', 'wb') as file:
                    pickle.dump(dict(test_auc=test_auc, test_loss=test_loss, validation_loss=best_loss,
                        validation_auc=best_auc, checkpoint_loss=best_loss, checkpoint_auc=best_auc,
                        best_probe_step=best_probe_step, y_true=y_true.tolist(), y_pred=y_pred.tolist(),
                        ssl_epochs=config['ssl_epochs'], probe_training_data='original_ABIDEI',
                        probe_training_samples=len(probe_data), augmented_probe_samples=0), file)
            verify_frozen_encoder(model, frozen_snapshot)
            del model, classifier, frozen_snapshot, frozen_encoder, best_probe
            del X_test, y_test, test_batches, checkpoint_batches
        results = dict(auc_scores=test_aucs, y_true=y_trues, y_pred=y_preds,
                       best_probe_steps=probe_steps, probe_training_data='original_ABIDEI',
                       probe_training_samples=len(probe_data), augmented_probe_samples=0)
        results_by_method[method] = results
        print(f'mean AUC = {np.mean(test_aucs):.3f} | std AUC = {np.std(test_aucs):.3f}')
        if config['save_results']:
            result_dir = Path(config['path_save']) / method
            result_dir.mkdir(parents=True, exist_ok=True)
            with open(result_dir / f'results_{seed1}_{seed2}.pkl', 'wb') as file:
                pickle.dump(results, file)
        del train_batches, train_data, labels, lengths
    return results_by_method


if __name__ == "__main__":
    config = {}
    config['path_data'] = r'/media/student1/ExtremePro/rDCM-GEN/data'
    config['path_save'] = r'/home/student1/Desktop/Charalampos_Lamprou/rDCM-GEN_results_HCP_2026/ASD_CLC/VarCoNet_OriginalOnlyProbe_FrozenEncoder'
    config['path_augmented_rdcmgen'] = r'/media/student1/ExtremePro/rDCM-GEN/rDCM-GEN_results/augmented_rDCM-GEN_noFCloss'
    config['path_augmented_crvae'] = r'/media/student1/ExtremePro/CRVAE_augmented'
    config['path_augmented_timevae'] = r'/media/student1/ExtremePro/TimeVAE_results/outputs/gen_data'
    config['path_augmented_timegan'] = r'/media/student1/ExtremePro/TimeGAN_results'
    config['path_augmented_t2idiff'] = r'/media/student1/ExtremePro/T2IDIFF'
    config['path_augmented_dsfm'] = r'/media/student1/ExtremePro/DSFM_results'
    config['methods'] = ['rDCM-GEN'] #,'DSFM','ABIDEI'] #['ABIDEI','TimeVAE', 'TimeGAN','CRVAE','T2IDIFF','DSFM','rDCM-GEN']
    config['atlas'] = 'SCH'
    config['model_config'] = {}
    config['model_config']['dim_feedforward'] = 1024
    config['model_config']['layers'] = 1
    config['model_config']['n_heads'] = 1
    config['model_config']['max_length'] = 200
    config['ssl_epochs'] = 50  # Stage 1: ONLY contrastive training.
    config['warm_up_epochs'] = 10
    config['lr'] = 1.5e-4
    config['tau'] = 0.054
    config['batch_size'] = 128
    config['max_length'] = 200
    config['min_length'] = 80
    config['train_length_limits'] = [config['min_length'],config['max_length']]
    config['epochs_cls'] = 300  # 299 full-batch updates, preserving the source convention.
    config['lr_cls'] = 1e-4
    config['num_classes'] = 2
    config['device'] = "cuda:0"
    config['save_results'] = False
    config['repeats'] = 10
    config['eval_batch_size'] = 128
    config['preprocess_batch_size'] = 128
    config['cache_on_gpu'] = False  # True needs space for raw signals + training activations.
    config['probe_eval_interval'] = 1  # Retains original validation-loss selection frequency.
    #for seed1 in range(42,43):
    #    for seed2 in range(10):
    config['seed1'] = 30  # Optional integer; changed RNG ordering prevents exact replay.
    config['seed2'] = 6  # Optional integer; changed RNG ordering prevents exact replay.
    results_by_method = main(config)
