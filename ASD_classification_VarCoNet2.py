"""Train and evaluate VarCoNet with subject-specific synthetic augmentation."""
import argparse

import math
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
    """Load exactly the two named files; adding probe samples happens explicitly."""
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
        #signals = signals[:,:5,:,:]
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



def probe_candidate_indices(config, total):
    start = 1 if config['probe_candidates_include_original'] else 0
    ids = config['probe_candidate_indices']
    ids = list(range(start, total)) if ids is None else list(ids)
    if (len(ids) < 2 or len(set(ids)) != len(ids)
            or any(not isinstance(i, int) or i < start or i >= total for i in ids)):
        raise ValueError('Need >=2 distinct valid probe candidate indices; exclude original 0 if present')
    baseline = config['probe_baseline_index']
    if baseline is not None and baseline not in ids:
        raise ValueError('probe_baseline_index is absent from candidates; change it or set None')
    return torch.tensor(ids, dtype=torch.long)



def load_probe_candidates(config, device, roi_num):
    root = config['probe_bank_root']
    signals = torch.load(os.path.join(root, config['probe_candidates_file']),
                         map_location='cpu', weights_only=True)
    labels = torch.load(os.path.join(root, config['probe_labels_file']),
                        map_location='cpu', weights_only=True)
    if signals.ndim != 4 or not len(signals):
        raise ValueError('Probe candidates must be nonempty [N, B, T, R]')
    n, b, t, r = signals.shape
    if r != roi_num or t > config['model_config']['max_length']:
        raise ValueError('Probe candidates have incompatible ROI count or sequence length')
    labels = prepare_labels(labels, n).reshape(-1).long()
    candidate_ids = probe_candidate_indices(config, b)
    flat, _ = prepare_signals(signals.reshape(n * b, t, r), device,
                              config['preprocess_batch_size'])
    signals = flat.reshape(n, b, t, r)
    if config['cache_on_gpu']:
        signals = signals.to(device)
    return signals, labels, candidate_ids



@torch.no_grad()
def encode_probe_candidates(model, signals, candidate_ids, config, device):
    model.eval()
    n, k = len(signals), len(candidate_ids)
    f = signals.shape[-1] * (signals.shape[-1] - 1) // 2
    storage = device if config['cache_probe_features_on_gpu'] else torch.device('cpu')
    result = torch.empty(n * k, f, dtype=torch.float32, device=storage)
    ids = candidate_ids.to(signals.device)
    size = config['eval_batch_size']
    for start in range(0, n * k, size):
        stop = min(start + size, n * k)
        flat_ids = torch.arange(start, stop, device=signals.device)
        batch = signals[flat_ids // k, ids[flat_ids % k]].to(device)
        z = model(batch)
        if not torch.isfinite(z).all():
            raise RuntimeError('Non-finite probe candidate embeddings')
        result[start:stop].copy_(z.to(storage))
    return result.reshape(n, k, f)



@torch.no_grad()
def make_attention_inputs(candidates, originals, projection, config):
    n, k, f = candidates.shape
    device = originals.device
    mean = originals.mean(0, keepdim=True)
    std = originals.std(0, keepdim=True, unbiased=False).clamp_min(1e-3)
    result = torch.empty(n * k, projection.shape[1], device=device)
    flat = candidates.reshape(n * k, f)
    for start in range(0, len(flat), config['probe_feature_batch_size']):
        stop = min(start + config['probe_feature_batch_size'], len(flat))
        z = F.normalize((flat[start:stop].to(device) - mean) / std, dim=1)
        result[start:stop] = F.normalize(z @ projection, dim=1)
    return result.reshape(n, k, -1)



class ProbeAttention(nn.Module):
    def __init__(self, subjects, candidates, input_dim, hidden_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim, bias=False)
        self.key = nn.Linear(input_dim, hidden_dim, bias=False)
        self.subject_logits = nn.Parameter(torch.zeros(subjects, candidates))

    def forward(self, features):
        q = F.normalize(self.query(features.mean(1)), dim=-1)
        k = F.normalize(self.key(features), dim=-1)
        return (q[:, None, :] * k).sum(-1) + self.subject_logits



def attention_probabilities(scores, temperature, exploration):
    p = (scores / temperature).softmax(-1)
    return (1 - exploration) * p + exploration / p.shape[-1]



def residual(logits, labels):
    return logits.softmax(-1) - F.one_hot(labels.long(), num_classes=2).to(logits.dtype)



@torch.no_grad()
def weighted_probe_gradient(weight, bias, originals, original_y, candidates, candidate_y,
                            probabilities, chunk_size):
    """Exact mean training gradient: M originals + total candidate weight N."""
    device = weight.device
    n, k, f = candidates.shape
    delta = residual(F.linear(originals, weight, bias), original_y)
    grad_w, grad_b = delta.t() @ originals, delta.sum(0)
    flat = candidates.reshape(n * k, f)
    labels = candidate_y.to(device).repeat_interleave(k)
    p = probabilities.detach().reshape(-1)
    for start in range(0, len(flat), chunk_size):
        stop = min(start + chunk_size, len(flat))
        x = flat[start:stop].to(device)
        delta = residual(F.linear(x, weight, bias), labels[start:stop])
        weighted = delta * p[start:stop, None]
        grad_w.add_(weighted.t() @ x)
        grad_b.add_(weighted.sum(0))
    denominator = len(originals) + n
    return grad_w / denominator, grad_b / denominator



@torch.no_grad()
def probe_lookahead(weight, bias, originals, original_y, candidates, candidate_y,
                    probabilities, val_x, val_y, step_size, chunk_size):
    """Exact d(validation CE after ONE SGD probe step)/d(candidate weights).

    For candidate j, g_j is its linear-probe CE gradient at the current probe.
    h_j = -step_size / (M+N) * dot(grad_val_at_virtual_probe, g_j).
    This is the same derivative as autograd through one virtual SGD step;
    no encoder graph or second-order autograd graph needs to be retained.
    """
    grad_w, grad_b = weighted_probe_gradient(
        weight, bias, originals, original_y, candidates, candidate_y, probabilities, chunk_size)
    virtual_w, virtual_b = weight - step_size * grad_w, bias - step_size * grad_b
    val_logits = F.linear(val_x, virtual_w, virtual_b)
    val_loss = F.cross_entropy(val_logits, val_y)
    val_delta = residual(val_logits, val_y) / len(val_x)
    val_grad_w, val_grad_b = val_delta.t() @ val_x, val_delta.sum(0)
    n, k, f = candidates.shape
    flat = candidates.reshape(n * k, f)
    labels = candidate_y.to(weight.device).repeat_interleave(k)
    hypergradient = weight.new_empty(n * k)
    factor = -step_size / (len(originals) + n)
    for start in range(0, len(flat), chunk_size):
        stop = min(start + chunk_size, len(flat))
        x = flat[start:stop].to(weight.device)
        delta = residual(F.linear(x, weight, bias), labels[start:stop])
        alignment = (delta * F.linear(x, val_grad_w, val_grad_b)).sum(-1)
        hypergradient[start:stop] = factor * alignment
    return hypergradient.reshape(n, k), val_loss.item(), virtual_w, virtual_b



@torch.no_grad()
def stable_meta_step_size(originals, candidates, requested, chunk_size):
    # A conservative curvature bound for two-logit CE, including the bias feature.
    # Candidate probabilities sum to one per subject, so per-subject maxima bound
    # the feature energy of every possible soft/hard candidate selection.
    n, k, f = candidates.shape
    flat = candidates.reshape(n * k, f)
    norms = originals.new_empty(n * k)
    for start in range(0, len(flat), chunk_size):
        stop = min(start + chunk_size, len(flat))
        norms[start:stop] = flat[start:stop].to(originals.device).square().sum(1)
    bound = (originals.square().sum() + norms.reshape(n, k).max(1).values.sum()) / (len(originals) + n) + 1
    return min(requested, 1.0 / bound.item())



def validate_probe_config(config):
    for key in ('probe_selection_steps', 'probe_hard_eval_interval', 'probe_feature_batch_size',
                'probe_attention_input_dim', 'probe_attention_hidden_dim'):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if config['probe_selector_warmup_steps'] < 0:
        raise ValueError('probe_selector_warmup_steps must be nonnegative')
    for key in ('probe_attention_lr', 'probe_meta_lr', 'probe_temperature_start', 'probe_temperature_end'):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if not 0 <= config['probe_exploration'] < 1 or config['probe_entropy_weight'] < 0:
        raise ValueError('Invalid probe exploration or entropy coefficient')



def validate_evaluation_splits(selection, checkpoint, test, labels):
    """Check disjoint, complete ABIDEII row partitions containing both classes."""
    labels = np.asarray(labels)
    arrays = [np.asarray(a) for a in (selection, checkpoint, test)]
    if labels.ndim != 1 or not np.isin(labels, [0, 1]).all():
        raise ValueError('Evaluation labels must be a 1D binary array')
    if any(a.ndim != 1 or a.dtype.kind not in 'iu' for a in arrays):
        raise ValueError('Evaluation partitions must contain 1D integer subject-row indices')
    if not np.array_equal(np.sort(np.concatenate(arrays)), np.arange(len(labels))):
        raise ValueError('Selection validation, checkpoint validation and test must partition all subjects without overlap')
    if any(len(a) < 2 or len(np.unique(labels[a])) != 2 for a in arrays):
        raise ValueError('Each of the three ABIDEII subsets must contain ASD and NC subjects')
    return tuple(a.astype(np.int64, copy=False) for a in arrays)



def split_evaluation(labels, seed):
    """Keep the former 50% test set; stratify the old validation half into two."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, [0, 1]).all():
        raise ValueError('Expected 1D binary labels for ABIDEII splitting')
    test, pool = train_test_split(np.arange(len(labels)), test_size=.5, shuffle=True,
                                  stratify=labels, random_state=seed)
    try:
        selection, checkpoint = train_test_split(pool, test_size=.5, shuffle=True,
            stratify=labels[pool], random_state=seed)
    except ValueError as exc:
        raise ValueError('Not enough subjects per class for stratified 25%/25%/50% evaluation') from exc
    return validate_evaluation_splits(selection, checkpoint, test, labels)



def fit_probe(X_train, y_train, X_val, y_val, config, initial_state=None, select_checkpoint=True):
    """Fit a hard subset: fixed final step for selection, or checkpoint using set B.

    select_checkpoint=False: evaluate ONLY the final Adam step on selection set A.
    select_checkpoint=True: choose the best Adam step using checkpoint set B.
    All gradient updates use X_train/y_train (ABIDEI) exclusively.
    """
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
        evaluate = step == last_step or (select_checkpoint and step % config['probe_eval_interval'] == 0)
        if evaluate:
            classifier.eval()
            with torch.no_grad():
                val_loss = F.cross_entropy(classifier(X_val), y_val.long()).item()
            if math.isfinite(val_loss) and val_loss < best_loss:
                best_loss, best_state, best_step = val_loss, cpu_state(classifier), step
    if best_state is None:
        raise RuntimeError('No finite hard-probe evaluation')
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



def select_probe_samples(originals, original_y, candidates, candidate_y, selection_x, selection_y,
                         selector, selector_optimizer, projection, file_indices, config,
                         initial_state=None, attention_inputs=None):
    """Set A only: learn attention and rank fixed-budget hard-subset probes."""
    # Explicit detaches are also a safeguard for direct callers of this function.
    originals, candidates, selection_x = originals.detach(), candidates.detach(), selection_x.detach()
    device = originals.device
    candidate_y = candidate_y.to(device).long()
    if attention_inputs is None:
        attention_inputs = make_attention_inputs(candidates, originals, projection, config)
    else:
        attention_inputs = attention_inputs.detach()
    n, k, f = candidates.shape
    if initial_state is None:
        initial_probe = MLP(f, 2).to(device)
        initial_state = cpu_state(initial_probe)
        del initial_probe
    working_probe = MLP(f, 2).to(device)
    working_probe.load_state_dict(initial_state)
    warm_optimizer = Adam(working_probe.parameters(), lr=config['lr_cls'])
    for _ in range(config['probe_selector_warmup_steps']):
        warm_optimizer.zero_grad(set_to_none=True)
        warm_loss = F.cross_entropy(working_probe(originals), original_y)
        warm_loss.backward()
        warm_optimizer.step()
    warm_optimizer.zero_grad(set_to_none=True)
    del warm_optimizer
    step_size = stable_meta_step_size(originals, candidates, config['probe_meta_lr'],
                                     config['probe_feature_batch_size'])
    best, tried, evaluations, history = None, {}, [], []
    step_count = config['probe_selection_steps']
    labels_hard = torch.cat((original_y, candidate_y))

    def evaluate_hard(local_ids, source, step, probabilities=None):
        nonlocal best
        local_cpu = local_ids.detach().cpu()
        key = local_cpu.numpy().tobytes()
        if key in tried:
            return
        rows = torch.arange(n, device=candidates.device)
        selected_z = candidates[rows, local_cpu.to(candidates.device)].to(device)
        train_x = torch.cat((originals, selected_z), dim=0)
        # Identical initialization for every subset, including fixed index 13.
        auc, loss, state, fit_step = fit_probe(train_x, labels_hard, selection_x, selection_y, config,
                                                 initial_state, select_checkpoint=False)
        info = dict(step=step, source=source, selection_loss=loss, selection_auc=auc, probe_steps=fit_step)
        evaluations.append(info)
        tried[key] = loss
        if best is None or loss < best['selection_loss']:
            best = dict(selection_loss=loss, selection_auc=auc, selection_probe_state=state,
                        local_indices=local_cpu.clone(),
                        selected_indices=file_indices[local_cpu].clone(),
                        source=source, selection_step=step,
                        attention_probabilities=None if probabilities is None else probabilities.detach().cpu().clone(),
                        selector_state=None if source == 'fixed_baseline' else cpu_state(selector))

    baseline = config['probe_baseline_index']
    if baseline is not None:
        local = int((file_indices == baseline).nonzero(as_tuple=True)[0][0])
        evaluate_hard(torch.full((n,), local, dtype=torch.long), 'fixed_baseline', 0)
    for step in range(1, step_count + 1):
        selector.train()
        fraction = (step - 1) / max(1, step_count - 1)
        temperature = config['probe_temperature_start'] * (
            config['probe_temperature_end'] / config['probe_temperature_start']) ** fraction
        selector_optimizer.zero_grad(set_to_none=True)
        scores = selector(attention_inputs)
        p = attention_probabilities(scores, temperature, config['probe_exploration'])
        h, virtual_ce, virtual_w, virtual_b = probe_lookahead(
            working_probe.fc.weight, working_probe.fc.bias,
            originals, original_y, candidates, candidate_y, p,
            selection_x, selection_y, step_size, config['probe_feature_batch_size'])
        if not math.isfinite(virtual_ce) or not torch.isfinite(h).all():
            raise RuntimeError('Non-finite probe lookahead; inspect data or reduce probe_meta_lr')
        # Its VALUE is not a CE; its gradient is exactly d(virtual validation CE)/d(theta).
        # h already contains the 1/(M+N) factor: do not divide this sum by N again.
        entropy = -(p * p.clamp_min(1e-12).log()).sum(1).mean()
        gradient_carrier = (p * h.detach()).sum() - config['probe_entropy_weight'] * entropy
        gradient_carrier.backward()
        grad = selector.subject_logits.grad
        if grad is None or not torch.isfinite(grad).all():
            raise RuntimeError('Missing or non-finite attention hypergradient')
        grad_rms = grad.square().mean().sqrt().item()
        selector_optimizer.step()
        # Continue the working probe with the exact soft-weight SGD step whose
        # validation loss was differentiated above. It is never the final probe.
        with torch.no_grad():
            working_probe.fc.weight.copy_(virtual_w)
            working_probe.fc.bias.copy_(virtual_b)
        history.append(dict(step=step, virtual_selection_ce=virtual_ce,
                            gradient_rms=grad_rms, temperature=temperature,
                            entropy=entropy.item(), effective_meta_lr=step_size))
        if step % config['probe_hard_eval_interval'] == 0 or step == step_count:
            selector.eval()
            with torch.no_grad():
                current_p = attention_probabilities(selector(attention_inputs), temperature,
                                                    config['probe_exploration'])
                local = current_p.argmax(1)
            evaluate_hard(local, 'attention', step, current_p)
    if best is None:
        raise RuntimeError('No hard probe subset was evaluated')
    best['hard_evaluations'] = evaluations
    best['attention_history'] = history
    best['effective_meta_lr'] = step_size
    best['probe_initial_state'] = initial_state
    return best



def select_and_checkpoint_probe(originals, original_y, candidates, candidate_y,
                                selection_x, selection_y, checkpoint_x, checkpoint_y,
                                selector, selector_optimizer, projection, file_indices, config,
                                initial_state=None, attention_inputs=None):
    """Seal the set-A choice before any set-B checkpoint evaluation."""
    selected = select_probe_samples(originals, original_y, candidates, candidate_y,
        selection_x, selection_y, selector, selector_optimizer, projection, file_indices, config,
        initial_state=initial_state, attention_inputs=attention_inputs)
    rows = torch.arange(len(candidates), device=candidates.device)
    local_ids = selected['local_indices'].to(candidates.device)
    selected_features = candidates[rows, local_ids].to(originals.device)
    train_x = torch.cat((originals.detach(), selected_features.detach()), dim=0)
    train_y = torch.cat((original_y, candidate_y.to(originals.device).long()), dim=0)
    auc, loss, state, step = fit_probe(train_x, train_y, checkpoint_x.detach(), checkpoint_y,
        config, initial_state=selected['probe_initial_state'], select_checkpoint=True)
    selected.update(checkpoint_loss=loss, checkpoint_auc=auc, probe_state=state, best_probe_step=step)
    return selected



def main(config):
    validate_probe_config(config)
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
    for key in ['ssl_epochs', 'selection_epochs']:
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
    candidate_signals, candidate_labels, candidate_ids = load_probe_candidates(config, device, roi_num)
    n_candidates_subjects, k = len(candidate_signals), len(candidate_ids)

    results_by_method = {}
    for method in config['methods']:
        print(f'\nMethod: {method}')
        if method == "CR-VAE":
            config['probe_bank_root'] = config['path_augmented_crvae']
        else:
            config['probe_bank_root'] = config['path_augmented_rdcmgen']
        config['probe_candidates_file'] = 'augmented_' + method + '_ABIDEI.pt'
        config['probe_labels_file'] = 'labels_augmented_' + method + '_ABIDEI.pt'
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
            #if b != 5:
            #    raise ValueError('Generated SSL data must contain exactly the fixed first five slices')
            train_batches = GeneratedPairBatches(train_data, subject_shape, config['batch_size'])
            print(f'SSL: {b} loaded slices per subject, indices 0..{b-1}; {n} subjects; random distinct pairs')
        else:
            train_batches = Batches((train_data, labels, lengths), config['batch_size'], shuffle=True)
        test_aucs, y_trues, y_preds, selections = [], [], [], []
        for run in range(config['repeats']):
            selection_idx, checkpoint_idx, test_idx = split_evaluation(
                eval_labels.numpy(), None if seed1 is None else seed1 + run)
            selection_batches = Batches((eval_data[selection_idx].to(cache_device),
                eval_labels[selection_idx].to(cache_device)), config['eval_batch_size'])
            checkpoint_batches = Batches((eval_data[checkpoint_idx].to(cache_device),
                eval_labels[checkpoint_idx].to(cache_device)), config['eval_batch_size'])
            print(f'ABIDEII run {run+1}: selection validation={len(selection_idx)} (~25%), '
                  f'checkpoint validation={len(checkpoint_idx)} (~25%), test={len(test_idx)} (~50%)')
            # Stage 1: SSL only. No MLP or attention model exists yet.
            model = VarCoNet(config['model_config'], roi_num).to(device)
            frozen_snapshot, ssl_history = pretrain_and_freeze(
                model, train_batches, config, device, f'{method} run {run+1} | SSL only')
            frozen_encoder = cpu_state(model)
            print(f"Encoder permanently frozen after {config['ssl_epochs']} SSL epochs. Caching embeddings once...")
            X_train, y_train = encode(model, probe_batches, device)
            X_selection, y_selection = encode(model, selection_batches, device)
            X_checkpoint, y_checkpoint = encode(model, checkpoint_batches, device)
            candidates = encode_probe_candidates(model, candidate_signals, candidate_ids, config, device)
            if any(x.requires_grad for x in [X_train, X_selection, X_checkpoint, candidates]):
                raise RuntimeError('Stage-2 feature caches must be detached from the encoder')
            verify_frozen_encoder(model, frozen_snapshot)
            # Stage 2: one persistent selector; every probe starts from the SAME state.
            with torch.random.fork_rng(devices=rng_devices):
                selector = ProbeAttention(n_candidates_subjects, k,
                    config['probe_attention_input_dim'], config['probe_attention_hidden_dim']).to(device)
                projection = torch.randn(roi_num * (roi_num - 1) // 2,
                    config['probe_attention_input_dim'], device=device) / math.sqrt(config['probe_attention_input_dim'])
                initial_probe = MLP(X_train.shape[1], 2).to(device)
                shared_initial_state = cpu_state(initial_probe)
                del initial_probe
            attention_inputs = make_attention_inputs(candidates, X_train, projection, config).detach()
            selector_optimizer = Adam(selector.parameters(), lr=config['probe_attention_lr'])
            best_loss, best_probe, best_selection = float('inf'), None, None
            best_selection_epoch, best_auc, best_probe_step = None, None, None
            previous_indices, selection_history = None, []
            with tqdm(range(1, config['selection_epochs'] + 1),
                      desc=f'{method} run {run+1} | selection (encoder frozen)') as progress:
                for selection_epoch in progress:
                    verify_frozen_encoder(model)
                    # Only cached features reach attention/probe fitting; no encoder calls.
                    with torch.random.fork_rng(devices=rng_devices):
                        selected = select_and_checkpoint_probe(
                            X_train, y_train, candidates, candidate_labels,
                            X_selection, y_selection, X_checkpoint, y_checkpoint,
                            selector, selector_optimizer, projection, candidate_ids, config,
                            initial_state=shared_initial_state, attention_inputs=attention_inputs)
                    checkpoint_loss, checkpoint_auc = selected['checkpoint_loss'], selected['checkpoint_auc']
                    changed = None if previous_indices is None else int(
                        (selected['selected_indices'] != previous_indices).sum().item())
                    previous_indices = selected['selected_indices'].clone()
                    improved = checkpoint_loss < best_loss
                    if improved:
                        best_loss, best_auc, best_selection_epoch = checkpoint_loss, checkpoint_auc, selection_epoch
                        best_probe_step = selected['best_probe_step']
                        best_probe, best_selection = selected['probe_state'], selected
                    selection_history.append(dict(
                        ssl_epochs=config['ssl_epochs'], selection_epoch=selection_epoch,
                        selection_loss=selected['selection_loss'], selection_auc=selected['selection_auc'],
                        checkpoint_loss=checkpoint_loss, checkpoint_auc=checkpoint_auc,
                        best_probe_step=selected['best_probe_step'], indices_changed=changed,
                        checkpoint_improved=improved,
                        selected_indices=selected['selected_indices'].tolist(),
                        source=selected['source'], selection_step=selected['selection_step'],
                        hard_evaluations=selected['hard_evaluations'], attention_history=selected['attention_history']))
                    progress.set_postfix(selector_ce=selected['selection_loss'],
                        checkpoint_ce=checkpoint_loss, checkpoint_auc=checkpoint_auc,
                        changed='first' if changed is None else changed, best_checkpoint_ce=best_loss)
                    del selected
            if best_probe is None:
                raise RuntimeError('No valid selection/probe checkpoint')
            verify_frozen_encoder(model, frozen_snapshot)
            print('Verified: encoder parameters, buffers and eval mode remained frozen throughout selection.')

            verify_frozen_encoder(model, frozen_snapshot)
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
            selections.append(best_selection['selected_indices'].tolist())
            print(f"ABIDEII test AUC = {test_auc:.3f}; frozen SSL epochs={config['ssl_epochs']}; "
                  f'selection round {best_selection_epoch}; '
                  f'probe step {best_probe_step}; checkpoint CE={best_loss:.4f}; subset={best_selection["source"]}')
            verify_frozen_encoder(model, frozen_snapshot)
            del model, selector, selector_optimizer, projection, classifier
            del X_train, y_train, X_selection, y_selection, X_checkpoint, y_checkpoint, candidates, attention_inputs
            del shared_initial_state, frozen_snapshot, previous_indices
            del frozen_encoder, best_probe, best_selection, X_test, y_test, test_batches, selection_batches, checkpoint_batches
        results = dict(auc_scores=test_aucs, y_true=y_trues, y_pred=y_preds, selected_probe_indices=selections)
        results_by_method[method] = results
        print(f'mean AUC = {np.mean(test_aucs):.3f} | std AUC = {np.std(test_aucs):.3f}')
        if config['save_results']:
            result_dir = os.path.join(config['path_save'], method)
            os.makedirs(result_dir, exist_ok=True)
            with open(os.path.join(result_dir, 'results.pkl'), 'wb') as file:
                pickle.dump(results, file)
        del train_batches, train_data, labels, lengths
    return results_by_method



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate VarCoNet with subject-specific synthetic augmentation.')
    parser.add_argument('--path_data', type=str, required=True)
    parser.add_argument('--path_save', type=str, required=True)
    parser.add_argument('--path_augmented_rdcmgen', type=str, required=True)
    parser.add_argument('--path_augmented_crvae', type=str, required=True)
    parser.add_argument('--methods', nargs="+", type=str, default=['CR-VAE', 'rDCM-GEN'])
    parser.add_argument('--model_dim_feedforward', type=int, default=1024)
    parser.add_argument('--model_layers', type=int, default=1)
    parser.add_argument('--model_n_heads', type=int, default=1)
    parser.add_argument('--model_max_length', type=int, default=200)
    parser.add_argument('--ssl_epochs', type=int, default=50)
    parser.add_argument('--selection_epochs', type=int, default=50)
    parser.add_argument('--warm_up_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1.5e-4)
    parser.add_argument('--tau', type=float, default=0.054)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_length', type=int, default=200)
    parser.add_argument('--min_length', type=int, default=80)
    parser.add_argument('--epochs_cls', type=int, default=300)
    parser.add_argument('--lr_cls', type=float, default=1e-4)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--save_results', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--preprocess_batch_size', type=int, default=128)
    parser.add_argument('--cache_on_gpu', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--probe_eval_interval', type=int, default=1)
    parser.add_argument('--probe_candidates_include_original', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--cache_probe_features_on_gpu', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--probe_feature_batch_size', type=int, default=1024)
    parser.add_argument('--probe_attention_input_dim', type=int, default=64)
    parser.add_argument('--probe_attention_hidden_dim', type=int, default=32)
    parser.add_argument('--probe_attention_lr', type=float, default=0.02)
    parser.add_argument('--probe_selection_steps', type=int, default=50)
    parser.add_argument('--probe_hard_eval_interval', type=int, default=1)
    parser.add_argument('--probe_selector_warmup_steps', type=int, default=50)
    parser.add_argument('--probe_meta_lr', type=float, default=0.1)
    parser.add_argument('--probe_temperature_start', type=float, default=1.0)
    parser.add_argument('--probe_temperature_end', type=float, default=0.3)
    parser.add_argument('--probe_exploration', type=float, default=0.02)
    parser.add_argument('--probe_entropy_weight', type=float, default=0.0)
    parser.add_argument('--seed1', type=int, default=30)
    parser.add_argument('--seed2', type=int, default=6)
    args = parser.parse_args()
    config = {}
    config['path_data'] = args.path_data
    config['path_save'] = args.path_save
    config['path_augmented_rdcmgen'] = args.path_augmented_rdcmgen
    config['path_augmented_crvae'] = args.path_augmented_crvae
    config['methods'] = args.methods 
    config['model_config'] = {}
    config['model_config']['dim_feedforward'] = args.model_dim_feedforward
    config['model_config']['layers'] = args.model_layers
    config['model_config']['n_heads'] = args.model_n_heads
    config['model_config']['max_length'] = args.model_max_length
    config['ssl_epochs'] = args.ssl_epochs  # Stage 1: ONLY contrastive training.
    config['selection_epochs'] = args.selection_epochs  # Stage 2: encoder permanently frozen.
    config['warm_up_epochs'] = args.warm_up_epochs
    config['lr'] = args.lr
    config['tau'] = args.tau
    config['batch_size'] = args.batch_size
    config['max_length'] = args.max_length
    config['min_length'] = args.min_length
    config['train_length_limits'] = [config['min_length'],config['max_length']]
    config['epochs_cls'] = args.epochs_cls
    config['lr_cls'] = args.lr_cls
    config['num_classes'] = args.num_classes
    config['device'] = args.device
    config['save_results'] = args.save_results
    config['repeats'] = args.repeats
    config['eval_batch_size'] = args.eval_batch_size
    config['preprocess_batch_size'] = args.preprocess_batch_size
    config['cache_on_gpu'] = args.cache_on_gpu  # True needs space for raw signals + training activations.
    config['probe_eval_interval'] = args.probe_eval_interval  # Retains original validation-loss selection frequency.
    config['probe_candidates_include_original'] = args.probe_candidates_include_original
    config['probe_candidate_indices'] = None  # None = every available augmentation.
    config['probe_baseline_index'] = None  # Compare to your existing slice; None disables comparison.
    config['cache_probe_features_on_gpu'] = args.cache_probe_features_on_gpu  # False saves GPU memory but transfers more data.
    config['probe_feature_batch_size'] = args.probe_feature_batch_size
    config['probe_attention_input_dim'] = args.probe_attention_input_dim
    config['probe_attention_hidden_dim'] = args.probe_attention_hidden_dim
    config['probe_attention_lr'] = args.probe_attention_lr
    config['probe_selection_steps'] = args.probe_selection_steps  # Attention updates per frozen selection round.
    config['probe_hard_eval_interval'] = args.probe_hard_eval_interval 
    config['probe_selector_warmup_steps'] = args.probe_selector_warmup_steps
    config['probe_meta_lr'] = args.probe_meta_lr  # Automatically capped by a feature-energy stability bound.
    config['probe_temperature_start'] = args.probe_temperature_start
    config['probe_temperature_end'] = args.probe_temperature_end
    config['probe_exploration'] = args.probe_exploration
    config['probe_entropy_weight'] = args.probe_entropy_weight
    #for seed1 in range(42,43):
    #    for seed2 in range(10):
    config['seed1'] = args.seed1  # Optional integer; changed RNG ordering prevents exact replay.
    config['seed2'] = args.seed2  # Optional integer; changed RNG ordering prevents exact replay.
    main(config)
