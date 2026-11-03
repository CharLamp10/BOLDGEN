"""Train and evaluate the original VarCoNet ASD-classification baseline."""
import argparse

import copy
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


def load_pair(asd_path, control_path, config):
    asd = np.load(asd_path)
    control = np.load(control_path)
    signals = np.concatenate((asd, control), axis=0)
    labels = torch.cat((torch.ones(len(asd), 1), torch.zeros(len(control), 1)))
    if 'ABIDEII' not in asd_path:
        signals2 = torch.load(
            os.path.join(config['path_augmented_rdcmgen'], 'augmented_rDCM-GEN_ABIDEI.pt'),
            map_location='cpu', weights_only=True)
        labels2 = torch.load(
            os.path.join(config['path_augmented_rdcmgen'], 'labels_augmented_rDCM-GEN_ABIDEI.pt'),
            map_location='cpu', weights_only=True)
        signals = np.concatenate((signals, signals2[:, 13, :, :].numpy()), axis=0)
        labels = torch.cat((labels, labels2.reshape(-1, 1)), dim=0)
    return signals, labels


def _load_training_data(config, method):
    root = config['path_data']
    return load_pair(os.path.join(root, 'ABIDEI_ASD_nilearn.npy'),
                     os.path.join(root, 'ABIDEI_NC_nilearn.npy'), config)


def load_training_data(config, method):
    loaded = _load_training_data(config, method)
    return loaded if len(loaded) == 3 else (*loaded, None)


def prepare_labels(labels, count):
    labels = torch.as_tensor(labels, dtype=torch.float32).reshape(-1, 1).cpu()
    if len(labels) != count or not torch.all((labels == 0) | (labels == 1)):
        raise ValueError('Expected one binary (0 or 1) label per sample')
    return labels.contiguous()


class MLP(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.fc = nn.Sequential(Linear(num_features, num_classes), nn.Softmax(dim=1))

    def forward(self, x):
        return self.fc(x)


class PositionalEncodingTrainable(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_seq_length, d_model))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MaskedInstanceNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def forward(self, x, mask):
        mask = mask.unsqueeze(-1).to(x.dtype)
        x_masked = x * mask
        valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean = x_masked.sum(dim=1, keepdim=True) / valid_counts
        var = ((x_masked - mean) * mask).pow(2).sum(dim=1, keepdim=True) / valid_counts
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        x_norm = x_norm * mask
        if self.affine:
            x_norm = x_norm * self.gamma + self.beta
        return x_norm


class CNN_Transformer(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, dim_feedforward, max_len):
        super().__init__()
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.conv1 = Conv1d(in_channels=1, out_channels=16, kernel_size=4, stride=2)
        self.IN = MaskedInstanceNorm1d(d_model)
        new_len = (max_len - 4) // 2 + 1
        self.pos_enc = PositionalEncodingTrainable(d_model, new_len)

    def forward(self, x, x_mask=None):
        b, k, d = x.shape
        mask = x[:, :, 0] != 0
        x = self.IN(x, mask)
        x = x.transpose(1, 2).contiguous().view(b * d, 1, k)
        x = self.conv1(x).mean(dim=1, keepdim=True)
        x = x.view(b, d, -1).transpose(1, 2)
        last_vals = x[:, -1:, :]
        matches = x == last_vals
        counts = matches.sum(dim=1)
        mask = (counts >= 2).unsqueeze(1).expand_as(matches)
        x = x.masked_fill(matches & mask, 0)
        x_mask = (x[:, :, 0] == 0).bool()
        x = self.pos_enc(x)
        x = x.masked_fill(x_mask.unsqueeze(-1), 0)
        x = self.transformer_encoder(x, src_key_padding_mask=x_mask)
        x = x.masked_fill(x_mask.unsqueeze(-1), 0)
        return x.transpose(1, 2)


@torch.no_grad()
def prepare_signals(signals, device, batch_size, standardize=False, min_crop=None):
    source = torch.as_tensor(signals)
    if source.ndim != 3 or not len(source) or source.shape[1] < 4:
        raise ValueError('Expected nonempty [samples, T, R], with T >= 4')
    result = torch.empty(source.shape, dtype=torch.float32)
    lengths = torch.empty(len(source), dtype=torch.long)
    for start in range(0, len(source), batch_size):
        x = source[start:start + batch_size].to(device=device, dtype=torch.float32)
        if standardize:
            x = (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True)
        if not torch.isfinite(x).all().item():
            raise ValueError('Non-finite signals after preprocessing; check NaNs/constant regions')
        zero_rows = (x == 0).all(dim=-1)
        first = zero_rows.long().argmax(dim=1)
        valid = torch.where(zero_rows.any(dim=1), first, x.shape[1])
        if min_crop is not None and (valid <= min_crop).any().item():
            raise ValueError('A sample is too short for the exclusive crop upper bound; lower min_length')
        result[start:start + len(x)].copy_(x.cpu())
        lengths[start:start + len(x)].copy_(valid.cpu())
    return result, lengths


class SubjectBatchSampler:
    """One generated sample per subject per batch; all N*B samples per pass."""
    def __init__(self, num_subjects, augmentations_per_subject, batch_size):
        if min(num_subjects, augmentations_per_subject, batch_size) < 1:
            raise ValueError('Subject counts, augmentation counts, and batch size must be positive')
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
            raise ValueError('Invalid batch size or mismatched tensors')
        self.tensors, self.batch_size, self.shuffle = tensors, batch_size, shuffle
        self.sampler = None
        if subject_shape is not None:
            N, B = subject_shape
            if N * B != len(tensors[0]):
                raise ValueError('Subject metadata does not match flattened samples')
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
    """One pair of distinct generated variants per subject per SSL epoch."""
    def __init__(self, num_subjects, num_variants, batch_size):
        if num_subjects < 2 or num_variants < 2 or batch_size < 2:
            raise ValueError('Need >=2 subjects, >=2 variants per subject, and batch_size >=2')
        self.n, self.b = num_subjects, num_variants
        self.batch_size = min(batch_size, num_subjects)
        self.groups = (self.n + self.batch_size - 1) // self.batch_size

    def __len__(self):
        return self.groups

    def __iter__(self):
        subjects = np.random.permutation(self.n)
        first = np.random.randint(self.b, size=self.n)
        second = np.random.randint(self.b - 1, size=self.n)
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
            raise ValueError('Flattened signals do not match subject metadata')
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
    """Generate two right-zero-padded crops per sample on the current device."""
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
    batch_idx = torch.arange(B, device=data.device)[:, None, None]
    views = data[batch_idx, positions.clamp(0, T - 1)]
    views = views.masked_fill((offsets >= lengths.unsqueeze(-1)).unsqueeze(-1), 0)
    return views[:, 0].contiguous(), views[:, 1].contiguous()


class VarCoNet(nn.Module):
    def __init__(self, model_config, roi_num):
        super().__init__()
        if roi_num % model_config['n_heads']:
            raise ValueError('ROI count must be divisible by n_heads')
        self.extract = CNN_Transformer(
            roi_num, model_config['layers'], model_config['n_heads'],
            model_config['dim_feedforward'], model_config['max_length'])
        self.register_buffer('triu', torch.triu_indices(roi_num, roi_num, offset=1), persistent=False)

    def forward(self, x):
        x = F.normalize(self.extract(x), p=2, dim=-1)
        similarity = x @ x.transpose(1, 2)
        return similarity[:, self.triu[0], self.triu[1]]


def contrastive_loss(z1, z2, tau):
    logits = F.normalize(z1, dim=1) @ F.normalize(z2, dim=1).t() / tau
    targets = torch.arange(len(z1), device=z1.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)) * 0.5


def train_epoch(model, batches, optimizer, config, device):
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
        loss = contrastive_loss(model(view1), model(view2), config['tau'])
        loss.backward()
        optimizer.step()
        total += loss.detach()
    return (total / len(batches)).item()


@torch.no_grad()
def encode(model, batches, device):
    # no_grad tensors can subsequently be used as inputs to a trainable probe.
    model.eval()
    outputs, labels = [], []
    for batch in batches:
        x, y = batch[:2]
        outputs.append(model(x.to(device, non_blocking=True)))
        labels.append(y.to(device, non_blocking=True).reshape(-1))
    return torch.cat(outputs), torch.cat(labels)


def fit_probe(X_train, y_train, X_val, y_val, config, description='Probe'):
    """Fit once on fixed features and restore the minimum-validation-BCE probe."""
    X_train, X_val = X_train.detach(), X_val.detach()
    classifier = MLP(X_train.shape[1], config['num_classes']).to(X_train.device)
    optimizer = Adam(classifier.parameters(), lr=config['lr_cls'])
    criterion = nn.BCELoss()
    train_target = F.one_hot(y_train, num_classes=2).float()
    val_target = F.one_hot(y_val, num_classes=2).float()
    best_loss, best_state, best_epoch = float('inf'), None, None
    # Preserve the supplied script: epochs_cls=300 performs 299 full-batch updates.
    last_epoch = config['epochs_cls'] - 1
    with tqdm(range(1, config['epochs_cls']), desc=description) as progress:
        for epoch in progress:
            classifier.train()
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(classifier(X_train), train_target)
            loss.backward()
            optimizer.step()
            if epoch % config['probe_eval_interval'] != 0 and epoch != last_epoch:
                continue
            classifier.eval()
            with torch.no_grad():
                val_loss = criterion(classifier(X_val), val_target).item()
            if np.isfinite(val_loss) and val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(classifier.state_dict())
            progress.set_postfix(train_loss=loss.item(), val_loss=val_loss,
                                 best_loss=best_loss, best_epoch=best_epoch)
    if best_state is None:
        raise RuntimeError('No finite probe validation loss; check training stability')
    classifier.load_state_dict(best_state)
    classifier.eval()
    with torch.no_grad():
        val_probs = classifier(X_val)[:, 1].cpu().numpy()
    val_auc = float(roc_auc_score(y_val.cpu().numpy(), val_probs))
    return classifier, val_auc, best_loss, best_epoch


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
    if config['epochs_cls'] < 2 or config['epochs'] < 1 or config['repeats'] < 1:
        raise ValueError('Require epochs_cls >= 2, epochs >= 1, repeats >= 1')
    if min(config['probe_eval_interval'], config['preprocess_batch_size'],
           config['batch_size'], config['eval_batch_size']) < 1:
        raise ValueError('Intervals and batch sizes must be positive')
    low, high = config['train_length_limits']
    if low < 4 or high <= low or high > config['model_config']['max_length'] or config['tau'] <= 0:
        raise ValueError('Invalid crop limits, max_length, or temperature')
    seed1, seed2 = config['seed1'], config['seed2']
    if seed1 is not None:
        np.random.seed(seed1)
    if seed2 is not None:
        torch.manual_seed(seed2)
    cache_device = device if config['cache_on_gpu'] else torch.device('cpu')

    eval_data, eval_labels = load_pair(
        os.path.join(config['path_data'], 'ABIDEII_ASD_nilearn.npy'),
        os.path.join(config['path_data'], 'ABIDEII_NC_nilearn.npy'), config)
    eval_labels = prepare_labels(eval_labels, len(eval_data)).reshape(-1).long()
    eval_data, _ = prepare_signals(eval_data, device, config['preprocess_batch_size'])
    if eval_data.shape[1] > config['model_config']['max_length']:
        raise ValueError("Evaluation sequence exceeds model_config['max_length']")
    roi_num = eval_data.shape[-1]

    # Fixed probe dataset for every method. load_pair retains the supplied
    # addition of rDCM-GEN variant 13 to the original ABIDEI signals.
    probe_data, probe_labels = load_pair(
        os.path.join(config['path_data'], 'ABIDEI_ASD_nilearn.npy'),
        os.path.join(config['path_data'], 'ABIDEI_NC_nilearn.npy'), config)
    probe_labels = prepare_labels(probe_labels, len(probe_data)).reshape(-1).long()
    probe_data, _ = prepare_signals(probe_data, device, config['preprocess_batch_size'])
    if (probe_data.shape[-1] != roi_num or
            probe_data.shape[1] > config['model_config']['max_length']):
        raise ValueError('Probe ROI count or sequence length is incompatible with the model')
    probe_data, probe_labels = probe_data.to(cache_device), probe_labels.to(cache_device)
    probe_batches = Batches((probe_data, probe_labels), config['eval_batch_size'])
    results_by_method = {}
    
    method = config['method']
    print(f'\nMethod: {method}')
    train_data, labels, subject_shape = load_training_data(config, method)
    labels = prepare_labels(labels, len(train_data)).reshape(-1).long()
    if train_data.shape[-1] != roi_num or train_data.shape[1] > config['model_config']['max_length']:
        raise ValueError('Training ROI count or sequence length is incompatible with the model')
    train_data, lengths = prepare_signals(
        train_data, device, config['preprocess_batch_size'],
        standardize=method in {'TimeVAE', 'TimeGAN', 'T2IDIFF', 'DSFM'},
        min_crop=low if subject_shape is None else None)
    train_data, labels, lengths = (t.to(cache_device) for t in (train_data, labels, lengths))
    if subject_shape is not None:
        N, B = subject_shape
        train_batches = GeneratedPairBatches(train_data, subject_shape, config['batch_size'])
        print(f'{N} subjects, {B} variants each; one random distinct pair per subject per epoch, no cropping')
    else:
        train_batches = Batches((train_data, labels, lengths), config['batch_size'], shuffle=True)
    test_aucs, y_trues, y_preds = [], [], []
    probe_epochs, val_losses, val_aucs, split_indices = [], [], [], []
    for run in range(config['repeats']):
        run_seed = None if seed1 is None else seed1 + run
        test_idx, val_pool_idx = train_test_split(
            np.arange(len(eval_data)), test_size=0.5, shuffle=True,
            stratify=eval_labels.numpy(), random_state=run_seed)
        unused_idx, val_idx = train_test_split(
            val_pool_idx, test_size=0.5, shuffle=True,
            stratify=eval_labels[val_pool_idx].numpy(), random_state=run_seed)
        if any(len(torch.unique(eval_labels[idx])) != 2 for idx in (val_idx, test_idx)):
            raise ValueError('Validation and test splits must each contain both classes for AUC')

        # Stage 1: finish SSL. No probe fitting or validation in this loop.
        model = VarCoNet(config['model_config'], roi_num).to(device)
        optimizer = Adam(model.parameters(), lr=config['lr'])
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_start_lr=1e-5, warmup_epochs=config['warm_up_epochs'],
            max_epochs=config['epochs'])
        with tqdm(range(1, config['epochs'] + 1),
                  desc=f'{method} run {run + 1}: SSL') as progress:
            for epoch in progress:
                loss = train_epoch(model, train_batches, optimizer, config, device)
                scheduler.step()
                progress.set_postfix(ssl_loss=loss)

        # Stage 2: use the FINAL SSL encoder, frozen in evaluation mode.
        # Clear the last SSL gradients and release the encoder optimizer.
        model.zero_grad(set_to_none=True)
        model.requires_grad_(False)
        model.eval()
        del optimizer, scheduler
        val_batches = Batches(
            (eval_data[val_idx].to(cache_device), eval_labels[val_idx].to(cache_device)),
            config['eval_batch_size'])
        X_train, y_train = encode(model, probe_batches, device)
        X_val, y_val = encode(model, val_batches, device)

        # Stage 3: fit ONE probe using fixed embeddings. Only the probe
        # epoch is selected by validation; the encoder is never selected.
        classifier, val_auc, val_loss, best_probe_epoch = fit_probe(
            X_train, y_train, X_val, y_val, config,
            description=f'{method} run {run + 1}: probe')
        del X_train, y_train, X_val, y_val, val_batches

        # Stage 4: the chosen probe sees test embeddings only now.
        test_batches = Batches((eval_data[test_idx], eval_labels[test_idx]),
                               config['eval_batch_size'])
        X_test, y_test = encode(model, test_batches, device)
        with torch.no_grad():
            y_pred = classifier(X_test)[:, 1].cpu().numpy()
        y_true = y_test.cpu().numpy()
        test_auc = float(roc_auc_score(y_true, y_pred))
        test_aucs.append(test_auc)
        y_trues.append(y_true.tolist())
        y_preds.append(y_pred.tolist())
        probe_epochs.append(best_probe_epoch)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)
        split_indices.append(dict(test=test_idx.tolist(), validation=val_idx.tolist(),
                                  unused=unused_idx.tolist()))
        print(f'AUC = {test_auc:.3f} | final SSL epoch = {config["epochs"]} | '
              f'best probe epoch = {best_probe_epoch} | validation BCE = {val_loss:.6f}')
        del model, classifier, X_test, y_test, test_batches

    results = dict(auc_scores=test_aucs, y_true=y_trues, y_pred=y_preds,
                   best_probe_epochs=probe_epochs, validation_losses=val_losses,
                   validation_aucs=val_aucs, final_ssl_epoch=config['epochs'],
                   split_indices=split_indices, config=copy.deepcopy(config))
    results_by_method[method] = results
    print(f'mean AUC = {np.mean(test_aucs):.3f} | std AUC = {np.std(test_aucs):.3f}')
    if config['save_results']:
        result_dir = os.path.join(config['path_save'], method)
        os.makedirs(result_dir, exist_ok=True)
        filename = 'results.pkl'
        with open(os.path.join(result_dir, filename), 'wb') as file:
            pickle.dump(results, file)
    return results_by_method


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate the original VarCoNet ASD-classification baseline.')
    parser.add_argument('--path_data', type=str, required=True)
    parser.add_argument('--path_save', type=str, required=True)
    parser.add_argument('--method', type=str, default='ABIDEI')
    parser.add_argument('--atlas', type=str, default='SCH')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--warm_up_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1.5e-4)
    parser.add_argument('--tau', type=float, default=0.054)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_length', type=int, default=200)
    parser.add_argument('--min_length', type=int, default=80)
    parser.add_argument('--epochs_cls', type=int, default=300)
    parser.add_argument('--lr_cls', type=float, default=1e-4)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--save_results', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--preprocess_batch_size', type=int, default=128)
    parser.add_argument('--cache_on_gpu', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--probe_eval_interval', type=int, default=1)
    parser.add_argument('--seed1', type=int, default=30)
    parser.add_argument('--seed2', type=int, default=6)
    args = parser.parse_args()
    config = {}
    config['path_data'] = args.path_data
    config['path_save'] = args.path_save
    config['method'] = args.method
    config['atlas'] = args.atlas
    config['model_config'] = dict(dim_feedforward=1024, layers=1, n_heads=1, max_length=200)
    config['epochs'] = args.epochs  # Complete all SSL epochs before fitting the probe.
    config['warm_up_epochs'] = args.warm_up_epochs
    config['lr'] = args.lr
    config['tau'] = args.tau
    config['batch_size'] = args.batch_size
    config['max_length'] = args.max_length
    config['min_length'] = args.min_length
    config['train_length_limits'] = [config['min_length'], config['max_length']]
    config['epochs_cls'] = args.epochs_cls  # Retains the original 299 full-batch probe updates.
    config['lr_cls'] = args.lr_cls
    config['num_classes'] = args.num_classes
    config['device'] = args.device
    config['save_results'] = args.save_results
    config['repeats'] = args.repeats
    config['eval_batch_size'] = args.eval_batch_size
    config['preprocess_batch_size'] = args.preprocess_batch_size
    config['cache_on_gpu'] = args.cache_on_gpu
    config['probe_eval_interval'] = args.probe_eval_interval  # Validate the probe after every probe update.
    # No encoder eval_interval: the encoder is trained to the final SSL epoch.
    config['seed1'] = args.seed1
    config['seed2'] = args.seed2
    main(config)