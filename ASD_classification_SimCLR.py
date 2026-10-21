"""Optimized generated-view Transformer contrastive training.

Run: python simclr_transformer_gpu.py after editing CONFIG at the bottom.
Requires torch, numpy, scikit-learn, tqdm, pl_bolts, and YOUR ORIGINAL utils.py.
The implementations of MaskedInstanceNorm1d, InfoNCE, and DualBranchContrast
remain imported from utils so their unseen behavior is not silently changed.

Inputs: [subjects, generated_variants, time, ROIs], one binary label per subject.
Default pairing preserves the original first-8 versus last-7 policy for B=15.
Set pair_mode='any' for any two distinct variants from the same subject.
One pair per subject per encoder epoch; distinct subjects across batch pairs.
Full-batch classifier fitting uses one random first-group variant per subject,
as in the original. Validation and testing now include EVERY batch.
The 199 probe updates, AUC checkpoint criterion, Softmax/BCE, encoder architecture,
and warmup schedule are retained. GPU AUC computation handles ties explicitly.
"""
import copy
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import InfoNCE, MaskedInstanceNorm1d, DualBranchContrast


class Transformer(nn.Module):
    def __init__(
        self,
        R,
        D,
        out_dim,
        num_layers,
        num_heads,
        dim_feedforward
    ):
        super().__init__()
        self.R = R
        self.D = D
        
        self.in_norm = MaskedInstanceNorm1d()
        
        # 1. Linear projection R -> D
        self.input_proj = nn.Linear(R, D)

        # 2. Learned mask token
        self.mask_token = nn.Parameter(torch.randn(D))

        # 3. Learned CLS token
        self.cls_token = nn.Parameter(torch.randn(D))

        # 4. Learned positional embedding for sequence of length (T+1)
        #    (we will dynamically expand to batch size)
        self.pos_emb = nn.Parameter(torch.randn(1, 1, D))  

        # 5. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(D,out_dim)


    def forward(self, x):
        """
        x: (B, T, R)
        returns:
            embedding: (B, out_dim)
        """
        x = x.float()
        B, T, R = x.shape
        
        mask = (x[:,:,0] != 0)
        x = self.in_norm(x, mask)
        
        x_mask = (x[:, :, 0] == 0).bool()

        # --- Step 1: project to embeddings ---
        x = self.input_proj(x)   # (B, T, D)

        # --- Step 3: prepend CLS token ---
        cls_tok = self.cls_token.unsqueeze(0).unsqueeze(1)   # (1, 1, D)
        cls_tok = cls_tok.expand(B, -1, -1)                    # (B, 1, D)

        x_seq = torch.cat([cls_tok, x], dim=1)        # (B, T+1, D)
        cls_mask = torch.zeros(x.shape[0], 1, dtype=torch.bool, device=x.device)
        x_mask = torch.cat([cls_mask, x_mask], dim=1)

        # --- Step 4: add positional embeddings ---
        x_seq = x_seq + self.pos_emb
        x_seq = x_seq.masked_fill(x_mask.unsqueeze(-1), 0)

        # --- Step 5: transformer ---
        z = self.transformer(x_seq,src_key_padding_mask=x_mask) # (B, T+1, D)

        # --- Step 6: extract CLS output for classification ---
        cls_out = z[:, 0, :]                                 # (B, D)
        cls_out = self.out_proj(cls_out)

        return cls_out

class MLP(nn.Module):
    def __init__(self, num_features, num_classes):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_features, num_classes),
            nn.Softmax(dim=1))

    def forward(self, x):
        z = self.fc(x)
        return z

def sample_pair_indices(n, variants, batch_size, mode='split', split=8, shuffle=True):
    """Yield first-view indices followed by matching second-view indices."""
    if n < 2 or variants < 2 or batch_size < 2:
        raise ValueError('Contrastive training requires >=2 subjects/variants and batch_size >=2')
    subjects = np.random.permutation(n) if shuffle else np.arange(n)
    if mode == 'split':
        if not 0 < split < variants:
            raise ValueError('variant_split must be between 1 and B-1')
        a = np.random.randint(0, split, size=n)
        b = np.random.randint(split, variants, size=n)
    elif mode == 'any':
        a = np.random.randint(variants, size=n)
        b = np.random.randint(variants - 1, size=n)
        b += b >= a
    else:
        raise ValueError("pair_mode must be 'split' or 'any'")
    groups = (n + batch_size - 1) // batch_size
    # Balanced batches avoid a singleton remainder whenever feasible.
    for ids in np.array_split(subjects, groups):
        yield np.concatenate((ids * variants + a[ids], ids * variants + b[ids])).tolist()


def contiguous_indices(n, batch_size):
    return [list(range(start, min(start + batch_size, n))) for start in range(0, n, batch_size)]


def indexed_batches(signals, labels, batches):
    """Select ONLY needed views before CPU->GPU transfer, or index GPU caches."""
    if signals.device.type == 'cpu':
        yield from DataLoader(TensorDataset(signals, labels), batch_sampler=batches,
                              num_workers=0, pin_memory=torch.cuda.is_available())
    else:
        for indices in batches:
            ids = torch.as_tensor(indices, dtype=torch.long, device=signals.device)
            yield signals[ids], labels[ids]


def prepare_labels(labels, expected):
    labels = torch.as_tensor(labels).reshape(-1).cpu()
    if len(labels) != expected or not ((labels == 0) | (labels == 1)).all():
        raise ValueError('Expected one binary label per subject')
    return labels.long()


def train_epoch(model, contrast, optimizer, signals, labels, subject_shape, config, device):
    model.train()
    n, variants = subject_shape
    indices = list(sample_pair_indices(n, variants, config['batch_size'],
                                       config['pair_mode'], config['variant_split'], config['shuffle']))
    total = torch.zeros((), device=device)
    for paired, _ in indexed_batches(signals, labels, indices):
        paired = paired.to(device, non_blocking=True)
        k = len(paired) // 2
        optimizer.zero_grad(set_to_none=True)
        z1, z2 = model(paired[:k]), model(paired[k:])
        loss = contrast(z1, z2)
        loss.backward()
        optimizer.step()
        total += loss.detach()
    return (total / len(indices)).item()


@torch.no_grad()
def encode(model, signals, labels, indices, device):
    # These ordinary no-grad tensors can subsequently train a linear classifier.
    model.eval()
    outputs, targets = [], []
    for x, y in indexed_batches(signals, labels, indices):
        outputs.append(model(x.to(device, non_blocking=True)))
        targets.append(y.to(device, non_blocking=True))
    if not outputs:
        raise ValueError('Cannot encode an empty dataset')
    return torch.cat(outputs), torch.cat(targets).reshape(-1)


@torch.no_grad()
def binary_auc(y_true, scores):
    """Mann-Whitney AUC with average ranks for ties; stays on scores.device.

    Caller checks the validation split contains both classes. Float64 ranks
    avoid loss of integer precision when computing rank sums for large splits.
    """
    order = scores.argsort()
    scores, targets = scores[order], y_true[order].to(torch.float64)
    n = len(scores)
    ranks = torch.arange(n, device=scores.device, dtype=torch.float64)
    starts = torch.cat((torch.ones(1, dtype=torch.bool, device=scores.device), scores[1:] != scores[:-1]))
    ends = torch.cat((scores[:-1] != scores[1:], torch.ones(1, dtype=torch.bool, device=scores.device)))
    lower = torch.where(starts, ranks, torch.zeros_like(ranks)).cummax(0).values
    upper = torch.where(ends, ranks, torch.full_like(ranks, n - 1)).flip(0).cummin(0).values.flip(0)
    avg_rank = (lower + upper) * 0.5 + 1
    positives = targets.sum()
    auc = ((avg_rank * targets).sum() - positives * (positives + 1) * 0.5) / (positives * (n - positives))
    return torch.where(torch.isfinite(scores).all(), auc, torch.full_like(auc, float('nan')))


def fit_classifier(X_train, y_train, X_val, y_val, config):
    classifier = MLP(X_train.shape[1], 2).to(X_train.device)
    optimizer = Adam(classifier.parameters(), lr=config['probe_lr'])
    criterion = nn.BCELoss()
    target = F.one_hot(y_train.long(), num_classes=2).float()
    best_auc = torch.full((), -float('inf'), device=X_train.device, dtype=torch.float64)
    best_state = copy.deepcopy(classifier.state_dict())
    for _ in range(config['probe_epochs']):
        classifier.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(classifier(X_train), target)
        loss.backward()
        optimizer.step()
        classifier.eval()
        with torch.no_grad():
            auc = binary_auc(y_val, classifier(X_val)[:, 1])
            improved = auc > best_auc
            # Small classifier weights stay on GPU; no per-step AUC .cpu/.item.
            for name, value in classifier.state_dict().items():
                best_state[name].copy_(torch.where(improved, value, best_state[name]))
            best_auc = torch.where(improved, auc, best_auc)
    score = best_auc.item()
    if not np.isfinite(score):
        raise RuntimeError('No finite validation AUC; inspect signals and model outputs')
    classifier.load_state_dict(best_state)
    return score, best_state


def main(config):
    device = torch.device(config['device'])
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; use CUDA-enabled PyTorch or device=cpu')
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f'{device} is not a visible GPU')
        torch.cuda.set_device(device)
    if min(config['epochs'], config['probe_epochs'], config['repeats'], config['eval_batch_size']) < 1:
        raise ValueError('Epoch counts, repetitions, and evaluation batch size must be positive')
    if config['tau'] <= 0:
        raise ValueError('tau must be positive')
    seed = config['seed']
    if seed is not None:
        np.random.seed(33)
        torch.manual_seed(9)
    eval_epochs = set(config['eval_epochs'])
    eval_epochs.add(config['epochs'])
    cache_device = device if config['cache_on_gpu'] else torch.device('cpu')
    asd = np.load(os.path.join(config['path_data'], 'ABIDEII_ASD_nilearn.npy'))
    control = np.load(os.path.join(config['path_data'], 'ABIDEII_NC_nilearn.npy'))
    eval_signals = torch.as_tensor(np.concatenate((asd, control)), dtype=torch.float32)
    eval_labels = torch.cat((torch.ones(len(asd)), torch.zeros(len(control)))).long()
    del asd, control
    if eval_signals.ndim != 3 or not torch.isfinite(eval_signals).all():
        raise ValueError('Evaluation data must be finite [subjects, T, R]')
    results_by_method = {}
    for method in config['methods']:
        if method == 'CRVAE':
            root, tag = config['path_augmented_crvae'], 'CR-VAE'
        elif method == 'rDCM-GEN':
            root, tag = config['path_augmented_rdcmgen'], 'rDCM-GEN'
        else:
            raise ValueError(f'Unsupported method: {method}')
        signals = torch.load(os.path.join(root, f'augmented_{tag}_ABIDEI.pt'), map_location='cpu', weights_only=True).float()
        labels = torch.load(os.path.join(root, f'labels_augmented_{tag}_ABIDEI.pt'), map_location='cpu', weights_only=True)
        if signals.ndim != 4 or not torch.isfinite(signals).all():
            raise ValueError('Generated data must be finite [N, B, T, R]')
        N, B, T, R = signals.shape
        labels = prepare_labels(labels, N)
        if R != eval_signals.shape[-1]:
            raise ValueError('Training/evaluation ROI counts differ')
        if N < 2 or B < 2 or config['batch_size'] < 2:
            raise ValueError('Need >=2 subjects/variants and batch_size >=2')
        if config['pair_mode'] not in {'split', 'any'}:
            raise ValueError("pair_mode must be 'split' or 'any'")
        if config['pair_mode'] == 'split' and not 0 < config['variant_split'] < B:
            raise ValueError('variant_split must be between 1 and B-1')
        signals = signals.reshape(N * B, T, R).to(cache_device)
        flat_labels = labels.repeat_interleave(B).to(cache_device)
        test_aucs, y_trues, y_preds = [], [], []
        print(f'\n{method}: {N} subjects, {B} variants; pairing={config["pair_mode"]}')
        for run in range(config['repeats']):
            test_idx, val_idx = train_test_split(np.arange(len(eval_signals)), test_size=0.3,
                shuffle=True, stratify=eval_labels.numpy(), random_state=None if seed is None else seed + run)
            val_signals, val_labels = eval_signals[val_idx].to(cache_device), eval_labels[val_idx].to(cache_device)
            if len(torch.unique(eval_labels[val_idx])) != 2 or len(torch.unique(eval_labels[test_idx])) != 2:
                raise ValueError('Validation and test splits must each contain both classes')
            mc = config['model_config']
            if mc['D'] % mc['num_heads']:
                raise ValueError('D must be divisible by num_heads')
            model = Transformer(R, mc['D'], mc['out_dim'], mc['num_layers'], mc['num_heads'], mc['dim_feedforward']).to(device)
            contrast = DualBranchContrast(loss=InfoNCE(tau=config['tau']), mode='L2L').to(device)
            optimizer = Adam(model.parameters(), lr=config['lr'])
            scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_start_lr=1e-5,
                warmup_epochs=config['warm_up_epochs'], max_epochs=config['epochs'])
            max_auc, best_encoder, best_classifier, best_epoch = -float('inf'), None, None, None
            with tqdm(range(1, config['epochs'] + 1), desc=f'{method} run {run + 1}') as progress:
                for epoch in progress:
                    loss = train_epoch(model, contrast, optimizer, signals, flat_labels, (N, B), config, device)
                    scheduler.step()
                    progress.set_postfix(loss=loss)
                    if epoch not in eval_epochs:
                        continue
                    # Original probe policy: one random variant per subject from first group.
                    # In 'any' mode, sample the probe variant from all available variants.
                    upper = config['variant_split'] if config['pair_mode'] == 'split' else B
                    selected = np.arange(N) * B + np.random.randint(upper, size=N)
                    probe_indices = [selected[start:start + config['eval_batch_size']].tolist()
                                     for start in range(0, N, config['eval_batch_size'])]
                    X_train, y_train = encode(model, signals, flat_labels, probe_indices, device)
                    X_val, y_val = encode(model, val_signals, val_labels,
                        contiguous_indices(len(val_labels), config['eval_batch_size']), device)
                    auc, classifier_state = fit_classifier(X_train, y_train, X_val, y_val, config)
                    if auc > max_auc:
                        max_auc, best_epoch = auc, epoch
                        best_encoder = copy.deepcopy(model.state_dict())
                        best_classifier = classifier_state
                    progress.set_postfix(loss=loss, val_auc=auc, best_auc=max_auc)
                    del X_train, y_train, X_val, y_val, classifier_state
            if best_encoder is None:
                raise RuntimeError('No valid encoder checkpoint selected')
            model.load_state_dict(best_encoder)
            classifier = MLP(mc['out_dim'], 2).to(device)
            classifier.load_state_dict(best_classifier)
            X_test, y_test = encode(model, eval_signals[test_idx], eval_labels[test_idx],
                contiguous_indices(len(test_idx), config['eval_batch_size']), device)
            classifier.eval()
            with torch.no_grad():
                y_pred = classifier(X_test)[:, 1].cpu().numpy()
            y_true = y_test.cpu().numpy()
            test_auc = float(roc_auc_score(y_true, y_pred))
            test_aucs.append(test_auc); y_trues.append(y_true); y_preds.append(y_pred)
            print(f'AUC = {test_auc:.3f}')
            if config['save_models']:
                path = os.path.join(config['path_save'], method, 'models')
                os.makedirs(path, exist_ok=True)
                torch.save({'encoder': {k: v.cpu() for k, v in best_encoder.items()},
                            'classifier': {k: v.cpu() for k, v in best_classifier.items()},
                            'config': config, 'roi_num': R, 'epoch': best_epoch,
                            'val_auc': max_auc, 'test_indices': test_idx.tolist(),
                            'val_indices': val_idx.tolist()}, os.path.join(path, f'run_{run + 1}.pt'))
            del model, contrast, optimizer, scheduler, classifier, best_encoder, best_classifier
            del X_test, y_test, val_signals, val_labels
        results = dict(auc_scores=test_aucs, y_true=y_trues, y_pred=y_preds)
        results_by_method[method] = results
        print(f'mean AUC = {np.mean(test_aucs):.3f} | std AUC = {np.std(test_aucs):.3f}')
        if config['save_results']:
            path = os.path.join(config['path_save'], method)
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, 'results.pkl'), 'wb') as file:
                pickle.dump(results, file)
        del signals, labels, flat_labels
    return results_by_method


if __name__ == '__main__':
    config = {}
    config['path_data'] = r'/media/student1/ExtremePro/rDCM-GEN/data'
    config['path_save'] = r'/home/student1/Desktop/Charalampos_Lamprou/rDCM-GEN_results_HCP_2026/ASD_CLC/SimCLR_Transformer'
    config['path_augmented_rdcmgen'] = r'/media/student1/ExtremePro/rDCM-GEN/rDCM-GEN_results/augmented_rDCM-GEN_noFCloss'
    config['path_augmented_crvae'] = r'/media/student1/ExtremePro/CRVAE_augmented'
    config['methods'] = ['rDCM-GEN','CRVAE']  
    config['atlas'] = 'SCH'
    config['TR'] = 0.72
    config['newTR'] = 2.0
    config['batch_size'] = 128
    config['shuffle'] = True
    config['epochs'] = 50
    config['tau'] = 0.1
    config['lr'] = 5e-5
    config['warm_up_epochs'] = 10
    config['eval_epochs'] = list(range(1,config['epochs']+1))
    config['model_config'] = {}
    config['model_config']['num_layers'] = 4
    config['model_config']['num_heads'] = 4
    config['model_config']['dim_feedforward'] = 1024
    config['model_config']['out_dim'] = 4096
    config['model_config']['D'] = 512
    config['device'] = "cuda:0"
    config['save_results'] = True
    config['save_models'] = True
    config['repeats'] = 10
    config['eval_batch_size'] = 128
    config['cache_on_gpu'] = False  # True requires space for all generated signals.
    config['pair_mode'] = 'split'  # 'any': any two distinct generated signals.
    config['variant_split'] = 8  # split mode: [0, 8) and [8, B).
    config['probe_epochs'] = 199  # Original range(1, 200).
    config['probe_lr'] = 1e-4
    config['seed'] = 42
    main(config)


