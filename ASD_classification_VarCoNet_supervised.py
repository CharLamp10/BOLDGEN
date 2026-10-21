"""Supervised VarCoNet: train feature extractor and classifier jointly.

Dependencies: torch, numpy, scikit-learn, tqdm, lightning-bolts (pl_bolts).
Run: python varconet_supervised.py after editing config below.

All input signals are used directly with ordinary shuffled training batches.
CRVAE/CR-VAE and rDCM-GEN: [N, B, T, R] -> [N*B, T, R]; each subject
label is repeated B times in matching order. All samples are used each epoch.
The existing CNN-Transformer and ROI cosine-similarity features are retained.
A two-logit linear head and CrossEntropyLoss train the entire model end to end.
Validation AUC selects the checkpoint; the test set is evaluated afterwards.
The source script's 50/50 ABIDEII validation/test split is preserved.
"""
import copy
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
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
    if method in ("CRVAE", "CR-VAE", "rDCM-GEN"):
        key, tag = (("path_augmented_crvae", "CR-VAE") if method in ("CRVAE", "CR-VAE")
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
                labels.repeat_interleave(B).reshape(-1, 1))
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
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)
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


class SupervisedVarCoNet(nn.Module):
    def __init__(self, model_config, roi_num, num_classes=2):
        super().__init__()
        self.encoder = VarCoNet(model_config, roi_num)
        self.classifier = nn.Linear(roi_num * (roi_num - 1) // 2, num_classes)

    def forward(self, x):
        # Return raw logits; CrossEntropyLoss applies log-softmax internally.
        return self.classifier(self.encoder(x))


@torch.no_grad()
def prepare_signals(signals, device, batch_size, standardize=False):
    """Convert once; preserve the source's method-specific standardization."""
    source = torch.as_tensor(signals)
    if source.ndim != 3 or not len(source) or source.shape[1] < 4 or source.shape[2] < 2:
        raise ValueError('Expected nonempty [samples, T, R] with T >=4 and R >=2')
    result = torch.empty(source.shape, dtype=torch.float32)
    for start in range(0, len(source), batch_size):
        x = source[start:start + batch_size].to(device=device, dtype=torch.float32)
        if standardize:
            x = (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True)
        if not torch.isfinite(x).all().item():
            raise ValueError('Non-finite signals after preprocessing; check NaNs/constant regions')
        result[start:start + len(x)].copy_(x.cpu())
    return result


class Batches:
    """Ordinary mini-batches from pinned CPU tensors or an optional GPU cache."""
    def __init__(self, data, labels, batch_size, shuffle=False):
        if batch_size < 1 or len(data) != len(labels) or not len(data):
            raise ValueError('Invalid batch size or mismatched/empty dataset')
        self.data, self.labels = data, labels
        self.batch_size, self.shuffle = batch_size, shuffle
        self.loader = None
        if data.device.type == 'cpu':
            self.loader = DataLoader(TensorDataset(data, labels), batch_size=batch_size,
                shuffle=shuffle, drop_last=False, num_workers=0,
                pin_memory=torch.cuda.is_available())

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.loader is not None:
            yield from self.loader
            return
        order = torch.randperm(len(self.data), device=self.data.device) if self.shuffle else None
        for start in range(0, len(self.data), self.batch_size):
            idx = slice(start, start + self.batch_size) if order is None else order[start:start + self.batch_size]
            yield self.data[idx], self.labels[idx]


def train_epoch(model, batches, optimizer, criterion, device):
    model.train()
    total_loss = torch.zeros((), device=device)
    count = 0
    for x, y in batches:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach() * len(x)
        count += len(x)
    return (total_loss / count).item()


@torch.no_grad()
def evaluate(model, batches, criterion, device):
    model.eval()
    targets, probabilities = [], []
    total_loss = torch.zeros((), device=device)
    count = 0
    for x, y in batches:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += criterion(logits, y) * len(x)
        count += len(x)
        probabilities.append(logits.softmax(dim=1)[:, 1])
        targets.append(y)
    y_true = torch.cat(targets).cpu().numpy()
    y_pred = torch.cat(probabilities).cpu().numpy()
    return float(roc_auc_score(y_true, y_pred)), (total_loss / count).item(), y_true, y_pred


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
    for key in ('epochs', 'repeats', 'batch_size', 'eval_batch_size', 'eval_interval', 'preprocess_batch_size'):
        if config[key] < 1:
            raise ValueError(f'{key} must be positive')
    if not 0 < config['validation_fraction'] < 1:
        raise ValueError('validation_fraction must be between 0 and 1')
    seed = config['seed']
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    cache_device = device if config['cache_on_gpu'] else torch.device('cpu')
    eval_data, eval_labels = load_pair(
        os.path.join(config['path_data'], 'ABIDEII_ASD_nilearn.npy'),
        os.path.join(config['path_data'], 'ABIDEII_NC_nilearn.npy'))
    eval_labels = prepare_labels(eval_labels, len(eval_data)).reshape(-1).long()
    eval_data = prepare_signals(eval_data, device, config['preprocess_batch_size'])
    if eval_data.shape[1] > config['model_config']['max_length']:
        raise ValueError("Evaluation sequence exceeds model_config['max_length']")
    roi_num = eval_data.shape[-1]
    results_by_method = {}
    for method in config['methods']:
        print(f'\nMethod: {method}')
        train_data, labels = load_training_data(config, method)
        labels = prepare_labels(labels, len(train_data)).reshape(-1).long()
        train_data = prepare_signals(train_data, device, config['preprocess_batch_size'],
            standardize=method in {'TimeVAE', 'TimeGAN', 'T2IDIFF', 'DSFM'})
        if train_data.shape[-1] != roi_num or train_data.shape[1] > config['model_config']['max_length']:
            raise ValueError('Training ROI count or sequence length is incompatible with the model')
        train_data, labels = train_data.to(cache_device), labels.to(cache_device)
        train_batches = Batches(train_data, labels, config['batch_size'], shuffle=True)
        print(f'{len(train_data)} labelled training samples; {len(train_batches)} batches per epoch')
        test_aucs, y_trues, y_preds = [], [], []
        for run in range(config['repeats']):
            test_idx, val_idx = train_test_split(
                np.arange(len(eval_data)), test_size=config['validation_fraction'], shuffle=True,
                stratify=eval_labels.numpy(), random_state=None if seed is None else seed + run)
            if len(torch.unique(eval_labels[val_idx])) != 2 or len(torch.unique(eval_labels[test_idx])) != 2:
                raise ValueError('Validation and test splits must each contain both classes for AUC')
            val_batches = Batches(eval_data[val_idx].to(cache_device),
                                  eval_labels[val_idx].to(cache_device), config['eval_batch_size'])
            model = SupervisedVarCoNet(config['model_config'], roi_num, config['num_classes']).to(device)
            criterion = nn.CrossEntropyLoss()
            # Both feature extractor and classification head receive label gradients.
            optimizer = Adam(model.parameters(), lr=config['lr'])
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer, warmup_start_lr=1e-5, warmup_epochs=config['warm_up_epochs'],
                max_epochs=config['epochs'])
            best_loss, best_state, best_epoch = float('inf'), None, None
            with tqdm(range(1, config['epochs'] + 1), desc=f'{method} run {run + 1}') as progress:
                for epoch in progress:
                    loss = train_epoch(model, train_batches, optimizer, criterion, device)
                    scheduler.step()
                    progress.set_postfix(loss=loss)
                    if epoch % config['eval_interval'] != 0 and epoch != config['epochs']:
                        continue
                    val_auc, val_loss, _, _ = evaluate(model, val_batches, criterion, device)
                    if val_loss < best_loss:
                        best_loss, best_epoch = val_loss, epoch
                        best_state = copy.deepcopy(model.state_dict())
                    progress.set_postfix(loss=loss, val_loss=val_loss, val_auc=val_auc, best_loss=best_loss)
            if best_state is None:
                raise RuntimeError('No valid validation checkpoint was selected')
            model.load_state_dict(best_state)
            test_batches = Batches(eval_data[test_idx], eval_labels[test_idx], config['eval_batch_size'])
            test_auc, _, y_true, y_pred = evaluate(model, test_batches, criterion, device)
            test_aucs.append(test_auc)
            y_trues.append(y_true.tolist())
            y_preds.append(y_pred.tolist())
            print(f'AUC = {test_auc:.3f}')
            if config['save_models']:
                model_dir = os.path.join(config['path_save'], method, 'models')
                os.makedirs(model_dir, exist_ok=True)
                torch.save({'model_state_dict': {k: v.cpu() for k, v in best_state.items()},
                    'config': config, 'roi_num': roi_num, 'epoch': best_epoch, 'val_loss': best_loss,
                    'test_indices': test_idx.tolist(), 'val_indices': val_idx.tolist()},
                    os.path.join(model_dir, f'run_{run + 1}.pt'))
            del model, optimizer, scheduler, best_state, test_batches, val_batches
        results = dict(auc_scores=test_aucs, y_true=y_trues, y_pred=y_preds)
        results_by_method[method] = results
        print(f'mean AUC = {np.mean(test_aucs):.3f} | std AUC = {np.std(test_aucs):.3f}')
        if config['save_results']:
            result_dir = os.path.join(config['path_save'], method)
            os.makedirs(result_dir, exist_ok=True)
            with open(os.path.join(result_dir, 'results.pkl'), 'wb') as file:
                pickle.dump(results, file)
        del train_batches, train_data, labels
    return results_by_method


if __name__ == "__main__":
    config = {}
    config['path_data'] = r'/media/student1/ExtremePro/rDCM-GEN/data'
    config['path_save'] = r'/home/student1/Desktop/Charalampos_Lamprou/rDCM-GEN_results_HCP_2026/ASD_CLC/VarCoNet_Supervised'
    config['path_augmented_rdcmgen'] = r'/media/student1/ExtremePro/rDCM-GEN/rDCM-GEN_results/augmented_rDCM-GEN_noFCloss'
    config['path_augmented_crvae'] = r'/media/student1/ExtremePro/CRVAE_augmented'
    config['path_augmented_timevae'] = r'/media/student1/ExtremePro/TimeVAE_results/outputs/gen_data'
    config['path_augmented_timegan'] = r'/media/student1/ExtremePro/TimeGAN_results'
    config['path_augmented_t2idiff'] = r'/media/student1/ExtremePro/T2IDIFF'
    config['path_augmented_dsfm'] = r'/media/student1/ExtremePro/DSFM_results'
    config['methods'] = ['ABIDEI','DSFM','rDCM-GEN','TimeVAE', 'TimeGAN','CRVAE','T2IDIFF']
    config['atlas'] = 'SCH'
    config['model_config'] = {}
    config['model_config']['layers'] = 1
    config['model_config']['dim_feedforward'] = 1024
    config['model_config']['layers'] = 1
    config['model_config']['n_heads'] = 1
    config['model_config']['max_length'] = 200
    config['epochs'] = 200
    config['warm_up_epochs'] = 10
    config['lr'] = 2e-5
    config['batch_size'] = 128
    config['num_classes'] = 2
    config['device'] = "cuda:0"
    config['save_results'] = True
    config['repeats'] = 10
    config['eval_batch_size'] = 128
    config['preprocess_batch_size'] = 128
    config['cache_on_gpu'] = False  # True needs space for raw signals + training activations.
    config['eval_interval'] = 1  # 5 is faster but changes which encoder epochs are eligible.
    config['seed'] = 42  # Optional integer; changed RNG ordering prevents exact replay.
    config['validation_fraction'] = 0.5
    config['save_models'] = False  # Optional checkpoint for each repetition.
    main(config)