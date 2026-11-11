"""Train and evaluate the BolT Transformer for cross-dataset ASD classification."""
import argparse

parent_path = r'/home/student1/Desktop/Charalampos_Lamprou/baselines'
import sys
sys.path.append(parent_path)
import copy
import os
import pickle
from numbers import Integral
import numpy as np
import torch
import torch.nn as nn
from BolT.Models.BolT.model import Model
from BolT.utils import Option
from BolT.Models.BolT.hyperparams import getHyper_bolT
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
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
    if method in ("CR-VAE", "BOLDGEN"):
        key, tag = (("path_augmented_crvae", "CR-VAE") if method =="CR-VAE"
                    else ("path_augmented_boldgen", "BOLDGEN"))
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
    if method == "T2I-Diff":
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


def train_step(model, x, y, device):
    """One mini-batch and one optimizer update, performed inside model.step."""
    model.model.train()
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    x = torch.permute(x, (0, 2, 1))
    x = (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=True)
    train_loss, _, _, _ = model.step(x, y, train=True)
    return train_loss.detach()


def configure_step_scheduler(model, max_steps):
    """Give BolT's OneCycleLR the same horizon for every training dataset.

    Preserve the scheduler settings chosen by the installed BolT model. Its
    state_dict supplies the phase layout because OneCycleLR has no public
    getters for pct_start or three_phase. Call before the first training step.
    model.step remains responsible for stepping the scheduler exactly once.
    """
    scheduler = getattr(model, 'scheduler', None)
    if scheduler is None:
        return
    if not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
        raise TypeError('Expected BolT OneCycleLR; adapt configure_step_scheduler '
                        'for your custom scheduler before using step-based training')
    state = scheduler.state_dict()
    groups = scheduler.optimizer.param_groups
    first = groups[0]
    phases = state['_schedule_phases']
    if len(phases) not in (2, 3):
        raise ValueError('Unsupported OneCycleLR phase layout')
    pct_start = (phases[0]['end_step'] + 1) / state['total_steps']
    anneal_strategy = state.get('_anneal_func_type')
    if anneal_strategy is None:
        # Older PyTorch versions store the annealing function itself.
        function_name = getattr(state.get('anneal_func'), '__name__', '')
        anneal_strategy = {'_annealing_cos': 'cos', '_annealing_linear': 'linear'}.get(function_name)
    if anneal_strategy not in ('cos', 'linear'):
        raise ValueError('Cannot determine the OneCycleLR annealing strategy')
    div_factor = first['max_lr'] / first['initial_lr']
    final_div_factor = first['initial_lr'] / first['min_lr']
    momentum = {}
    if state['cycle_momentum']:
        momentum = dict(base_momentum=[g['base_momentum'] for g in groups],
                        max_momentum=[g['max_momentum'] for g in groups])
    model.scheduler = torch.optim.lr_scheduler.OneCycleLR(
        scheduler.optimizer,
        max_lr=[g['max_lr'] for g in groups],
        total_steps=max_steps,
        pct_start=pct_start,
        anneal_strategy=anneal_strategy,
        cycle_momentum=state['cycle_momentum'],
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        three_phase=len(phases) == 3,
        **momentum)


def train_steps(model, train_batches, val_batches, criterion, device,
                max_steps, eval_every_n_steps, description):
    """Train to a fixed update budget, validating on a global step counter."""
    best_loss, best_state, best_step = float('inf'), None, None
    train_iterator = iter(train_batches)
    total_loss = torch.zeros((), device=device)
    count = 0
    with tqdm(range(1, max_steps + 1), desc=description, unit='step') as progress:
        for step in progress:
            try:
                x, y = next(train_iterator)
            except StopIteration:
                # Restart only after exhausting the data; shuffle=True reshuffles.
                train_iterator = iter(train_batches)
                x, y = next(train_iterator)
            total_loss += train_step(model, x, y, device) * len(x)
            count += len(x)
            if step % eval_every_n_steps != 0 and step != max_steps:
                continue
            # Sample-weighted training loss since the previous validation.
            loss = (total_loss / count).item()
            val_auc, val_loss, _, _ = evaluate(model, val_batches, criterion, device)
            if val_loss < best_loss:
                best_loss, best_step = val_loss, step
                best_state = copy.deepcopy(model.model.state_dict())
            progress.set_postfix(loss=loss, val_loss=val_loss,
                                 val_auc=val_auc, best_loss=best_loss)
            total_loss.zero_()
            count = 0
    if best_state is None:
        raise RuntimeError('No valid validation checkpoint was selected')
    return best_state, best_step, best_loss


@torch.no_grad()
def evaluate(model, batches, criterion, device):
    model.model.eval()
    targets, probabilities = [], []
    total_loss = torch.zeros((), device=device)
    count = 0
    for x, y in batches:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x = torch.permute(x,(0,2,1))
        x = (x - torch.mean(x, dim=-1, keepdim=True))/torch.std(x, dim=-1, keepdim=True)
        test_loss, _, test_probs, _ = model.step(x, y, train=False)
        total_loss += test_loss * len(x)
        count += len(x)
        probabilities.append(test_probs[:, 1])
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
    for key in ('repeats', 'batch_size', 'eval_batch_size',
                'eval_every_n_steps', 'preprocess_batch_size'):
        if isinstance(config[key], bool) or not isinstance(config[key], Integral) or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
        config[key] = int(config[key])
    if not 0 < config['validation_fraction'] < 1:
        raise ValueError('validation_fraction must be between 0 and 1')
    seed1 = config['seed1']
    seed2 = config['seed2']
    if seed1 is not None:
        np.random.seed(seed1)
    if seed2 is not None:
        torch.manual_seed(seed2)
    cache_device = device if config['cache_on_gpu'] else torch.device('cpu')
    eval_data, eval_labels = load_pair(
        os.path.join(config['path_data'], 'ABIDEII_ASD_nilearn.npy'),
        os.path.join(config['path_data'], 'ABIDEII_NC_nilearn.npy'))
    eval_labels = prepare_labels(eval_labels, len(eval_data)).reshape(-1).long()
    eval_data = prepare_signals(eval_data, device, config['preprocess_batch_size'])
    roi_num = eval_data.shape[-1]
    results_by_method = {}
    for method in config['methods']:
        print(f'\nMethod: {method}')
        if method == 'ABIDEI':
            config['max_steps'] = 16
        else:
            config['max_steps'] = 240
        train_data, labels = load_training_data(config, method)
        labels = prepare_labels(labels, len(train_data)).reshape(-1).long()
        train_data = prepare_signals(train_data, device, config['preprocess_batch_size'],
            standardize=method in {'TimeVAE', 'TimeGAN', 'T2I-Diff', 'DSFM'})
        train_data, labels = train_data.to(cache_device), labels.to(cache_device)
        train_batches = Batches(train_data, labels, config['batch_size'], shuffle=True)
        print(f'{len(train_data)} labelled training samples; '
              f'{len(train_batches)} batches per data pass; '
              f'{config["max_steps"]} training steps per run')
        test_aucs, y_trues, y_preds = [], [], []
        for run in range(config['repeats']):
            test_idx, val_idx = train_test_split(
                np.arange(len(eval_data)), test_size=config['validation_fraction'], shuffle=True,
                stratify=eval_labels.numpy(), random_state=None if seed1 is None else seed1 + run)
            if len(torch.unique(eval_labels[val_idx])) != 2 or len(torch.unique(eval_labels[test_idx])) != 2:
                raise ValueError('Validation and test splits must each contain both classes for AUC')
            val_batches = Batches(eval_data[val_idx].to(cache_device),
                                  eval_labels[val_idx].to(cache_device), config['eval_batch_size'])
            details = Option({
                "device" : device,
                "nOfTrains" : len(train_data),
                "nOfClasses" : config['num_classes'],
                "batchSize" : config['batch_size'],
                # Compatibility with BolT's constructor only. The training loop
                # and the final scheduler horizon are both controlled by steps.
                "nOfEpochs" : max(1, (config['max_steps'] + len(train_batches) - 1)
                                 // len(train_batches))
            })
            hyperParams = getHyper_bolT(rois = roi_num)
            model = Model(hyperParams, details)
            configure_step_scheduler(model, config['max_steps'])
            criterion = nn.CrossEntropyLoss()
            # Both feature extractor and classification head receive label gradients.
            best_state, best_step, best_loss = train_steps(
                model, train_batches, val_batches, criterion, device,
                config['max_steps'], config['eval_every_n_steps'],
                description=f'{method} run {run + 1}')
            model.model.load_state_dict(best_state)
            test_batches = Batches(eval_data[test_idx], eval_labels[test_idx], config['eval_batch_size'])
            test_auc, _, y_true, y_pred = evaluate(model, test_batches, criterion, device)
            test_aucs.append(test_auc)
            y_trues.append(y_true.tolist())
            y_preds.append(y_pred.tolist())
            print(f'AUC = {test_auc:.3f} | best step = {best_step}')
            if config['save_models']:
                model_dir = os.path.join(config['path_save'], method, 'models')
                os.makedirs(model_dir, exist_ok=True)
                torch.save({'model_state_dict': {k: v.cpu() for k, v in best_state.items()},
                    'config': config, 'roi_num': roi_num, 'step': best_step, 'val_loss': best_loss,
                    'test_indices': test_idx.tolist(), 'val_indices': val_idx.tolist()},
                    os.path.join(model_dir, f'run_{run + 1}.pt'))
            del model, best_state, test_batches, val_batches
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
    parser = argparse.ArgumentParser(description='Train and evaluate the BolT Transformer for cross-dataset ASD classification.')
    parser.add_argument('--path_data', type=str, required=True)
    parser.add_argument('--path_save', type=str, required=True)
    parser.add_argument('--path_augmented_boldgen', type=str, required=True)
    parser.add_argument('--path_augmented_crvae', type=str, required=True)
    parser.add_argument('--path_augmented_timevae', type=str, required=True)
    parser.add_argument('--path_augmented_timegan', type=str, required=True)
    parser.add_argument('--path_augmented_t2idiff', type=str, required=True)
    parser.add_argument('--path_augmented_dsfm', type=str, required=True)
    parser.add_argument('--methods', nargs="+", type=str, default=['ABIDEI', 'DSFM', 'T2I-Diff', 'BOLDGEN', 'CR-VAE', 'TimeVAE', 'TimeGAN'])
    parser.add_argument('--atlas', type=str, default='SCH')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--device', type=str, default="cuda:1")
    parser.add_argument('--save_results', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--preprocess_batch_size', type=int, default=128)
    parser.add_argument('--cache_on_gpu', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--eval_every_n_steps', type=int, default=4)
    parser.add_argument('--seed1', type=int, default=30)
    parser.add_argument('--seed2', type=int, default=6)
    parser.add_argument('--validation_fraction', type=float, default=0.5)
    parser.add_argument('--save_models', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    config = {}
    config['path_data'] = args.path_data
    config['path_save'] = args.path_save
    config['path_augmented_boldgen'] = args.path_augmented_boldgen
    config['path_augmented_crvae'] = args.path_augmented_crvae
    config['path_augmented_timevae'] = args.path_augmented_timevae
    config['path_augmented_timegan'] = args.path_augmented_timegan
    config['path_augmented_t2idiff'] = args.path_augmented_t2idiff
    config['path_augmented_dsfm'] = args.path_augmented_dsfm
    config['methods'] = args.methods
    config['atlas'] = args.atlas
    # 15 * ceil(916 / 128) = 120 updates: the original ABIDEI training budget.
    # Keep this fixed across methods; increase it for a longer experiment.
    config['batch_size'] = args.batch_size
    config['num_classes'] = args.num_classes
    config['device'] = args.device
    config['save_results'] = args.save_results
    config['repeats'] = args.repeats
    config['eval_batch_size'] = args.eval_batch_size
    config['preprocess_batch_size'] = args.preprocess_batch_size
    config['cache_on_gpu'] = args.cache_on_gpu  # True needs space for raw signals + training activations.
    config['eval_every_n_steps'] = args.eval_every_n_steps  # Validate every 8 updates, and at max_steps.
    config['seed1'] = args.seed1  # Optional integer; changed RNG ordering prevents exact replay.
    config['seed2'] = args.seed2  # Optional integer; changed RNG ordering prevents exact replay.
    config['validation_fraction'] = args.validation_fraction
    config['save_models'] = args.save_models  # Optional checkpoint for each repetition.
    main(config)