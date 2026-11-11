"""Train and evaluate Transformer masked sequence modeling on augmented HCP data."""
import argparse

import math
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as FF
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Sequence, Union
import os
import copy
import pickle
from scipy.signal import butter, filtfilt, resample


def preprocess_HCP(data, cat):
    data_new = []
    for i, y in enumerate(data):
        if cat == "full_original":
            y_train = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
            order = 4
            cutoff = 0.1
            fs = 1/config['TR']
            b, a = butter(order, cutoff, btype='low', fs=fs)
            y = filtfilt(b, a, y.T).astype('float32')
        elif cat == "half_original":
            y = (y[125:708] - np.mean(y[125:708], axis=0, keepdims=True))/np.std(y[125:708], axis=0, keepdims=True)
            order = 4
            cutoff = 0.1
            fs = 1/config['TR']
            b, a = butter(order, cutoff, btype='low', fs=fs)
            y = filtfilt(b, a, y.T).astype('float32')
        elif cat == "full_subsampled":
            y_train = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
            order = 4
            cutoff = 0.1
            fs = 1/config['TR']
            b, a = butter(order, cutoff, btype='low', fs=fs)
            y = filtfilt(b, a, y.T).T.astype('float32')
            rate = config['newTR']/config['TR']
            new_samples = int(np.round(y_train.shape[0] / rate))
            y = resample(y, new_samples)
            y = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
            y = y.T
        elif cat == "half_subsampled":
            y = (y[125:708] - np.mean(y[125:708], axis=0, keepdims=True))/np.std(y[125:708], axis=0, keepdims=True)
            order = 4
            cutoff = 0.1
            fs = 1/config['TR']
            b, a = butter(order, cutoff, btype='low', fs=fs)
            y = filtfilt(b, a, y.T).T.astype('float32')
            rate = config['newTR']/config['TR']
            new_samples = int(np.round(y.shape[0] / rate))
            y = resample(y, new_samples)
            y = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
            y = y.T
        if not np.any(np.isnan(y)) and not np.all(y == 0):
            data_new.append(y)
    return data_new


class fMRIDataset(Dataset):
    """
    Synthetic multivariate fMRI-like time series:
    - low-rank temporal components + ROI mixing + drift + noise
    Each sample: [num_rois, num_timepoints]
    """
    def __init__(self, data):
        self.data = data
        self.len_data = len(data)

    def __len__(self):
        return self.len_data

    def __getitem__(self, idx):
        signal = self.data[idx]
        roi_signal = (signal - signal.mean(axis=1, keepdims=True)) / (signal.std(axis=1, keepdims=True) + 1e-6)
        return roi_signal  # shape: [num_rois, num_timepoints]


# -------------------------
# Masking utilities (point-wise)
# -------------------------

def random_timepoint_mask(
    batch_size: int,
    num_timepoints: int,
    n_mask_points: int,
    idxs: Optional[Union[Sequence[int], Sequence[Sequence[int]]]] = None
) -> np.ndarray:
    """
    Create boolean mask [batch_size, num_timepoints] where True = masked.

    Args:
        batch_size: number of samples
        num_timepoints: length of time dimension
        n_mask_points: number of masked points per sample (ignored if idxs is given)
        idxs:
            - None → random indices per sample
            - Sequence[int] → same indices used for all samples
            - Sequence[Sequence[int]] → per-sample indices (len == batch_size)
    """
    masks = np.zeros((batch_size, num_timepoints), dtype=bool)

    for i in range(batch_size):
        if idxs is None:
            # random behavior (original)
            if n_mask_points >= num_timepoints:
                masks[i, :] = True
            else:
                chosen = np.random.choice(
                    num_timepoints, size=n_mask_points, replace=False
                )
        else:
            # provided indices
            if isinstance(idxs[0], (list, tuple, np.ndarray)):
                # per-sample idxs
                chosen = idxs[i]
            else:
                # shared idxs
                chosen = idxs

        masks[i, chosen] = True

    return masks



# -------------------------
# Model components
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerMaskRecon(nn.Module):
    def __init__(self,
                 num_rois,
                 d_model,
                 nhead,
                 num_encoder_layers,
                 num_decoder_layers,
                 dim_feedforward,
                 dropout,
                 max_timepoints):
        super().__init__()
        self.num_rois = num_rois
        self.d_model = d_model

        # project ROI features -> token embedding
        self.input_proj = nn.Linear(num_rois, d_model)

        # learned mask token for masked timepoints
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.1)

        self.pos_encoder = PositionalEncoding(num_rois, max_len=max_timepoints)

        encoder_layer = nn.TransformerEncoderLayer(d_model=num_rois,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation='gelu',
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=num_rois,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation='gelu',
                                                   batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.output_proj = nn.Linear(num_rois, num_rois)
        self.final_ln = nn.LayerNorm(num_rois)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, mask_bool: torch.Tensor) -> torch.Tensor:
        """
        x: [B, F, T]
        mask_bool: [B, T] boolean indicating masked positions (True = masked)
        returns recon: [B, F, T]
        """
        B, F, T = x.shape
        assert F == self.num_rois
    
        # transpose to [B, T, F]
        x_seq = x.permute(0, 2, 1).contiguous()
    
        # project to d_model
        #src = self.input_proj(x_seq)  # [B, T, d_model]
    
        # add positional encoding
        src = self.pos_encoder(x_seq)
    
        # For encoder input we keep masked tokens zeroed so encoder learns from visible context
        src_enc = src.clone()
        src_enc[mask_bool] = 0.0
    
        # Encode
        memory = self.encoder(src_enc)  # [B, T, d_model]
    
        # prepare decoder tgt: masked positions get mask_token, visible keep embeddings
        tgt = src.clone()
        mask_token_full = self.mask_token.expand(B, T, -1)  # [B, T, d_model]
        tgt[mask_bool] = mask_token_full[mask_bool]
    
        # Add positional encoding also to tgt
        tgt = self.pos_encoder(tgt)
    
        # Decode
        decoded = self.decoder(tgt=tgt, memory=memory)  # [B, T, d_model]
    
        # Project back to ROI space
        out = self.output_proj(decoded)  # [B, T, num_rois]
    
        # Permute to [B, num_rois, T] for return, but apply LayerNorm over the ROI dimension:
        # first permute to [B, T, F] -> apply LayerNorm(num_rois) over last dim -> permute back to [B, F, T]
        out = out  # currently [B, T, F]
        out = self.final_ln(out)     # final_ln was defined as nn.LayerNorm(num_rois)
        out = out.permute(0, 2, 1).contiguous()  # [B, F, T]

        return out


# -------------------------
# Loss + training/validation
# -------------------------
def compute_loss(pred: torch.Tensor,
                 target: torch.Tensor,
                 mask_bool: torch.Tensor,
                 mask_loss_weight: float = 1.0,
                 unmask_loss_weight: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    pred, target: [B, F, T]
    mask_bool: [B, T]
    Returns total loss (weighted), masked_mse, unmasked_mse
    """
    device = pred.device
    B, F, T = pred.shape
    mask = mask_bool.unsqueeze(1).expand(-1, F, -1).to(device)  # [B, F, T]

    mse = FF.mse_loss(pred, target, reduction='none')  # [B, F, T]
    masked_mse = mse[mask].mean() if mask.any() else torch.tensor(0.0, device=device)
    unmasked_mse = mse[~mask].mean() if (~mask).any() else torch.tensor(0.0, device=device)
    loss = mask_loss_weight * masked_mse + unmask_loss_weight * unmasked_mse
    return loss, masked_mse.detach(), unmasked_mse.detach()


def train_one_epoch(model, dataloader, optimizer, device, epoch, mask_params, grad_clip=1.0):
    model.train()
    running_loss = 0.0
    running_masked = 0.0
    batches = len(dataloader)
    pbar = tqdm(enumerate(dataloader), total=batches, desc=f"Train E{epoch}")
    for i, batch in pbar:
        batch = batch.float().to(device)  # [B, F, T]
        B, F, T = batch.shape

        # generate point-wise masks
        n_mask_points = mask_params.get('n_mask_points', max(1, T // 10))
        mask_np = random_timepoint_mask(B, T, n_mask_points=n_mask_points)
        mask_t = torch.from_numpy(mask_np).to(device)

        optimizer.zero_grad()
        pred = model(batch, mask_t)
        loss, masked_mse, unmasked_mse = compute_loss(pred, batch, mask_t,
                                                      mask_loss_weight=0.7, #1.0
                                                      unmask_loss_weight=0.3) #0.0
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += loss.item()
        running_masked += float(masked_mse)
        pbar.set_postfix({'loss': running_loss / (i + 1), 'masked_mse': running_masked / (i + 1)})
    return running_loss / batches, running_masked / batches


def validate(model, dataloader, device, mask_params, idxs):
    model.eval()
    running_masked = 0.0
    running_unmasked = 0.0
    original = []
    predicted = []
    masks = []
    masked_mses = []
    unmasked_mses = []
    n = len(dataloader)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = batch.float().to(device)
            original.append(batch.cpu())
            B, F, T = batch.shape
            n_mask_points = mask_params.get('n_mask_points', max(1, T // 10))
            mask_np = random_timepoint_mask(B, T, n_mask_points=n_mask_points, idxs=idxs)
            mask_t = torch.from_numpy(mask_np).to(device)
            masks.append(mask_t.cpu())
            pred = model(batch, mask_t)
            predicted.append(pred.cpu())
            _, masked_mse, unmasked_mse = compute_loss(pred, batch, mask_t,
                                               mask_loss_weight=0.7, unmask_loss_weight=0.3) #1.0, 0.0
            for p, b, mt in zip(pred, batch, mask_t):
                _,m_mse,unm_mse = compute_loss(p.unsqueeze(0), b.unsqueeze(0), mt.unsqueeze(0),
                                       mask_loss_weight=0.7, unmask_loss_weight=0.3)
                masked_mses.append(m_mse.item())
                unmasked_mses.append(unm_mse.item())
            running_masked += float(masked_mse)
            running_unmasked += float(unmasked_mse)
    original = torch.cat(original, dim=0)
    predicted = torch.cat(predicted, dim=0)
    return running_masked / n, running_unmasked / n, original, predicted, mask_t, masked_mses, unmasked_mses


# -------------------------
# Main
# -------------------------
def main(config):
    device = torch.device(config['device'])
    np.random.seed(42)
    idxs = np.random.choice(
        config['T'], size=config['n_mask_points'], replace=False
    )
    for method in config['methods']:               
        path = config['path_data'] 
        
        data = np.load(os.path.join(path,'train_data_HCP_' + config['atlas'] + '.npz'))
        train_data_hcp = []
        for key in data:
            if data[key].shape[0] == 1200:
                temp = data[key][183:-184]
                train_data_hcp.append(temp)
        
        data = np.load(os.path.join(path,'val_data_HCP_' + config['atlas'] + '.npz'))
        val_data = []
        for key in data:
            if data[key].shape[0] == 1200:
                temp = data[key][183:-184]
                val_data.append(temp)
    
            
        data = np.load(os.path.join(path,'test_data_HCP_' + config['atlas'] + '.npz'))
        test_data = []
        for key in data:
            if data[key].shape[0] == 1200:
                temp = data[key][183:-184]
                test_data.append(temp)
        
        train_data_hcp = preprocess_HCP(train_data_hcp, config['case'])
        val_data = preprocess_HCP(val_data, config['case'])
        test_data = preprocess_HCP(test_data, config['case'])
        
        if method == "HCP":
            train_data = torch.from_numpy(np.stack(train_data_hcp))
        elif method == "CR-VAE":
            train_data = torch.load(os.path.join(config['path_augmented_crvae'], 'augmented_CR-VAE_' + config['case'] + '.pt'))
            train_data = train_data.view(-1, train_data.shape[2], train_data.shape[3])
            train_data = train_data.permute(0,2,1)
        elif method == "BOLDGEN":
            train_data = torch.load(os.path.join(config['path_augmented_boldgen'], 'augmented_BOLDGEN_' + config['case'] + '.pt'))
            train_data = train_data.view(-1, train_data.shape[2], train_data.shape[3])
            train_data = train_data.permute(0,2,1)
        elif method == "TimeVAE":
            train_data = np.load(os.path.join(config['path_augmented_timevae'], config['case'], 'timeVAE_' + config['case'] + '_prior_samples.npy'))
            train_data = torch.from_numpy(train_data).permute(0,2,1)
            train_data = (train_data - train_data.mean(dim=2, keepdim=True))/train_data.std(dim=2, keepdim=True)
        elif method == "TimeGAN":
            train_data = np.load(os.path.join(config['path_augmented_timegan'], 'augmented_TimeGAN_HCP_2026_' + config['case'] + '.npy'))
            train_data = torch.from_numpy(train_data).permute(0,2,1)
            train_data = (train_data - train_data.mean(dim=2, keepdim=True))/train_data.std(dim=2, keepdim=True)
        elif method == "T2I-Diff":
            train_data = np.load(os.path.join(config['path_augmented_t2idiff'], config['case'], 'gen_sig_' + config['case'] + '_fold1.npy'))
            train_data = torch.from_numpy(train_data).permute(0,2,1)
            train_data = (train_data - train_data.mean(dim=2, keepdim=True))/train_data.std(dim=2, keepdim=True)
        elif method == "DSFM":
            train_data = np.load(os.path.join(config['path_augmented_dsfm'], config['case'], 'samples', 'samples.npy'))
            train_data = torch.from_numpy(train_data).permute(0,2,1)
            train_data = (train_data - train_data.mean(dim=2, keepdim=True))/train_data.std(dim=2, keepdim=True)
        
        if config['data_scale'] == "small":
            train_data = train_data[:5000,:,:config['T']]
        elif config['data_scale'] == "medium":
            train_data = train_data[:10000,:,:config['T']]
        else:
            train_data = train_data[:,:,:config['T']]
        
        train_dataset = fMRIDataset(train_data)
        val_dataset = fMRIDataset(val_data)
        test_dataset = fMRIDataset(test_data)
    
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=False, num_workers=2)
    
        model_config = config['model_config']
        if model_config['n_enc_layers'] == 5:
            model_size = "heavyweight"
        elif model_config['n_enc_layers'] == 2:
            model_size = "lightweight"
            
        model = TransformerMaskRecon(num_rois=config['R'],
                                     d_model=config['R'],
                                     nhead=model_config['nheads'], #4
                                     num_encoder_layers=model_config['n_enc_layers'],
                                     num_decoder_layers=model_config['n_dec_layers'],
                                     dim_feedforward=4 * config['R'],
                                     dropout=model_config['dropout'],
                                     max_timepoints=config['T'])
        model.to(device)
    
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, config['epochs'] // 3), gamma=0.5)
    
        mask_params = {'n_mask_points': config['n_mask_points']}
        train_losses = []
        val_masked, val_unmasked = [], []
    
        best_val = float('inf')
        for epoch in range(1, config['epochs'] + 1):
            tr_loss, tr_masked = train_one_epoch(model, train_loader, optimizer, device, epoch, mask_params, grad_clip=1.0)
            scheduler.step()
            val_masked_mse, val_unmasked_mse,_,_,_,_,_ = validate(model, val_loader, device, mask_params, idxs=idxs)
    
            train_losses.append(tr_loss)
            val_masked.append(val_masked_mse)
            val_unmasked.append(val_unmasked_mse)
    
            print(f"Epoch {epoch:02d} | Train loss {tr_loss:.6f} masked_mse {tr_masked:.6f} | Val Masked {val_masked_mse:.6f} Val Unasked {val_unmasked_mse:.6f}")
    
            if val_masked_mse + val_unmasked_mse < best_val:
                best_val = val_masked_mse + val_unmasked_mse
                best_state_dict = copy.deepcopy(model.state_dict())
    
        print("Training finished. Best val loss:", best_val)
        
        model = TransformerMaskRecon(num_rois=config['R'],
                                     d_model=config['R'],
                                     nhead=model_config['nheads'], #4
                                     num_encoder_layers=model_config['n_enc_layers'],
                                     num_decoder_layers=model_config['n_dec_layers'],
                                     dim_feedforward=4 * config['R'],
                                     dropout=model_config['dropout'],
                                     max_timepoints=config['T'])
        model.to(device)
        model.load_state_dict(best_state_dict)
        test_loss, test_masked_mse, original, predicted, mask_t, masked_mses, unmasked_mses = validate(model, test_loader, device, mask_params, idxs)
    
        results = {}
        results['original'] = original.numpy()
        results['pred'] = predicted.numpy()
        results['mask'] = idxs
        results['test_masked_mse'] = masked_mses
        results['test_unmasked_mse'] = unmasked_mses
        
        if config['save_models']:
            if not os.path.exists(os.path.join(config['path_save'],'models','MSM', model_size + '_' + config['data_scale'], config['case'])):
                os.makedirs(os.path.join(config['path_save'],'models','MSM', model_size + '_' + config['data_scale'], config['case']),exist_ok=True)   
            torch.save(best_state_dict, os.path.join(config['path_save'],'models','MSM', model_size + '_' + config['data_scale'], config['case'],'model_' + method + '.pth'))
        
        if config['save_results']:
            os.makedirs(os.path.join(config['path_save'], 'MSM', model_size + '_' + config['data_scale'], config['case'], method), exist_ok=True)
            with open(os.path.join(config['path_save'], 'MSM', model_size + '_' + config['data_scale'], config['case'], method, 'results.pkl'), 'wb') as f:
                pickle.dump(results,f)
        
        for sample_i in range(1,11):
            rois = [0, 99]
            t = np.arange(config['T'])
            
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            for ax in axes:
                ax.tick_params(axis='both', which='major', labelsize=18)
            
            for ax, roi_i in zip(axes, rois):
                ax.plot(
                    t,
                    original.cpu().numpy()[sample_i, roi_i],
                    label='target'
                )
                ax.plot(
                    t,
                    predicted.cpu().numpy()[sample_i, roi_i],
                    label='pred',
                    alpha=0.8
                )
            
                masked = mask_t.cpu().numpy()[sample_i]
                ax.scatter(
                    t[masked],
                    original.cpu().numpy()[sample_i, roi_i][masked],
                    color='red',
                    s=20,
                    label='masked',
                    zorder=3
                )
            
                ax.set_title(f"ROI {roi_i + 1}", fontsize=22)
            
            # Legend only in the first subplot
            #axes[0].legend(loc='upper right', fontsize=12)
            
            axes[-1].set_xlabel('Time (sec)', fontsize=22)
            
            plt.tight_layout()
            
            if config['save_plots']:
                save_dir = os.path.join('plots', 'reconstructed_plots', model_size + '_' + config['data_scale'], config['case'], method)
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(
                    os.path.join(save_dir, method + '_subject_' + str(sample_i) + '_rois.png'),
                    dpi=600,
                    bbox_inches='tight'
                )
            
            plt.show()


    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate Transformer masked sequence modeling on augmented HCP data.')
    parser.add_argument('--path_data', type=str, required=True)
    parser.add_argument('--path_save', type=str, required=True)
    parser.add_argument('--path_augmented_boldgen', type=str, required=True)
    parser.add_argument('--path_augmented_crvae', type=str, required=True)
    parser.add_argument('--path_augmented_timevae', type=str, required=True)
    parser.add_argument('--path_augmented_timegan', type=str, required=True)
    parser.add_argument('--path_augmented_t2idiff', type=str, required=True)
    parser.add_argument('--path_augmented_dsfm', type=str, required=True)
    parser.add_argument('--methods', nargs="+", type=str, default=['HCP', 'TimeVAE', 'TimeGAN','CR-VAE','T2I-Diff', 'DSFM','BOLDGEN'])
    parser.add_argument('--case', type=str, default="full_subsampled")
    parser.add_argument('--TR', type=float, default=0.72)
    parser.add_argument('--newTR', type=float, default=2.0)
    parser.add_argument('--R', type=int, default=100)
    parser.add_argument('--atlas', type=str, default='SCH')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--data_scale', type=str, default="large")
    parser.add_argument('--model_n_enc_layers', type=int, default=2)
    parser.add_argument('--model_n_dec_layers', type=int, default=2)
    parser.add_argument('--model_nheads', type=int, default=4)
    parser.add_argument('--model_dropout', type=float, default=0.1)
    parser.add_argument('--device', type=str, default="cuda:1")
    parser.add_argument('--save_results', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--save_models', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--save_plots', action=argparse.BooleanOptionalAction, default=True)
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
    config['case'] = args.case
    config['TR'] = args.TR
    config['newTR'] = args.newTR
    if config['case'] == "full_original":
        config['T'] = 833
    elif config['case'] == "half_subsampled":
        config['T'] = 210
    elif config['case'] == "half_original":
        config['T'] = 583
    elif config['case'] == "full_subsampled":
        config['T'] = 300
    config['R'] = args.R
    config['atlas'] = args.atlas
    config['batch_size'] = args.batch_size
    config['epochs'] = args.epochs
    config['lr'] = args.lr
    config['weight_decay'] = args.weight_decay
    config['n_mask_points'] = config['T'] // 4
    config['data_scale'] = args.data_scale
    config['model_config'] = {}
    config['model_config']['n_enc_layers'] = args.model_n_enc_layers
    config['model_config']['n_dec_layers'] = args.model_n_dec_layers
    config['model_config']['nheads'] = args.model_nheads
    config['model_config']['dropout'] = args.model_dropout
    config['device'] = args.device
    config['save_results'] = args.save_results
    config['save_models'] = args.save_models
    config['save_plots'] = args.save_plots
    main(config)
    
