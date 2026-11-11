"""Generate subject-specific augmented HCP BOLD signals (Module 2)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm 
from utils import rDCM_generate
import argparse
import pickle
from scipy.signal import butter, filtfilt, resample

def welch_psd(x, nperseg=1024, noverlap=None, nfft=None, window="hann", eps=1e-12):
    """
    Computes Welch’s power spectral density (PSD) estimate for batched multichannel signals.

    Args:
        x (torch.Tensor): Input tensor of shape (N, C, T), (C, T), or (T,).
        fs (float): Sampling frequency.
        nperseg (int): Window length.
        noverlap (int or None): Number of overlapping samples. Defaults to nperseg // 2.
        nfft (int or None): FFT length. Defaults to nperseg.
        window (str): Window type ("hann" or "hamming").
        eps (float): Numerical stability guard.

    Returns:
        freqs (torch.Tensor): Frequency vector of shape (F,)
        psd (torch.Tensor): Power spectral density, shape (N, C, F)
    """
    # ------------------------------------------------------
    # Handle shape: (T,), (C, T), or (N, C, T)
    if x.dim() == 1:
        x = x.unsqueeze(0).unsqueeze(0)   # (1, 1, T)
    elif x.dim() == 2:
        x = x.unsqueeze(0)                # (1, C, T)
    N, C, T = x.shape

    # ------------------------------------------------------
    if noverlap is None:
        noverlap = nperseg // 2
    if nfft is None:
        nfft = nperseg
    step = nperseg - noverlap

    nseg = 1 + max(0, (T - nperseg) // step)
    if nseg <= 0:
        raise ValueError("Signal too short for given nperseg/noverlap.")

    # ------------------------------------------------------
    # Window creation
    if window == "hann":
        w = torch.hann_window(nperseg, periodic=True, device=x.device, dtype=x.dtype)
    elif window == "hamming":
        w = torch.hamming_window(nperseg, periodic=True, device=x.device, dtype=x.dtype)
    else:
        raise ValueError("Unsupported window type.")
    scale = w.pow(2).sum()  # normalization factor

    # ------------------------------------------------------
    # Build frame indices: (nseg, nperseg)
    idx0 = torch.arange(0, nseg * step, step, device=x.device)
    idx = idx0.unsqueeze(-1) + torch.arange(nperseg, device=x.device)  # (nseg, nperseg)

    # Extract frames: (N, C, nseg, nperseg)
    frames = x[..., idx]  # shape broadcasting magic

    # Detrend (remove mean per frame)
    frames = frames - frames.mean(dim=-1, keepdim=True)

    # Apply window
    frames = frames * w.view(1, 1, 1, -1)

    # ------------------------------------------------------
    # FFT and PSD computation
    Xf = torch.fft.rfft(frames, n=nfft, dim=-1)             # (N, C, nseg, F)
    ps = (Xf.abs() ** 2) / (scale)                    # Power spectral density
    psd = ps.mean(dim=2)                                   # Average over segments → (N, C, F)

    # Frequency vector
    #freqs = torch.fft.rfftfreq(nfft, d=1.0 / fs).to(x.device)

    # Numerical guard
    psd = psd.clamp_min(eps)

    return psd


# --------------------------
# Full Model + Loss
# --------------------------
class fALFFDecoderModel(nn.Module):
    def __init__(self, A, transit, decay, epsilon, beta, TR, R, T, reps, wind, overlap):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(reps, R, T*16))
        self.A = A
        self.transit = transit
        self.decay = decay
        self.epsilon = epsilon
        self.beta = beta
        self.TR = TR
        self.R = R
        self.T = T
        self.wind = wind
        self.overlap = overlap

    def forward(self):
        y_gen, x_gen = rDCM_generate(self.T,
                              self.R,
                              self.TR,
                              self.beta,
                              self.embedding.permute(0,2,1),
                              self.A,
                              0.0,
                              self.transit,
                              self.decay,
                              self.epsilon)
        X_gen = welch_psd(x_gen.permute(0,2,1), self.wind, self.overlap)
        return y_gen, X_gen


def rmssd(x: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """
    Compute ROI-wise RMSSD (root mean square of successive differences).

    Args:
        x: Tensor of shape (B, T, R) where
           B = batch size, T = time points, R = ROIs.
        eps: Optional small value added inside sqrt for numerical stability.

    Returns:
        Tensor of shape (B, R) containing RMSSD per ROI for each batch item.

    Formula:
        RMSSD = sqrt( mean_t( (x[t] - x[t-1])^2 ) )
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x to have shape (B, T, R); got {tuple(x.shape)}")
    B, T, R = x.shape
    if T < 2:
        raise ValueError(f"T must be >= 2 to compute successive differences; got T={T}")

    # successive differences along time: (B, T-1, R)
    dx = x[:, 1:, :] - x[:, :-1, :]

    # mean squared successive differences over time: (B, R)
    mssd = dx.pow(2).mean(dim=1)

    # rmssd: (B, R)
    return torch.sqrt(mssd + eps)



def loss_function(X_gen, X, Y_gen, Y, y_gen, y, epoch): 
    triu_indices = torch.triu_indices(y_gen.shape[0], y_gen.shape[0], offset=1)
    #FC = torch.corrcoef(y.T)
    #fc_loss = 0.0
    #for y_g in y_gen:
    #    FC_gen = torch.corrcoef(y_g.T)
    #    fc_loss += 1-torch.corrcoef(torch.stack([FC[triu_indices[0], triu_indices[1]],FC_gen[triu_indices[0], triu_indices[1]]]))[0,1]
    #fc_loss = 10*fc_loss/y_gen.shape[0]    
    mse_states = F.mse_loss(X_gen, X.expand(X_gen.shape[0],-1,-1))*100000
    mse_bold = F.mse_loss(Y_gen, Y.expand(Y_gen.shape[0],-1,-1))
    mse_rmssd = F.mse_loss(rmssd(y.unsqueeze(0)).repeat(y_gen.shape[0],1), rmssd(y_gen))*100
    y_gen = y_gen.contiguous().view(y_gen.shape[0], -1)
    FC_samples = torch.corrcoef(y_gen)
    triu_indices = torch.triu_indices(y_gen.shape[0], y_gen.shape[0], offset=1)
    similarity_loss = 10*torch.sum(torch.abs(FC_samples[triu_indices[0], triu_indices[1]]))/(y_gen.shape[0]*(y_gen.shape[0]-1)/2)
    if epoch % 10  == 0:
        print('')
        print('MSE states:', mse_states.item()) 
        print('MSE BOLD:', mse_bold.item()) 
        #print('FC Loss:', fc_loss.item())
        print('Similarity Loss:', similarity_loss.item())
        print('rMSSD Loss:', mse_rmssd.item())
    return mse_states + mse_bold + similarity_loss + mse_rmssd #+fc_loss


# --------------------------
# Example usage
# --------------------------
def main(config):
    TR = config['TR']
    device = torch.device(config['device'])      

    names = []
    with open(os.path.join(config['data_path'],'names_train.txt'), 'r') as f:
        for line in f:
            temp = line.strip()
            names.append(temp)
            
    data = np.load(os.path.join(config['data_path'],'train_data_HCP_' + config['atlas'] + '.npz'))
    train_data = []
    names_new = []
    for key,name in zip(data,names):
        if data[key].shape[0] == 1200:
            if config['case'] == "full_original":
                y = data[key][183:-184]
                y = (y - np.mean(y, axis=0, keepdims=True))/(np.std(y, axis=0, keepdims=True) + 1e-6)
                order = 4
                cutoff = 0.1
                fs = 1/TR
                newTR = TR
                b, a = butter(order, cutoff, btype='low', fs=fs)
                y = filtfilt(b, a, y.T).T.astype('float32')
            elif config['case'] == "half_original":
                y = data[key][183:-184]
                y = y[125:708]
                y = (y - np.mean(y, axis=0, keepdims=True))/(np.std(y, axis=0, keepdims=True) + 1e-6)
                order = 4
                cutoff = 0.1
                fs = 1/TR
                newTR = TR
                b, a = butter(order, cutoff, btype='low', fs=fs)
                y = filtfilt(b, a, y.T).T.astype('float32')
            elif config['case'] == "full_subsampled":
                y = data[key][183:-184]
                y = (y - np.mean(y, axis=0, keepdims=True))/(np.std(y, axis=0, keepdims=True) + 1e-6)
                order = 4
                cutoff = 0.1
                fs = 1/TR
                b, a = butter(order, cutoff, btype='low', fs=fs)
                y = filtfilt(b, a, y.T).T.astype('float32')
                newTR = 2
                rate = newTR/TR
                new_samples = int(np.round(y.shape[0] / rate))
                y = resample(y, new_samples)
                y = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
            elif config['case'] == "half_subsampled":
                y = data[key][183:-184]
                y = y[125:708]
                y = (y - np.mean(y, axis=0, keepdims=True))/(np.std(y, axis=0, keepdims=True) + 1e-6)
                order = 4
                cutoff = 0.1
                fs = 1/TR
                b, a = butter(order, cutoff, btype='low', fs=fs)
                y = filtfilt(b, a, y.T).T.astype('float32')
                newTR = 2
                rate = newTR/TR
                new_samples = int(np.round(y.shape[0] / rate))
                y = resample(y, new_samples)
                y = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
            if not np.any(np.isnan(y)):
                train_data.append(torch.from_numpy(y))
                names_new.append(name)
                
    
    T,R = train_data[0].shape
    if not os.path.exists(os.path.join(config['save_path'], 'BOLDGEN_Module1_results_gathered', config['case'], 'fmris.pt')):
        path = os.path.join(config['save_path'],'neural_noise', config['case'])
        dire = os.listdir(path)
        neural_noises = torch.zeros(len(names_new),T*16,R)
        As = torch.zeros(len(names_new),R,R)
        betas = torch.zeros(len(names_new))
        transits = torch.zeros(len(names_new),R,1)
        decays = torch.zeros(len(names_new),R,1)
        epsilons =  torch.zeros(len(names_new))
        fmris = torch.zeros(len(names_new),T,R)
        names_BOLDGEN = []
        for temp in names_new:
            names_BOLDGEN.append(temp)
        names_BOLDGEN = np.array(names_BOLDGEN, dtype=str)
        for i,file in enumerate(dire):
            try:
                with open(os.path.join(path, file), 'rb') as f:
                    data = pickle.load(f)
                pos = np.where(np.array(names_new) == file[:-4])
                if file[:-4] in names_new:
                    try:
                        fmris[i] = train_data[pos[0][0]]
                        names_BOLDGEN[i] = np.array(names_new)[pos[0][0]]
                        neural_noises[i] = data['neural_noises'][-1]
                        As[i] = data['outputs'][-1][0]['Ep']['A']
                        betas[i] = data['betas'][-1]
                        transits[i] = data['outputs'][-1][0]['Ep']['transit']
                        decays[i] = data['outputs'][-1][0]['Ep']['decay']
                        epsilons[i] = data['outputs'][-1][0]['Ep']['epsilon']
                    except:
                        print(i)
            except:
                print(i)
        
        pos_delete = torch.unique(torch.where(torch.all(fmris == 0, dim=1))[0])
        mask = torch.ones(len(names_new), dtype=torch.bool, device=neural_noises.device)
        mask[pos_delete] = False
        neural_noises = neural_noises[mask]
        As = As[mask]
        betas = betas[mask]
        transits = transits[mask]
        decays = decays[mask]
        epsilons = epsilons[mask]
        fmris = fmris[mask]
        names_BOLDGEN = names_BOLDGEN[mask]
        names_BOLDGEN = names_BOLDGEN.tolist()
        
    
        os.makedirs(os.path.join(config['save_path'], 'BOLDGEN_Module1_results_gathered', config['case']), exist_ok=True)
        torch.save(neural_noises, os.path.join(config['save_path'], 'BOLDGEN_Module1_results_gathered', config['case'], 'neural_noises.pt'))
        torch.save(As, os.path.join(config['save_path'], 'BOLDGEN_Module1_results_gathered', config['case'], 'As.pt'))
        torch.save(betas, os.path.join(config['save_path'], 'BOLDGEN_Module1_results_gathered', config['case'], 'betas.pt'))
        torch.save(transits, os.path.join(config['save_path'], 'BOLDGEN_Module1_results_gathered', config['case'], 'transits.pt'))
        torch.save(decays, os.path.join(config['save_path'], 'BOLDGEN_Module1_results_gathered', config['case'], 'decays.pt'))
        torch.save(epsilons, os.path.join(config['save_path'], 'BOLDGEN_Module1_results_gathered', config['case'], 'epsilons.pt'))
        torch.save(fmris, os.path.join(config['save_path'], 'BOLDGEN_Module1_results_gathered', config['case'], 'fmris.pt'))
        with open(os.path.join(config['save_path'], 'BOLDGEN_Module1_results_gathered', config['case'],'names_BOLDGEN.txt'), 'w') as f:
            for item in names_BOLDGEN:
                f.write("%s\n" % item)
    else:
        neural_noises = torch.load(os.path.join(config['save_path'], 'BOLDGEN_Module1_results_gathered', config['case'], 'neural_noises.pt'))
        As = torch.load(os.path.join(config['save_path'], 'BOLDGEN_Module1_results_gathered', config['case'], 'As.pt'))
        betas = torch.load(os.path.join(config['save_path'], 'BOLDGEN_Module1_results_gathered', config['case'], 'betas.pt'))
        transits = torch.load(os.path.join(config['save_path'], 'BOLDGEN_Module1_results_gathered', config['case'], 'transits.pt'))
        decays = torch.load(os.path.join(config['save_path'], 'BOLDGEN_Module1_results_gathered', config['case'], 'decays.pt'))
        epsilons = torch.load(os.path.join(config['save_path'], 'BOLDGEN_Module1_results_gathered', config['case'], 'epsilons.pt'))
        fmris = torch.load(os.path.join(config['save_path'], 'BOLDGEN_Module1_results_gathered', config['case'], 'fmris.pt'))
        names_BOLDGEN = []
        with open(os.path.join(config['save_path'], 'BOLDGEN_Module1_results_gathered', config['case'],'names_BOLDGEN.txt'), 'r') as f:
            for line in f:
                temp = line.strip()
                names_BOLDGEN.append(temp)
        
    neural_noises = neural_noises[config['start']:config['finish']]
    As = As[config['start']:config['finish']]
    betas = betas[config['start']:config['finish']]
    transits = transits[config['start']:config['finish']]
    decays = decays[config['start']:config['finish']]
    epsilons = epsilons[config['start']:config['finish']]
    fmris = fmris[config['start']:config['finish']]
    names_BOLDGEN = names_BOLDGEN[config['start']:config['finish']]
    
    epochs = config['epochs']
    for i, (y, neural_noise, A, beta, transit, decay, epsilon, name) in enumerate(zip(fmris, neural_noises, As, betas, transits, decays, epsilons, names_BOLDGEN)):
        if not os.path.exists(os.path.join(config['save_path'], 'augmented_BOLDGEN_noFCloss', config['case'], name + '.pt')):
            T,R = y.shape
            _,x = rDCM_generate(T,
                                R,
                                newTR,
                                beta,
                                neural_noise.unsqueeze(dim=0),
                                A,
                                0.0,
                                transit,
                                decay,
                                epsilon)
            X = welch_psd(x.double().T, 4*int(T/4), 4*int(T/8))
            y = y.double()
            Y = welch_psd(y.T, int(T/4), int(T/8))
            model = fALFFDecoderModel(A=A,
                                      transit=transit,
                                      decay=decay,
                                      epsilon=epsilon,
                                      beta=beta.to(device),
                                      TR=newTR,
                                      R=R, 
                                      T=T,
                                      reps=config['reps'],
                                      wind=4*int(T/4),
                                      overlap=4*int(T/8)).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-3)
            min_loss = 10e20
            with tqdm(total=epochs, desc='(T)') as pbar:
                for epoch in range(epochs):
                    # Training step example
                    model.train()
                    optimizer.zero_grad()
                    y_gen, X_gen = model()
                    Y_gen = welch_psd(y_gen.permute(0,2,1), int(T/4), int(T/8))
                    loss = loss_function(X_gen, X.to(device), Y_gen, Y.to(device), y_gen, y.to(device), epoch)
                    loss.backward()
                    optimizer.step()
                    pbar.set_postfix({'loss': loss.item()})
                    pbar.update()
                    if loss.item() < min_loss:
                        min_loss = loss.item()
                        y_gen_best = y_gen.detach().cpu()
            os.makedirs(os.path.join(config['save_path'], 'augmented_BOLDGEN_noFCloss', config['case']), exist_ok=True)
            torch.save(y_gen_best, os.path.join(config['save_path'], 'augmented_BOLDGEN_noFCloss', config['case'], name + '.pt'))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run')
    parser.add_argument('--data_path', type=str, 
                        required=True)
    parser.add_argument('--save_path', type=str, 
                        required=True)
    parser.add_argument('--atlas', type=str, default='SCH')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--TR', type=float, default=0.72)
    parser.add_argument('--reps', type=int, default=15)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--finish', type=int, default=1250)
    parser.add_argument('--case', type=str, default="full_subsampled")
    parser.add_argument('--device', type=str, default='cuda:2')
    args = parser.parse_args()
    config = {
        'data_path': args.data_path,
        'save_path': args.save_path,
        'atlas': args.atlas,
        'epochs': args.epochs,
        'TR': args.TR,
        'reps': args.reps,
        'start': args.start,
        'finish': args.finish,
        'case': args.case,
        'device': args.device
        }
    main(config)


