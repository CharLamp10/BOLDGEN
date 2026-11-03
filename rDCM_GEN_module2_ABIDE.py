import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm 
from utils import rDCM_generate
import argparse
import pickle

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
    device = torch.device(config['device'])
    data_path = os.path.join(config['data_path'], 'ROIsignals_SCH_' + config['dataset'])
    data_dir = os.listdir(data_path)    
    neural_noise_path = os.path.join(config['save_path'],'neural_noise', config['dataset'])     
    neural_noise_dir = os.listdir(neural_noise_path)
    neural_noise_dir = neural_noise_dir[config['start']:config['finish']]       
    epochs = config['epochs']
    for i, name in enumerate(neural_noise_dir):
        if not os.path.exists(os.path.join(config['save_path'], 'augmented_rDCM-GEN_noFCloss', config['dataset'], name + '.pt')):
            pos = np.where(np.array(data_dir) == name[:-4])[0][0]
            y = np.load(os.path.join(data_path, data_dir[pos], 'ROISignals.npy'))
            y = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
            TR = np.load(os.path.join(data_path, data_dir[pos], 'TR.npy'))
            with open(os.path.join(neural_noise_path, name), 'rb') as f:
                neural_noises = pickle.load(f)
            neural_noise = neural_noises['neural_noises'][-1]
            A = neural_noises['outputs'][-1][0]['Ep']['A']
            beta = neural_noises['betas'][-1]
            transit = neural_noises['outputs'][-1][0]['Ep']['transit']
            decay = neural_noises['outputs'][-1][0]['Ep']['decay']
            epsilon = neural_noises['outputs'][-1][0]['Ep']['epsilon']
            T,R = y.shape
            _,x = rDCM_generate(T,
                                R,
                                TR,
                                beta,
                                neural_noise.unsqueeze(dim=0),
                                A,
                                0.0,
                                transit,
                                decay,
                                epsilon)
            X = welch_psd(x.double().T, 4*int(T/4), 4*int(T/8))
            y = torch.from_numpy(y).double()
            Y = welch_psd(y.T, int(T/4), int(T/8))
            model = fALFFDecoderModel(A=A,
                                      transit=transit,
                                      decay=decay,
                                      epsilon=epsilon,
                                      beta=beta.to(device),
                                      TR=TR,
                                      R=R, 
                                      T=T,
                                      reps=config['reps'],
                                      wind=4*int(T/4),
                                      overlap=4*int(T/8)).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
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
            os.makedirs(os.path.join(config['save_path'], 'augmented_rDCM-GEN_noFCloss', config['dataset']), exist_ok=True)
            torch.save(y_gen_best, os.path.join(config['save_path'], 'augmented_rDCM-GEN_noFCloss', config['dataset'], name + '.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run')
    parser.add_argument('--data_path', type=str, 
                        default=r'/media/student1/ExtremePro/rDCM-GEN/data')
    parser.add_argument('--dataset', type=str, 
                        default='ABIDEI')
    parser.add_argument('--save_path', type=str, 
                        default=r'/media/student1/ExtremePro/rDCM-GEN/rDCM-GEN_results')
    parser.add_argument('--atlas', type=str, default='SCH')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--reps', type=int, default=15)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--finish', type=int, default=1235)
    parser.add_argument('--device', type=str, default='cuda:1')
    args = parser.parse_args()
    config = {
        'data_path': args.data_path,
        'dataset': args.dataset,
        'save_path': args.save_path,
        'atlas': args.atlas,
        'epochs': args.epochs,
        'reps': args.reps,
        'start': args.start,
        'finish': args.finish,
        'device': args.device
        }
    main(config)


