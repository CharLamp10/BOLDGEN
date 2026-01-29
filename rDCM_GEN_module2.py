import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm 
from utils import rDCM_generate
import argparse


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


def welch_cpsd(x, nperseg=1024, noverlap=None, nfft=None, window="hann", eps=1e-12):
    """
    Computes Welch’s cross power spectral density (CPSD) estimate for batched multichannel signals.

    Args:
        x (torch.Tensor): Input tensor of shape (N, C, T), (C, T), or (T,).
        nperseg (int): Window length.
        noverlap (int or None): Number of overlapping samples. Defaults to nperseg // 2.
        nfft (int or None): FFT length. Defaults to nperseg.
        window (str): Window type ("hann" or "hamming").
        eps (float): Diagonal jitter for numerical stability (added on channel-diagonal).

    Returns:
        cpsd (torch.Tensor): Cross power spectral density, shape (N, C, C, F)
                             cpsd[n, i, j, f] = S_{ij}(f) between channel i and j.
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
    frames = x[..., idx]  # (N, C, nseg, nperseg)

    # Detrend (remove mean per frame)
    frames = frames - frames.mean(dim=-1, keepdim=True)

    # Apply window
    frames = frames * w.view(1, 1, 1, -1)

    # ------------------------------------------------------
    # FFT
    Xf = torch.fft.rfft(frames, n=nfft, dim=-1)  # (N, C, nseg, F)
    F = Xf.shape[-1]

    # ------------------------------------------------------
    # CPSD computation:
    # S_ij(f) = (1 / (scale * nseg)) * sum_s X_i(s, f) * conj(X_j(s, f))
    # Use einsum to sum over segment dimension.
    nseg = Xf.shape[2]
    norm = scale * nseg

    # Xf: (N, C, nseg, F)
    # -> cpsd: (N, C, C, F)
    cpsd = torch.einsum("ncsf,nksf->nckf", Xf, Xf.conj()) / norm

    # Optional numerical guard: add jitter on the diagonal (per frequency)
    if eps is not None and eps > 0:
        eye = torch.eye(C, device=x.device, dtype=Xf.dtype).view(1, C, C, 1)
        cpsd = cpsd + eps * eye

    return cpsd.abs()




# --------------------------
# Full Model + Loss
# --------------------------
class fALFFDecoderModel(nn.Module):
    def __init__(self, A, transit, decay, epsilon, beta, TR, R, T, reps):
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
        #Y_gen = welch_psd(y_gen.permute(0,2,1), 512, 256)
        X_gen = welch_psd(x_gen.permute(0,2,1), 512, 256)
        return y_gen, X_gen


def loss_function(X_gen, X, Y_gen, Y, y_gen, y, epoch): 
    triu_indices = torch.triu_indices(y_gen.shape[0], y_gen.shape[0], offset=1)
    FC = torch.corrcoef(y.T)
    loss2 = 0.0
    for y_g in y_gen:
        FC_gen = torch.corrcoef(y_g.T)
        loss2 += 1-torch.corrcoef(torch.stack([FC[triu_indices[0], triu_indices[1]],FC_gen[triu_indices[0], triu_indices[1]]]))[0,1]
    loss2 = loss2/y_gen.shape[0]    
    mse_loss1 = F.mse_loss(X_gen, X.expand(X_gen.shape[0],-1,-1))*10000
    mse_loss2 = F.mse_loss(Y_gen, Y.expand(Y_gen.shape[0],-1,-1))
    y_gen = y_gen.contiguous().view(y_gen.shape[0], -1)
    FC_samples = torch.corrcoef(y_gen)
    triu_indices = torch.triu_indices(y_gen.shape[0], y_gen.shape[0], offset=1)
    loss3 = torch.sum(torch.abs(FC_samples[triu_indices[0], triu_indices[1]]))/(y_gen.shape[0]*(y_gen.shape[0]-1)/2)
    if epoch % 10  == 0:
        print('')
        print('MSE1:', mse_loss1.item()) 
        print('MSE2:', mse_loss2.item()) 
        print('Loss 2:', loss2.item())
        print('Loss 3:', loss3.item())
    return mse_loss1 + mse_loss2 + loss2 + loss3


# --------------------------
# Example usage
# --------------------------
def main(config):
    device = torch.device(config['device'])
    neural_noise = torch.load(os.path.join(config['save_path'], 'neural_noise', 'neural_noises.pt'))
    A = torch.load(os.path.join(config['save_path'], 'neural_noise', 'As.pt'))
    beta = torch.load(os.path.join(config['save_path'], 'neural_noise', 'betas.pt'))
    transit = torch.load(os.path.join(config['save_path'], 'neural_noise', 'transits.pt'))
    decay = torch.load(os.path.join(config['save_path'], 'neural_noise', 'decays.pt'))
    epsilon = torch.load(os.path.join(config['save_path'], 'neural_noise', 'epsilons.pt'))
    names_rdcm = []
    with open(os.path.join(config['save_path'], 'neural_noise','names_rdcm.txt'), 'r') as f:
        for line in f:
            temp = line.strip()
            names_rdcm.append(temp[:-4])

    names = []
    with open(os.path.join(config['data_path'],'names_train.txt'), 'r') as f:
        for line in f:
            temp = line.strip()
            names.append(temp[:-4])
            
    data = np.load(os.path.join(config['data_path'],'train_data_HCP_' + config['atlas'] + '_resampled.npz'))
    train_data = []
    for key in data:
        y = data[key][88:-88]
        y = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
        train_data.append(torch.from_numpy(y))
    
    T,R = train_data[0].shape
    neural_noises = torch.zeros(len(names),T*16,R)
    As = torch.zeros(len(names),R,R)
    betas = torch.zeros(len(names))
    transits = torch.zeros(len(names),R,1)
    decays = torch.zeros(len(names),R,1)
    epsilons =  torch.zeros(len(names))
    fmris = torch.zeros(len(names),T,R)
    names_final = []
    pos_delete = []
    for i,(name,fmri) in enumerate(zip(names,train_data)):
        pos = np.where(np.array(names_rdcm) == name)[0]
        if len(pos) > 0:
            neural_noises[i] = neural_noise[pos[0]]
            As[i] = A[pos[0]]
            betas[i] = beta[pos[0]]
            transits[i] = transit[pos[0]]
            decays[i] = decay[pos[0]]
            epsilons[i] = epsilon[pos[0]]
            fmris[i] = fmri
            names_final.append(name)
        else:
             pos_delete.append(i)

    pos_delete = np.unique(np.array(pos_delete))
    mask = torch.ones(len(names), dtype=torch.bool, device=neural_noises.device)
    mask[pos_delete] = False
    neural_noises = neural_noises[mask]
    As = As[mask]
    betas = betas[mask]
    transits = transits[mask]
    decays = decays[mask]
    epsilons = epsilons[mask]
    fmris = fmris[mask]
    
    neural_noises = neural_noises[config['start']:config['finish']]
    As = As[config['start']:config['finish']]
    betas = betas[config['start']:config['finish']]
    transits = transits[config['start']:config['finish']]
    decays = decays[config['start']:config['finish']]
    epsilons = epsilons[config['start']:config['finish']]
    fmris = fmris[config['start']:config['finish']]
    names_final = names_final[config['start']:config['finish']]
    with open(os.path.join(config['save_path_new'],'names_rDCM.txt'), 'w') as f:
        for item in names_final:
            f.write("%s\n" % item)
    
    TR = config['TR']
    epochs = config['epochs']
    for (y, neural_noise, A, beta, transit, decay, epsilon, name) in zip(fmris, neural_noises, As, betas, transits, decays, epsilons, names_final):
        if not os.path.exists(os.path.join(config['save_path_new'], 'augmented_new', name + '.pt')):
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
            X = welch_psd(x.double().T, 512, 256)
            y = y.double()
            Y = welch_psd(y.T, 128, 64)
            model = fALFFDecoderModel(A=A,
                                      transit=transit,
                                      decay=decay,
                                      epsilon=epsilon,
                                      beta=beta.to(device),
                                      TR=TR,
                                      R=R, 
                                      T=T,
                                      reps=config['reps']).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
            with tqdm(total=epochs, desc='(T)') as pbar:
                for epoch in range(epochs):
                    # Training step example
                    model.train()
                    optimizer.zero_grad()
                    y_gen, X_gen = model()
                    Y_gen = welch_psd(y_gen.permute(0,2,1), 128, 64)
                    loss = loss_function(X_gen, X.to(device), Y_gen, Y.to(device), y_gen, y.to(device), epoch)
                    loss.backward()
                    optimizer.step()
                    pbar.set_postfix({'loss': loss.item()})
                    pbar.update()  
            os.makedirs(os.path.join(config['save_path_new'], 'augmented_new'), exist_ok=True)
            torch.save(y_gen, os.path.join(config['save_path_new'], 'augmented_new', name + '.pt'))
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run')
    parser.add_argument('--data_path', type=str, 
                        default=r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_GNN_data/HCP')
    parser.add_argument('--save_path', type=str, 
                        default=r'/media/student1/Extreme Pro/VarCoNet_GNN_results')
    parser.add_argument('--save_path_new', type=str, 
                        default=r'/media/student1/Extreme Pro')
    parser.add_argument('--atlas', type=str, default='SCH')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--TR', type=float, default=1.5)
    parser.add_argument('--reps', type=int, default=15)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--finish', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    config = {
        'data_path': args.data_path,
        'save_path': args.save_path,
        'save_path_new': args.save_path_new,
        'atlas': args.atlas,
        'epochs': args.epochs,
        'TR': args.TR,
        'reps': args.reps,
        'start': args.start,
        'finish': args.finish,
        'device': args.device
        }
    main(config)


