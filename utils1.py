"""Provide additional rDCM and neural-noise generation utilities."""

import torch
import numpy as np
from rdcm import tapas_rdcm_model_specification, tapas_rdcm_set_options,tapas_rdcm_get_convolution_bm,tapas_dcm_euler_make_indices
import torch.fft as fft


def safe_clone(obj):
    """
    Recursively clones a DCM-like structure containing nested dicts, lists, and tensors.
    Works even when copy.deepcopy fails (e.g., GPU tensors, lambda objects).
    """
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    elif isinstance(obj, dict):
        return {k: safe_clone(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_clone(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(safe_clone(v) for v in obj)
    else:
        # For non-tensor leaf nodes (numbers, strings, None, etc.)
        return obj

def tapas_dcm_euler_gen(DCM, Ep):
    """
    Batched version of tapas_dcm_euler_gen.

    Args:
        DCM (dict): Model structure.
        Ep (dict): Data-generating parameters.

    Returns:
        y (torch.Tensor): Generated BOLD signal (batch × time × regions).
        x (torch.Tensor): Generated neuronal signal (batch × time × regions).
    """

    # --------------------------------------------------------
    # Number of regions
    nr = Ep['A'].shape[0]
    A = Ep['A'].T.clone()  # (nr, nr)

    # --------------------------------------------------------
    # Prepare empty B and D arrays
    mArrayB = torch.zeros((nr, nr, nr), dtype=torch.float32, device=A.device).permute(1, 0, 2).clone()
    Ep['D'] = torch.zeros((nr, nr, nr), dtype=torch.float32, device=A.device)
    mArrayD = Ep['D'].permute(1, 0, 2).clone()

    # --------------------------------------------------------
    # Driving inputs
    # U: (N, T, R)
    U = DCM['U']['u']
    if U.ndim == 2:
        U = U.unsqueeze(0)  # convert (T,R) -> (1,T,R)
    N, T, R = U.shape

    C = U @ (Ep['C'].T / 16.0).float()  # (N, T, nr)

    # --------------------------------------------------------
    # Hemodynamic constants
    H = torch.tensor([0.64, 0.32, 2.00, 0.32, 0.32], dtype=torch.float32, device=A.device)

    # --------------------------------------------------------
    # Constants for hemodynamic model
    oxygenExtractionFraction = torch.full((nr, 1), 0.32, dtype=torch.float32, device=A.device)
    alphainv = 1.0 / H[3]
    tau = H[2] * torch.exp(Ep['transit'])
    gamma = H[1]
    kappa = H[0] * torch.exp(Ep['decay'])
    epsilon = torch.exp(Ep['epsilon']) * torch.ones((nr, 1), dtype=torch.float32, device=A.device)

    # -------------------------------------------------------
    paramList = [
        DCM['U']['dt'],
        T,
        nr,
        R,
        0,
        1,
        1
    ]

    # -------------------------------------------------------
    # Batched Euler integration
    x = dcm_euler_integration_vectorized_batch(
        A, C, U, mArrayB, mArrayD,
        oxygenExtractionFraction, alphainv, tau, gamma, kappa, paramList
    )

    return x


def dcm_euler_integration_vectorized_batch(A, C, U, B, D, rho, alphainv, tau, gamma, kappa, paramList):
    """
    Batched Euler integration for multiple input sequences.

    Args:
        A (nr, nr)
        C (N, T, nr)
        U (N, T, R)
        B, D: connectivity tensors
        ...
        paramList: same as before

    Returns:
        x_out (N, T, nr)
    """
    # Unpack parameters
    timeStep = float(paramList[0])
    nTime = int(paramList[1])
    nStates = int(paramList[2])
    nInputs = int(paramList[3])
    dcmTypeB = 0
    dcmTypeD = 0

    N = U.shape[0]
    device = A.device

    # Allocate output
    x_out = torch.zeros((N, nTime, nStates), dtype=torch.float64, device=device)

    # ------------------------------------------------------------
    # Main Euler integration loop (batched)
    # ------------------------------------------------------------
    for iStep in range(nTime - 1):
        x_t = x_out[:, iStep, :]  # (N, nr)

        # --------------------------------------------------------
        # Update x (neuronal activity)
        # A-term
        Ax = x_t @ A.double()  # (N, nr)
        x_new = x_t + timeStep * (Ax + C[:, iStep, :].double())

        # B-term (bilinear)
        if dcmTypeB != 0:
            UB_effect = torch.zeros_like(x_t)
            for kIter in range(nInputs):
                UB_effect += (U[:, iStep, kIter].unsqueeze(1)) * (x_t @ B[:, :, kIter].double())
            x_new += timeStep * UB_effect

        # D-term (nonlinear)
        if dcmTypeD != 0:
            temp_D = torch.stack([x_t @ D[:, :, k].double() for k in range(nStates)], dim=-1)  # (N, nr, nr)
            x_new += timeStep * torch.sum(x_t.unsqueeze(2) * temp_D, dim=1)

        x_out[:, iStep + 1, :] = x_new

    return x_out




def tapas_rdcm_generate(DCM, options, SNR):
    """
    tapas_rdcm_generate(DCM, options, SNR)

    Generates synthetic fMRI data under a given signal-to-noise ratio (SNR)
    with the fixed hemodynamic convolution kernel.

    Args:
        DCM (dict): Model structure.
        options (dict): Estimation options.
        SNR (float): Signal-to-noise ratio (Inf for noise-free simulation).

    Returns:
        DCM (dict): Updated model structure with synthetic data (Y.y, y, x).
    """

    # --------------------------------------------------------
    # Set dt if specified in options
    if options and 'y_dt' in options and options['y_dt']:
        DCM['Y']['dt'] = options['y_dt']

    r_dt = DCM['Y']['dt'] / DCM['U']['dt']
    N = DCM['U']['u'].shape[1]
    nr = DCM['a'].shape[0]


    # --------------------------------------------------------
    # Generate or use stored HRF
    if ('h' not in options) or (len(options['h']) != DCM['U']['u'].shape[1]):
        options['DCM'] = DCM
        options['conv_dt'] = DCM['U']['dt']
        options['conv_length'] = DCM['U']['u'].shape[0]
        options['conv_full'] = True
        options['h'] = tapas_rdcm_get_convolution_bm(options)

    h = options['h']

    # --------------------------------------------------------
    # Generate neuronal using Euler integration
    DCM_dummy = safe_clone(DCM)
    DCM_dummy['U']['u'] = DCM_dummy['U']['u'][0]
    DCM = tapas_dcm_euler_make_indices(DCM)
    x = tapas_dcm_euler_gen(DCM, DCM['Tp']) #_,x = ...
    # Triplicate output for convolution padding
    x = torch.cat([x,x,x], dim=1)
    
    # --------------------------------------------------------
    # Allocate data array
    B = x.shape[0]
    y = torch.zeros((B, N, nr), dtype=torch.float32, device=x.device)

    # --------------------------------------------------------
    # Convolve neuronal signal with HRF
    H_pad = torch.cat([h.to(x.device), torch.zeros((N * 3 - len(h),), dtype=torch.float32, device=x.device)])
    H_fft = torch.fft.fft(H_pad)
    xi_fft = torch.fft.fft(x.permute(0,2,1).contiguous().view(-1, x.shape[1]), dim=1)
    tmp = torch.fft.ifft(xi_fft * H_fft).real
    tmp = tmp.view(B, nr, N*3).permute(0,2,1)
    y = tmp[:,N:2 * N,:]

    # --------------------------------------------------------
    # Downsample to match acquisition rate
    step = int(round(r_dt))
    y = y[:,::step, :]

    # --------------------------------------------------------
    # Add Gaussian noise according to SNR
    if torch.isfinite(torch.tensor(SNR)) and SNR > 0:
        std_per_region = torch.std(y, dim=0)
        noise = torch.randn_like(y) * std_per_region / SNR
        y_noise = y + noise
    else:
        y_noise = y.clone()

    # --------------------------------------------------------
    # Save outputs
    #DCM['Y']['y'] = y_noise
    #DCM['y'] = y
    #DCM['x'] = x[:,N:2 * N, :]

    return y_noise, y, x[:,N:2 * N, :]



import torch
import torch.fft as fft

def pink_from_normal(x, beta):
    """
    Batched differentiable pink-noise transform.

    Args:
        x (torch.Tensor): Input tensor of shape (N, L, R) — typically standard normal noise.
        beta (float or torch.Tensor): Pinkness exponent β (can be scalar or tensor on same device).

    Returns:
        pink (torch.Tensor): Pink noise of shape (N, L, R), zero-mean and unit-variance per (N,R).
    """
    # Unpack shape
    if len(x.shape) < 3:
        x = x.unsqueeze(0)
    N, L, R = x.shape

    # FFT along time dimension (L)
    Xf = fft.rfft(x, dim=1)  # (N, F, R)

    # Frequency vector for scaling (F bins)
    F = Xf.shape[1]
    freqs = torch.linspace(0.001, 0.5, F, dtype=x.dtype, device=x.device)

    # Compute 1/f^(β/2) scaling factor
    # Shape (1, F, 1) for broadcasting across N and R
    scaling = (1.0 / (freqs ** (beta / 2.0))).view(1, F, 1)

    # Apply scaling in frequency domain
    Xf_scaled = Xf * scaling

    # Inverse FFT to time domain
    pink = fft.irfft(Xf_scaled, n=L, dim=1)

    # Normalize per batch & channel (over time)
    mean = pink.mean(dim=1, keepdim=True)
    std = pink.std(dim=1, keepdim=True) + 1e-8
    pink = (pink - mean) / std

    return pink  # (N, L, R)



def rDCM_generate(T,R,TR,beta,neural_noise,A,scale,transit,decay,epsilon):
    #with torch.no_grad():
    Y = {
         'y': torch.randn(T,R),
         'dt': TR
         }
    
    U = {
         'u': torch.zeros((T*16,R))
         }
    
    args = {
            'a': np.ones((R,R)),
            'c': np.eye(R,R)
            }
    DCM = tapas_rdcm_model_specification(Y, U, args)
    options = tapas_rdcm_set_options(DCM)
    #std =  torch.std(neural_noise, dim=0, keepdim=True)
    #random_noise = torch.randn_like(neural_noise)*(scale*std)
    #random_noise = random_noise*(scale*std)
    #neural_noise = neural_noise + random_noise
    #neural_noise = (neural_noise - torch.mean(neural_noise, dim=0, keepdims=True))/torch.std(neural_noise, dim=0, keepdims=True)
    neural_pink = pink_from_normal(neural_noise, beta)# + random_noise)
    if len(neural_pink.shape) < 3:
        neural_pink = neural_pink.unsqueeze(0)
    DCM['U']['u'] = neural_pink
    DCM['Tp'] = {}
    DCM['Tp']['A'] = A.to(neural_pink.device)
    DCM['Tp']['C'] = 0.05*torch.eye(A.shape[0]).to(neural_pink.device)
    DCM['Tp']['transit'] = transit.to(neural_pink.device)
    DCM['Tp']['decay'] = decay.to(neural_pink.device)
    DCM['Tp']['epsilon'] = epsilon.to(neural_pink.device)
    _, y_gen, x_gen = tapas_rdcm_generate(DCM, options, float('inf'))
    del DCM
    #y_gen.requires_grad = False
        
    return y_gen.squeeze(), x_gen.squeeze()



