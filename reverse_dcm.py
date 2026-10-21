# torch>=2.1
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import torch.fft as fft
from rdcm import tapas_rdcm_model_specification, tapas_rdcm_set_options, tapas_dcm_euler_make_indices, tapas_rdcm_get_convolution_bm




def tapas_dcm_euler_gen(DCM, Ep):
    """
    tapas_dcm_euler_gen(DCM, Ep)

    Generates synthetic fMRI data under a given signal-to-noise ratio (SNR)
    using the fixed hemodynamic convolution kernel.

    Args:
        DCM (dict): Model structure.
        Ep (dict): Data-generating parameters.

    Returns:
        y (torch.Tensor): Generated BOLD signal (time � regions).
        x (torch.Tensor): Generated neuronal signal.
    """

    # --------------------------------------------------------
    # Number of regions
    nr = Ep['A'].shape[0]
    A = Ep['A'].T.clone()

    # --------------------------------------------------------
    # Transpose each nr�nr matrix in B

    mArrayB = torch.zeros((nr,nr,nr)).permute(1, 0, 2).clone()    

    # --------------------------------------------------------
    Ep['D'] = torch.zeros((nr, nr, nr), dtype=torch.float32)

    # Transpose each nr�nr matrix in D
    mArrayD = Ep['D'].permute(1, 0, 2).clone()
    if mArrayD.numel() == 0:
        mArrayD = torch.zeros((nr, nr, nr), dtype=torch.float32)

    # --------------------------------------------------------
    # Driving inputs
    # MATLAB: C = DCM.U.u * (Ep.C'/16)
    C = DCM['U']['u'] @ (Ep['C'].T / 16.0).float()
    U = DCM['U']['u']  # "full" U

    # --------------------------------------------------------
    # Hemodynamic constants
    H = torch.tensor([0.64, 0.32, 2.00, 0.32, 0.32], dtype=torch.float32)

    # --------------------------------------------------------
    # Constants for hemodynamic model
    oxygenExtractionFraction = torch.full((nr, 1), 0.32, dtype=torch.float32)
    alphainv = 1.0 / H[3]
    tau = H[2] * torch.exp(Ep['transit'])
    gamma = H[1]
    kappa = H[0] * torch.exp(Ep['decay'])
    epsilon = torch.exp(Ep['epsilon']) * torch.ones((nr, 1), dtype=torch.float32)

    # --------------------------------------------------------
    # Parameter list (for compatibility with MATLAB signature)
    # [DCM.U.dt, size(U,1), nr, size(U,2), 0, 1, 1]
    paramList = [
        DCM['U']['dt'],
        U.shape[0],
        nr,
        U.shape[1],
        0,
        1,
        1
    ]

    #x, _, _, v, q = dcm_euler_integration_vectorized(
    #    A, C, U, mArrayB, mArrayD,
    #    oxygenExtractionFraction, alphainv, tau, gamma, kappa, paramList
    #)
    x = dcm_euler_integration_vectorized(
        A, C, U, mArrayB, mArrayD,
        oxygenExtractionFraction, alphainv, tau, gamma, kappa, paramList
        )

    # --------------------------------------------------------
    # Constants for BOLD signal equation
    #relaxationRateSlope = 25.0
    #frequencyOffset = 40.3
    #oxygenExtractionFraction = torch.full((1, nr), 0.4, dtype=torch.float32)
    #echoTime = 0.04
    #restingVenousVolume = 4.0

    # --------------------------------------------------------
    # Coefficients of BOLD signal equation
    #coefficientK1 = 4.3 * frequencyOffset * echoTime * oxygenExtractionFraction
    #coefficientK2 = (epsilon.T) * (relaxationRateSlope * oxygenExtractionFraction * echoTime)
    #coefficientK3 = 1.0 - epsilon.T

    # --------------------------------------------------------
    # Get the Euler indices
    #Indices = DCM['M']['idx'].to(torch.long) - 1  # MATLAB is 1-based

    # --------------------------------------------------------
    # Compute BOLD signal time course
    # MATLAB: bsxfun(@times, K1, (1 - q)) + bsxfun(@times, K2, (1 - q./v)) + bsxfun(@times, K3, (1 - v))
    #q_sel = q[Indices, :]
    #v_sel = v[Indices, :]

    #term1 = coefficientK1 * (1.0 - q_sel)
    #term2 = coefficientK2 * (1.0 - (q_sel / v_sel))
    #term3 = coefficientK3 * (1.0 - v_sel)

    #y = restingVenousVolume * (term1 + term2 + term3)

    return x#y, x


def dcm_euler_integration_vectorized(A, C, U, B, D, rho, alphainv, tau, gamma, kappa, paramList):
    # Unpack parameters
    timeStep = float(paramList[0])
    nTime = int(paramList[1])
    nStates = int(paramList[2])
    nInputs = int(paramList[3])
    dcmTypeB = 0
    dcmTypeD = 0

    # Allocate outputs
    x_out = torch.zeros((nTime, nStates), dtype=torch.float64, device=A.device)
    #s_out = torch.zeros((nTime, nStates), dtype=torch.float64)
    #f1_out = torch.ones((nTime, nStates), dtype=torch.float64)
    #v1_out = torch.ones((nTime, nStates), dtype=torch.float64)
    #q1_out = torch.ones((nTime, nStates), dtype=torch.float64)

    # Log-variables for hemodynamics
    #oldf1 = torch.zeros(nStates, dtype=torch.float64)
    #oldv1 = torch.zeros(nStates, dtype=torch.float64)
    #oldq1 = torch.zeros(nStates, dtype=torch.float64)

    # ------------------------------------------------------------
    # Main Euler integration loop (only over time)
    # ------------------------------------------------------------
    for iStep in range(nTime - 1):
        x_t = x_out[iStep]  # shape [nStates]

        # --------------------------------------------------------
        # Update x (neuronal activity)
        # A-term
        x_new = x_t + timeStep * (x_t @ A.double() + C[iStep].double())

        # B-term (bilinear)
        if dcmTypeB != 0:
            # Compute contribution: sum_k U[iStep,k] * (x_t @ B[:,:,k])
            UB_effect = torch.zeros_like(x_t)
            for kIter in range(nInputs):
                UB_effect += U[iStep, kIter] * (x_t @ B[:, :, kIter].double())
            x_new += timeStep * UB_effect

        # D-term (nonlinear)
        if dcmTypeD != 0:
            # Compute x_t[k] * (x_t @ D[:,:,k]) for all k, then sum
            temp_D = torch.stack([x_t @ D[:, :, k].double() for k in range(nStates)], dim=0)
            x_new += timeStep * torch.sum(x_t.unsqueeze(1) * temp_D, dim=0)

        x_out[iStep + 1] = x_new

        # --------------------------------------------------------
        # Update s (vasodilatory signal)
        #s_t = s_out[iStep]
        #f_t = f1_out[iStep]
        #s_new = s_t + timeStep * (x_t - kappa.squeeze() * s_t - gamma * (f_t - 1.0))
        #s_out[iStep + 1] = s_new

        # --------------------------------------------------------
        # Update hemodynamics (f, v, q)
        #oldf1 += timeStep * (s_t / f_t)

        #temp1 = v1_out[iStep].pow(alphainv)
        #temp2 = (1.0 - (1.0 - rho.squeeze()).pow(1.0 / f_t)) / rho.squeeze()

        #oldv1 += (timeStep * ((f_t - temp1) / (tau.squeeze() * v1_out[iStep]))).squeeze()
        #oldq1 += (timeStep * ((f_t * temp2 - temp1 * q1_out[iStep] / v1_out[iStep]) /
        #                     (tau.squeeze() * q1_out[iStep]))).squeeze()

        #f1_out[iStep + 1] = torch.exp(oldf1)
        #v1_out[iStep + 1] = torch.exp(oldv1)
        #q1_out[iStep + 1] = torch.exp(oldq1)

    return x_out#, s_out, f1_out, v1_out, q1_out



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
    N = DCM['U']['u'].shape[0]
    nr = DCM['a'].shape[0]


    # --------------------------------------------------------
    # Generate or use stored HRF
    if ('h' not in options) or (len(options['h']) != DCM['U']['u'].shape[0]):
        options['DCM'] = DCM
        options['conv_dt'] = DCM['U']['dt']
        options['conv_length'] = DCM['U']['u'].shape[0]
        options['conv_full'] = True
        options['h'] = tapas_rdcm_get_convolution_bm(options)

    h = options['h']

    # --------------------------------------------------------
    # Generate neuronal signal using Euler integration
    DCM = tapas_dcm_euler_make_indices(DCM)
    x = tapas_dcm_euler_gen(DCM, DCM['Tp']) #_,x = ...
    # Triplicate output for convolution padding
    x = torch.cat([x,x,x], dim=0)
    
    # --------------------------------------------------------
    # Allocate data array
    y = torch.zeros((N, nr), dtype=torch.float32, device=x.device)

    # --------------------------------------------------------
    # Convolve neuronal signal with HRF
    H_pad = torch.cat([h.to(x.device), torch.zeros((N * 3 - len(h),), dtype=torch.float32, device=x.device)])
    H_fft = torch.fft.fft(H_pad)
    for i in range(nr):
        xi_fft = torch.fft.fft(x[:, i])
        tmp = torch.fft.ifft(xi_fft * H_fft).real
        y[:, i] = tmp[N:2 * N]

    # --------------------------------------------------------
    # Downsample to match acquisition rate
    step = int(round(r_dt))
    y = y[::step, :]

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
    DCM['Y']['y'] = y_noise
    DCM['y'] = y
    DCM['x'] = x[N:2 * N, :]

    return y_noise, y, x[N:2 * N, :]


class PinkNoiseParam(nn.Module):
    def __init__(self, beta_init=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta_init))
    
    def forward(self, x):
        """
        Returns a differentiable multivariate pink-noise matrix [L, R].
        """
        length = x.shape[0]
        Xf = fft.rfft(x, dim=0)  # FFT along the time dimension
        
        # Frequency vector for scaling
        freqs = torch.linspace(0.001, 0.5, Xf.shape[0], dtype=x.dtype, device=self.beta.device)
        
        # Create 1/f^(β/2) scaling factor (broadcast along channels)
        scaling = 1.0 / (freqs ** (self.beta / 2.0))
        scaling = scaling[:, None]  # [F, 1] to broadcast over R
        
        Xf_scaled = Xf * scaling
        
        # Back to time domain
        pink = fft.irfft(Xf_scaled, n=length, dim=0)
        
        # Normalize per channel (zero mean, unit variance)
        pink = (pink - pink.mean(dim=0, keepdim=True)) / (pink.std(dim=0, keepdim=True) + 1e-8)
        
        return pink  # [L, R]



# -----------------------------
# Full Generative Model
# -----------------------------
class GenerativeModel(nn.Module):
    def __init__(self, y, TR, neural_init=None, beta_init=1.0):
        super().__init__()
        self.T, self.R = y.shape
        self.TR = TR
        if neural_init == None:
            self.neural_noise = nn.Parameter(torch.randn(self.T*16, self.R))
        else:
            self.neural_noise = nn.Parameter(neural_init)
        self.pink_from_normal = PinkNoiseParam(beta_init=beta_init)
        self.y = y
        self.TR = TR


    def forward(self, out):
        Y = {
             'y': self.y,
             'dt': self.TR
             }

        U = {
             'u': torch.zeros((self.T*16,self.R))
             }
        
        a = np.ones((self.R,self.R))
        args = {
                'a': a,
                'c': np.eye(self.R,self.R)
                }
        DCM = tapas_rdcm_model_specification(Y, U, args)
        options = tapas_rdcm_set_options(DCM)
        neural_pink = self.pink_from_normal(self.neural_noise)
        self.neural_pink = neural_pink.squeeze().detach()
        DCM['U']['u'] = neural_pink.squeeze()
        DCM['Tp'] = {}
        DCM['Tp']['A'] = out[0]['Ep']['A'].to(self.neural_pink.device)
        DCM['Tp']['C'] = 0.05*torch.eye(DCM['Tp']['A'].shape[0]).to(self.neural_pink.device)
        DCM['Tp']['transit'] = out[0]['Ep']['transit']
        DCM['Tp']['decay'] = out[0]['Ep']['decay']
        DCM['Tp']['epsilon'] = out[0]['Ep']['epsilon']
        _, y_gen, _ = tapas_rdcm_generate(DCM, options, float('inf'))
        del DCM
        return y_gen
        

if __name__ == '__main__':
    # -----------------------------
    # Example training loop (sketch)
    # -----------------------------
    """
    Assume you have observed fMRI y with shape (B, R, T_y).
    Our model outputs valid length T_hat = T - K + 1 because of 'valid' causal conv.
    Easiest is to choose model T so that T_hat == T_y.
    """
    
    TR = 1.5
    
    B = 1
    path = r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_GNN_data/HCP'
    atlas = 'SCH'
    data = np.load(os.path.join(path,'train_data_HCP_' + atlas + '_resampled_all.npz'))
    train_data = []
    for key in data:
        train_data.append((data[key] - np.mean(data[key], axis=0, keepdims=True))/np.std(data[key], axis=0, keepdims=True))
    y_obs = torch.from_numpy(train_data[0])
    
    
    model = GenerativeModel(y=y_obs, TR=TR)
    
    opt = torch.optim.Adam([
        {"params": model.parameters(), "lr": 3e-4}
    ])
    
    
    for step in range(100):
        opt.zero_grad()
        y_hat = model()
        mse = F.mse_loss(y_hat, y_obs)
        loss = mse
        loss.backward()
        opt.step()
        print(loss.item())
