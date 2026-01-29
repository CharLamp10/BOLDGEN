import torch
import time
import numpy as np
import random
import sys
from torch.special import psi, gammaln
import math
from torch.utils.cpp_extension import load



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
    # Allocate data array
    y = torch.zeros((N, nr), dtype=torch.float32)

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
    # Triplicate input for convolution padding
    #DCM['U']['u'] = torch.cat([DCM['U']['u'], DCM['U']['u'], DCM['U']['u']], dim=0)
    DCM = tapas_dcm_euler_make_indices(DCM)
    _, x = tapas_dcm_euler_gen(DCM, DCM['Tp'])
    x = torch.cat([x,x,x], dim=0)
    #DCM['U']['u'] = DCM['U']['u'][:N, :]

    # --------------------------------------------------------
    # Convolve neuronal signal with HRF
    H_pad = torch.cat([h, torch.zeros((N * 3 - len(h),), dtype=torch.float32)])
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

    return DCM


def tapas_rdcm_compute_signals(DCM, output, options):
    """
    tapas_rdcm_compute_signals(DCM, output, options)

    Computes true and predicted signals in time and frequency domains.

    Args:
        DCM (dict): Model structure.
        output (dict): Model inversion results.
        options (dict): Estimation options.

    Returns:
        output (dict): Updated output structure with computed signals and residuals.
    """

    # --------------------------------------------------------
    # True or measured signal
    output['signal'] = {}
    output['signal']['y_source'] = DCM['Y']['y'].reshape(-1)

    # --------------------------------------------------------
    # Simulated (noise-free) or VBL predicted signal
    if options['type'] == 's':
        if options.get('compute_signal', 0):
            # Generate noise-free signal
            DCMs = tapas_rdcm_generate(DCM, options, float('inf'))
            output['signal']['y_clean'] = DCMs['Y']['y'].reshape(-1)

            # MSE for noise-free signal
            diff = output['signal']['y_source'] - output['signal']['y_clean']
            output.setdefault('residuals', {})
            output['residuals']['y_mse_clean'] = torch.mean(diff**2).item()

    else:
        if 'y' in DCM:
            output['signal']['y_pred_vl'] = DCM['y'].reshape(-1)
            diff = output['signal']['y_source'] - output['signal']['y_pred_vl']
            output.setdefault('residuals', {})
            output['residuals']['y_mse_vl'] = torch.mean(diff**2).item()
        else:
            output.setdefault('residuals', {})
            output['residuals']['y_mse_vl'] = None

    # --------------------------------------------------------
    # Store source derivative (frequency domain)
    yd_source_fft = output['temp']['yd_source_fft'].clone()
    yd_source_fft[~torch.isfinite(yd_source_fft)] = 0
    output['signal']['yd_source_fft'] = yd_source_fft.reshape(-1)

    # --------------------------------------------------------
    # Compute predicted signal (time domain) if requested
    if options.get('compute_signal', 0):
        # Use posterior parameters
        DCM['Tp'] = output['Ep']

        # No baseline for simulated data
        if options['type'] == 's':
            DCM['Tp']['baseline'] = torch.zeros((DCM['Tp']['C'].shape[0], 1))

        # Adjust sampling rate
        r_dt = DCM['Y']['dt'] / DCM['U']['dt']
        DCM['Y']['dt'] = DCM['U']['dt']

        # Apply posterior probabilities (sparse rDCM)
        if 'Ip' in output:
            DCM['Tp']['A'] = DCM['Tp']['A'] * output['Ip']['A']
            DCM['Tp']['C'] = DCM['Tp']['C'] * output['Ip']['C']

        # Generate predicted signal
        DCM_rDCM = tapas_rdcm_generate(DCM, options, float('inf'))

        # Add confounds (e.g., baseline)
        nT, nR = DCM['Y']['y'].shape
        for t in range(nT):
            for r in range(nR):
                idx = int(1 + r_dt * t) - 1  # MATLAB -> Python index correction
                if idx < DCM_rDCM['U']['X0'].shape[0]:
                    DCM_rDCM['Y']['y'][t, r] += torch.matmul(
                        DCM_rDCM['Tp']['baseline'][r, :].float(),
                        DCM_rDCM['U']['X0'][idx, :].float().T
                    )

        # Flatten to vector
        output['signal']['y_pred_rdcm'] = DCM_rDCM['Y']['y'].reshape(-1)

    # --------------------------------------------------------
    # Store predicted derivative (frequency domain)
    yd_pred_rdcm_fft = output['temp']['yd_pred_rdcm_fft']
    output['signal']['yd_pred_rdcm_fft'] = yd_pred_rdcm_fft.reshape(-1)

    # Remove temporary fields
    if 'temp' in output:
        del output['temp']

    # --------------------------------------------------------
    # Assign region names
    if 'name' in DCM['Y']:
        output['signal']['name'] = DCM['Y']['name']

    # --------------------------------------------------------
    # Compute residuals and MSE for rDCM predicted signal
    if options.get('compute_signal', 0):
        diff = output['signal']['y_source'] - output['signal']['y_pred_rdcm']
        output['residuals']['y_mse_rdcm'] = torch.mean(diff**2).item()
        output['residuals']['R_rdcm'] = diff

    # --------------------------------------------------------
    # Store driving inputs
    output['inputs'] = {
        'u': DCM['U']['u'],
        'name': DCM['U']['name'],
        'dt': DCM['U']['dt']
    }

    return output



def tapas_rdcm_ep2par(Ep):
    """
    tapas_rdcm_ep2par(Ep)

    Vectorizes Ep structure into a single parameter vector.

    Args:
        Ep (dict): Parameter structure with fields 'A', 'B', and 'C'.

    Returns:
        par (torch.Tensor): 1D vectorized parameter tensor.
    """

    # Flatten and concatenate all parameter fields
    par = torch.cat([
        Ep['A'].reshape(-1),
        Ep['B'].reshape(-1),
        Ep['C'].reshape(-1)
    ])

    return par


def tapas_rdcm_compute_statistics(DCM, output, options):
    """
    tapas_rdcm_compute_statistics(DCM, output, options)

    Computes posterior parameter statistics for rDCM output.

    Args:
        DCM (dict): Model structure
        output (dict): Model inversion results
        options (dict): Estimation options

    Returns:
        output (dict): Updated output structure with computed statistics
    """

    # --------------------------------------------------------
    # Get posterior estimates
    output['allParam'] = {}
    output['allParam']['par_est'] = tapas_rdcm_ep2par(output['Ep'])

    # --------------------------------------------------------
    # Get true or reference parameters
    try:
        output['allParam']['par_true'] = tapas_rdcm_ep2par(DCM['Tp'])
    except Exception:
        try:
            output['allParam']['par_true'] = tapas_rdcm_ep2par(DCM['Ep'])
        except Exception:
            output['allParam']['par_true'] = None

    # --------------------------------------------------------
    # Identify true present connections
    if output['allParam']['par_true'] is not None:
        output['allParam']['idx'] = output['allParam']['par_true'] != 0
    else:
        output['allParam']['idx'] = None

    # --------------------------------------------------------
    # Compute statistics
    if output['allParam']['par_true'] is not None:
        par_est = output['allParam']['par_est']
        par_true = output['allParam']['par_true']
        idx = output['allParam']['idx']

        diff = par_est[idx] - par_true[idx]
        mse = torch.mean(diff**2)
        mse_n = mse / torch.norm(par_true[idx])
        sign_flip = torch.sum((par_est[idx] * par_true[idx]) < 0)

        output['statistics'] = {
            'mse_n': mse_n.item(),
            'mse': mse.item(),
            'sign': int(sign_flip.item())
        }
    else:
        output['statistics'] = {'mse_n': None, 'mse': None, 'sign': None}

    # --------------------------------------------------------
    # Compute signal predictions
    output = tapas_rdcm_compute_signals(DCM, output, options)

    return output


def tapas_rdcm_get_prior_all(DCM):
    """
    tapas_rdcm_get_prior_all(DCM)

    Returns prior parameters on model parameters (theta) and noise precision (tau)
    for the full connectivity model  used by the sparse rDCM variant.

    Args:
        DCM (dict): Model structure containing fields a, b, c, d.

    Returns:
        m0 (torch.Tensor): Prior means (nr � (nr*nu + nu))
        l0 (torch.Tensor): Prior precisions (same shape as m0)
        a0 (float): Prior shape parameter for noise precision
        b0 (float): Prior rate parameter for noise precision
    """

    # --------------------------------------------------------
    # Get SPM-style priors assuming full connectivity
    pE, pC, _ = tapas_rdcm_spm_dcm_fmri_priors(
        torch.ones_like(DCM['a']),
        torch.ones_like(DCM['b']),
        torch.ones_like(DCM['c']),
        torch.ones_like(DCM['d'])
    )

    # --------------------------------------------------------
    # Number of regions and inputs
    nr, nu = pE['C'].shape

    # --------------------------------------------------------
    # Set prior mean of endogenous parameters to zero except diagonal
    pE['A'] = torch.zeros_like(pE['A']) + torch.diag(torch.diag(pE['A']))

    # --------------------------------------------------------
    # Construct prior mean (m0)
    m0 = torch.cat([
        pE['A'],
        pE['B'].reshape(nr, nr * nu),
        pE['C']
    ], dim=1)

    # --------------------------------------------------------
    # Convert prior covariance to precision (elementwise reciprocal)
    pC['A'] = 1.0 / pC['A']
    pC['B'] = 1.0 / pC['B']
    pC['C'] = 1.0 / pC['C']
    pC['D'] = 1.0 / pC['D']
    pC['transit'] = 1.0 / pC['transit']
    pC['decay'] = 1.0 / pC['decay']
    pC['epsilon'] = 1.0 / pC['epsilon']

    # --------------------------------------------------------
    # Construct prior precision (l0)
    l0 = torch.cat([
        pC['A'],
        pC['B'].reshape(nr, nr * nu),
        pC['C']
    ], dim=1)

    # --------------------------------------------------------
    # Priors on noise precision (gamma distribution)
    a0 = 2.0
    b0 = 1.0

    return m0, l0, a0, b0


def tapas_rdcm_sparse(DCM, X, Y, args):
    """
    tapas_rdcm_sparse(DCM, X, Y, args)

    Variational Bayesian inversion of a linear DCM with sparsity constraints
    (Fr�ssle et al., 2018).

    Args:
        DCM (dict): Model structure
        X (torch.Tensor): Design matrix (predictors)
        Y (torch.Tensor): Data (dependent variable)
        args (dict): Arguments controlling inversion options

    Returns:
        output (list[dict]): Output list (single dict for this variant)
    """

    # --------------------------------------------------------
    # Initialization
    nr = DCM['Y']['y'].shape[1]
    pr = 1e-5

    # --------------------------------------------------------
    # Add confound regressors
    Nc = DCM['U']['X0'].shape[1]
    for _ in range(Nc):
        DCM['b'] = torch.cat((DCM['b'], DCM['b'][:, :, [0]]), dim=2)
        DCM['c'] = torch.cat((DCM['c'], torch.ones((DCM['c'].shape[0], 1))), dim=1)

    # Remove confounds for simulations
    if args['type'] == 's':
        DCM['c'][:, -Nc:] = 0.0

    # --------------------------------------------------------
    # Priors
    m0, l0, a0, b0 = tapas_rdcm_get_prior_all(DCM)

    idx_x = torch.cat([
        torch.ones(DCM['a'].shape[0]),
        torch.zeros(DCM['b'].shape[0] * DCM['b'].shape[2]),
        torch.ones(DCM['c'].shape[1])
    ]) == 1

    D = int(idx_x.sum().item())
    ord_regions = list(range(nr))

    # --------------------------------------------------------
    # Initialize results
    mN = torch.zeros((nr, D), dtype=torch.float64)
    sN = [torch.zeros((D, D), dtype=torch.float64) for _ in range(nr)]
    aN = torch.zeros(nr, dtype=torch.float64)
    bN = torch.zeros(nr, dtype=torch.float64)
    z = torch.zeros((nr, D), dtype=torch.float64)
    logF = torch.zeros(nr, dtype=torch.float64)
    yd_pred_rdcm_fft = torch.full_like(DCM['Y']['y'], torch.nan, dtype=torch.complex64)

    # --------------------------------------------------------
    # Default inversion settings
    args.setdefault('iter', 100)
    args.setdefault('restrictInputs', 1)
    args.setdefault('diagnostics_logF', 0)
    args.setdefault('verbose', 1)

    # --------------------------------------------------------
    # Iterate over regions
    for k in ord_regions:
        mN_iter, sN_iter, aN_iter, bN_iter, z_iter, l0_iter = [], [], [], [], [], []
        logF_iter = torch.full((args['iter'],), float('nan'))
        logF_terms = {
            'log_lik': torch.full((args['iter'],), float('nan')),
            'log_p_weight': torch.full((args['iter'],), float('nan')),
            'log_p_prec': torch.full((args['iter'],), float('nan')),
            'log_p_z': torch.full((args['iter'],), float('nan')),
            'log_q_weight': torch.full((args['iter'],), float('nan')),
            'log_q_prec': torch.full((args['iter'],), float('nan')),
            'log_q_z': torch.full((args['iter'],), float('nan'))
        }

        idx_y = ~torch.isnan(Y[:, k])
        X_r = X[idx_y, :]
        Y_r = Y[idx_y, k]
        N_eff = idx_y.sum().item()

        l0_r = torch.diag(l0[k, idx_x])
        m0_r = m0[k, idx_x].unsqueeze(1)

        for iter_ in range(args['iter']):
            # ------------------------------------------------
            # Initialize feature informativeness p0
            p0 = torch.ones(D, 1) * args.get('p0_temp', 0.5)

            if args.get('p0_inform', False):
                p0[:DCM['a'].shape[0], 0] *= DCM['a'][k, :]

            p0[k, 0] = 1.0  # self-connectivity
            p0[-1, 0] = 0.0 if args['type'] == 's' else 1.0  # baseline

            if args['restrictInputs'] == 1:
                p0[-(DCM['c'].shape[1] - 1):-1, 0] = DCM['c'][k, :-1]

            # ------------------------------------------------
            # Initialize z, t, aN
            z_r = p0.clone()
            t = a0 / b0
            aN_r = a0 + N_eff / 2.0

            W = (X_r[:, idx_x].T @ X_r[:, idx_x]).to(torch.float64)
            v = (X_r[:, idx_x].T @ Y_r).unsqueeze(1).to(torch.float64)

            count = 500
            logF_old = -float('inf')
            convergence = False

            while not convergence:
                Z = torch.diag(z_r.flatten())
                G = Z @ W @ Z
                diag_mask = torch.eye(D, dtype=torch.bool)
                G[diag_mask] = z_r.flatten() * torch.diag(W)

                sN_r = torch.linalg.inv(t * G + l0_r)
                mN_r = sN_r @ (t * Z @ v + l0_r @ m0_r)

                # ------------------------------------------------
                # Update z (sparsity indicators)
                Wb = W * (mN_r @ mN_r.T)
                Ws = W * sN_r
                A = Wb + Ws
                g = torch.log(p0 / (1 - p0)) + t * mN_r * v + 0.5 * t * torch.diag(A).unsqueeze(1)

                for i in torch.randperm(D):
                    z_r[i] = 1
                    g[i] = g[i] - t * (z_r.T @ A[:, i])
                    z_r[i] = 1.0 / (1.0 + torch.exp(-g[i]))

                # ------------------------------------------------
                # Update covariance and noise parameters
                Z = torch.diag(z_r.flatten())
                G = Z @ W @ Z
                G[diag_mask] = z_r.flatten() * torch.diag(W)

                QF = (Y_r.T @ Y_r) / 2.0 - (mN_r.T @ Z @ v) + (mN_r.T @ G @ mN_r) / 2.0 + torch.trace(G @ sN_r) / 2.0
                bN_r = b0 + QF
                t = aN_r / bN_r

                # sparsify small parameters
                mN_r[torch.abs(mN_r) < 1e-5] = 0
                z_r[mN_r == 0] = 0
                z_idx = ((z_r > pr ** 2) & (z_r < 1.0)).flatten()

                # ------------------------------------------------
                # Compute free energy
                log_lik = N_eff * (psi(aN_r) - torch.log(bN_r)) / 2.0 - N_eff * torch.log(torch.tensor(2 * torch.pi)) / 2.0 - t * QF
                log_p_weight = 0.5 * tapas_rdcm_spm_logdet(l0_r) - D * torch.log(torch.tensor(2 * torch.pi)) / 2.0 \
                               - 0.5 * (mN_r - m0_r).T @ l0_r @ (mN_r - m0_r) - 0.5 * torch.trace(l0_r @ sN_r)
                log_p_prec = a0 * torch.log(b0) - gammaln(a0) + (a0 - 1) * (psi(aN_r) - torch.log(bN_r)) - b0 * t
                log_p_z = torch.sum(torch.log(1 - p0[z_idx]) + z_r[z_idx] * torch.log(p0[z_idx] / (1 - p0[z_idx])))
                log_q_weight = 0.5 * tapas_rdcm_spm_logdet(sN_r) + D * (1 + torch.log(torch.tensor(2 * torch.pi))) / 2.0
                log_q_prec = aN_r - torch.log(bN_r) + gammaln(aN_r) + (1 - aN_r) * psi(aN_r)
                log_q_z = torch.sum(-(1 - z_r[z_idx]) * torch.log(1 - z_r[z_idx]) - z_r[z_idx] * torch.log(z_r[z_idx]))

                logF_temp = (log_lik + log_p_prec + log_p_weight + log_p_z + log_q_prec + log_q_weight + log_q_z).real

                # convergence check
                if (logF_old - logF_temp).abs().item() ** 2 < pr ** 2:
                    convergence = True
                logF_old = logF_temp
                count -= 1
                if count <= 0:
                    break

            # ------------------------------------------------
            # Save iteration
            mN_iter.append(mN_r.clone())
            sN_iter.append(sN_r.clone())
            z_iter.append(z_r.clone())
            aN_iter.append(aN_r.clone())
            bN_iter.append(bN_r.clone())
            l0_iter.append(l0_r.clone())
            logF_iter[iter_] = logF_temp.real

        # --------------------------------------------------------
        # Select best iteration
        best = torch.argmax(logF_iter).item()
        mN[k, :] = mN_iter[best].flatten().real
        z[k, :] = z_iter[best].flatten().real
        sN[k][idx_x, idx_x] = sN_iter[best].real
        aN[k] = aN_iter[best].real
        bN[k] = bN_iter[best].real
        logF[k] = logF_iter[best]

        # Predicted signal
        yd_pred_rdcm_fft[:, k] = (X[:, idx_x] @ mN_iter[best]).flatten()

    # --------------------------------------------------------
    # Wrap results in standard output format
    output = [{}]
    output[0] = tapas_rdcm_store_parameters(DCM, mN, sN, aN, bN, logF, None, idx_x, z, args)

    # Priors and metadata
    output[0]['priors'] = {'m0': m0, 'l0': l0, 'a0': a0, 'b0': b0, 'p0': args.get('p0_temp', 0.5)}
    output[0]['inversion'] = 'tapas_rdcm_sparse'
    output[0]['temp'] = {'yd_source_fft': Y, 'yd_pred_rdcm_fft': yd_pred_rdcm_fft}

    return output



def tapas_rdcm_store_parameters(DCM, mN_cut, sN, aN, bN, logF, logF_term, idx_x, z_cut, args):
    """
    tapas_rdcm_store_parameters(DCM, mN_cut, sN, aN, bN, logF, logF_term, idx_x, z_cut, args)

    Wraps estimated parameters and statistics into the output structure.

    Args:
        DCM (dict): Model structure
        mN_cut (torch.Tensor): Posterior means of connectivity parameters
        sN (list[torch.Tensor]): Posterior covariances of connectivity parameters (per region)
        aN (torch.Tensor): Posterior shape parameter of measurement noise
        bN (torch.Tensor): Posterior rate parameter of measurement noise
        logF (torch.Tensor): Negative free energy per region
        logF_term (dict or None): Components of the negative free energy
        idx_x (torch.Tensor or None): Parameter index mask
        z_cut (torch.Tensor or None): Posterior binary indicators
        args (dict): Argument structure

    Returns:
        output (dict): Output structure containing all posterior quantities
    """

    output = {}

    # --------------------------------------------------------
    # Remove confound regressors
    Nc = DCM['U']['X0'].shape[1]
    DCM['b'] = DCM['b'][:, :, :DCM['b'].shape[2] - Nc]
    DCM['c'] = DCM['c'][:, :DCM['c'].shape[1] - Nc]

    # --------------------------------------------------------
    # Get number of regions and inputs
    nr, nu = DCM['c'].shape

    # --------------------------------------------------------
    # Handle vectorized posterior case
    if min(mN_cut.shape) == 1:
        mN = torch.zeros((nr, len(idx_x)), dtype=torch.float32)
        mN[idx_x.expand(nr, -1)] = mN_cut
    else:
        mN = mN_cut.clone()

    # --------------------------------------------------------
    # === Mean (posterior expectation) ===
    output['Ep'] = tapas_rdcm_empty_par(DCM)
    output['Ep']['A'] = mN[:nr, :nr]
    output['Ep']['B'] = mN[:nr, nr:nr + nr * nu].reshape(nr, nr, nu)
    output['Ep']['C'] = mN[:nr, -Nc - nu:-Nc]
    output['Ep']['baseline'] = mN[:nr, -Nc:] if Nc > 0 else mN[:nr, 0:0]


    # Scale driving inputs for real data
    if args['type'] == 'r':
        output['Ep']['C'] = output['Ep']['C'] * 16.0

    # --------------------------------------------------------
    # === Variance (posterior covariance) ===
    D = output['Ep']['A'].numel() + output['Ep']['C'].numel()

    if (z_cut is None) and (args.get('evalCp', 0) == 1):
        output['Cp'] = torch.zeros((D, D), dtype=torch.float32)
        nr_A = output['Ep']['A'].numel()
        nr_C = output['Ep']['C'].numel()
        nr_A_B = output['Ep']['A'][0, :].numel() + \
                 output['Ep']['B'][0, :, :].numel() + \
                 output['Ep']['B'][0, :, 0].numel()

        indexA = torch.arange(1, nr_A + 1).reshape(nr, nr)
        indexC = torch.arange(1, nr_C + 1).reshape(nr, nu)

        # Fill Cp for endogenous parameters
        for k in range(nr):
            for i in range(nr):
                for j in range(nr):
                    output['Cp'][indexA[k, i] - 1, indexA[k, j] - 1] = sN[k][i, j]

        # Fill Cp for driving input parameters
        for k in range(nr):
            for i in range(nu):
                for j in range(nu):
                    output['Cp'][nr_A + indexC[k, i] - 1, nr_A + indexC[k, j] - 1] = \
                        sN[k][nr_A_B + i, nr_A_B + j]

        # Cross-covariance terms
        for k in range(nr):
            for i in range(nr):
                for j in range(nu):
                    output['Cp'][indexA[k, i] - 1, nr_A + indexC[k, j] - 1] = \
                        sN[k][i, nr_A_B + j]
                    output['Cp'][nr_A + indexC[k, j] - 1, indexA[k, i] - 1] = \
                        sN[k][i, nr_A_B + j]

    output['sN'] = sN

    # --------------------------------------------------------
    # === Precision parameters ===
    output['t'] = aN / bN
    output['aN'] = aN
    output['bN'] = bN

    # --------------------------------------------------------
    # === Connection probabilities ===
    if z_cut is not None:
        if min(mN_cut.shape) == 1:
            z = torch.zeros((nr, len(idx_x)), dtype=torch.float32)
            z[idx_x.expand(nr, -1)] = z_cut
        else:
            z = z_cut.clone()

        output['Ip'] = tapas_rdcm_empty_par(DCM)
        output['Ip']['A'] = z[:, :nr]
        output['Ip']['B'] = z[:, nr:nr + nr * nu].reshape(nr, nr, nu)
        output['Ip']['C'] = z[:, -nu:-1]
    else:
        output['Ip'] = tapas_rdcm_empty_par(DCM)
        output['Ip']['A'] = DCM['a']
        output['Ip']['B'] = DCM['b']
        output['Ip']['C'] = DCM['c']

    # --------------------------------------------------------
    # === Free energy ===
    output['logF'] = torch.sum(logF)
    output['logF_r'] = logF

    if logF_term is not None:
        output['logF_term'] = {
            'log_lik': torch.sum(logF_term['log_lik']),
            'log_p_weight': torch.sum(logF_term['log_p_weight']),
            'log_p_prec': torch.sum(logF_term['log_p_prec']),
            'log_p_z': torch.sum(logF_term['log_p_z']),
            'log_q_weight': torch.sum(logF_term['log_q_weight']),
            'log_q_prec': torch.sum(logF_term['log_q_prec']),
            'log_q_z': torch.sum(logF_term['log_q_z'])
        }

    return output


def tapas_rdcm_spm_logdet(C):
    """
    tapas_rdcm_spm_logdet(C)

    Returns the log of the determinant of a positive semi-definite matrix C.
    Faithful PyTorch translation of MATLAB's spm_logdet.

    Args:
        C (torch.Tensor): Positive semi-definite matrix (n�n)

    Returns:
        H (torch.Tensor): log(det(C))
    """

    # --------------------------------------------------------
    TOL = 1e-16
    C = C.clone().detach().to(torch.float64)

    # --------------------------------------------------------
    # Attempt fast diagonal evaluation
    s = torch.diag(C)
    mask = (s > TOL) & (s < 1.0 / TOL)

    if mask.any():
        C_sub = torch.diag(s[mask])
        H = torch.sum(torch.log(torch.diag(C_sub)))
    else:
        H = torch.tensor(0.0, dtype=torch.float64)

    # --------------------------------------------------------
    # If matrix is non-diagonal, fall back to det computation
    if not torch.allclose(C, torch.diag(torch.diag(C))):
        n = C.shape[0]
        a = torch.exp(H / n) if n > 0 else torch.tensor(1.0)
        try:
            H = H + torch.logdet(C / a)
        except RuntimeError:
            # torch.logdet fails for singular matrices
            H = torch.tensor(float('-inf'))

    # --------------------------------------------------------
    # If rank deficient or logdet invalid, use SVD fallback
    if not torch.isfinite(H) or not torch.isreal(H):
        try:
            s_vals = torch.linalg.svdvals(C)
            valid = (s_vals > TOL) & (s_vals < 1.0 / TOL)
            H = torch.sum(torch.log(s_vals[valid]))
        except RuntimeError:
            # Numerical instability fallback
            H = torch.tensor(float('-inf'))

    return H.real


def tapas_rdcm_spm_dcm_fmri_priors(A, B, C, D, options=None, predicted_U=False, A_prior=None):
    """
    tapas_rdcm_spm_dcm_fmri_priors(A, B, C, D, options=None)

    Returns priors for a (one-state) DCM for fMRI.

    Args:
        A (torch.Tensor): Connectivity matrix (n�n)
        B (torch.Tensor): Bilinear coupling (n�n�nu)
        C (torch.Tensor): Input matrix (n�nu)
        D (torch.Tensor): Nonlinear coupling (n�n�n)
        options (dict, optional): Settings for two_state or endogenous model

    Returns:
        pE (dict): Prior expectations
        pC (dict): Prior covariances
        x (torch.Tensor): Prior (initial) states
    """

    # --------------------------------------------------------
    # Number of regions
    n = A.shape[0]

    # --------------------------------------------------------
    # Handle options
    if options is None:
        options = {}
    two_state = options.get('two_state', 0)
    endogenous = options.get('endogenous', 0)

    if D is None:
        D = torch.zeros((n, n, 0), dtype=torch.float32)

    # --------------------------------------------------------
    # Prior (initial) states and shrinkage priors on A
    if two_state:
        x = torch.zeros((n, 6))
    else:
        x = torch.zeros((n, 5))

    if endogenous:
        a = 128.0
    else:
        a = 8.0

    # --------------------------------------------------------
    # Connectivity priors
    if two_state:
        # enforce optimisation of intrinsic (I�E) connections
        A = ((A + torch.eye(n)) > 0).float()

        # prior expectations
        pE = {
            'A': A * 32.0 - 32.0,
            'B': B * 0.0,
            'C': C * 0.0,
            'D': D * 0.0
        }

        # prior covariances
        pC = {
            'A': A / 4.0,
            'B': B / 4.0,
            'C': C * 4.0,
            'D': D / 4.0
        }

    else:
        # enforce self-inhibition
        A = (A > 0).float()
        A = A - torch.diag(torch.diag(A))  # zero out diagonal

        # prior expectations
        pE = {
            'A': A / (64.0 * n) - torch.eye(n) / 2.0,
            'B': B * 0.0,
            'C': C * 0.0,
            'D': D * 0.0
        }

        # prior covariances
        pC = {
            'A': A * a / n + torch.eye(n) / (8.0 * n),
            'B': B.clone(),
            'C': C.clone(),
            'D': D.clone()
        }
        if predicted_U:
            pE['C'] = pE['C'] + torch.cat([(0.05/16)*torch.eye(n), torch.zeros((n,1))], dim=1)
            pC['C'] = pC['C'] - torch.cat([torch.eye(n), torch.ones((n,1))], dim=1) + torch.cat([1e-16*torch.eye(n), torch.zeros((n,1))], dim=1)
        
        if isinstance(A_prior, list):
            pE['A'] = A_prior[0]
            pC['A'] = A_prior[1]*torch.ones(pC['A'].shape)
        
    # --------------------------------------------------------
    # Add hemodynamic priors
    pE['transit'] = torch.zeros((n, 1))
    pE['decay']   = torch.zeros((n, 1))
    pE['epsilon'] = torch.zeros((1, 1))

    pC['transit'] = torch.full((n, 1), math.exp(-6))
    pC['decay']   = torch.full((n, 1), math.exp(-6))
    pC['epsilon'] = torch.full((1, 1), math.exp(-6))

    return pE, pC, x


def tapas_rdcm_get_prior(DCM, predicted_U, A_prior):
    """
    tapas_rdcm_get_prior(DCM)

    Returns prior parameters on model parameters (theta) and noise precision (tau)
    for the specified DCM connectivity pattern.

    Args:
        DCM (dict): Model structure containing fields a, b, c, d.

    Returns:
        m0 (torch.Tensor): Prior means (nr � (nr*nu + nu))
        l0 (torch.Tensor): Prior precisions (same shape as m0)
        a0 (float): Prior shape parameter for noise precision
        b0 (float): Prior rate parameter for noise precision
    """

    # --------------------------------------------------------
    # Number of regions and inputs
    nr, nu = DCM['c'].shape

    # --------------------------------------------------------
    # Get priors on model parameters
    pE, pC, _ = tapas_rdcm_spm_dcm_fmri_priors(DCM['a'], DCM['b'], DCM['c'], DCM['d'], predicted_U=predicted_U, A_prior=A_prior)

    # --------------------------------------------------------
    # Set prior mean of endogenous parameters (A) to zero except diagonal
    if torch.all(torch.diag(pE['A']) == -0.5):
        pE['A'] = torch.zeros_like(pE['A']) + torch.diag(torch.diag(pE['A']))

    # --------------------------------------------------------
    # Construct prior mean m0
    m0 = torch.cat([
        pE['A'],
        pE['B'].reshape(nr, nr * nu),
        pE['C']
    ], dim=1)

    # --------------------------------------------------------
    # Convert prior covariance to precision
    pC['A'] = 1.0 / pC['A']
    pC['B'] = 1.0 / pC['B']
    pC['C'] = 1.0 / pC['C']
    pC['D'] = 1.0 / pC['D']
    pC['transit'] = 1.0 / pC['transit']
    pC['decay'] = 1.0 / pC['decay']
    pC['epsilon'] = 1.0 / pC['epsilon']

    # --------------------------------------------------------
    # Reduce prior precision of baseline term (if present)
    if torch.any(pC['C'][:, -1] != 0) and torch.any(pC['C'][:, -1] != torch.inf):
        pC['C'][:, -1] = 1e-8
    elif torch.all(pC['C'][:, -1] == torch.inf):
        pC['C'][:, -1] = 1e16
        

    # --------------------------------------------------------
    # Construct prior precision l0
    l0 = torch.cat([
        pC['A'],
        pC['B'].reshape(nr, nr * nu),
        pC['C']
    ], dim=1)

    # --------------------------------------------------------
    # Priors on noise precision (gamma distribution)
    a0 = 2.0
    b0 = 1.0

    return m0, l0, a0, b0



def tapas_rdcm_ridge(DCM, X, Y, args, predicted_U, A_prior):
    """
    tapas_rdcm_ridge(DCM, X, Y, args)

    Variational Bayesian inversion of a linear DCM using regression DCM (rDCM).
    Faithful PyTorch translation of the MATLAB version.

    Args:
        DCM (dict): Model structure
        X (torch.Tensor): Design matrix (predictors)
        Y (torch.Tensor): Data (dependent variable)
        args (dict): Argument structure (type, etc.)

    Returns:
        output (dict): Output structure containing posterior parameters, priors, and results.
    """

    # --------------------------------------------------------
    # Precision limit
    pr = 1e-5

    # --------------------------------------------------------
    # Add confound regressor dimensions
    Nc = DCM['U']['X0'].shape[1]
    for _ in range(Nc):
        DCM['b'] = torch.cat((DCM['b'], DCM['b'][:, :, [0]]), dim=2)
        new_c = torch.ones((DCM['c'].shape[0], 1))
        DCM['c'] = torch.cat((DCM['c'], new_c), dim=1)

    # --------------------------------------------------------
    # Get number of regions and inputs
    nr, nu = DCM['c'].shape

    # No baseline regressor for simulations
    if args['type'] == 's':
        DCM['c'][:, -1] = 0.0

    # --------------------------------------------------------
    # Define relevant parameters and get priors
    idx = (torch.cat([
        DCM['a'],
        DCM['b'].reshape(nr, nr * nu),
        DCM['c']
    ], dim=1) > 0)

    m0, l0, a0, b0 = tapas_rdcm_get_prior(DCM, predicted_U, A_prior)

    # --------------------------------------------------------
    # Initialize result arrays
    mN = torch.zeros_like(idx, dtype=torch.float64)
    sN = [None] * nr
    aN = torch.zeros(nr, dtype=torch.float64)
    bN = torch.zeros(nr, dtype=torch.float64)
    logF = torch.zeros(nr, dtype=torch.float64)
    yd_pred_rdcm_fft = torch.full_like(DCM['Y']['y'], torch.nan, dtype=torch.complex64)

    # --------------------------------------------------------
    # Iterate over regions
    for k in range(nr):
        idx_y = ~torch.isnan(Y[:, k])
        X_r = X[idx_y][:, idx[k, :]]
        Y_r = Y[idx_y, k]
        N_eff = idx_y.sum().item()
        D_r = idx[k, :].sum().item()

        # ----------------------------------------------------
        # Read priors
        l0_r = torch.diag(l0[k, idx[k, :]].clone())
        m0_r = m0[k, idx[k, :]].clone().unsqueeze(1)

        # Compute X'X and X'Y
        X_r = X_r.to(torch.complex128)
        Y_r = Y_r.to(torch.complex128)
        W = X_r.conj().T @ X_r
        v = X_r.conj().T @ Y_r.unsqueeze(1)

        # ----------------------------------------------------
        # Initialize parameters
        t = a0 / b0
        aN_r = a0 + N_eff / 2.0
        logF_old = -float('inf')
        convergence = False
        count = 500

        # ----------------------------------------------------
        # VB update loop
        while not convergence:
            sN_r = torch.linalg.inv(t * W + l0_r)
            l0_r = l0_r.to(m0_r.dtype)
            mN_r = sN_r @ (t * v + l0_r @ m0_r)
            QF = ((Y_r.unsqueeze(1) - X_r @ mN_r).conj().T @ (Y_r.unsqueeze(1) - X_r @ mN_r)) / 2.0 + torch.trace(W @ sN_r) / 2.0
            bN_r = b0 + QF
            t = aN_r / bN_r

            # ------------------------------------------------
            # Model evidence components
            log_lik = N_eff * (psi(torch.tensor(aN_r)) - torch.log(bN_r)) / 2.0 - N_eff * torch.log(torch.tensor(2 * torch.pi)) / 2.0 - QF * t
            l0_r = l0_r.to(mN_r.dtype)
            log_p_weight = (
                0.5 * tapas_rdcm_spm_logdet(l0_r)
                - D_r * torch.log(torch.tensor(2 * torch.pi)) / 2.0
                - 0.5 * ((mN_r - m0_r).conj().T @ l0_r @ (mN_r - m0_r))
                - 0.5 * torch.trace(l0_r @ sN_r)
            )
            log_p_prec = (
                a0 * torch.log(torch.tensor(b0))
                - gammaln(torch.tensor(a0))
                + (a0 - 1) * (psi(torch.tensor(aN_r)) - torch.log(bN_r))
                - b0 * t
            )
            log_q_weight = 0.5 * tapas_rdcm_spm_logdet(sN_r) + D_r * (1 + torch.log(torch.tensor(2 * torch.pi))) / 2.0
            log_q_prec = aN_r - torch.log(bN_r) + gammaln(torch.tensor(aN_r)) + (1 - aN_r) * psi(torch.tensor(aN_r))

            logF[k] = log_lik + log_p_prec + log_p_weight + log_q_prec + log_q_weight

            # Check convergence
            if (logF_old - logF[k]).abs().item() ** 2 < pr ** 2:
                convergence = True
            logF_old = logF[k].real
            count -= 1
            if count <= 0:
                break

        # ----------------------------------------------------
        # Final free energy and posterior storage
        mN[k, idx[k, :]] = mN_r.flatten().real
        sN_k = torch.zeros((idx[k, :].sum(), idx[k, :].sum()))
        sN_k[:, :] = sN_r.real
        sN[k] = sN_k
        aN[k] = aN_r.real
        bN[k] = bN_r.real

        # Predicted signal in frequency domain
        yd_pred_rdcm_fft[:, k] = (X[:, idx[k, :]] @ mN_r).flatten()

    # --------------------------------------------------------
    # Store results in output dictionary
    output = tapas_rdcm_store_parameters(DCM, mN, sN, aN, bN, logF.real, None, idx, None, args)

    # Store priors
    output['priors'] = {'m0': m0, 'l0': l0, 'a0': a0, 'b0': b0}

    # Metadata
    output['inversion'] = 'tapas_rdcm_ridge'
    output['temp'] = {
        'yd_source_fft': Y,
        'yd_pred_rdcm_fft': yd_pred_rdcm_fft
    }

    return output


def tapas_rdcm_reduce_zeros(X, Y):
    """
    tapas_rdcm_reduce_zeros(X, Y)

    If there are more zero-valued frequencies than informative ones,
    subsample those frequencies to balance the dataset.

    Args:
        X (torch.Tensor): Design matrix (predictors)
        Y (torch.Tensor): Data (dependent variable)

    Returns:
        (X, Y): Possibly modified tensors (Y may have NaNs replacing dropped rows)
    """

    # --------------------------------------------------------
    # Compute combined signal magnitude (similar to sum(abs([Y X]),2))
    data = torch.sum(torch.abs(torch.cat((Y, X), dim=1)), dim=1)

    # --------------------------------------------------------
    # Identify zero and nonzero frequencies
    idx_0 = torch.where(data == 0)[0]
    n0 = len(idx_0)
    n1 = torch.sum(data > 0).item()

    # --------------------------------------------------------
    # If there are too many zero frequencies, randomly drop some
    if n0 > n1:
        # Create binary mask [zeros(1,n1) ones(1,n0-n1)]
        mask = torch.cat((
            torch.zeros(n1, dtype=torch.int32),
            torch.ones(n0 - n1, dtype=torch.int32)
        ))

        # Random permutation of zero-frequency indices
        perm = torch.randperm(n0)
        idx_del = mask[perm] > 0

        # Mark deleted indices as NaN in Y
        Y[idx_0[idx_del], :] = torch.nan

    return X, Y


def tapas_rdcm_filter(y_fft, u_fft, h_fft, Ny, options):
    """
    tapas_rdcm_filter(y_fft, u_fft, h_fft, Ny, options)

    Specifies informative frequencies and filters the Fourier-transformed signal.

    Args:
        y_fft (torch.Tensor): Initial Fourier-transformed signal (complex, N�nr)
        u_fft (torch.Tensor): Fourier-transformed driving inputs (complex, N�nu)
        h_fft (torch.Tensor): Fourier-transformed HRF (complex, N�1)
        Ny (int): Original signal length (for padding detection)
        options (dict): Estimation options (must contain filter_str and filtu)

    Returns:
        y_fft (torch.Tensor): Filtered Fourier-transformed signal
        idx (torch.BoolTensor): Boolean mask of frequencies to include
    """

    # --------------------------------------------------------
    # Dimensionality
    N, nr = y_fft.shape
    prec = 1e-4

    # --------------------------------------------------------
    # Frequencies not smoothed out by HRF convolution
    h_idx = (torch.abs(h_fft) > prec)

    # Frequencies with non-zero input structure
    if options.get('filtu', 1) == 1:
        u_idx = (torch.sum(torch.abs(u_fft), dim=1) > prec)
    else:
        u_idx = torch.ones(u_fft.shape[0], dtype=torch.bool)

    # --------------------------------------------------------
    # Padding detection
    if N != Ny:
        # MATLAB: [ones(1+round(Ny/2)); zeros(N-Ny-1); ones(Ny/2)]
        y_idx = torch.cat((
            torch.ones(1 + round(Ny / 2), dtype=torch.bool),
            torch.zeros(N - Ny - 1, dtype=torch.bool),
            torch.ones(round(Ny / 2), dtype=torch.bool)
        ), dim=0)
    else:
        y_idx = torch.ones(N, dtype=torch.bool)

    # Indices where h, u, and y are present
    str_idx = h_idx & u_idx & y_idx

    # --------------------------------------------------------
    # Filtering based on noise variance threshold
    thr = options.get('filter_str', 1.0)

    y_real = y_fft.real
    y_imag = y_fft.imag

    if torch.sum(~str_idx) > 1:
        std_real = torch.std(y_real[~str_idx, :], dim=0, keepdim=True)
        std_imag = torch.std(y_imag[~str_idx, :], dim=0, keepdim=True)
        std_real = std_real.repeat(N, 1)
        std_imag = std_imag.repeat(N, 1)
    else:
        std_real = torch.zeros((N, nr))
        std_imag = torch.zeros((N, nr))

    idx = (torch.abs(y_real) > thr * std_real) | (torch.abs(y_imag) > thr * std_imag)

    # --------------------------------------------------------
    # High-pass filtering
    if options.get('filtu', 1) == 1:
        hpf = 16
        freq = round(7 * N / hpf)
    else:
        hpf = max(16 + (thr - 1) * 4, 16)
        freq = round(7 * N / hpf)

    # MATLAB: idx_freq = [ones(1+freq); zeros(N-2*freq-1); ones(freq)]
    idx_freq = torch.cat((
        torch.ones(1 + freq, dtype=torch.bool),
        torch.zeros(N - 2 * freq - 1, dtype=torch.bool),
        torch.ones(freq, dtype=torch.bool)
    ), dim=0).unsqueeze(1).repeat(1, nr)

    # Combine all filters
    idx = idx & str_idx.unsqueeze(1) & idx_freq

    # --------------------------------------------------------
    # FFT symmetry constraints
    idx[0, :] = True  # DC component always included
    idx = idx | idx[torch.cat((torch.tensor([0]), torch.arange(N - 1, 0, -1))), :]

    # Apply filter
    y_fft_filtered = y_fft.clone()
    y_fft_filtered[~idx] = 0

    # --------------------------------------------------------
    # Include all frequencies between first and last informative bins
    for i in range(nr):
        # last informative frequency in first half
        first = None
        for k in range(round(N / 2) + 1):
            if idx[k, i]:
                first = k
        # first informative frequency in second half
        second = None
        for k in range(round(N / 2), N):
            if idx[k, i]:
                second = k
                break

        if first is not None:
            idx[:first + 1, i] = True
        if second is not None:
            idx[second:, i] = True

    return y_fft_filtered, idx



def tapas_rdcm_create_regressors(DCM, options):
    """
    tapas_rdcm_create_regressors(DCM, options)

    Transforms initial DCM signal into a set of regressors X (design matrix)
    and Y (data) in the Fourier domain, performing HRF convolution and optional filtering.

    Args:
        DCM (dict): Model structure containing fields Y.y, U.u, U.dt, etc.
        options (dict): Estimation options.

    Returns:
        X (torch.Tensor): Design matrix (predictors)
        Y (torch.Tensor): Data (dependent variable)
        DCM (dict): Possibly updated model structure
        args (dict): Additional arguments (P, r_dt, type, evalCp)
    """

    # --------------------------------------------------------
    # Circular shift of the stimulus input function
    u_shift = int(options.get('u_shift', 0))
    DCM['U']['u'] = torch.roll(DCM['U']['u'], shifts=u_shift, dims=0)

    # --------------------------------------------------------
    # Unwrap DCM
    y = DCM['Y']['y'].clone()
    u = DCM['U']['u'].clone()
    u_dt = float(DCM['U']['dt'])
    y_dt = float(options['y_dt'])
    r_dt = y_dt / u_dt
    Nu, nu = u.shape
    Ny, nr = y.shape

    # Complex imaginary unit
    ic = 1j

    # --------------------------------------------------------
    # Fourier transform of HRF
    h = options['h']
    h_fft = torch.fft.fft(h)
    h_fft_tr = torch.fft.fft(h[0:int(r_dt):])  # HRF sampled at data rate

    # --------------------------------------------------------
    # Fourier transform of BOLD signal
    y_fft = torch.fft.fft(y, dim=0)

    # --------------------------------------------------------
    # Convolution of stimulus input and HRF
    u_fft = torch.fft.fft(u, dim=0)
    u = torch.fft.ifft(u_fft * h_fft.unsqueeze(1), dim=0).real

    # --------------------------------------------------------
    # Add constant confound if not provided
    if 'X0' not in DCM['U']:
        if DCM['U']['name'][0] != 'null':
            DCM['U']['X0'] = torch.ones((u.shape[0], 1), dtype=torch.float32)
            options['filtu'] = 1
        else:
            DCM['U']['X0'] = torch.zeros((u.shape[0], 0), dtype=torch.float32)
            options['filtu'] = 0

    # Add confounds
    u = torch.cat((u, DCM['U']['X0']), dim=1)

    # --------------------------------------------------------
    # Upsampling / interpolation of BOLD data
    if options.get('padding', 0):
        mid = Ny // 2
        y_fft[mid, :] = y_fft[mid, :] / 2
        pad_len = Nu - Ny - 1
        top = y_fft[:mid + 1, :]
        zeros_pad = torch.zeros((pad_len, nr), dtype=torch.complex64)
        bottom = y_fft[mid + 1:, :]
        y_fft = r_dt * torch.cat((top, zeros_pad, bottom), dim=0)
        Ny = Nu
        u_fft = torch.fft.fft(u / 16.0, dim=0)
    else:
        if r_dt > 1:
            step = int(r_dt)
            u = u[::step, :]
            h_fft = torch.fft.fft(h[::step])
        u_fft = torch.fft.fft(u, dim=0)

    # --------------------------------------------------------
    # Filtering (placeholder: delegate to tapas_rdcm_filter)
    if options.get('filter_str', 0) != 0:
        y_fft, idx = tapas_rdcm_filter(y_fft, u_fft, h_fft, y.shape[0], options)
    else:
        idx = torch.ones_like(y_fft, dtype=torch.bool)

    # --------------------------------------------------------
    # Coefficients for Fourier-domain derivative
    coef = torch.exp(2 * np.pi * ic * torch.arange(0, Ny, dtype=torch.float32) / Ny) - 1
    coef = coef.unsqueeze(1)  # for broadcasting across regions

    # Derivative (finite difference approximation)
    yd_fft = coef * y_fft / y_dt

    # Mask out filtered frequencies
    yd_fft[~idx] = torch.nan

    # --------------------------------------------------------
    # Bilinear term (not supported in rDCM v1.x)
    yu_fft = torch.zeros((Ny, nr * (nu + DCM['U']['X0'].shape[1])),
                         dtype=torch.complex64)
    if options.get('bilinear', 0):
        y_unconv = torch.fft.ifft(y_fft / h_fft_tr.unsqueeze(1), dim=0).real
        for i in range(nr):
            for j in range(nu + DCM['U']['X0'].shape[1]):
                conv_term = torch.fft.fft(y_unconv[:, i] * u[:, j], dim=0)
                yu_fft[:, (j * nr) + i] = conv_term * h_fft

    # --------------------------------------------------------
    # Combine regressors
    if options['type'] == 'r':
        X = torch.cat((y_fft, yu_fft, u_fft), dim=1)
    else:
        if 'scale_u' in options:
            X = torch.cat((y_fft, yu_fft, u_fft), dim=1)
        else:
            X = torch.cat((y_fft, yu_fft, u_fft / r_dt), dim=1)

    # --------------------------------------------------------
    # Create Y matrix and reduce zero rows
    Y = yd_fft
    X, Y = tapas_rdcm_reduce_zeros(X, Y)

    # --------------------------------------------------------
    # Output argument dictionary
    args = {
        'P': [],
        'r_dt': r_dt,
        'type': options['type'],
        'evalCp': options['evalCp']
    }

    return X, Y, DCM, args



def dcm_euler_integration(A, C, U, B, D, rho, alphainv, tau, gamma, kappa, paramList):
    """
    dcm_euler_integration(A, C, U, B, D, rho, alphainv, tau, gamma, kappa, paramList)

    Faithful PyTorch translation of the C MEX Euler integrator used in TAPAS rDCM.

    Integrates the DCM-fMRI dynamical system using Eulers method.

    Args:
        A (torch.Tensor): Connection matrix (transposed).
        C (torch.Tensor): Input contribution (U * C'/16).
        U (torch.Tensor): Input matrix.
        B (torch.Tensor): Bilinear connection tensor (transposed).
        D (torch.Tensor): Nonlinear connection tensor (transposed).
        rho (torch.Tensor): Hemodynamic constant per region (nStates � 1).
        alphainv (float): Hemodynamic constant.
        tau (torch.Tensor): Hemodynamic constant per region (nStates � 1).
        gamma (float): Hemodynamic constant.
        kappa (torch.Tensor): Hemodynamic constant per region (nStates � 1).
        paramList (list or tuple): [timeStep, nTime, nStates, nInputs, _, dcmTypeB, dcmTypeD]

    Returns:
        x, s, f1, v1, q1 (torch.Tensor): Integrated neural and hemodynamic states
                                         each of shape (nTime � nStates)
    """

    # Unpack parameters
    timeStep = float(paramList[0])
    nTime = int(paramList[1])
    nStates = int(paramList[2])
    nInputs = int(paramList[3])
    dcmTypeB = int(paramList[5])
    dcmTypeD = int(paramList[6])

    # Allocate output tensors
    x_out = torch.zeros((nTime, nStates), dtype=torch.float64)
    s_out = torch.zeros((nTime, nStates), dtype=torch.float64)
    f1_out = torch.ones((nTime, nStates), dtype=torch.float64)
    v1_out = torch.ones((nTime, nStates), dtype=torch.float64)
    q1_out = torch.ones((nTime, nStates), dtype=torch.float64)

    # Initialize old hemodynamic log-variables
    oldf1 = torch.zeros((nStates,), dtype=torch.float64)
    oldv1 = torch.zeros((nStates,), dtype=torch.float64)
    oldq1 = torch.zeros((nStates,), dtype=torch.float64)

    # ------------------------------------------------------------
    # Main Euler integration loop
    # ------------------------------------------------------------
    for iStep in range(nTime - 1):
        for jState in range(nStates):

            # ----------------------------------------------------
            # Update x (neuronal activity)
            temp1 = 0.0
            for kIter in range(nStates):
                temp1 += x_out[iStep, kIter] * A[kIter, jState]

            x_new = x_out[iStep, jState] + timeStep * (temp1 + C[iStep, jState])

            # B matrix update (bilinear term)
            if dcmTypeB != 0:
                for kIter in range(nInputs):
                    temp1 = 0.0
                    for mState in range(nStates):
                        temp1 += x_out[iStep, mState] * B[mState, jState, kIter]
                    x_new += timeStep * (U[iStep, kIter] * temp1)

            # D matrix update (nonlinear term)
            if dcmTypeD != 0:
                for kIter in range(nStates):
                    temp1 = 0.0
                    for mState in range(nStates):
                        temp1 += x_out[iStep, mState] * D[mState, jState, kIter]
                    x_new += timeStep * (x_out[iStep, kIter] * temp1)

            x_out[iStep + 1, jState] = x_new

            # ----------------------------------------------------
            # Update s (vasodilatory signal)
            s_new = s_out[iStep, jState] + timeStep * (
                x_out[iStep, jState]
                - kappa[jState] * s_out[iStep, jState]
                - gamma * (f1_out[iStep, jState] - 1.0)
            )
            s_out[iStep + 1, jState] = s_new

            # ----------------------------------------------------
            # Update f (blood flow)
            oldf1[jState] = oldf1[jState] + timeStep * (
                s_out[iStep, jState] / f1_out[iStep, jState]
            )

            # ----------------------------------------------------
            # Update v (blood volume)
            temp1 = torch.pow(v1_out[iStep, jState], alphainv)
            temp2 = (1.0 - torch.pow((1.0 - rho[jState]), (1.0 / f1_out[iStep, jState]))) / rho[jState]
            oldv1[jState] = oldv1[jState] + timeStep * (
                (f1_out[iStep, jState] - temp1)
                / (tau[jState] * v1_out[iStep, jState])
            )

            # ----------------------------------------------------
            # Update q (deoxyhemoglobin content)
            oldq1[jState] = oldq1[jState] + timeStep * (
                (
                    f1_out[iStep, jState] * temp2
                    - temp1 * q1_out[iStep, jState] / v1_out[iStep, jState]
                )
                / (tau[jState] * q1_out[iStep, jState])
            )

            # ----------------------------------------------------
            # Exponentiate to obtain current step states
            f1_out[iStep + 1, jState] = torch.exp(oldf1[jState])
            v1_out[iStep + 1, jState] = torch.exp(oldv1[jState])
            q1_out[iStep + 1, jState] = torch.exp(oldq1[jState])

    return x_out, s_out, f1_out, v1_out, q1_out



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
    mArrayB = Ep['B'].permute(1, 0, 2).clone()

    # --------------------------------------------------------
    # Create empty D if needed
    if torch.sum(Ep['D']) == 0:
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
    dcm_ext = load(
        name="dcm_euler_ext",
        sources=["dcm_euler_integration_extension.cpp"],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False
    )
    # --------------------------------------------------------
    # Neuronal signal and hemodynamic parameters (v, q)
    x, _, _, v, q = dcm_ext.dcm_euler_integration(
        A.double(), C.double(), U.double(), mArrayB.double(), mArrayD.double(),
        oxygenExtractionFraction.double().flatten(), alphainv.double().flatten(), tau.double().flatten(),
        gamma.double().flatten(), kappa.double().flatten(), paramList
        )

    #x, _, _, v, q = dcm_euler_integration_vectorized(
    #    A, C, U, mArrayB, mArrayD,
    #    oxygenExtractionFraction, alphainv, tau, gamma, kappa, paramList
    #)

    # --------------------------------------------------------
    # Constants for BOLD signal equation
    relaxationRateSlope = 25.0
    frequencyOffset = 40.3
    oxygenExtractionFraction = torch.full((1, nr), 0.4, dtype=torch.float32)
    echoTime = 0.04
    restingVenousVolume = 4.0

    # --------------------------------------------------------
    # Coefficients of BOLD signal equation
    coefficientK1 = 4.3 * frequencyOffset * echoTime * oxygenExtractionFraction
    coefficientK2 = (epsilon.T) * (relaxationRateSlope * oxygenExtractionFraction * echoTime)
    coefficientK3 = 1.0 - epsilon.T

    # --------------------------------------------------------
    # Get the Euler indices
    Indices = DCM['M']['idx'].to(torch.long) - 1  # MATLAB is 1-based

    # --------------------------------------------------------
    # Compute BOLD signal time course
    # MATLAB: bsxfun(@times, K1, (1 - q)) + bsxfun(@times, K2, (1 - q./v)) + bsxfun(@times, K3, (1 - v))
    q_sel = q[Indices, :]
    v_sel = v[Indices, :]

    term1 = coefficientK1 * (1.0 - q_sel)
    term2 = coefficientK2 * (1.0 - (q_sel / v_sel))
    term3 = coefficientK3 * (1.0 - v_sel)

    y = restingVenousVolume * (term1 + term2 + term3)

    return y, x


def tapas_dcm_euler_make_indices(DCM):
    """
    tapas_dcm_euler_make_indices(DCM)

    Get the indices for the Euler integration.
    Equivalent to the MATLAB function tapas_dcm_euler_make_indices.m

    Args:
        DCM (dict): Model structure.

    Returns:
        DCM (dict): Updated model structure with time indices in DCM['M']['idx'].
    """

    # --------------------------------------------------------
    # Length of input
    L = DCM['U']['u'].shape[0]

    # Number of regions
    nr = DCM['a'].shape[0] if isinstance(DCM['a'], torch.Tensor) else 1

    # --------------------------------------------------------
    # Check whether delay is specified
    if 'delay' not in DCM:
        DCM['delay'] = torch.ones((nr,), dtype=torch.int32)

    # --------------------------------------------------------
    # Create index array
    # MATLAB: Indices = repmat(1:L, 1, 1)
    Indices = torch.arange(1, L + 1, dtype=torch.int32)

    # --------------------------------------------------------
    # Get indices that coincide with data timepoints
    idx = torch.zeros(L, dtype=torch.int32)

    # MATLAB: idx(DCM.delay:floor(DCM.Y.dt/DCM.U.dt):end) = 1;
    delay_start = int(DCM['delay'][0].item()) if isinstance(DCM['delay'], torch.Tensor) else int(DCM['delay'])
    step = int(torch.floor(torch.tensor(DCM['Y']['dt'] / DCM['U']['dt'])).item())
    idx[delay_start - 1::step] = 1  # MATLAB uses 1-based indexing

    # --------------------------------------------------------
    # Assign timings
    valid_indices = Indices[idx > 0]

    if 'M' not in DCM:
        DCM['M'] = {}
    DCM['M']['idx'] = valid_indices.clone()

    return DCM


def tapas_rdcm_empty_par(DCM):
    """
    tapas_rdcm_empty_par(DCM)

    Creates an empty valid parameter structure for a given DCM.
    Equivalent to the MATLAB function tapas_rdcm_empty_par.m

    Args:
        DCM (dict): Model structure.

    Returns:
        Ep (dict): Parameter structure with zero-initialized tensors.
    """

    # --------------------------------------------------------
    # Number of regions and inputs
    c = DCM['c']
    if c.ndim == 0:
        nr, nu = 1, 1
    else:
        nr, nu = c.shape

    # --------------------------------------------------------
    # Empty parameter structure
    Ep = {}
    Ep['A'] = torch.zeros((nr, nr), dtype=torch.float32)
    Ep['B'] = torch.zeros((nr, nr, nu), dtype=torch.float32)
    Ep['C'] = torch.zeros((nr, nu), dtype=torch.float32)
    Ep['D'] = torch.zeros((nr, nr, 0), dtype=torch.float32)
    Ep['transit'] = torch.zeros((nr, 1), dtype=torch.float32)
    Ep['decay'] = torch.zeros((nr, 1), dtype=torch.float32)
    Ep['epsilon'] = torch.tensor(0.0, dtype=torch.float32)

    return Ep


def tapas_rdcm_get_convolution_bm(options):
    """
    tapas_rdcm_get_convolution_bm(options)
    
    Creates a fixed hemodynamic response function (HRF) by convolving a single
    event (impulse) with the standard Balloon model from DCM.
    
    Args:
        options (dict): Options structure containing the DCM and relevant parameters.
    
    Returns:
        h (torch.Tensor): Hemodynamic response function (HRF).
    """

    # --------------------------------------------------------
    # Get the DCM
    DCM = options['DCM']

    # Number of input data points
    N = DCM['U']['u'].shape[0]

    # --------------------------------------------------------
    # Check if full convolution should be generated
    if ('conv_full' in options) and options['conv_full']:
        r_dt = 1
    else:
        r_dt = N / options['conv_length']

    # --------------------------------------------------------
    # Construct dummy DCM
    DCM_dummy = {}

    DCM_dummy['a'] = torch.tensor([[-1.0]])
    DCM_dummy['b'] = torch.tensor([[0.0]])
    DCM_dummy['c'] = torch.tensor([[1.0]])
    DCM_dummy['d'] = torch.zeros((1, 1, 0), dtype=torch.float32)

    DCM_dummy['Ep'] = tapas_rdcm_empty_par(DCM_dummy)
    DCM_dummy['Ep']['A'] = DCM_dummy['a']
    DCM_dummy['Ep']['C'] = DCM_dummy['c'] * 16.0

    # --------------------------------------------------------
    # Set input of fake DCM
    DCM_dummy['U'] = {}
    DCM_dummy['U']['u'] = torch.zeros((DCM['U']['u'].shape[0],1))
    # MATLAB: DCM.U.u(1:r_dt,:) = 1
    num_impulse_points = int(max(1, round(r_dt)))
    DCM_dummy['U']['u'][:num_impulse_points, :] = 1.0

    # --------------------------------------------------------
    # Set additional parameters
    DCM_dummy['U']['dt'] = DCM['U']['dt']
    DCM_dummy['Y'] = {'dt': DCM['U']['dt']}
    DCM_dummy['n'] = 1
    DCM_dummy['v'] = N
    DCM_dummy['ns'] = N

    # --------------------------------------------------------
    # Create the HRF from the dummy DCM
    DCM_dummy = tapas_dcm_euler_make_indices(DCM_dummy)
    y,_ = tapas_dcm_euler_gen(DCM_dummy, DCM_dummy['Ep'])

    # --------------------------------------------------------
    # Sample HRF at data sampling rate
    step = int(max(1, round(r_dt)))
    h = y[::step, 0].clone()  # MATLAB: y(1:r_dt:end,1)

    return h



def tapas_rdcm_set_options(DCM, input_options=None, type='r'):
    """
    tapas_rdcm_set_options(DCM, input_options=None, type='r')
    
    Sets options for the rDCM analysis to the default settings.
    Equivalent to the MATLAB function tapas_rdcm_set_options.m

    Args:
        DCM (dict): Model structure.
        input_options (dict or None): Estimation options (if None, defaults are used).
        type (str): 'r' for empirical data or 's' for synthetic data.

    Returns:
        options (dict): Estimation options.
    """

    # --------------------------------------------------------
    # Set the data type (empirical or simulated data)
    options = {}
    options['type'] = type

    # --------------------------------------------------------
    # Default settings
    if type == 's':
        options['SNR'] = 3
        options['y_dt'] = 0.5
        options['padding'] = 0
    else:
        options['y_dt'] = DCM['Y']['dt']
        options['padding'] = 0

    # --------------------------------------------------------
    # Check for bilinear terms (not included in current rDCM)
    if ('b' in DCM) and torch.any(DCM['b'] != 0):
        options['bilinear'] = 1
    else:
        options['bilinear'] = 0

    # --------------------------------------------------------
    # Additional default settings
    options['padding'] = 0                   # zero-padding of signal
    options['u_shift'] = 0                   # circular shift of input
    options['filter_str'] = 1                # filtering strength (STD)
    options['coef'] = 1
    options['visualize'] = 1
    options['compute_signal'] = 1
    options['evalCp'] = 0                    # full covariance matrix (optional)

    # --------------------------------------------------------
    # Overwrite defaults with provided input_options
    if input_options is not None and len(input_options) > 0:
        # get all relevant fields from default options
        names = list(options.keys())
        # convolution (h) can exist even if absent in defaults
        names.append('h')

        for name in names:
            if name in input_options:
                options[name] = input_options[name]

    # --------------------------------------------------------
    # rDCM with sparsity constraints (optional fields)
    if input_options is not None:
        if 'p0_all' in input_options:
            options['p0_all'] = input_options['p0_all']
        if 'iter' in input_options:
            options['iter'] = input_options['iter']
        if 'restrictInputs' in input_options:
            options['restrictInputs'] = input_options['restrictInputs']
        if 'p0_inform' in input_options:
            options['p0_inform'] = input_options['p0_inform']

    # --------------------------------------------------------
    # Consistency checks
    if type == 'r':
        options['y_dt'] = DCM['Y']['dt']

    # Sampling ratio between input and signal
    r_dt = options['y_dt'] / DCM['U']['dt']

    # No up/subsampling needed
    if r_dt == 1:
        options['padding'] = 0
        
        
    # --------------------------------------------------------
    # Compute fixed hemodynamic response function (HRF)
    if ('h' not in options) or (len(options['h']) != DCM['U']['u'].shape[0]):
        options['DCM'] = DCM
        options['conv_dt'] = DCM['U']['dt']
        options['conv_length'] = DCM['U']['u'].shape[0]
        options['conv_full'] = True
        options['h'] = tapas_rdcm_get_convolution_bm(options)
    
    return options


def tapas_rdcm_estimate(DCM, type, options=None, methods=1, predicted_U=False, A_prior=None):
    """
    tapas_rdcm_estimate(DCM, type, options=None, methods=1)
    
    Main analysis function for regression DCM (rDCM) analysis.
    
    Args:
        DCM (dict or str): Either the DCM model structure (dict) or filename to load.
        type (str): 'r' for empirical data or 's' for simulated data.
        options (dict, optional): Estimation options. If None, default options are set.
        methods (int): (1) rDCM (original) or (2) rDCM with sparsity.

    Returns:
        output (dict): Output structure.
        options (dict): Estimation options.
    """
    # --------------------------------------------------------
    # Store RNG seed
    rng_seed = torch.seed()
    np.random.seed(int(rng_seed % (2**32 - 1)))
    random.seed(int(rng_seed % (2**32 - 1)))

    # --------------------------------------------------------
    # Display header
    if not ('M' in DCM and 'noprint' in DCM['M'] and DCM['M']['noprint']):
        print('\n========================================================')
        print('Regression dynamic causal modeling (rDCM)')
        print('========================================================\n')

    # --------------------------------------------------------
    # Load the DCM if it's a filename
    if isinstance(DCM, str):
        # Assuming DCM file is a .pt or .pkl serialized dict
        DCM = torch.load(DCM)

    # --------------------------------------------------------
    # Start timer
    current_timer = time.time()

    # --------------------------------------------------------
    # Check for endogenous DCMs (no exogenous driving effects)
    if (('c' not in DCM or DCM['c'] is None) or
        ('U' not in DCM or 'u' not in DCM['U'] or DCM['U']['u'] is None) or
        ('name' not in DCM['U'] or DCM['U']['name'][0] == 'null')):

        T = DCM['Y']['y'].shape[0]
        DCM['U'] = {}
        DCM['U']['u'] = torch.zeros((T * 16, 1), dtype=torch.float32)
        DCM['U']['name'] = ['null']
        DCM['U']['dt'] = DCM['Y']['dt'] / 16.0

        n = DCM['n']
        DCM['b'] = torch.zeros((n, n, DCM['U']['u'].shape[1]), dtype=torch.float32)
        DCM['c'] = torch.zeros((n, DCM['U']['u'].shape[1]), dtype=torch.float32)
        DCM['d'] = torch.zeros((n, n, 0), dtype=torch.float32)

        if options is None:
            options = {}
        options['compute_signal'] = 0
    
    # --------------------------------------------------------
    # Create options
    options = tapas_rdcm_set_options(DCM, options, type)

    # --------------------------------------------------------
    # Create regressors
    X, Y, DCM, args = tapas_rdcm_create_regressors(DCM, options)

    # --------------------------------------------------------
    # Display start
    if not ('M' in DCM and 'noprint' in DCM['M'] and DCM['M']['noprint']):
        print('Run model inversion')

    # --------------------------------------------------------
    # Evaluate results (model inversion)
    if methods == 1:
        # rDCM (original)
        output = tapas_rdcm_ridge(DCM, X, Y, args, predicted_U, A_prior)

    elif methods == 2:
        # Specify default grid for optimizing sparsity hyperparameter p0
        if 'p0_all' not in options:
            options['p0_all'] = np.arange(0.05, 0.95 + 0.001, 0.05)

        output_all = [None] * len(options['p0_all'])
        F_all = torch.full((len(options['p0_all']),), float('nan'))

        reverse_str = ''

        if 'restrictInputs' in options:
            args['restrictInputs'] = options['restrictInputs']

        for p0_counter, p0_val in enumerate(options['p0_all']):
            args['p0_temp'] = float(p0_val)
            if 'iter' in options:
                args['iter'] = options['iter']
            if 'p0_inform' in options:
                args['p0_inform'] = options['p0_inform']

            if len(options['p0_all']) != 1:
                args['verbose'] = 0
            else:
                args['verbose'] = 1

            if not ('M' in DCM and 'noprint' in DCM['M'] and DCM['M']['noprint']):
                msg = f'Processing p0: {p0_counter+1}/{len(options["p0_all"])}'
                sys.stdout.write('\r' + msg)
                sys.stdout.flush()

            # Run sparse rDCM
            output_temp = tapas_rdcm_sparse(DCM, X, Y, args)
            output_all[p0_counter] = output_temp[0]

            # Get the negative free energy
            F_all[p0_counter] = output_all[p0_counter]['logF']

        # Find optimal hyperparameter (p0)
        F_max_ind = int(torch.argmax(F_all))
        output = output_all[F_max_ind]

    # --------------------------------------------------------
    # Output elapsed time
    time_rDCM_VBinv = time.time() - current_timer

    # --------------------------------------------------------
    # Display finalizing
    if not ('M' in DCM and 'noprint' in DCM['M'] and DCM['M']['noprint']):
        print('\nFinalize results')

    # --------------------------------------------------------
    # Compute statistics and predicted signal
    output = tapas_rdcm_compute_statistics(DCM, output, options)

    # --------------------------------------------------------
    # Store metadata
    if 'time' not in output:
        output['time'] = {}
    output['time']['time_rDCM_VBinv'] = time_rDCM_VBinv

    output['rngSeed'] = int(rng_seed)
    output['ver'] = '2022_v1.4'

    return output, options



def tapas_rdcm_model_specification(Y, U=None, args=None):
    """
    tapas_rdcm_model_specification(Y, U=None, args=None)

    Specifies a whole-brain DCM structure for use with the TAPAS rDCM pipeline.

    Args:
        Y (dict): Data structure with keys:
            - 'y': BOLD data (N � R)
            - 'dt': sampling time [s]
            - 'name' (optional): list of region names
        U (dict, optional): Input structure with keys:
            - 'u': inputs ((16*N) � U) or (N � U)
            - 'name' (optional): list of input names
        args (dict, optional): Optional fields 'a' and 'c' to override connectivity matrices.

    Returns:
        DCM (dict): Dynamic Causal Model structure.
    """

    # --------------------------------------------------------
    # Validate data and input fields
    if 'y' not in Y or (U is not None and 'u' not in U):
        print("\nERROR: Data (y) and/or inputs (u) not correctly specified!")
        print("Please double-check...\n")
        return None

    DCM = {}
    DCM['a'] = torch.tensor([])
    DCM['b'] = torch.tensor([])
    DCM['c'] = torch.tensor([])
    DCM['d'] = torch.tensor([])

    # --------------------------------------------------------
    # Data setup
    DCM['Y'] = {'y': torch.tensor(Y['y'], dtype=torch.float32), 'dt': float(Y['dt'])}

    # Region names
    if 'name' in Y and isinstance(Y['name'], list):
        DCM['Y']['name'] = Y['name']
    else:
        DCM['Y']['name'] = [f"region_{i+1}" for i in range(DCM['Y']['y'].shape[1])]

    # --------------------------------------------------------
    # Input specification
    if U is not None and len(U) > 0:
        N_y, R = DCM['Y']['y'].shape
        U_u = torch.tensor(U['u'], dtype=torch.float32)
        if U_u.shape[0] == N_y * 16:
            u_temp = U_u
        elif U_u.shape[0] == N_y:
            u_temp = torch.zeros((N_y * 16, U_u.shape[1]))
            for i in range(U_u.shape[1]):
                uu = U_u[:, i].repeat_interleave(16)
                u_temp[:, i] = uu
        else:
            print("\nERROR: Dimensionality of data (y) and inputs (u) does not match!")
            print("Please double-check...\n")
            return None

        DCM['U'] = {
            'u': u_temp,
            'dt': DCM['Y']['dt'] / 16.0,
            'name': U.get('name', [f"input_{i+1}" for i in range(U_u.shape[1])])
        }

        # --------------------------------------------------------
        # Connectivity matrices for task-based DCM
        R = DCM['Y']['y'].shape[1]
        U_dim = U_u.shape[1]
        DCM['a'] = torch.ones((R, R))
        DCM['b'] = torch.zeros((R, R, U_dim))
        DCM['c'] = torch.ones((R, U_dim))
        DCM['d'] = torch.zeros((R, R, 0))

        # Overwrite defaults if specified
        if args:
            if 'a' in args:
                DCM['a'] = torch.tensor(args['a'], dtype=torch.float32)
            if 'c' in args:
                DCM['c'] = torch.tensor(args['c'], dtype=torch.float32)

    else:
        # --------------------------------------------------------
        # Resting-state model
        print("\nNOTE: No inputs specified! Assuming resting-state model...\n")

        R = DCM['Y']['y'].shape[1]
        DCM['a'] = torch.ones((R, R))
        DCM['b'] = torch.zeros((R, R, 0))
        DCM['c'] = torch.zeros((R, 0))
        DCM['d'] = torch.zeros((R, R, 0))

        if args and 'a' in args:
            DCM['a'] = torch.tensor(args['a'], dtype=torch.float32)

    # --------------------------------------------------------
    # Model dimensions
    DCM['v'] = DCM['Y']['y'].shape[0]
    DCM['n'] = DCM['Y']['y'].shape[1]

    # --------------------------------------------------------
    # Default delay and options fields
    DCM['delays'] = torch.full((DCM['v'], 1), DCM['Y']['dt'] / 2.0)
    DCM['options'] = {
        'nonlinear': 0,
        'two_state': 0,
        'stochastic': 0,
        'centre': 0,
        'endogenous': 0
    }

    return DCM

if __name__ == '__main__':
    from scipy.io import loadmat
    data = loadmat('data.mat')
    
    # Example usage
    torch.set_default_device("cpu")
    cpu = torch.device('cpu')
    
    Y = {
         'y': data['y'],
         'dt': data['dt']
         }
    
    U = {
         'u': data['u'].toarray()
         }
    
    args = {
            'a': np.ones((3,3)),
            'c': np.ones((3,3))
            }
    
    data = loadmat('test_rdcm_data.mat')
    Y = {
         'y': data['y'],
         'dt': data['TR']
         }
    
    # --- fMRI data (3 regions, 10 timepoints)
    #Y = {
    #    'y': torch.randn(360, 100, device=cpu).numpy(),  # 10 time points � 3 regions
    #    'dt': 2.0,  # TR = 2s
    #}
    
    # --- Input (1 task regressor, length matches Y.y)
    #U = {
    #    'u': torch.sin(torch.linspace(0, 2 * torch.pi, 360, device=cpu)).unsqueeze(1).numpy(),
    #    'name': ['stimulus']
    #}
    
    # --- Optional arguments (custom connectivity)
    #args = {
    #    'a': torch.ones((100,100), device=cpu).numpy(),  # self-connections only
    #    'c': torch.zeros((100,1), device=cpu).numpy()  # input to first region
    #}
    #args['c'][0] = 1
    
    # --- Create DCM structure
    #DCM = tapas_rdcm_model_specification(Y, U, args)
    DCM = tapas_rdcm_model_specification(Y)
    
    
    # --- Inspect result
    with torch.no_grad():
        out = tapas_rdcm_estimate(DCM ,'r', None, 1)



