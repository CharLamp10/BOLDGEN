import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, resample



def plot_three_subplots(x, data, title, cat, filename):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    indices = [0, 99]
    for ax, idx in zip(axes, indices):
        ax.plot(x, data[:, idx], linewidth=2.5)
        ax.set_ylabel(f'ROI {idx+1}', fontsize=24)
        ax.tick_params(axis='both', labelsize=22)

    axes[-1].set_xlabel('Time (sec)', fontsize=24)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.join('plots', 'BOLD_plots', cat), exist_ok=True)
    plt.savefig(
        os.path.join('plots', 'BOLD_plots', cat, filename),
        dpi=600,
        bbox_inches='tight'
    )
    plt.show()



def main(config):           
    data = np.load(os.path.join(config['path_data'],'train_data_HCP_SCH.npz'))
    train_data = []
    for key in data:
        y = data[key][183:-184,:]
        if config['case'] == "full_original":
            y = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
            order = 4
            cutoff = 0.1
            fs = 1/config['TR']
            newTR = config['TR']
            b, a = butter(order, cutoff, btype='low', fs=fs)
            y = filtfilt(b, a, y.T).T.astype('float32')
        elif config['case'] == "half_original":
            y = y[125:708]
            y = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
            order = 4
            cutoff = 0.1
            fs = 1/config['TR']
            newTR = config['TR']
            b, a = butter(order, cutoff, btype='low', fs=fs)
            y = filtfilt(b, a, y.T).T.astype('float32')
        elif config['case'] == "full_subsampled":
            y = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
            order = 4
            cutoff = 0.1
            fs = 1/config['TR']
            b, a = butter(order, cutoff, btype='low', fs=fs)
            y = filtfilt(b, a, y.T).T.astype('float32')
            newTR = 2
            rate = newTR/config['TR']
            new_samples = int(np.round(y.shape[0] / rate))
            y = resample(y, new_samples)
            y = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
        elif config['case'] == "half_subsampled":
            y = y[125:708]
            y = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
            order = 4
            cutoff = 0.1
            fs = 1/config['TR']
            b, a = butter(order, cutoff, btype='low', fs=fs)
            y = filtfilt(b, a, y.T).T.astype('float32')
            newTR = 2
            rate = newTR/config['TR']
            new_samples = int(np.round(y.shape[0] / rate))
            y = resample(y, new_samples)
            y = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
        train_data.append(y)
            
    augmented_rdcmgen = torch.load(os.path.join(config['path_augmented_rdcmgen'], 'augmented_rDCM-GEN_' + config['case'] + '.pt')).numpy()
    augmented_crvae = torch.load(os.path.join(config['path_augmented_crvae'], 'augmented_CR-VAE_' + config['case'] + '.pt')).numpy()
    augmented_timevae = np.load(os.path.join(config['path_augmented_timevae'], config['case'], 'timeVAE_' + config['case'] + '_prior_samples.npy'))
    augmented_timegan = np.load(os.path.join(config['path_augmented_timegan'], 'augmented_TimeGAN_HCP_2026_' + config['case'] + '.npy'))
    augmented_t2idiff = np.load(os.path.join(config['path_augmented_t2idiff'], config['case'], 'gen_sig_' + config['case'] + '_fold1.npy'))
    augmented_dsfm = np.load(os.path.join(config['path_augmented_dsfm'], config['case'], 'samples', 'samples.npy'))
    
    i = 26 #10, 13, 23
    data = train_data[i]
    data_rdcmgen = augmented_rdcmgen[i,7][:data.shape[0]] #1, 7
    data_crvae = augmented_crvae[i,5][:data.shape[0]] #2, 5
    data_timevae = augmented_timevae[0][:data.shape[0]]
    data_timevae = (data_timevae - np.mean(data_timevae, axis=0, keepdims=True))/np.std(data_timevae, axis=0, keepdims=True)
    data_timegan = augmented_timegan[0][:data.shape[0]]
    data_timegan = (data_timegan - np.mean(data_timegan, axis=0, keepdims=True))/np.std(data_timegan, axis=0, keepdims=True)
    data_t2idiff = augmented_t2idiff[0][:data.shape[0]]
    data_t2idiff = (data_t2idiff - np.mean(data_t2idiff, axis=0, keepdims=True))/np.std(data_t2idiff, axis=0, keepdims=True)
    data_dsfm = augmented_dsfm[0][:data.shape[0]]
    data_dsfm = (data_dsfm - np.mean(data_dsfm, axis=0, keepdims=True))/np.std(data_dsfm, axis=0, keepdims=True)
    
    if 'full' in config['case']:
        dur = 600
    else:
        dur = 420
    
    x = np.linspace(0, dur, data.shape[0])

    plot_three_subplots(x, data, 'Original', config['case'], 'time_series_original.png')
    
    plot_three_subplots(x, data_timevae, 'TimeVAE', config['case'], 'time_series_TimeVAE.png')
    
    plot_three_subplots(x, data_timegan, 'TimeGAN', config['case'], 'time_series_TimeGAN.png')
    
    plot_three_subplots(x, data_crvae, 'CRVAE', config['case'], 'time_series_CRVAE.png')
    
    plot_three_subplots(x, data_t2idiff, 'T2IDIFF', config['case'], 'time_series_T2IDIFF.png')
    
    plot_three_subplots(x, data_dsfm, 'DSFM', config['case'], 'time_series_DSFM.png')
    
    plot_three_subplots(x, data_rdcmgen, 'rDCM-GEN', config['case'], 'time_series_rDCM-GEN.png')


if __name__ == "__main__":
    config = {}
    config['path_data'] = r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_GNN_data/HCP_2026'
    config['path_save'] = r'/home/student1/Desktop/Charalampos_Lamprou/rDCM-GEN'
    config['path_augmented_rdcmgen'] = r'/media/student1/ExtremePro/rDCM-GEN/rDCM-GEN_results/augmented_rDCM-GEN_noFCloss'
    config['path_augmented_crvae'] = r'/media/student1/ExtremePro/CRVAE_augmented'
    config['path_augmented_timevae'] = r'/media/student1/ExtremePro/TimeVAE_results/outputs/gen_data'
    config['path_augmented_timegan'] = r'/media/student1/ExtremePro/TimeGAN_results'
    config['path_augmented_t2idiff'] = r'/media/student1/ExtremePro/T2IDIFF'
    config['path_augmented_dsfm'] = r'/media/student1/ExtremePro/DSFM_results'
    config['case'] = 'full_original'
    config['TR'] = 0.72
    main(config)
