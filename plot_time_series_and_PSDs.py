import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.size': 20,          # base font size
    'axes.titlesize': 22,     # subplot titles
    'axes.labelsize': 16,     # x/y labels
    'xtick.labelsize': 12,    # x tick labels
    'ytick.labelsize': 12,    # y tick labels
    'legend.fontsize': 18,   # legend text
    'figure.titlesize': 24   # suptitle
})

def plot_three_subplots(x, data, title, filename):
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    indices = [0, 49, 99]
    for ax, idx in zip(axes, indices):
        ax.plot(x, data[:400, idx])
        ax.set_ylabel(f'ROI {idx+1}')

    axes[-1].set_xlabel('Time (sec)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join('plots', filename), dpi=600, bbox_inches='tight')
    plt.show()



def main(config):           
    data = np.load(os.path.join(config['path_data'],'train_data_HCP_SCH_resampled.npz'))
    train_data = []
    for key in data:
        temp = data[key]
        temp = (temp - np.mean(temp, axis=0, keepdims=True))/np.std(temp, axis=0, keepdims=True)
        train_data.append(temp)
            
    augmented_rdcm = torch.load(config['path_augmented_rdcm']).numpy()
    augmented_crvae = torch.load(config['path_augmented_crvae']).numpy()
    augmented_timevae = np.load(config['path_augmented_timevae'])['data']
    augmented_timegan = np.load(config['path_augmented_timegan'])
    augmented_diffusion_ts = np.load(config['path_augmented_diffusion_ts'])
    
    data = train_data[0]
    data_rdcm = augmented_rdcm[0,0]
    data_crvae = augmented_crvae[0,5]
    data_noise05 = train_data[0] + np.random.randn(*data.shape)*0.5
    data_noise075 = train_data[0] + np.random.randn(*data.shape)*0.75
    data_timevae = augmented_timevae[1000]
    data_timegan = augmented_timegan[1000]
    data_diffusion_ts = augmented_diffusion_ts[1000]
    
    x = np.linspace(0, 600, 400)

    plot_three_subplots(x, data, 'Original', 'time_series_original.png')

    plot_three_subplots(x, data_noise05, 'Noise-6dB', 'time_series_Noise-6dB.png')
    
    plot_three_subplots(x, data_noise075, 'Noise-3dB', 'time_series_Noise-3dB.png')
    
    plot_three_subplots(x, data_timevae, 'TimeVAE', 'time_series_TimeVAE.png')
    
    plot_three_subplots(x, data_timegan, 'TimeGAN', 'time_series_TimeGAN.png')
    
    plot_three_subplots(x, data_crvae, 'CRVAE', 'time_series_CRVAE.png')
    
    plot_three_subplots(x, data_diffusion_ts, 'Diffusion-TS', 'time_series_Diffusion-TS.png')
    
    plot_three_subplots(x, data_rdcm, 'rDCM-GEN', 'time_series_rDCM-GEN.png')


if __name__ == "__main__":
    config = {}
    config['path_data'] = r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_GNN_data/HCP'
    config['path_save'] = r'/home/student1/Desktop/Charalampos_Lamprou/rDCM-GEN_results'
    config['path_augmented_rdcm'] = r'/media/student1/Extreme Pro/augmented_new/augmented_rDCM-GEN.pt'
    config['path_augmented_crvae'] = r'/media/student1/Extreme Pro/CRVAE_models/augmented_CRVAE.pt'
    config['path_augmented_timevae'] = r'/media/student1/Extreme Pro/TimeVAE_results/outputs/gen_data/HCP_train/timeVAE_HCP_train_prior_samples.npz'
    config['path_augmented_timegan'] = r'/media/student1/Extreme Pro/TimeGAN_results/augmented_TimeGAN.npy'
    config['path_augmented_diffusion_ts'] = r'/media/student1/Extreme Pro/Diffusion-TS_results/augmented_Diffusion-TS.npy'
    main(config)
