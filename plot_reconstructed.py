import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
from MSM import random_timepoint_mask


path = r'/home/student1/Desktop/Charalampos_Lamprou/rDCM-GEN_results/MSM'
methods = ['Noise-6dB','Noise-3dB', 'CRVAE', 'TimeVAE', 'TimeGAN','Diffusion-TS','rDCM-GEN']
T = 400
R = 100

for method in methods:
    with open(os.path.join(path, method, 'results.pkl'), 'rb') as f:
        result = pickle.load(f)
    mask = result['mask']
    original = result['original']
    predicted = result['pred']
    mask_np = random_timepoint_mask(original.shape[0], T, n_mask_points=100, idxs=mask)
    
    for sample_i in range(1,11):
        rois = [0, 99]
        t = np.arange(T)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        for ax in axes:
            ax.tick_params(axis='both', which='major', labelsize=18)
        
        for ax, roi_i in zip(axes, rois):
            ax.plot(
                t,
                original[sample_i, roi_i],
                label='target'
            )
            ax.plot(
                t,
                predicted[sample_i, roi_i],
                label='pred',
                alpha=0.8
            )
        
            masked = mask_np[sample_i]
            ax.scatter(
                t[masked],
                original[sample_i, roi_i][masked],
                color='red',
                s=20,
                label='masked',
                zorder=3
            )
        
            ax.set_title(f"ROI {roi_i + 1}", fontsize=22)
        
        # Legend only in the first subplot
        #axes[0].legend(loc='upper right', fontsize=16)
        
        axes[-1].set_xlabel('Time (sec)', fontsize=22)
        
        plt.tight_layout()
        
        save_dir = os.path.join('plots', 'reconstructed_plots', method)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, method + '_subject_' + str(sample_i) + '_rois.png'),
            dpi=600,
            bbox_inches='tight'
        )
        
        plt.show()