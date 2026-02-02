import torch
import os
import numpy as np
import pandas as pd


def main(config):
    data = np.load(os.path.join(config['path_data'],'train_data_HCP_SCH_resampled.npz'))
    train_data = []
    for key in data:
        train_data.append(data[key])
    
    data_rDCM = torch.load(os.path.join(config['path_rdcm'])).detach().cpu().numpy()[0:3][:,5,:,:]
    data_timeVAE = np.load(config['path_timeVae'])['data'][50:53]
    data_timeVAE = (data_timeVAE - np.mean(data_timeVAE, axis=1, keepdims=True))/np.std(data_timeVAE, axis=1, keepdims=True)
    data_timeGAN = np.load(config['path_timeGan'])[50:53]
    data_timeGAN = (data_timeGAN - np.mean(data_timeGAN, axis=1, keepdims=True))/np.std(data_timeGAN, axis=1, keepdims=True)
    data_diffusionTS = np.load(config['path_diffusion_ts'])[50:53]
    data_diffusionTS = (data_diffusionTS - np.mean(data_diffusionTS, axis=1, keepdims=True))/np.std(data_diffusionTS, axis=1, keepdims=True)
    data_crvae = torch.load(config['path_crvae']).cpu().numpy()[0:3][:,5,:,:]
    data_crvae = (data_crvae - np.mean(data_crvae, axis=1, keepdims=True))/np.std(data_crvae, axis=1, keepdims=True)
    data_orig = train_data[0:3]
    for i, data in enumerate(data_orig):
        data = data[88:-88]
        data_orig[i] = (data - np.mean(data, axis=0, keepdims=True))/np.std(data, axis=0, keepdims=True)
    data_orig = np.concatenate(data_orig, axis=0)
    data_noise05 = data_orig + np.random.randn(*data_orig.shape)*0.5
    data_noise075 = data_orig + np.random.randn(*data_orig.shape)*0.75
    data_rDCM = np.reshape(data_rDCM, (data_rDCM.shape[0]*data_rDCM.shape[1], data_rDCM.shape[2]))
    data_crvae = np.reshape(data_crvae, (data_crvae.shape[0]*data_crvae.shape[1], data_crvae.shape[2]))
    data_timeVAE = np.reshape(data_timeVAE, (data_timeVAE.shape[0]*data_timeVAE.shape[1], data_timeVAE.shape[2]))
    data_timeGAN = np.reshape(data_timeGAN, (data_timeGAN.shape[0]*data_timeGAN.shape[1], data_timeGAN.shape[2]))
    data_diffusionTS = np.reshape(data_diffusionTS, (data_diffusionTS.shape[0]*data_diffusionTS.shape[1], data_diffusionTS.shape[2]))
            
    result = np.concatenate((data_orig, data_noise05), axis=0)
    labels = [0 for i in range(1200)] + [1 for i in range(1200)]
    labels = np.array(labels)
    labels = pd.DataFrame(labels)
    labels.to_csv(os.path.join(config['path_save'], 'Noise-6dB_labels.csv'))
    os.makedirs(os.path.join(config['path_save'], 'Noise-6dB'), exist_ok=True)
    np.save(os.path.join(config['path_save'], 'Noise-6dB', 'sub-04_Noise-6dB_samples.npy'), result)
    
    result = np.concatenate((data_orig, data_noise075), axis=0)
    labels = [0 for i in range(1200)] + [1 for i in range(1200)]
    labels = np.array(labels)
    labels = pd.DataFrame(labels)
    labels.to_csv(os.path.join(config['path_save'], 'Noise-3dB_labels.csv'))
    os.makedirs(os.path.join(config['path_save'], 'Noise-3dB'), exist_ok=True)
    np.save(os.path.join(config['path_save'], 'Noise-3dB', 'sub-04_Noise-3dB_samples.npy'), result)
    
    result = np.concatenate((data_orig, data_timeVAE), axis=0)
    labels = [0 for i in range(1200)] + [1 for i in range(1200)]
    labels = np.array(labels)
    labels = pd.DataFrame(labels)
    labels.to_csv(os.path.join(config['path_save'], 'TimeVAE_labels.csv'))
    os.makedirs(os.path.join(config['path_save'], 'TimeVAE'), exist_ok=True)
    np.save(os.path.join(config['path_save'], 'TimeVAE', 'sub-04_TimeVAE_samples.npy'), result)
    
    result = np.concatenate((data_orig, data_timeGAN), axis=0)
    labels = [0 for i in range(1200)] + [1 for i in range(1200)]
    labels = np.array(labels)
    labels = pd.DataFrame(labels)
    labels.to_csv(os.path.join(config['path_save'], 'TimeGAN_labels.csv'))
    os.makedirs(os.path.join(config['path_save'], 'TimeGAN'), exist_ok=True)
    np.save(os.path.join(config['path_save'], 'TimeGAN', 'sub-04_TimeGAN_samples.npy'), result)
    
    result = np.concatenate((data_orig, data_diffusionTS), axis=0)
    labels = [0 for i in range(1200)] + [1 for i in range(1200)]
    labels = np.array(labels)
    labels = pd.DataFrame(labels)
    labels.to_csv(os.path.join(config['path_save'], 'Diffusion-TS_labels.csv'))
    os.makedirs(os.path.join(config['path_save'], 'Diffusion-TS'), exist_ok=True)
    np.save(os.path.join(config['path_save'], 'Diffusion-TS', 'sub-04_Diffusion-TS_samples.npy'), result)
    
    result = np.concatenate((data_orig, data_crvae), axis=0)
    labels = [0 for i in range(1200)] + [1 for i in range(1200)]
    labels = np.array(labels)
    labels = pd.DataFrame(labels)
    labels.to_csv(os.path.join(config['path_save'], 'CRVAE_labels.csv'))
    os.makedirs(os.path.join(config['path_save'], 'CRVAE'), exist_ok=True)
    np.save(os.path.join(config['path_save'], 'CRVAE', 'sub-04_CRVAE_samples.npy'), result)
    
    result = np.concatenate((data_orig, data_rDCM), axis=0)
    labels = [0 for i in range(1200)] + [1 for i in range(1200)]
    labels = np.array(labels)
    labels = pd.DataFrame(labels)
    labels.to_csv(os.path.join(config['path_save'], 'rDCM-GEN_labels.csv'))
    os.makedirs(os.path.join(config['path_save'], 'rDCM-GEN'), exist_ok=True)
    np.save(os.path.join(config['path_save'], 'rDCM-GEN', 'sub-04_rDCM-GEN_samples.npy'), result)
    
    
    result = np.concatenate((data_orig[:300], data_noise05[:300], data_noise075[:300],  data_timeVAE[:300], data_timeGAN[:300], data_diffusionTS[:300], data_crvae[:300], data_rDCM[:300]), axis=0)
    labels = [0 for i in range(300)] + [1 for i in range(300)] + [2 for i in range(300)] + [3 for i in range(300)] + [4 for i in range(300)] + [5 for i in range(300)] + [6 for i in range(300)] + [7 for i in range(300)]
    labels = np.array(labels)
    labels = pd.DataFrame(labels)
    labels.to_csv(os.path.join(config['path_save'], 'comparison_labels.csv'))
    os.makedirs(os.path.join(config['path_save'], 'comparison'), exist_ok=True)
    np.save(os.path.join(config['path_save'], 'comparison', 'sub-04_comparison_samples.npy'), result)
    
if __name__ == "__main__":
    config = {}
    config['path_rdcm'] = r'/media/student1/Extreme Pro/augmented_new/augmented_rDCM-GEN.pt'
    config['path_crvae'] = r'/media/student1/Extreme Pro/CRVAE_models/augmented_CRVAE.pt'
    config['path_timeGan'] = r'/media/student1/Extreme Pro/TimeGAN_results/augmented_TimeGAN.npy'
    config['path_timeVae'] = r'/media/student1/Extreme Pro/TimeVAE_results/outputs/gen_data/HCP_train/timeVAE_HCP_train_prior_samples.npz'
    config['path_diffusion_ts'] = r'/media/student1/Extreme Pro/Diffusion-TS_results/augmented_Diffusion-TS.npy'
    config['path_data'] = r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_GNN_data/HCP'
    config['path_save'] = r'/home/student1/Desktop/Charalampos_Lamprou/BCNE/data'
    main(config)


