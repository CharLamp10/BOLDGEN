import torch
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample



def preprocess_HCP(data, cat):
    for i, y in enumerate(data):
        if cat == "full_original":
            y_train = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
            order = 4
            cutoff = 0.1
            fs = 1/config['TR']
            b, a = butter(order, cutoff, btype='low', fs=fs)
            y = filtfilt(b, a, y.T).T.astype('float32')
        elif cat == "half_original":
            y = (y[125:708] - np.mean(y[125:708], axis=0, keepdims=True))/np.std(y[125:708], axis=0, keepdims=True)
            order = 4
            cutoff = 0.1
            fs = 1/config['TR']
            b, a = butter(order, cutoff, btype='low', fs=fs)
            y = filtfilt(b, a, y.T).T.astype('float32')
        elif cat == "full_subsampled":
            y_train = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
            order = 4
            cutoff = 0.1
            fs = 1/config['TR']
            b, a = butter(order, cutoff, btype='low', fs=fs)
            y = filtfilt(b, a, y.T).T.astype('float32')
            rate = config['newTR']/config['TR']
            new_samples = int(np.round(y_train.shape[0] / rate))
            y = resample(y, new_samples)
            y = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
        elif cat == "half_subsampled":
            y = (y[125:708] - np.mean(y[125:708], axis=0, keepdims=True))/np.std(y[125:708], axis=0, keepdims=True)
            order = 4
            cutoff = 0.1
            fs = 1/config['TR']
            b, a = butter(order, cutoff, btype='low', fs=fs)
            y = filtfilt(b, a, y.T).T.astype('float32')
            rate = config['newTR']/config['TR']
            new_samples = int(np.round(y_train.shape[0] / rate))
            y = resample(y, new_samples)
            y = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
        data[i] = y
    return data



def main(config):
    data = np.load(os.path.join(config['path_data'],'train_data_HCP_SCH.npz'))
    train_data = []
    for key in data:
        temp = data[key][183:-184]
        train_data.append(temp[:400])
    
    data_crvae = torch.load(os.path.join(config['path_augmented_crvae'], 'augmented_CR-VAE_' + config['case'] + '.pt'))
    data_rDCM = torch.load(os.path.join(config['path_augmented_rdcmgen'], 'augmented_rDCM-GEN_' + config['case'] + '.pt'))
    data_timeVAE = np.load(os.path.join(config['path_augmented_timevae'], config['case'], 'timeVAE_HCP_2026_' + config['case'] + '_prior_samples.npz'))['data']
    data_timeGAN = np.load(os.path.join(config['path_augmented_timegan'], 'augmented_TimeGAN_HCP_2026_' + config['case'] + '.npy'))
    data_t2idiff = np.load(os.path.join(config['path_augmented_t2idiff'], config['case'], 'gen_sig_' + config['case'] + '_fold1.npy'))
    data_dsfm = np.load(os.path.join(config['path_augmented_dsfm'], config['case'], 'samples', 'samples.npy'))
    
    data_rDCM = data_rDCM.numpy()[0:3][:,5,:400,:]
    data_rDCM = (data_rDCM - np.mean(data_rDCM, axis=1, keepdims=True))/np.std(data_rDCM, axis=1, keepdims=True)
    data_timeVAE = data_timeVAE[50:53,:400,:]
    data_timeVAE = (data_timeVAE - np.mean(data_timeVAE, axis=1, keepdims=True))/np.std(data_timeVAE, axis=1, keepdims=True)
    data_timeGAN = data_timeGAN[50:53,:400,:]
    data_timeGAN = (data_timeGAN - np.mean(data_timeGAN, axis=1, keepdims=True))/np.std(data_timeGAN, axis=1, keepdims=True)
    data_t2idiff = data_t2idiff[0:3,:400,:]
    data_t2idiff = (data_t2idiff - np.mean(data_t2idiff, axis=1, keepdims=True))/np.std(data_t2idiff, axis=1, keepdims=True)
    data_crvae = data_crvae.numpy()[0:3][:,5,:400,:]
    data_crvae = (data_crvae - np.mean(data_crvae, axis=1, keepdims=True))/np.std(data_crvae, axis=1, keepdims=True)
    data_dsfm = data_dsfm[0:3,:400,:]
    data_dsfm = (data_dsfm - np.mean(data_dsfm, axis=1, keepdims=True))/np.std(data_dsfm, axis=1, keepdims=True)
    data_orig = train_data[0:3]
    
    data_orig = preprocess_HCP(data_orig, config['case'])
        
    data_orig = np.concatenate(data_orig, axis=0)
    data_rDCM = np.reshape(data_rDCM, (data_rDCM.shape[0]*data_rDCM.shape[1], data_rDCM.shape[2]))
    data_crvae = np.reshape(data_crvae, (data_crvae.shape[0]*data_crvae.shape[1], data_crvae.shape[2]))
    data_timeVAE = np.reshape(data_timeVAE, (data_timeVAE.shape[0]*data_timeVAE.shape[1], data_timeVAE.shape[2]))
    data_timeGAN = np.reshape(data_timeGAN, (data_timeGAN.shape[0]*data_timeGAN.shape[1], data_timeGAN.shape[2]))
    data_t2idiff = np.reshape(data_t2idiff, (data_t2idiff.shape[0]*data_t2idiff.shape[1], data_t2idiff.shape[2]))
    data_dsfm = np.reshape(data_dsfm, (data_dsfm.shape[0]*data_dsfm.shape[1], data_dsfm.shape[2]))
               
    result = np.concatenate((data_orig, data_timeVAE), axis=0)
    labels = [0 for i in range(1200)] + [1 for i in range(1200)]
    labels = np.array(labels)
    labels = pd.DataFrame(labels)
    labels.to_csv(os.path.join(config['path_save'], 'TimeVAE_labels.csv'))
    os.makedirs(os.path.join(config['path_save'], 'TimeVAE'), exist_ok=True)
    np.save(os.path.join(config['path_save'], 'TimeVAE', 'sub-01_TimeVAE_samples.npy'), result)
    
    result = np.concatenate((data_orig, data_timeGAN), axis=0)
    labels = [0 for i in range(1200)] + [1 for i in range(1200)]
    labels = np.array(labels)
    labels = pd.DataFrame(labels)
    labels.to_csv(os.path.join(config['path_save'], 'TimeGAN_labels.csv'))
    os.makedirs(os.path.join(config['path_save'], 'TimeGAN'), exist_ok=True)
    np.save(os.path.join(config['path_save'], 'TimeGAN', 'sub-01_TimeGAN_samples.npy'), result)
    
    result = np.concatenate((data_orig, data_t2idiff), axis=0)
    labels = [0 for i in range(1200)] + [1 for i in range(1200)]
    labels = np.array(labels)
    labels = pd.DataFrame(labels)
    labels.to_csv(os.path.join(config['path_save'], 'T2IDIFF_labels.csv'))
    os.makedirs(os.path.join(config['path_save'], 'T2IDIFF'), exist_ok=True)
    np.save(os.path.join(config['path_save'], 'T2IDIFF', 'sub-01_T2IDIFF_samples.npy'), result)
    
    result = np.concatenate((data_orig, data_dsfm), axis=0)
    labels = [0 for i in range(1200)] + [1 for i in range(1200)]
    labels = np.array(labels)
    labels = pd.DataFrame(labels)
    labels.to_csv(os.path.join(config['path_save'], 'DSFM_labels.csv'))
    os.makedirs(os.path.join(config['path_save'], 'DSFM'), exist_ok=True)
    np.save(os.path.join(config['path_save'], 'DSFM', 'sub-01_DSFM_samples.npy'), result)
    
    result = np.concatenate((data_orig, data_crvae), axis=0)
    labels = [0 for i in range(1200)] + [1 for i in range(1200)]
    labels = np.array(labels)
    labels = pd.DataFrame(labels)
    labels.to_csv(os.path.join(config['path_save'], 'CRVAE_labels.csv'))
    os.makedirs(os.path.join(config['path_save'], 'CRVAE'), exist_ok=True)
    np.save(os.path.join(config['path_save'], 'CRVAE', 'sub-01_CRVAE_samples.npy'), result)
    
    result = np.concatenate((data_orig, data_rDCM), axis=0)
    labels = [0 for i in range(1200)] + [1 for i in range(1200)]
    labels = np.array(labels)
    labels = pd.DataFrame(labels)
    labels.to_csv(os.path.join(config['path_save'], 'rDCM-GEN_labels.csv'))
    os.makedirs(os.path.join(config['path_save'], 'rDCM-GEN'), exist_ok=True)
    np.save(os.path.join(config['path_save'], 'rDCM-GEN', 'sub-01_rDCM-GEN_samples.npy'), result)
    
    
    result = np.concatenate((data_orig[:300],  data_timeVAE[:300], data_timeGAN[:300], data_t2idiff[:300], data_dsfm[:300], data_crvae[:300], data_rDCM[:300]), axis=0)
    labels = [0 for i in range(300)] + [1 for i in range(300)] + [2 for i in range(300)] + [3 for i in range(300)] + [4 for i in range(300)] + [5 for i in range(300)] + [6 for i in range(300)]
    labels = np.array(labels)
    labels = pd.DataFrame(labels)
    labels.to_csv(os.path.join(config['path_save'], 'comparison_labels.csv'))
    os.makedirs(os.path.join(config['path_save'], 'comparison'), exist_ok=True)
    np.save(os.path.join(config['path_save'], 'comparison', 'sub-01_comparison_samples.npy'), result)
    
if __name__ == "__main__":
    config = {}
    config['path_augmented_rdcmgen'] = r'/media/student1/ExtremePro/rDCM-GEN/rDCM-GEN_results/augmented_rDCM-GEN_noFCloss'
    config['path_augmented_crvae'] = r'/media/student1/ExtremePro/CRVAE_augmented'
    config['path_augmented_timevae'] = r'/media/student1/ExtremePro/TimeVAE_results/outputs/gen_data'
    config['path_augmented_timegan'] = r'/media/student1/ExtremePro/TimeGAN_results'
    config['path_augmented_t2idiff'] = r'/media/student1/ExtremePro/T2IDIFF'
    config['path_augmented_dsfm'] = r'/media/student1/ExtremePro/DSFM_results'
    config['path_data'] = r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_GNN_data/HCP_2026'
    config['path_save'] = r'/home/student1/Desktop/Charalampos_Lamprou/BCNE/data'
    config['case'] = "full_original"
    config['TR'] = 0.72
    config['newTR'] = 2.0
    main(config)


