parent_path = r'/home/student1/Desktop/Charalampos_Lamprou/Diffusion-TS'
import sys
sys.path.append(parent_path)
import os
import torch
import numpy as np
from Models.ts2vec.ts2vec import TS2Vec
from Utils.context_fid import calculate_fid
from scipy.signal import butter, filtfilt, resample
import pickle


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
            new_samples = int(np.round(y.shape[0] / rate))
            y = resample(y, new_samples)
            y = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
        data[i] = y
    return data



def main(config):
    for method in config['methods']:               
        path = config['path_data']

        data = np.load(os.path.join(path,'train_data_HCP_' + config['atlas'] + '.npz'))
        train_data = []
        for key in data:
            if data[key].shape[0] == 1200:
                temp = data[key][183:-184]
                train_data.append(temp)    
        
        data = np.load(os.path.join(path,'val_data_HCP_' + config['atlas'] + '.npz'))
        for key in data:
            if data[key].shape[0] == 1200:
                temp = data[key][183:-184]
                train_data.append(temp)
    
            
        data = np.load(os.path.join(path,'test_data_HCP_' + config['atlas'] + '.npz'))
        for key in data:
            if data[key].shape[0] == 1200:
                temp = data[key][183:-184]
                train_data.append(temp)
              
        train_real = preprocess_HCP(train_data[:3109], config['case'])
        test_real = preprocess_HCP(train_data[-1000:], config['case'])
        
        train_real = np.stack(train_real)
        test_real = np.stack(test_real)
        
        train_real = train_real[:,:256,:]
        test_real = test_real[:,:256,:]
        
        if method == "CR-VAE":
            test_gen = torch.load(os.path.join(config['path_augmented_crvae'], 'augmented_CR-VAE_' + config['case'] + '.pt'))
            test_gen = test_gen.view(-1, test_gen.shape[2], test_gen.shape[3])
            test_gen = test_gen.numpy()
        elif method == "rDCM-GEN":
            test_gen = torch.load(os.path.join(config['path_augmented_rdcmgen'], 'augmented_rDCM-GEN_' + config['case'] + '.pt'))
            test_gen = test_gen.view(-1, test_gen.shape[2], test_gen.shape[3])
            test_gen = test_gen.numpy()
        elif method == "TimeVAE":
            test_gen = np.load(os.path.join(config['path_augmented_timevae'], config['case'], 'timeVAE_' + config['case'] + '_prior_samples.npy'))
            test_gen = (test_gen - test_gen.mean(axis=2, keepdims=True))/test_gen.std(axis=2, keepdims=True)
        elif method == "TimeGAN":
            test_gen = np.load(os.path.join(config['path_augmented_timegan'], 'augmented_TimeGAN_HCP_2026_' + config['case'] + '.npy'))
            test_gen = (test_gen - test_gen.mean(axis=2, keepdims=True))/test_gen.std(axis=2, keepdims=True)
        elif method == "T2I-Diff":
            test_gen = np.load(os.path.join(config['path_augmented_t2idiff'], config['case'], 'gen_sig_' + config['case'] + '_fold1.npy'))
            test_gen = (test_gen - test_gen.mean(axis=2, keepdims=True))/test_gen.std(axis=2, keepdims=True)
        elif method == "DSFM":
            test_gen = np.load(os.path.join(config['path_augmented_dsfm'], config['case'], 'samples', 'samples.npy'))
            test_gen = (test_gen - test_gen.mean(axis=2, keepdims=True))/test_gen.std(axis=2, keepdims=True)
        
        scores = []
        for i in range(10):
            rng = np.random.default_rng()
            test_gen = test_gen[rng.permutation(test_gen.shape[0])]
            test_gen = test_gen[:test_real.shape[0],:256,:]
            
            encoder = TS2Vec(
                input_dims=train_real.shape[-1],
                output_dims=320,
                device=3)  
            encoder.fit(train_real)
            z_real = encoder.encode(test_real, encoding_window="full_series")
            z_generated = encoder.encode(test_gen, encoding_window="full_series")
            
            score = calculate_fid(z_real, z_generated)
            scores.append(score)
    
        results = {}
        results['context_FID'] = scores
        
        if config['save_results']:
            os.makedirs(os.path.join(config['path_save'], 'cFID', config['case'], method), exist_ok=True)
            with open(os.path.join(config['path_save'], 'cFID', config['case'], method, 'results.pkl'), 'wb') as f:
                pickle.dump(results,f)
    return results


if __name__ == '__main__':
    config = {}
    config['path_data'] = r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_GNN_data/HCP_2026'
    config['path_save'] = r'/home/student1/Desktop/Charalampos_Lamprou/rDCM-GEN_results_HCP_2026'
    config['path_augmented_rdcmgen'] = r'/media/student1/ExtremePro/rDCM-GEN/rDCM-GEN_results/augmented_rDCM-GEN_noFCloss'
    config['path_augmented_crvae'] = r'/media/student1/ExtremePro/CR-VAE_augmented'
    config['path_augmented_timevae'] = r'/media/student1/ExtremePro/TimeVAE_results/outputs/gen_data'
    config['path_augmented_timegan'] = r'/media/student1/ExtremePro/TimeGAN_results'
    config['path_augmented_t2idiff'] = r'/media/student1/ExtremePro/T2I-Diff'
    config['path_augmented_dsfm'] = r'/media/student1/ExtremePro/DSFM_results'
    config['methods'] = ['CR-VAE', 'TimeVAE', 'TimeGAN','T2I-Diff', 'DSFM','rDCM-GEN']
    config['case'] = "full_subsampled"
    config['TR'] = 0.72
    config['newTR'] = 2.0
    config['R'] = 100
    config['atlas'] = 'SCH'
    config['batch_size'] = 256
    config['weight_decay'] = 1e-5
    config['save_results'] = True
    main(config)