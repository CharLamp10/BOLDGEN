import os
import numpy as np
import torch
import pickle
from scipy.signal import butter, filtfilt, resample


def main(config):
    names = []
    with open(os.path.join(config['path_data'],'names_train.txt'), 'r') as f:
        for line in f:
            temp = line.strip()
            names.append(temp)
            
    data = np.load(os.path.join(config['path_data'],'train_data_HCP_SCH.npz'))
    train_data = []
    for key in data:
        train_data.append(data[key][183:-184,:])
    
    data_orig = []
    TR = config['TR']
    for i, (y,name) in enumerate(zip(train_data,names)):
        if config['case'] == "full_original":
            y = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
            order = 4
            cutoff = 0.1
            fs = 1/TR
            newTR = TR
            b, a = butter(order, cutoff, btype='low', fs=fs)
            y = filtfilt(b, a, y.T).T.astype('float32')
        elif config['case'] == "half_original":
            y = y[125:708]
            y = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
            order = 4
            cutoff = 0.1
            fs = 1/TR
            newTR = TR
            b, a = butter(order, cutoff, btype='low', fs=fs)
            y = filtfilt(b, a, y.T).T.astype('float32')
        elif config['case'] == "full_subsampled":
            y = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
            order = 4
            cutoff = 0.1
            fs = 1/TR
            b, a = butter(order, cutoff, btype='low', fs=fs)
            y = filtfilt(b, a, y.T).T.astype('float32')
            newTR = 2
            rate = newTR/TR
            new_samples = int(np.round(y.shape[0] / rate))
            y = resample(y, new_samples)
            y = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
        elif config['case'] == "half_subsampled":
            y = y[125:708]
            y = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
            order = 4
            cutoff = 0.1
            fs = 1/TR
            b, a = butter(order, cutoff, btype='low', fs=fs)
            y = filtfilt(b, a, y.T).T.astype('float32')
            newTR = 2
            rate = newTR/TR
            new_samples = int(np.round(y.shape[0] / rate))
            y = resample(y, new_samples)
            y = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
        data_orig.append(y)
    
    path_rdcm_gen = os.path.join(config['path_augmented_rdcm_gen'], config['case'])
    dir_rdcm_gen = os.listdir(path_rdcm_gen)
    similarities = []
    for file in dir_rdcm_gen:
        data_rdcm_gen = torch.load(os.path.join(path_rdcm_gen, file))
        pos = np.where(np.array(names) == file[:-3])[0][0]
        y_orig = torch.from_numpy(data_orig[pos])
        FC = torch.corrcoef(y_orig.T)
        for y_gen in data_rdcm_gen:
            FC_gen = torch.corrcoef(y_gen.T)
            sim = np.corrcoef(FC.flatten().numpy(), FC_gen.flatten().numpy())[0,1]
            if not np.isnan(sim):
                similarities.append(sim)           
    
    return similarities
    
if __name__ == '__main__':
    config = {}
    config['path_data'] = r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_GNN_data/HCP_2026'
    config['path_save'] = r'/home/student1/Desktop/Charalampos_Lamprou/rDCM-GEN_results'
    config['case'] = 'full_original'
    config['path_augmented_rdcm_gen'] = r'/media/student1/ExtremePro/rDCM-GEN/rDCM-GEN_results/augmented_rDCM-GEN_noFCloss'
    config['TR'] = 0.72
    result = main(config)