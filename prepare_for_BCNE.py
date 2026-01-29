import torch
import os
import numpy as np
import pandas as pd

'''
path_augmented = r'/media/student1/Extreme Pro/augmented'
path_data = r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_GNN_data/HCP'
path_save = r'/home/student1/Desktop/Charalampos_Lamprou/BCNE/data'
names = ['101309_rest1', '101410_rest1', '101915_rest1', '102008_rest1',
         '102109_rest1', '102311_rest1', '102816_rest1', '103818_rest1',
         '105014_rest1', '105216_rest1', '105620_rest1', '105923_rest1',
         '106521_rest1', '107220_rest1', '107422_rest1']

names_train = []
with open(os.path.join(path_data,'names_train.txt'), 'r') as f:
    for line in f:
        temp = line.strip()
        names_train.append(temp[:-7])
        
data = np.load(os.path.join(path_data,'train_data_HCP_SCH_resampled.npz'))
train_data = []
for key in data:
    train_data.append(data[key])

result = []
labels = []
count = 0
directory = os.listdir(path_augmented)
for name in directory:
    if name[:-6] in names:
        data_aug = torch.load(os.path.join(path_augmented, name)).detach().cpu().numpy()
        pos = np.where(np.array(names_train) == name[:-6])[0][0]
        data_orig = train_data[pos]
        data_orig = (data_orig - np.mean(data_orig, axis=0, keepdims=True))/np.std(data_orig, axis=0, keepdims=True)
        result.append(data_orig[100:120])
        for j in range(20):
            labels.append(count)
        for i in range(9):
            result.append(data_aug[i,100:120])
            for j in range(20):
                labels.append(count)
        count += 1
        
result = np.concatenate(result, axis=0)
labels = np.array(labels)
labels = pd.DataFrame(labels)
labels.to_csv(os.path.join(path_save, 'rDCM_labels.csv'))
np.save(os.path.join(path_save, 'rDCM', 'sub-04_rDCM_samples.npy'), result)


result = []
labels = []
count = 0
directory = os.listdir(path_augmented)
for name in directory:
    if name[:-6] in names:
        pos = np.where(np.array(names_train) == name[:-6])[0][0]
        data_orig = train_data[pos]
        data_orig = (data_orig - np.mean(data_orig, axis=0, keepdims=True))/np.std(data_orig, axis=0, keepdims=True)
        result.append(data_orig[100:300])
        for j in range(200):
            labels.append(count)
        count += 1
        
result = np.concatenate(result, axis=0)
labels = np.array(labels)
labels = pd.DataFrame(labels)
labels.to_csv(os.path.join(path_save, 'labels.csv'))
np.save(os.path.join(path_save, 'rDCM', 'sub-04_samples.npy'), result)
'''

path_rDCM = r'/media/student1/Extreme Pro/augmented'
path_crvae = r'/media/student1/Extreme Pro/CRVAE_models/augmented_cr_vae.pt'
path_timeGAN = r'/media/student1/Extreme Pro/TimeGAN_results/augmented_TimeGAN.npy'
path_timeVAE = r'/media/student1/Extreme Pro/TimeVAE_results/outputs/gen_data/HCP_train/timeVAE_HCP_train_prior_samples.npz'
path_data = r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_GNN_data/HCP'
path_save = r'/home/student1/Desktop/Charalampos_Lamprou/BCNE/data'
names = ['101309_rest1', '101410_rest1', '101915_rest1', '102008_rest1',
         '102109_rest1', '102311_rest1', '102816_rest1', '103818_rest1',
         '105014_rest1', '105216_rest1', '105620_rest1', '105923_rest1',
         '106521_rest1', '107220_rest1', '107422_rest1']
name = '101309_rest1'

names_train = []
with open(os.path.join(path_data,'names_train.txt'), 'r') as f:
    for line in f:
        temp = line.strip()
        names_train.append(temp[:-7])
        
data = np.load(os.path.join(path_data,'train_data_HCP_SCH_resampled.npz'))
train_data = []
for key in data:
    train_data.append(data[key])


data_rDCM = torch.load(os.path.join(path_rDCM, 'augmented_train.pt')).detach().cpu().numpy()[0:3][:,5,:,:]
data_timeVAE = np.load(path_timeVAE)['data'][50:53]
data_timeVAE = (data_timeVAE - np.mean(data_timeVAE, axis=1, keepdims=True))/np.std(data_timeVAE, axis=1, keepdims=True)
data_timeGAN = np.load(path_timeGAN)[50:53]
data_timeGAN = (data_timeGAN - np.mean(data_timeGAN, axis=1, keepdims=True))/np.std(data_timeGAN, axis=1, keepdims=True)
data_crvae = torch.load(path_crvae).cpu().numpy()[0:3][:,5,:,:]
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

        
result = np.concatenate((data_orig, data_noise05), axis=0)
labels = [0 for i in range(1200)] + [1 for i in range(1200)]
labels = np.array(labels)
labels = pd.DataFrame(labels)
labels.to_csv(os.path.join(path_save, 'noise05_labels.csv'))
os.makedirs(os.path.join(path_save, 'noise05'), exist_ok=True)
np.save(os.path.join(path_save, 'noise05', 'sub-04_noise05_samples.npy'), result)

result = np.concatenate((data_orig, data_noise075), axis=0)
labels = [0 for i in range(1200)] + [1 for i in range(1200)]
labels = np.array(labels)
labels = pd.DataFrame(labels)
labels.to_csv(os.path.join(path_save, 'noise075_labels.csv'))
os.makedirs(os.path.join(path_save, 'noise075'), exist_ok=True)
np.save(os.path.join(path_save, 'noise075', 'sub-04_noise075_samples.npy'), result)

result = np.concatenate((data_orig, data_timeVAE), axis=0)
labels = [0 for i in range(1200)] + [1 for i in range(1200)]
labels = np.array(labels)
labels = pd.DataFrame(labels)
labels.to_csv(os.path.join(path_save, 'TimeVAE_labels.csv'))
os.makedirs(os.path.join(path_save, 'TimeVAE'), exist_ok=True)
np.save(os.path.join(path_save, 'TimeVAE', 'sub-04_TimeVAE_samples.npy'), result)

result = np.concatenate((data_orig, data_timeGAN), axis=0)
labels = [0 for i in range(1200)] + [1 for i in range(1200)]
labels = np.array(labels)
labels = pd.DataFrame(labels)
labels.to_csv(os.path.join(path_save, 'TimeGAN_labels.csv'))
os.makedirs(os.path.join(path_save, 'TimeGAN'), exist_ok=True)
np.save(os.path.join(path_save, 'TimeGAN', 'sub-04_TimeGAN_samples.npy'), result)

result = np.concatenate((data_orig, data_crvae), axis=0)
labels = [0 for i in range(1200)] + [1 for i in range(1200)]
labels = np.array(labels)
labels = pd.DataFrame(labels)
labels.to_csv(os.path.join(path_save, 'CRVAE_labels.csv'))
os.makedirs(os.path.join(path_save, 'CRVAE'), exist_ok=True)
np.save(os.path.join(path_save, 'CRVAE', 'sub-04_CRVAE_samples.npy'), result)

result = np.concatenate((data_orig, data_rDCM), axis=0)
labels = [0 for i in range(1200)] + [1 for i in range(1200)]
labels = np.array(labels)
labels = pd.DataFrame(labels)
labels.to_csv(os.path.join(path_save, 'rDCM2_labels.csv'))
os.makedirs(os.path.join(path_save, 'rDCM2'), exist_ok=True)
np.save(os.path.join(path_save, 'rDCM2', 'sub-04_rDCM2_samples.npy'), result)


result = np.concatenate((data_orig[:300], data_noise05[:300], data_noise075[:300],  data_timeVAE[:300], data_timeGAN[:300], data_crvae[:300], data_rDCM[:300]), axis=0)
labels = [0 for i in range(300)] + [1 for i in range(300)] + [2 for i in range(300)] + [3 for i in range(300)] + [4 for i in range(300)] + [5 for i in range(300)] + [6 for i in range(300)]
labels = np.array(labels)
labels = pd.DataFrame(labels)
labels.to_csv(os.path.join(path_save, 'comparison_labels.csv'))
os.makedirs(os.path.join(path_save, 'comparison'), exist_ok=True)
np.save(os.path.join(path_save, 'comparison', 'sub-04_comparison_samples.npy'), result)


