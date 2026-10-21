import os
import torch
from scipy.signal import resample
import numpy as np


cat = "ABIDEI"

if cat == "ABIDEI":
    labels = np.load(r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_GNN_data/ABIDEI/fmriprep/ABIDEI_nilearn_classes.npy')
    names = []
    with open(r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_GNN_data/ABIDEI/fmriprep/ABIDEI_nilearn_names.txt', 'r') as f:
        for line in f:
            names.append(line.strip())

path_rdcmgen = r'/media/student1/ExtremePro/rDCM-GEN/rDCM-GEN_results/augmented_rDCM-GEN_noFCloss'
path_crvae = r'/media/student1/ExtremePro/CRVAE_augmented'


dir_rdcmgen = os.listdir(os.path.join(path_rdcmgen, cat))
dir_crvae = os.listdir(os.path.join(path_crvae, cat))

newTR = 1.5

data_rdcmgen = []
data_orig_rdcmgen = []
names_rdcmgen = []
labels_rdcmgen = []
for i,file in enumerate(dir_rdcmgen):
    data = torch.load(os.path.join(path_rdcmgen, cat, file))
    if torch.any(torch.isnan(data)) or torch.any(torch.isinf(data)) or torch.any(data > 20):
        continue
    if cat == "ABIDEI":
        pos = np.where(np.array(names) == file[:11])[0]
        try:
            label = labels[pos[0]]
        except:
            label = labels[pos]
        path_abide = r'/media/student1/ExtremePro/rDCM-GEN/data/ROIsignals_SCH_ABIDEI'
        TR = np.load(os.path.join(path_abide, file[:-3], 'TR.npy'))
        data_orig = np.load(os.path.join(path_abide, file[:-3], 'ROISignals.npy'))
        rate = newTR/TR
        new_samples = int(np.round(data.shape[1] / rate))
        data = torch.from_numpy(resample(data.numpy(), new_samples, axis=1))
        data_orig = torch.from_numpy(resample(data_orig, new_samples, axis=0))
    if cat == "ABIDEI":
        if data.shape[1] >= 200 and label.size > 0:
            data = (data[:,:200,:] - torch.mean(data[:,:200,:], dim=1, keepdim=True))/torch.std(data[:,:200,:], dim=1, keepdim=True)
            data_orig = (data_orig[:200,:] - torch.mean(data_orig[:200,:], dim=1, keepdim=True))/torch.std(data_orig[:200,:], dim=1, keepdim=True)
            data_rdcmgen.append(data)
            data_orig_rdcmgen.append(data_orig)
            names_rdcmgen.append(file[:-3])
            labels_rdcmgen.append(label)
    else:
        data = (data - torch.mean(data, dim=1, keepdim=True))/torch.std(data, dim=1, keepdim=True)
        data_rdcmgen.append(data)
        names_rdcmgen.append(file[:-3])


data_crvae = []
data_orig_crvae = []
names_crvae = []
labels_crvae = []
for file in dir_crvae:
    data = torch.load(os.path.join(path_crvae, cat, file))
    if torch.any(torch.isnan(data)) or torch.any(torch.isinf(data)) or torch.any(data > 20):
        continue
    if cat == "ABIDEI":
        pos = np.where(np.array(names) == file[:11])[0]
        try:
            label = labels[pos[0]]
        except:
            label = labels[pos]
        path_abide = r'/media/student1/ExtremePro/rDCM-GEN/data/ROIsignals_SCH_ABIDEI'
        TR = np.load(os.path.join(path_abide, file[:-3], 'TR.npy'))
        data_orig = np.load(os.path.join(path_abide, file[:-3], 'ROISignals.npy'))
        rate = newTR/TR
        new_samples = int(np.round(data.shape[1] / rate))
        data = torch.from_numpy(resample(data.numpy(), new_samples, axis=1))
        data_orig = torch.from_numpy(resample(data_orig, new_samples, axis=0))
    if cat == "ABIDEI":
        if data.shape[1] >= 200 and label.size > 0 and file[:-3] in names_rdcmgen:
            data = (data[:,:200,:] - torch.mean(data[:,:200,:], dim=1, keepdim=True))/torch.std(data[:,:200,:], dim=1, keepdim=True)
            data_orig = (data_orig[:200,:] - torch.mean(data_orig[:200,:], dim=1, keepdim=True))/torch.std(data_orig[:200,:], dim=1, keepdim=True)
            data_crvae.append(data)
            data_orig_crvae.append(data_orig)
            names_crvae.append(file[:-3])
            labels_crvae.append(label)
    else:
        data = (data - torch.mean(data, dim=1, keepdim=True))/torch.std(data, dim=1, keepdim=True)
        data_crvae.append(data)
        names_crvae.append(file[:-3])
    

if cat == "ABIDEI":
    pos_delete = []
    for i,name in enumerate(names_rdcmgen):
        if name not in names_crvae:
            pos_delete.append(i)
        
    for pos in pos_delete[::-1]:
        data_rdcmgen.pop(pos)
        data_orig_rdcmgen.pop(pos)
        names_rdcmgen.pop(pos)
        labels_rdcmgen.pop(pos)
    
    
data_rdcmgen = torch.stack(data_rdcmgen)
if cat == "ABIDEI":
    labels_rdcmgen = torch.from_numpy(np.array(labels_rdcmgen))
    data_orig_rdcmgen = torch.stack(data_orig_rdcmgen)
    
data_crvae = torch.stack(data_crvae)   
if cat == "ABIDEI":
    labels_crvae = torch.from_numpy(np.array(labels_crvae))
    data_orig_crvae = torch.stack(data_orig_crvae)


torch.save(data_rdcmgen, os.path.join(path_rdcmgen, 'augmented_rDCM-GEN_' + cat + '.pt'))
if cat == "ABIDEI":
    torch.save(labels_rdcmgen, os.path.join(path_rdcmgen, 'labels_augmented_rDCM-GEN_' + cat + '.pt'))
    data_rdcmgen2 = torch.cat((data_orig_rdcmgen.unsqueeze(dim=1), data_rdcmgen), dim=1)
    torch.save(data_rdcmgen2, os.path.join(path_rdcmgen, 'augmented_rDCM-GEN_orig_' + cat + '.pt'))
with open(os.path.join(path_rdcmgen,'names_rDCM-GEN_' + cat + '.txt'), 'w') as f:
    for item in names_rdcmgen:
        f.write("%s\n" % item)

torch.save(data_crvae, os.path.join(path_crvae, 'augmented_CR-VAE_' + cat + '.pt'))
if cat == "ABIDEI":
    torch.save(labels_crvae, os.path.join(path_crvae, 'labels_augmented_CR-VAE_' + cat + '.pt'))
    data_crvae2 = torch.cat((data_orig_crvae.unsqueeze(dim=1), data_crvae), dim=1)
    torch.save(data_crvae2, os.path.join(path_crvae, 'augmented_CR-VAE_orig_' + cat + '.pt'))
with open(os.path.join(path_crvae,'names_CR-VAE_' + cat + '.txt'), 'w') as f:
    for item in names_crvae:
        f.write("%s\n" % item)
