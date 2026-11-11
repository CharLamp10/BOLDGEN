"""Collect, harmonize, and save generated and empirical time series."""
import argparse

import os
import torch
from scipy.signal import resample
import numpy as np



def main():
    parser = argparse.ArgumentParser(description='Collect, harmonize, and save generated and empirical time series.')
    parser.add_argument("--labels_path", type=str, required=True)
    parser.add_argument("--names_path", type=str, required=True)
    parser.add_argument("--path_boldgen", type=str, required=True)
    parser.add_argument("--path_crvae", type=str, required=True)
    parser.add_argument("--roi_path", type=str, required=True)
    parser.add_argument("--cat", default='ABIDEI')
    parser.add_argument("--newTR", type=float, default=1.5)
    args = parser.parse_args()

    cat = args.cat

    if cat == "ABIDEI":
        labels = np.load(args.labels_path)
        names = []
        with open(args.names_path, 'r') as f:
            for line in f:
                names.append(line.strip())

    path_boldgen = args.path_boldgen
    path_crvae = args.path_crvae


    dir_boldgen = os.listdir(os.path.join(path_boldgen, cat))
    dir_crvae = os.listdir(os.path.join(path_crvae, cat))

    newTR = args.newTR

    data_boldgen = []
    data_orig_boldgen = []
    names_boldgen = []
    labels_boldgen = []
    for i,file in enumerate(dir_boldgen):
        data = torch.load(os.path.join(path_boldgen, cat, file))
        if torch.any(torch.isnan(data)) or torch.any(torch.isinf(data)) or torch.any(data > 20):
            continue
        if cat == "ABIDEI":
            pos = np.where(np.array(names) == file[:11])[0]
            try:
                label = labels[pos[0]]
            except:
                label = labels[pos]
            path_abide = args.roi_path
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
                data_boldgen.append(data)
                data_orig_boldgen.append(data_orig)
                names_boldgen.append(file[:-3])
                labels_boldgen.append(label)
        else:
            data = (data - torch.mean(data, dim=1, keepdim=True))/torch.std(data, dim=1, keepdim=True)
            data_boldgen.append(data)
            names_boldgen.append(file[:-3])


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
            path_abide = args.roi_path
            TR = np.load(os.path.join(path_abide, file[:-3], 'TR.npy'))
            data_orig = np.load(os.path.join(path_abide, file[:-3], 'ROISignals.npy'))
            rate = newTR/TR
            new_samples = int(np.round(data.shape[1] / rate))
            data = torch.from_numpy(resample(data.numpy(), new_samples, axis=1))
            data_orig = torch.from_numpy(resample(data_orig, new_samples, axis=0))
        if cat == "ABIDEI":
            if data.shape[1] >= 200 and label.size > 0 and file[:-3] in names_boldgen:
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
        for i,name in enumerate(names_boldgen):
            if name not in names_crvae:
                pos_delete.append(i)
        
        for pos in pos_delete[::-1]:
            data_boldgen.pop(pos)
            data_orig_boldgen.pop(pos)
            names_boldgen.pop(pos)
            labels_boldgen.pop(pos)
    
    
    data_boldgen = torch.stack(data_boldgen)
    if cat == "ABIDEI":
        labels_boldgen = torch.from_numpy(np.array(labels_boldgen))
        data_orig_boldgen = torch.stack(data_orig_boldgen)
    
    data_crvae = torch.stack(data_crvae)   
    if cat == "ABIDEI":
        labels_crvae = torch.from_numpy(np.array(labels_crvae))
        data_orig_crvae = torch.stack(data_orig_crvae)


    torch.save(data_boldgen, os.path.join(path_boldgen, 'augmented_BOLDGEN_' + cat + '.pt'))
    if cat == "ABIDEI":
        torch.save(labels_boldgen, os.path.join(path_boldgen, 'labels_augmented_BOLDGEN_' + cat + '.pt'))
        data_boldgen2 = torch.cat((data_orig_boldgen.unsqueeze(dim=1), data_boldgen), dim=1)
        torch.save(data_boldgen2, os.path.join(path_boldgen, 'augmented_BOLDGEN_orig_' + cat + '.pt'))
    with open(os.path.join(path_boldgen,'names_BOLDGEN_' + cat + '.txt'), 'w') as f:
        for item in names_boldgen:
            f.write("%s\n" % item)

    torch.save(data_crvae, os.path.join(path_crvae, 'augmented_CR-VAE_' + cat + '.pt'))
    if cat == "ABIDEI":
        torch.save(labels_crvae, os.path.join(path_crvae, 'labels_augmented_CR-VAE_' + cat + '.pt'))
        data_crvae2 = torch.cat((data_orig_crvae.unsqueeze(dim=1), data_crvae), dim=1)
        torch.save(data_crvae2, os.path.join(path_crvae, 'augmented_CR-VAE_orig_' + cat + '.pt'))
    with open(os.path.join(path_crvae,'names_CR-VAE_' + cat + '.txt'), 'w') as f:
        for item in names_crvae:
            f.write("%s\n" % item)


if __name__ == "__main__":
    main()
