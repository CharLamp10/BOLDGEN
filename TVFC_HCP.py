"""Estimate and compare sliding-window functional brain states on HCP."""
import argparse

import os
import torch
from kmeans_pytorch import kmeans
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
            new_samples = int(np.round(y.shape[0] / rate))
            y = resample(y, new_samples)
            y = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
        data[i] = y
    return data



def silhouette_score_torch(x, labels, centroids):
    N, D = x.shape
    K = centroids.size(0)

    # distance to all centroids
    dist_to_centroids = (
        x.pow(2).sum(1, keepdim=True)
        - 2 * x @ centroids.T
        + centroids.pow(2).sum(1)
    )

    # intra-cluster distance
    a = torch.zeros(N, device=x.device)
    for k in range(K):
        mask = (labels == k)
        pts = x[mask]
        if pts.shape[0] > 1:
            dmat = torch.cdist(pts, pts)
            a[mask] = dmat.sum(1) / (pts.shape[0] - 1)
        else:
            a[mask] = 0.0

    # nearest other cluster
    one_hot = torch.nn.functional.one_hot(labels, K).bool()
    b = dist_to_centroids.masked_fill(one_hot, float("inf")).min(1).values

    s = (b - a) / torch.maximum(a, b)
    return s.mean()


def main(config):

    path = config['path_data']
    device = torch.device(config['device'])

    names = []
    with open(os.path.join(path,'names_train.txt'), 'r') as f:
        for line in f:
            temp = line.strip()
            names.append(temp[:-4])
            
    data = np.load(os.path.join(path,'train_data_HCP_' + config['atlas'] + '.npz'))
    train_data = []
    for key in data:
        train_data.append(data[key][183:-184])
    
    train_data = preprocess_HCP(train_data[:config['subjects']], config['case'])

    augmented_crvae = torch.load(os.path.join(config['path_augmented_crvae'], 'augmented_CR-VAE_' + config['case'] + '.pt'))
    augmented_rdcmgen = torch.load(os.path.join(config['path_augmented_rdcmgen'], 'augmented_rDCM-GEN_' + config['case'] + '.pt'))
    augmented_timevae = np.load(os.path.join(config['path_augmented_timevae'], config['case'], 'timeVAE_' + config['case'] + '_prior_samples.npy'))
    augmented_timegan = np.load(os.path.join(config['path_augmented_timegan'], 'augmented_TimeGAN_HCP_2026_' + config['case'] + '.npy'))
    augmented_t2idiff = np.load(os.path.join(config['path_augmented_t2idiff'], config['case'], 'gen_sig_' + config['case'] + '_fold1.npy'))
    augmented_dsfm = np.load(os.path.join(config['path_augmented_dsfm'], config['case'], 'samples', 'samples.npy'))
        
    
    T,R = train_data[0].shape            
    data_rdcmgen = augmented_rdcmgen[:config['subjects']]
    data_rdcmgen = data_rdcmgen[:,0,:,:]
    data_crvae = augmented_crvae[:config['subjects']]
    data_crvae = data_crvae[:,0,:,:]
    data_timegan = torch.from_numpy(augmented_timegan[:config['subjects']])
    data_timevae = torch.from_numpy(augmented_timevae[:config['subjects']])
    data_t2idiff = torch.from_numpy(augmented_t2idiff[:config['subjects']])
    data_dsfm = torch.from_numpy(augmented_dsfm[:config['subjects']])
    triu_indices = torch.triu_indices(R, R, offset=1)
    
    if 'HCP' in config['methods']:
        x_all = []
        for dat in train_data:
            for i in range(T - config['wind'] + 1):
                x = dat[i:i+config['wind']]
                x = np.corrcoef(x.T)
                x = x[triu_indices[0], triu_indices[1]]
                x_all.append(x)
        x = np.stack(x_all)
        x = torch.from_numpy(x).float()
        
        scores = []
        centroids = []
        Ks = [2,3,4,5,6]
        for K in Ks:
            cluster_ids, cluster_centers = kmeans(X=x, num_clusters=K, distance='euclidean', device=device)
            scores.append(silhouette_score_torch(x, cluster_ids, cluster_centers))
            centroids.append(cluster_centers)
            print(K)
            
        pos = np.where(np.array(scores) == np.max(scores))[0][0]
        K = Ks[pos]
        centroids = centroids[pos]
        for i,centroid in enumerate(centroids):
            M = torch.zeros((R, R), dtype=x.dtype)
            M[triu_indices[0], triu_indices[1]] = centroid
            M[triu_indices[1], triu_indices[0]] = centroid   # symmetric
            M.fill_diagonal_(1.0)   # if needed
            plt.figure(figsize=(10, 8))
            sns.heatmap(M.cpu().numpy(), cmap='coolwarm', center=0, square=True, vmin=-1, vmax=1)
            plt.xticks([])
            plt.yticks([])
            os.makedirs(os.path.join('plots', 'TVFC', config['case']), exist_ok=True)
            plt.savefig(os.path.join("plots","TVFC", config['case'], "state_" + str(i) + ".png"), dpi=600, bbox_inches='tight')
            plt.show()
    
    
    if 'CR-VAE' in config['methods']:
        x_all = []
        for dat in data_crvae:
            for i in range(T - config['wind'] + 1):
                x = dat[i:i+config['wind']]
                x = np.corrcoef(x.T)
                x = x[triu_indices[0], triu_indices[1]]
                x_all.append(x)
        x = np.stack(x_all)
        x = torch.from_numpy(x).float()
        
        cluster_ids, cluster_centers = kmeans(X=x, num_clusters=K, distance='euclidean', device=device)
            
        for i,centroid in enumerate(cluster_centers):
            M = torch.zeros((R, R), dtype=x.dtype)
            M[triu_indices[0], triu_indices[1]] = centroid
            M[triu_indices[1], triu_indices[0]] = centroid   # symmetric
            M.fill_diagonal_(1.0)   # if needed
            plt.figure(figsize=(10, 8))
            sns.heatmap(M.cpu().numpy(), cmap='coolwarm', center=0, square=True, vmin=-1, vmax=1)
            plt.xticks([])
            plt.yticks([])
            os.makedirs(os.path.join('plots', 'TVFC', config['case']), exist_ok=True)
            plt.savefig(os.path.join("plots","TVFC", config['case'], "CR-VAE_state_" + str(i) + ".png"), dpi=600, bbox_inches='tight')
            plt.show()
    
    if 'TimeVAE' in config['methods']:
        x_all = []
        for dat in data_timevae:
            for i in range(T - config['wind'] + 1):
                x = dat[i:i+config['wind']]
                x = np.corrcoef(x.T)
                x = x[triu_indices[0], triu_indices[1]]
                x_all.append(x)
        x = np.stack(x_all)
        x = torch.from_numpy(x).float()
    
        cluster_ids, cluster_centers = kmeans(X=x, num_clusters=K, distance='euclidean', device=device)   
            
        for i,centroid in enumerate(cluster_centers):
            M = torch.zeros((R, R), dtype=x.dtype)
            M[triu_indices[0], triu_indices[1]] = centroid
            M[triu_indices[1], triu_indices[0]] = centroid   # symmetric
            M.fill_diagonal_(1.0)   # if needed
            plt.figure(figsize=(10, 8))
            sns.heatmap(M.cpu().numpy(), cmap='coolwarm', center=0, square=True, vmin=-1, vmax=1)
            plt.xticks([])
            plt.yticks([])
            os.makedirs(os.path.join('plots', 'TVFC', config['case']), exist_ok=True)
            plt.savefig(os.path.join("plots","TVFC", config['case'], "TimeVAE_state_" + str(i) + ".png"), dpi=600, bbox_inches='tight')
            plt.show()
    
    if 'TimeGAN' in config['methods']:
        x_all = []
        for dat in data_timegan:
            for i in range(T - config['wind'] + 1):
                x = dat[i:i+config['wind']]
                x = np.corrcoef(x.T)
                x = x[triu_indices[0], triu_indices[1]]
                x_all.append(x)
        x = np.stack(x_all)
        x = torch.from_numpy(x).float()
        
        cluster_ids, cluster_centers = kmeans(X=x, num_clusters=K, distance='euclidean', device=device)

        for i,centroid in enumerate(cluster_centers):
            M = torch.zeros((R, R), dtype=x.dtype)
            M[triu_indices[0], triu_indices[1]] = centroid
            M[triu_indices[1], triu_indices[0]] = centroid   # symmetric
            M.fill_diagonal_(1.0)   # if needed
            plt.figure(figsize=(10, 8))
            sns.heatmap(M.cpu().numpy(), cmap='coolwarm', center=0, square=True, vmin=-1, vmax=1)
            plt.xticks([])
            plt.yticks([])
            os.makedirs(os.path.join('plots', 'TVFC', config['case']), exist_ok=True)
            plt.savefig(os.path.join("plots","TVFC", config['case'], "TimeGAN_state_" + str(i) + ".png"), dpi=600, bbox_inches='tight')
            plt.show()
            
    if 'T2I-Diff' in config['methods']:
        x_all = []
        for dat in data_t2idiff:
            for i in range(T - config['wind'] + 1):
                x = dat[i:i+config['wind']]
                x = np.corrcoef(x.T)
                x = x[triu_indices[0], triu_indices[1]]
                x_all.append(x)
        x = np.stack(x_all)
        x = torch.from_numpy(x).float()
        
        cluster_ids, cluster_centers = kmeans(X=x, num_clusters=K, distance='euclidean', device=device)
        
        for i,centroid in enumerate(cluster_centers):
            M = torch.zeros((R, R), dtype=x.dtype)
            M[triu_indices[0], triu_indices[1]] = centroid
            M[triu_indices[1], triu_indices[0]] = centroid   # symmetric
            M.fill_diagonal_(1.0)   # if needed
            plt.figure(figsize=(10, 8))
            sns.heatmap(M.cpu().numpy(), cmap='coolwarm', center=0, square=True, vmin=-1, vmax=1)
            plt.xticks([])
            plt.yticks([])
            os.makedirs(os.path.join('plots', 'TVFC', config['case']), exist_ok=True)
            plt.savefig(os.path.join("plots","TVFC", config['case'], "T2IDIFF_state_" + str(i) + ".png"), dpi=600, bbox_inches='tight')
            plt.show()
            
    if 'DSFM' in config['methods']:
        x_all = []
        for dat in data_dsfm:
            for i in range(T - config['wind'] + 1):
                x = dat[i:i+config['wind']]
                x = np.corrcoef(x.T)
                x = x[triu_indices[0], triu_indices[1]]
                x_all.append(x)
        x = np.stack(x_all)
        x = torch.from_numpy(x).float()
        
        cluster_ids, cluster_centers = kmeans(X=x, num_clusters=K, distance='euclidean', device=device)
        
        for i,centroid in enumerate(cluster_centers):
            M = torch.zeros((R, R), dtype=x.dtype)
            M[triu_indices[0], triu_indices[1]] = centroid
            M[triu_indices[1], triu_indices[0]] = centroid   # symmetric
            M.fill_diagonal_(1.0)   # if needed
            plt.figure(figsize=(10, 8))
            sns.heatmap(M.cpu().numpy(), cmap='coolwarm', center=0, square=True, vmin=-1, vmax=1)
            plt.xticks([])
            plt.yticks([])
            os.makedirs(os.path.join('plots', 'TVFC', config['case']), exist_ok=True)
            plt.savefig(os.path.join("plots","TVFC", config['case'], "DSFM_state_" + str(i) + ".png"), dpi=600, bbox_inches='tight')
            plt.show()
    
    if 'rDCM-GEN' in config['methods']:
        x_all = []
        for dat in data_rdcmgen:
            for i in range(T - config['wind'] + 1):
                x = dat[i:i+config['wind']]
                x = np.corrcoef(x.T)
                x = x[triu_indices[0], triu_indices[1]]
                x_all.append(x)
        x = np.stack(x_all)
        x = torch.from_numpy(x).float()
        
        cluster_ids, cluster_centers = kmeans(X=x, num_clusters=K, distance='euclidean', device=device)
            
        for i,centroid in enumerate(cluster_centers):
            M = torch.zeros((R, R), dtype=x.dtype)
            M[triu_indices[0], triu_indices[1]] = centroid
            M[triu_indices[1], triu_indices[0]] = centroid   # symmetric
            M.fill_diagonal_(1.0)   # if needed
            plt.figure(figsize=(10, 8))
            sns.heatmap(M.cpu().numpy(), cmap='coolwarm', center=0, square=True, vmin=-1, vmax=1)
            plt.xticks([])
            plt.yticks([])
            os.makedirs(os.path.join('plots', 'TVFC', config['case']), exist_ok=True)
            plt.savefig(os.path.join("plots","TVFC", config['case'], "rDCM-GEN_state_" + str(i) + ".png"), dpi=600, bbox_inches='tight')
            plt.show()



        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate and compare sliding-window functional brain states on HCP.')
    parser.add_argument('--path_data', type=str, required=True)
    parser.add_argument('--path_save', type=str, required=True)
    parser.add_argument('--path_augmented_rdcmgen', type=str, required=True)
    parser.add_argument('--path_augmented_crvae', type=str, required=True)
    parser.add_argument('--path_augmented_timevae', type=str, required=True)
    parser.add_argument('--path_augmented_timegan', type=str, required=True)
    parser.add_argument('--path_augmented_t2idiff', type=str, required=True)
    parser.add_argument('--path_augmented_dsfm', type=str, required=True)
    parser.add_argument('--atlas', type=str, default='SCH')
    parser.add_argument('--TR', type=float, default=0.72)
    parser.add_argument('--newTR', type=float, default=2.0)
    parser.add_argument('--device', type=str, default="cuda:3")
    parser.add_argument('--case', type=str, default="full_subsampled")
    parser.add_argument('--methods', nargs="+", type=str, default=['HCP', 'rDCM-GEN', 'CR-VAE', 'TimeVAE', 'TimeGAN', 'T2I-Diff', 'DSFM'])
    args = parser.parse_args()
    config = {}
    config['path_data'] = args.path_data
    config['path_save'] = args.path_save
    config['path_augmented_rdcmgen'] = args.path_augmented_rdcmgen
    config['path_augmented_crvae'] = args.path_augmented_crvae
    config['path_augmented_timevae'] = args.path_augmented_timevae
    config['path_augmented_timegan'] = args.path_augmented_timegan
    config['path_augmented_t2idiff'] = args.path_augmented_t2idiff
    config['path_augmented_dsfm'] = args.path_augmented_dsfm
    config['atlas'] = args.atlas
    config['TR'] = args.TR
    config['newTR'] = args.newTR
    config['device'] = args.device
    config['case'] = args.case
    if config['case'] == "full_original" or config['case'] == "half_original":
        config['wind'] = 62
    elif config['case'] == "full_subsampled" or config['case'] == "half_subsampled":
        config['wind'] = 22
    if config['case'] == "full_original":
        config['subjects'] = 100
    elif config['case'] == "half_original":
        config['subjects'] = 140
    elif config['case'] == "full_subsampled":
        config['subjects'] = 100
    elif config['case'] == "half_subsampled":
        config['subjects'] = 145
    config['methods'] = args.methods
    results = main(config)   
    