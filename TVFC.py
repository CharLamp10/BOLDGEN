import os
import torch
from kmeans_pytorch import kmeans
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
            
    data = np.load(os.path.join(path,'train_data_HCP_' + config['atlas'] + '_resampled.npz'))
    train_data = []
    for key in data:
        temp = data[key][88:-88]
        train_data.append((temp - np.mean(temp, axis=0, keepdims=True))/np.std(temp, axis=0, keepdims=True))
    
            
    augmented_rdcm = torch.load(config['path_augmented_rdcm'])
    augmented_crvae = torch.load(config['path_augmented_crvae'])
    augmented_timeVAE = torch.from_numpy(np.load(config['path_augmented_timeVae'])['data'])
    augmented_timeGAN = torch.from_numpy(np.load(config['path_augmented_timeGan']))
    augmented_diffusionTS = torch.from_numpy(np.load(config['path_augmented_diffusion_ts']))
    
    T,R = train_data[0].shape            
    data_rdcm = augmented_rdcm[:200]
    data_rdcm = data_rdcm[:,0,:,:]
    data_crvae = augmented_crvae[:200]
    data_crvae = data_crvae[:,0,:,:]
    data_timeGAN = augmented_timeGAN[:200]
    data_timeVAE = augmented_timeVAE[:200]
    data_diffusionTS = augmented_diffusionTS[1000:1200]
    triu_indices = torch.triu_indices(R, R, offset=1)
    
    if 'HCP' in config['methods']:
        x_all = []
        for dat in train_data:
            for i in range(T - 30 + 1):
                x = dat[i:i+30]
                x = np.corrcoef(x.T)
                x = x[triu_indices[0], triu_indices[1]]
                x_all.append(x)
        x = np.stack(x_all)
        x = torch.from_numpy(x).float()
        
        scores = []
        centroids = []
        for K in [2]:#,3,4,5,6]:
            cluster_ids, cluster_centers = kmeans(X=x, num_clusters=K, distance='euclidean', device=device)
            scores.append(silhouette_score_torch(x, cluster_ids, cluster_centers))
            centroids.append(cluster_centers)
            print(K)
            
        pos = np.where(np.array(scores) == np.max(scores))[0][0]
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
            os.makedirs(os.path.join('plots', 'TVFC'), exist_ok=True)
            plt.savefig(os.path.join("plots","TVFC", "state_" + str(i) + ".png"), dpi=600, bbox_inches='tight')
            plt.show()
    
    for noise in [0.5, 0.75]:
        if (noise == 0.5 and 'Noise-6dB' in config['methods']) or (noise == 0.75 and 'Noise-3dB' in config['methods']):
            x_all = []
            for dat in train_data:
                for i in range(T - 30 + 1):
                    dat = dat + np.random.randn(*dat.shape)*noise
                    x = dat[i:i+30]
                    x = np.corrcoef(x.T)
                    x = x[triu_indices[0], triu_indices[1]]
                    x_all.append(x)
            x = np.stack(x_all)
            x = torch.from_numpy(x).float()
            
            scores = []
            centroids = []
            for K in [2]:#,3,4,5,6]:
                cluster_ids, cluster_centers = kmeans(X=x, num_clusters=K, distance='euclidean', device = torch.device("cuda:0"))
                scores.append(silhouette_score_torch(x, cluster_ids, cluster_centers))
                centroids.append(cluster_centers)
                print(K)
                
            pos = np.where(np.array(scores) == np.max(scores))[0][0]
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
                os.makedirs(os.path.join('plots', 'TVFC'), exist_ok=True)
                if noise == 0.5:
                    plt.savefig(os.path.join("plots","TVFC", "Noise-6dB_state_" + str(i) + ".png"), dpi=600, bbox_inches='tight')
                elif noise == 0.75:
                    plt.savefig(os.path.join("plots","TVFC", "Noise-3dB_state_" + str(i) + ".png"), dpi=600, bbox_inches='tight')
                plt.show()
    
    if 'CRVAE' in config['methods']:
        x_all = []
        for dat in data_crvae:
            for i in range(T - 30 + 1):
                x = dat[i:i+30]
                x = np.corrcoef(x.T)
                x = x[triu_indices[0], triu_indices[1]]
                x_all.append(x)
        x = np.stack(x_all)
        x = torch.from_numpy(x).float()
        
        scores = []
        centroids = []
        for K in [2]:#,3,4,5,6]:
            cluster_ids, cluster_centers = kmeans(X=x, num_clusters=K, distance='euclidean', device = torch.device("cuda:0"))
            scores.append(silhouette_score_torch(x, cluster_ids, cluster_centers))
            centroids.append(cluster_centers)
            print(K)
            
        pos = np.where(np.array(scores) == np.max(scores))[0][0]
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
            os.makedirs(os.path.join('plots', 'TVFC'), exist_ok=True)
            plt.savefig(os.path.join("plots","TVFC", "CRVAE_state_" + str(i) + ".png"), dpi=600, bbox_inches='tight')
            plt.show()
    
    if 'TimeVAE' in config['methods']:
        x_all = []
        for dat in data_timeVAE:
            for i in range(T - 30 + 1):
                x = dat[i:i+30]
                x = np.corrcoef(x.T)
                x = x[triu_indices[0], triu_indices[1]]
                x_all.append(x)
        x = np.stack(x_all)
        x = torch.from_numpy(x).float()
        
        scores = []
        centroids = []
        for K in [2]:#,3,4,5,6]:
            cluster_ids, cluster_centers = kmeans(X=x, num_clusters=K, distance='euclidean', device = torch.device("cuda:0"))
            scores.append(silhouette_score_torch(x, cluster_ids, cluster_centers))
            centroids.append(cluster_centers)
            print(K)
        
            
        pos = np.where(np.array(scores) == np.max(scores))[0][0]
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
            os.makedirs(os.path.join('plots', 'TVFC'), exist_ok=True)
            plt.savefig(os.path.join("plots","TVFC", "TimeVAE_state_" + str(i) + ".png"), dpi=600, bbox_inches='tight')
            plt.show()
    
    if 'TimeGAN' in config['methods']:
        x_all = []
        for dat in data_timeGAN:
            for i in range(T - 30 + 1):
                x = dat[i:i+30]
                x = np.corrcoef(x.T)
                x = x[triu_indices[0], triu_indices[1]]
                x_all.append(x)
        x = np.stack(x_all)
        x = torch.from_numpy(x).float()
        
        scores = []
        centroids = []
        for K in [2]:#,3,4,5,6]:
            cluster_ids, cluster_centers = kmeans(X=x, num_clusters=K, distance='euclidean', device = torch.device("cuda:0"))
            scores.append(silhouette_score_torch(x, cluster_ids, cluster_centers))
            centroids.append(cluster_centers)
            print(K)
        
            
        pos = np.where(np.array(scores) == np.max(scores))[0][0]
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
            os.makedirs(os.path.join('plots', 'TVFC'), exist_ok=True)
            plt.savefig(os.path.join("plots","TVFC", "TimeGAN_state_" + str(i) + ".png"), dpi=600, bbox_inches='tight')
            plt.show()
            
    if 'Diffusion-TS' in config['methods']:
        x_all = []
        for dat in data_diffusionTS:
            for i in range(T - 30 + 1):
                x = dat[i:i+30]
                x = np.corrcoef(x.T)
                x = x[triu_indices[0], triu_indices[1]]
                x_all.append(x)
        x = np.stack(x_all)
        x = torch.from_numpy(x).float()
        
        scores = []
        centroids = []
        for K in [2]:#,3,4,5,6]:
            cluster_ids, cluster_centers = kmeans(X=x, num_clusters=K, distance='euclidean', device = torch.device("cuda:0"))
            scores.append(silhouette_score_torch(x, cluster_ids, cluster_centers))
            centroids.append(cluster_centers)
            print(K)
        
            
        pos = np.where(np.array(scores) == np.max(scores))[0][0]
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
            os.makedirs(os.path.join('plots', 'TVFC'), exist_ok=True)
            plt.savefig(os.path.join("plots","TVFC", "Diffusion-TS_state_" + str(i) + ".png"), dpi=600, bbox_inches='tight')
            plt.show()
    
    if 'rDCM-GEN' in config['methods']:
        x_all = []
        for dat in data_rdcm:
            for i in range(T - 30 + 1):
                x = dat[i:i+30]
                x = np.corrcoef(x.T)
                x = x[triu_indices[0], triu_indices[1]]
                x_all.append(x)
        x = np.stack(x_all)
        x = torch.from_numpy(x).float()
        
        scores = []
        centroids = []
        for K in [2]:#,3,4,5,6]:
            cluster_ids, cluster_centers = kmeans(X=x, num_clusters=K, distance='euclidean', device = torch.device("cuda:0"))
            scores.append(silhouette_score_torch(x, cluster_ids, cluster_centers))
            centroids.append(cluster_centers)
            print(K)
        
        pos = np.where(np.array(scores) == np.max(scores))[0][0]
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
            os.makedirs(os.path.join('plots', 'TVFC'), exist_ok=True)
            plt.savefig(os.path.join("plots","TVFC", "rDCM-GEN_state_" + str(i) + ".png"), dpi=600, bbox_inches='tight')
            plt.show()



        
if __name__ == '__main__':
    config = {}
    config['path_data'] = r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_GNN_data/HCP'
    config['path_augmented_rdcm'] = r'/media/student1/Extreme Pro/augmented_new/augmented_rDCM-GEN.pt'
    config['path_augmented_crvae'] = r'/media/student1/Extreme Pro/CRVAE_models/augmented_CRVAE.pt'
    config['path_augmented_timeVae'] = r'/media/student1/Extreme Pro/TimeVAE_results/outputs/gen_data/HCP_train/timeVAE_HCP_train_prior_samples.npz'
    config['path_augmented_timeGan'] = r'/media/student1/Extreme Pro/TimeGAN_results/augmented_TimeGAN.npy'
    config['path_augmented_diffusion_ts'] = r'/media/student1/Extreme Pro/Diffusion-TS_results/augmented_Diffusion-TS.npy'
    config['atlas'] = 'SCH'
    config['device'] = "cuda:1"
    config['methods'] = ['CRVAE','TimeGAN','Diffusion-TS']#['HCP', 'rDCM-GEN', 'CRVAE', 'Noise-3dB', 'Noise-6dB', 'TimeVAE', 'TimeGAN', 'Diffusion-TS']
    results = main(config)   
    