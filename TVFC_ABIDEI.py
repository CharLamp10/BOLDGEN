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

    train_data = np.load(os.path.join(path, 'ABIDEI_NC_nilearn.npy'))

    augmented_crvae = torch.load(os.path.join(config['path_augmented_crvae'], 'augmented_CR-VAE_ABIDEI.pt'))
    labels_crvae = torch.load(os.path.join(config['path_augmented_crvae'], 'labels_augmented_CR-VAE_ABIDEI.pt'))
    augmented_rdcmgen = torch.load(os.path.join(config['path_augmented_rdcmgen'], 'augmented_rDCM-GEN_ABIDEI.pt'))
    labels_rdcmgen = torch.load(os.path.join(config['path_augmented_rdcmgen'], 'labels_augmented_rDCM-GEN_ABIDEI.pt'))
    augmented_timevae = np.load(os.path.join(config['path_augmented_timevae'], 'ABIDEI_NC', 'timeVAE_ABIDEI_NC_prior_samples.npy'))
    augmented_timegan = np.load(os.path.join(config['path_augmented_timegan'], 'augmented_TimeGAN_ABIDEI_NC.npy'))
    augmented_t2idiff = np.load(os.path.join(config['path_augmented_t2idiff'], 'ABIDEI', 'gen_sig_ABIDEI_fold1.npy'))
    augmented_dsfm = np.load(os.path.join(config['path_augmented_dsfm'], 'ABIDEI', 'samples', 'control_bold.npy'))
        
    pos = torch.where(labels_crvae == 0)[0]
    augmented_crvae = augmented_crvae[pos]
    
    pos = torch.where(labels_rdcmgen == 0)[0]
    augmented_rdcmgen = augmented_rdcmgen[pos]
    
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
    
    if 'ABIDEI' in config['methods']:
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
        Ks = [2,3]#,4,5,6]
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
            os.makedirs(os.path.join('plots', 'TVFC', 'ABIDEI'), exist_ok=True)
            plt.savefig(os.path.join("plots","TVFC", 'ABIDEI', "state_" + str(i) + ".png"), dpi=600, bbox_inches='tight')
            plt.show()
    
    '''
    if 'CRVAE' in config['methods']:
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
            os.makedirs(os.path.join('plots', 'TVFC', 'ABIDEI'), exist_ok=True)
            plt.savefig(os.path.join("plots","TVFC", 'ABIDEI', "CRVAE_state_" + str(i) + ".png"), dpi=600, bbox_inches='tight')
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
            os.makedirs(os.path.join('plots', 'TVFC', 'ABIDEI'), exist_ok=True)
            plt.savefig(os.path.join("plots","TVFC", 'ABIDEI', "TimeVAE_state_" + str(i) + ".png"), dpi=600, bbox_inches='tight')
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
            os.makedirs(os.path.join('plots', 'TVFC', 'ABIDEI'), exist_ok=True)
            plt.savefig(os.path.join("plots","TVFC", 'ABIDEI', "TimeGAN_state_" + str(i) + ".png"), dpi=600, bbox_inches='tight')
            plt.show()
            
    if 'T2IDIFF' in config['methods']:
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
            os.makedirs(os.path.join('plots', 'TVFC', 'ABIDEI'), exist_ok=True)
            plt.savefig(os.path.join("plots","TVFC", 'ABIDEI', "T2IDIFF_state_" + str(i) + ".png"), dpi=600, bbox_inches='tight')
            plt.show()
    '''     
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
            os.makedirs(os.path.join('plots', 'TVFC', 'ABIDEI'), exist_ok=True)
            plt.savefig(os.path.join("plots","TVFC", 'ABIDEI', "DSFM_state_" + str(i) + ".png"), dpi=600, bbox_inches='tight')
            plt.show()
    '''
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
            os.makedirs(os.path.join('plots', 'TVFC', 'ABIDEI'), exist_ok=True)
            plt.savefig(os.path.join("plots","TVFC", 'ABIDEI', "rDCM-GEN_state_" + str(i) + ".png"), dpi=600, bbox_inches='tight')
            plt.show()
    '''



        
if __name__ == '__main__':
    config = {}
    config['path_data'] = r'/media/student1/ExtremePro/rDCM-GEN/data'
    config['path_save'] = r'/home/student1/Desktop/Charalampos_Lamprou/rDCM-GEN_results_HCP_2026'
    config['path_augmented_rdcmgen'] = r'/media/student1/ExtremePro/rDCM-GEN/rDCM-GEN_results/augmented_rDCM-GEN_noFCloss'
    config['path_augmented_crvae'] = r'/media/student1/ExtremePro/CRVAE_augmented'
    config['path_augmented_timevae'] = r'/media/student1/ExtremePro/TimeVAE_results/outputs/gen_data'
    config['path_augmented_timegan'] = r'/media/student1/ExtremePro/TimeGAN_results'
    config['path_augmented_t2idiff'] = r'/media/student1/ExtremePro/T2IDIFF'
    config['path_augmented_dsfm'] = r'/media/student1/ExtremePro/DSFM_results'
    config['atlas'] = 'SCH'
    config['TR'] = 0.72
    config['newTR'] = 2.0
    config['device'] = "cuda:2"
    config['wind'] = 30
    config['subjects'] = 200
    config['methods'] = ['ABIDEI', 'rDCM-GEN', 'CRVAE', 'TimeVAE', 'TimeGAN', 'T2IDIFF', 'DSFM']
    results = main(config)   
    