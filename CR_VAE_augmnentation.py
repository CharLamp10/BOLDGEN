import os
import torch
import numpy as np
parent_path = r'/home/student1/Desktop/Charalampos_Lamprou/CR-VAE'
import sys
sys.path.append(parent_path)
from models.cgru import CRVAE

def main(config):
    device = torch.device(config['device'])
    names_train = []
    with open(os.path.join(config['data_path'],'names_train.txt'), 'r') as f:
        for line in f:
            temp = line.strip()
            names_train.append(temp[:-7])
    
    T = 400
    R = 100
    fmri_crvae = torch.zeros(len(names_train), 15, T, R)
    
    with torch.no_grad():
        for i,name in enumerate(names_train):
            for j in range(15):
                state_dict = torch.load(os.path.join(config['path_crvae'],'model_'+ name + '.pth'))
                GC_est = np.load(os.path.join(config['path_crvae'],'GC_'+ name + '.npy'))
                crvae = CRVAE(R, GC_est, 64, device)
                crvae.load_state_dict(state_dict)
                crvae = crvae.to(device)
                X_t = torch.zeros((1,T,R), device=device)
                crvae.eval()
                temp, _ = crvae(X_t, 400, torch.randn((1,400,100), device=device), mode = 'test', phase = 1)
                fmri_crvae[i,j] = temp.cpu()
            print(i)
    
    positions = []
    for i in range(fmri_crvae.shape[0]):
        if torch.sum(fmri_crvae[i] == 0) >= 400*100:
            positions.append(i)
    
    mask = torch.ones(fmri_crvae.shape[0], dtype=torch.bool)   
    mask[positions] = False
    fmri_crvae = fmri_crvae[mask]
    names_crvae = [names_train[i] for i in range(len(names_train)) if i not in positions]
    torch.save(fmri_crvae, os.path.join(config['path_crvae'], 'augmented_cr_vae.pt'))
    with open(os.path.join(config['path_crvae'],'names_CRVAE.txt'), 'w') as f:
        for item in names_crvae:
            f.write("%s\n" % item)

if __name__ == "__main__":
    config = {}
    config['path_crvae'] = r'/media/student1/Extreme Pro/CRVAE_models'
    config['data_path'] = r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_GNN_data/HCP'
    main(config)

