import os
import torch
import numpy as np
parent_path = r'/home/student1/Desktop/Charalampos_Lamprou/CR-VAE'
import sys
sys.path.append(parent_path)
from models.cgru import CRVAE

path_crvae = r'/home/student1/Desktop/Charalampos_Lamprou/CR-VAE'
path_crvae_models = r'/media/student1/Extreme Pro/CRVAE_models'
data_path = r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_GNN_data/HCP'

device = torch.device("cuda:2")

names_train = []
with open(os.path.join(data_path,'names_train.txt'), 'r') as f:
    for line in f:
        temp = line.strip()
        names_train.append(temp[:-7])

T = 400
R = 100
fmri_crvae = torch.zeros(len(names_train), 15, T, R)

with torch.no_grad():
    for i,name in enumerate(names_train):
        if i > 773:
            for j in range(15):
                state_dict = torch.load(os.path.join(path_crvae_models,'model_'+ name + '.pth'))
                GC_est = np.load(os.path.join(path_crvae_models,'GC_'+ name + '.npy'))
                crvae = CRVAE(R, GC_est, 64, device)
                crvae.load_state_dict(state_dict)
                crvae = crvae.to(device)
                X_t = torch.zeros((1,T,R), device=device)
                crvae.eval()
                temp, _ = crvae(X_t, 400, torch.randn((1,400,100), device=device), mode = 'test', phase = 1)
                fmri_crvae[i,j] = temp.cpu()
            print(i)

torch.save(fmri_crvae, os.path.join(path_crvae_models, 'augmented_cr_vae.pt'))

