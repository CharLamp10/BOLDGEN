import os
import torch
import numpy as np
parent_path = r'/home/student1/Desktop/Charalampos_Lamprou/CR-VAE'
import sys
sys.path.append(parent_path)
from models.cgru import CRVAE
from scipy.signal import resample
import argparse


def resample_signal(signal):
    ratio = 0.72/1.5
    new_len = int(np.round(signal.shape[0]*ratio))
    signal = resample(signal,new_len)
    return signal



def main(config):
    device = torch.device(config['device'])
    dire = os.listdir(config['path_crvae'])
    directory = []
    for file in dire:
        if 'model' in file:
            directory.append(file)
    T = 833
    R = 100
    directory = directory[config['start']:config['finish']]
    with torch.no_grad():
        for i,name in enumerate(directory):
            if not os.path.exists(os.path.join(config['path_save'], name[6:-4] + '.pt')):
                fmri_crvae = torch.zeros(15, 400, R)
                for j in range(15):
                    state_dict = torch.load(os.path.join(config['path_crvae'], name))
                    GC_est = np.load(os.path.join(config['path_crvae'], 'GC_' + name[6:-4]  + '.npy'))
                    crvae = CRVAE(R, GC_est, 64, device)
                    crvae.load_state_dict(state_dict)
                    crvae = crvae.to(device)
                    X_t = torch.zeros((1,T,R), device=device)
                    crvae.eval()
                    temp, _ = crvae(X_t, T, torch.randn((1,T,R), device=device), mode = 'test', phase = 1)
                    temp = temp.squeeze().cpu().numpy()
                    temp = resample_signal(temp)
                    fmri_crvae[j] = torch.from_numpy(temp)
                torch.save(fmri_crvae, os.path.join(config['path_save'], name[6:-4] + '.pt'))
            print(i)
                


    
            
    '''
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
    '''

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description='Run')
   parser.add_argument('--path_crvae', type=str, 
                       default=r'/media/student1/ExtremePro/CRVAE_models_HCP_2026')
   parser.add_argument('--path_save', type=str,
                       default=r'/media/student1/ExtremePro/CRVAE_augmented')
   parser.add_argument('--start', type=int, default=0)
   parser.add_argument('--finish', type=int, default=1000)
   parser.add_argument('--device', type=str, default='cuda:0')
   args = parser.parse_args()
   config = {
       'path_crvae': args.path_crvae,
       'path_save': args.path_save,
       'start': args.start,
       'finish': args.finish,
       'device': args.device
       }
   main(config)


