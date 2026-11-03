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
    dire = os.listdir(os.path.join(config['path_crvae'], config['case']))
    if config['case'] == "full_original":
        T = 833
    elif config['case'] == "full_subsampled":
        T = 300
    elif config['case'] == "half_original":
        T = 583
    elif config['case'] == "half_subsampled":
        T = 210
    elif config['case'] == "ABIDEI":
        T = 200
    R = 100
    directory = []
    for file in dire:
        if 'model' in file:
            directory.append(file)
    directory = directory[config['start']:config['finish']]
    os.makedirs(os.path.join(config['path_save'], config['case']), exist_ok=True)
    with torch.no_grad():
        for i,name in enumerate(directory):
            if not os.path.exists(os.path.join(config['path_save'], config['case'], name[6:-4] + '.pt')):
                fmri_crvae = torch.zeros(15, T, R)
                for j in range(15):
                    state_dict = torch.load(os.path.join(config['path_crvae'], config['case'], name))
                    GC_est = np.load(os.path.join(config['path_crvae'], config['case'], 'GC_' + name[6:-4]  + '.npy'))
                    crvae = CRVAE(R, GC_est, 64, device)
                    crvae.load_state_dict(state_dict)
                    crvae = crvae.to(device)
                    X_t = torch.zeros((1,T,R), device=device)
                    crvae.eval()
                    temp, _ = crvae(X_t, T, torch.randn((1,T,R), device=device), mode = 'test', phase = 1)
                    temp = temp.squeeze().cpu().numpy()
                    #temp = resample_signal(temp)
                    fmri_crvae[j] = torch.from_numpy(temp)
                torch.save(fmri_crvae, os.path.join(config['path_save'], config['case'], name[6:-4] + '.pt'))
            print(i)
                

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description='Run')
   parser.add_argument('--path_crvae', type=str, 
                       default=r'/media/student1/ExtremePro/CR-VAE_models')
   parser.add_argument('--path_save', type=str,
                       default=r'/media/student1/ExtremePro/CR-VAE_augmented')
   parser.add_argument('--start', type=int, default=0)
   parser.add_argument('--finish', type=int, default=1000)
   parser.add_argument('--case', type=str, default="ABIDEI")
   parser.add_argument('--device', type=str, default='cuda:0')
   args = parser.parse_args()
   config = {
       'path_crvae': args.path_crvae,
       'path_save': args.path_save,
       'start': args.start,
       'finish': args.finish,
       'case': args.case,
       'device': args.device
       }
   main(config)


