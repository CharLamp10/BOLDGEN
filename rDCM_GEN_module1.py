import torch
import torch.nn.functional as F
from rdcm import tapas_rdcm_estimate, tapas_rdcm_model_specification
from reverse_dcm import GenerativeModel
import numpy as np
from tqdm import tqdm
import os
import pickle
import argparse
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

def main(config):
    TR = 0.72
    device = torch.device(config['device']) if torch.cuda.is_available() else torch.device("cpu")
    names_train = []
    with open(os.path.join(config['data_path'],'names_train.txt'), 'r') as f:
        for line in f:
            names_train.append(line.strip())
            
    data = np.load(os.path.join(config['data_path'],'train_data_HCP_SCH.npz'))
    train_data = []
    for key in data:
        train_data.append(data[key][183:-184,:])
    
    train_data = train_data[config['start']:config['finish']]
    names_train = names_train[config['start']:config['finish']]
    for i, (y,name) in enumerate(zip(train_data,names_train)):
        if not os.path.exists(os.path.join(config['save_path'], 'neural_noise', name + '.pkl')):
            y = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
            order = 4
            cutoff = 0.1
            fs = 1/TR
            b, a = butter(order, cutoff, btype='low', fs=fs)
            y = filtfilt(b, a, y.T).T.astype('float32')
            R = y.shape[-1]
            Y = {
                 'y': y,
                 'dt': TR
                 }
            corr = np.corrcoef(y.T)
            
            thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            outs = []
            for thr in thresholds:
                DCM = tapas_rdcm_model_specification(Y)
                mask = np.abs(corr) >= thr
                DCM['a'][~mask] = 0
                with torch.no_grad():
                    out = tapas_rdcm_estimate(DCM, 'r', None, 1)
                outs.append(out)
            
            # Extract logF values
            logFs = [out[0]['logF'] for out in outs]
            
            # Find index of maximum logF
            best_idx = np.argmax(logFs)
            
            # Select best output
            out = outs[best_idx]
            threshold = thresholds[best_idx]
            mask = np.abs(corr) >= threshold
    
            epochs = 100
            flag = True
            rep = 0
            neural_noises = []
            pink_neural_noises = []
            betas = []
            outputs = []
            NFE = [out[0]['logF']]
            losses_all = []
            lr = 7.5e-4
            A_var = [1e-4, 5e-5, 1e-5, 1e-5, 1e-6]
            flag_stop = False
            while flag and rep < len(A_var):
                if rep == 0:
                    model = GenerativeModel(y=y, TR=TR, mask=mask).to(device)
                else:
                    model = GenerativeModel(y=y, TR=TR, mask=mask, 
                                            neural_init=neural_noise, 
                                            beta_init=beta).to(device)
                
                opt = torch.optim.Adam([
                    {"params": model.parameters(), "lr": lr}
                ])
                losses = []
                with tqdm(total=epochs, desc='(T)') as pbar:
                    for step in range(1,epochs+1):
                        opt.zero_grad()
                        if rep == 0:
                            y_hat = model(out)
                        else:
                            y_hat = model(outputs[-1])
                        loss = F.mse_loss(y_hat, torch.from_numpy(y).float().to(y_hat.device))
                        if torch.isinf(loss) or torch.isnan(loss):
                            flag_stop = True
                            break
                        loss.backward()
                        opt.step()
                        losses.append(loss.item())
                        pbar.set_postfix({'loss': loss.item()})
                        pbar.update()
                plt.figure()
                plt.plot(y[:,0])
                plt.plot(y_hat[:,0].detach().cpu())
                
                if flag_stop:
                    break
                neural_noise = model.neural_noise.detach().cpu()
                neural_pink = model.neural_pink.detach().cpu()
                beta = model.pink_from_normal.beta.detach().cpu()
                losses_all.append(losses)
                del model
                U = {
                     'u': neural_pink
                     }
                args = {
                        'a': np.ones((R,R)),
                        'c': np.eye(R)
                        }
                DCM = tapas_rdcm_model_specification(Y, U, args)
                DCM['a'][~mask] = 0
                with torch.no_grad():
                    if rep == 0:
                        new_out = tapas_rdcm_estimate(DCM ,'r', None, 1, True, [out[0]['Ep']['A'], A_var[rep]])
                    else:
                        new_out = tapas_rdcm_estimate(DCM ,'r', None, 1, True, [outputs[-1][0]['Ep']['A'], A_var[rep]])
                if rep == 0:
                    outputs.append(new_out)
                    neural_noises.append(neural_noise)
                    pink_neural_noises.append(neural_pink)
                    betas.append(beta)
                    NFE.append(new_out[0]['logF'])
                if rep >= 1:
                    if new_out[0]['logF'] > outputs[-1][0]['logF']:
                        outputs.append(new_out)
                        neural_noises.append(neural_noise)
                        pink_neural_noises.append(neural_pink)
                        betas.append(beta)
                        NFE.append(new_out[0]['logF'])
                    else:
                        DCM = tapas_rdcm_model_specification(Y, U, args)
                        DCM['a'][~mask] = 0
                        temp = tapas_rdcm_estimate(DCM ,'r', None, 1, True, [outputs[-1][0]['Ep']['A'], 1e-16])
                        if temp[0]['logF'] > outputs[-1][0]['logF']:
                            neural_noises.append(neural_noise)
                            pink_neural_noises.append(neural_pink)
                            betas.append(beta)
                            NFE.append(temp[0]['logF'])
                        else:
                            flag = False
                rep += 1
            result = {}
            result['outputs'] = outputs
            result['neural_noises'] = neural_noises
            result['pink_neural_noises'] = pink_neural_noises
            result['betas'] = betas
            result['NFE'] = NFE
            result['losses_all'] = losses_all
            if not os.path.exists(os.path.join(config['save_path'], 'neural_noise')):
                os.makedirs(os.path.join(config['save_path'], 'neural_noise'), exist_ok=True)
            with open(os.path.join(config['save_path'], 'neural_noise', name + '.pkl'), 'wb') as f:
                pickle.dump(result, f)
            del DCM
            print('--------------SUBJECT ' + str(i) + '--------------')
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run')
    parser.add_argument('--data_path', type=str, 
                        default=r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_GNN_data/HCP_2026')
    parser.add_argument('--save_path', type=str, 
                        default=r'/media/student1/ExtremePro/rDCM-GEN/rDCM-GEN_results')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--finish', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    config = {
        'data_path': args.data_path,
        'save_path': args.save_path,
        'start': args.start,
        'finish': args.finish,
        'device': args.device
        }
    main(config)
            
