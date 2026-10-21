import torch
import torch.nn.functional as F
from rdcm import tapas_rdcm_estimate, tapas_rdcm_model_specification
from reverse_dcm import GenerativeModel
import numpy as np
from tqdm import tqdm
import os
import pickle
import argparse
from scipy.signal import butter, filtfilt, resample
import matplotlib.pyplot as plt

def main(config):
    TR = config['TR']
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
        if not os.path.exists(os.path.join(config['save_path'], 'neural_noise', config['case'], name + '.pkl')):
            if config['case'] == "full_original":
                y = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
                order = 4
                cutoff = 0.1
                fs = 1/TR
                newTR = TR
                b, a = butter(order, cutoff, btype='low', fs=fs)
                y = filtfilt(b, a, y.T).T.astype('float32')
            elif config['case'] == "half_original":
                y = y[125:708]
                y = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
                order = 4
                cutoff = 0.1
                fs = 1/TR
                newTR = TR
                b, a = butter(order, cutoff, btype='low', fs=fs)
                y = filtfilt(b, a, y.T).T.astype('float32')
            elif config['case'] == "full_subsampled":
                y = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
                order = 4
                cutoff = 0.1
                fs = 1/TR
                b, a = butter(order, cutoff, btype='low', fs=fs)
                y = filtfilt(b, a, y.T).T.astype('float32')
                newTR = 2
                rate = newTR/TR
                new_samples = int(np.round(y.shape[0] / rate))
                y = resample(y, new_samples)
                y = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
            elif config['case'] == "half_subsampled":
                y = y[125:708]
                y = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
                order = 4
                cutoff = 0.1
                fs = 1/TR
                b, a = butter(order, cutoff, btype='low', fs=fs)
                y = filtfilt(b, a, y.T).T.astype('float32')
                newTR = 2
                rate = newTR/TR
                new_samples = int(np.round(y.shape[0] / rate))
                y = resample(y, new_samples)
                y = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
            
            R = y.shape[-1]
            Y = {
                 'y': y,
                 'dt': newTR
                 }
            
            DCM = tapas_rdcm_model_specification(Y)

            with torch.no_grad():
                out = tapas_rdcm_estimate(DCM, 'r', None, 1)
    
            epochs = config['epochs']
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
            #A_var = [5e-5, 1e-5, 1e-5, 1e-5, 1e-6]
            flag_stop = False
            while flag and rep < len(A_var):
                if rep == 0:
                    model = GenerativeModel(y=y, TR=newTR).to(device)
                else:
                    model = GenerativeModel(y=y, TR=newTR, 
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
                #plt.figure()
                #plt.plot(y[:,0])
                #plt.plot(y_hat[:,0].detach().cpu())
                
                if flag_stop or np.std(losses) > 20:
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
                with torch.no_grad():
                    if rep == 0:
                        new_out = tapas_rdcm_estimate(DCM ,'r', None, 1, True, [out[0]['Ep']['A'], A_var[rep]])
                    else:
                        new_out = tapas_rdcm_estimate(DCM ,'r', None, 1, True, [outputs[-1][0]['Ep']['A'], A_var[rep]])
                if rep == 0 and losses[-1] < losses[0]:
                    outputs.append(new_out)
                    neural_noises.append(neural_noise)
                    pink_neural_noises.append(neural_pink)
                    betas.append(beta)
                    NFE.append(new_out[0]['logF'])
                elif rep == 0 and losses[-1] >= losses[0]:
                    flag = False
                if rep >= 1:
                    if new_out[0]['logF'] > outputs[-1][0]['logF'] and losses_all[-1][-1] < losses_all[-2][-1]:
                        outputs.append(new_out)
                        neural_noises.append(neural_noise)
                        pink_neural_noises.append(neural_pink)
                        betas.append(beta)
                        NFE.append(new_out[0]['logF'])
                    elif new_out[0]['logF'] < outputs[-1][0]['logF'] and losses_all[-1][-1] < losses_all[-2][-1]:
                        DCM = tapas_rdcm_model_specification(Y, U, args)
                        temp = tapas_rdcm_estimate(DCM ,'r', None, 1, True, [outputs[-1][0]['Ep']['A'], 1e-16])
                        if temp[0]['logF'] > outputs[-1][0]['logF']:
                            neural_noises.append(neural_noise)
                            pink_neural_noises.append(neural_pink)
                            betas.append(beta)
                            NFE.append(temp[0]['logF'])
                        else:
                            flag = False
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
            if not os.path.exists(os.path.join(config['save_path'], 'neural_noise', config['case'])):
                os.makedirs(os.path.join(config['save_path'], 'neural_noise', config['case']), exist_ok=True)
            with open(os.path.join(config['save_path'], 'neural_noise', config['case'], name + '.pkl'), 'wb') as f:
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
    parser.add_argument('--finish', type=int, default=2000)
    parser.add_argument('--TR', type=float, default=0.72)
    parser.add_argument('--epochs', type=float, default=100)
    parser.add_argument('--case', type=str, default="full_subsampled")
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    config = {
        'data_path': args.data_path,
        'save_path': args.save_path,
        'start': args.start,
        'finish': args.finish,
        'TR': args.TR,
        'epochs': args.epochs,
        'case': args.case,
        'device': args.device
        }
    main(config)
            
