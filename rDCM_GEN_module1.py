import torch
import torch.nn.functional as F
from rdcm import tapas_rdcm_estimate, tapas_rdcm_model_specification
from reverse_dcm import GenerativeModel
import numpy as np
from tqdm import tqdm
import os
import pickle
import argparse

def main(config):
    TR = 1.5
    device = torch.device(config['device']) if torch.cuda.is_available() else torch.device("cpu")
    names = []
    with open(os.path.join(config['data_path'],'names_train_all.txt'), 'r') as f:
        for line in f:
            names.append(line.strip())
            
    data = np.load(os.path.join(config['data_path'],'train_data_HCP_SCH_resampled_all.npz'))
    train_data = []
    for key in data:
        train_data.append(data[key][88:-88,:])
    
    train_data = train_data[config['start']:config['finish']]
    names = names[config['start']:config['finish']]
    for i, (y,name) in enumerate(zip(train_data,names)):
        y = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
        R = y.shape[-1]
        Y = {
             'y': y,
             'dt': TR
             }
    DCM = tapas_rdcm_model_specification(Y)
    with torch.no_grad():
        out = tapas_rdcm_estimate(DCM ,'r', None, 1)
    
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
    A_var = [1e-4, 5e-5, 1e-5, 1e-5, 1e-6, 1e-7]
    flag_stop = False
    while flag and rep < len(A_var):
        if rep == 0:
            model = GenerativeModel(y=y, TR=TR)
        else:
            model = GenerativeModel(y=y, TR=TR, 
                                    neural_init=neural_noise, 
                                    beta_init=beta)
        
        opt = torch.optim.Adam([
            {"params": model.parameters(), "lr": lr}
        ])
        losses = []
        with tqdm(total=epochs, desc='(T)') as pbar:
            for step in range(1,epochs+1):
                if flag_stop:
                    continue
                opt.zero_grad()
                if rep == 0:
                    y_hat = model(out)
                else:
                    y_hat = model(outputs[-1])
                loss = F.mse_loss(y_hat, torch.from_numpy(y).float())
                if torch.isinf(loss) or torch.isnan(loss):
                    flag_stop = True
                    break
                loss.backward()
                opt.step()
                losses.append(loss.item())
                pbar.set_postfix({'loss': loss.item()})
                pbar.update()
        
        if flag_stop:
            break
        neural_noise = model.neural_noise.detach()
        neural_pink = model.neural_pink.detach()
        beta = model.pink_from_normal.beta.detach()
        losses_all.append(losses)
        del model
        U = {
             'u': neural_pink
             }
        
        DCM = tapas_rdcm_model_specification(Y)
        with torch.no_grad():
            out = tapas_rdcm_estimate(DCM ,'r', None, 1)
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
        A_var = [1e-4, 5e-5, 1e-5, 1e-5, 1e-6, 1e-7]
        flag_stop = False
        while flag and rep < len(A_var):
            if rep == 0:
                model = GenerativeModel(y=y, TR=TR).to(device)
            else:
                model = GenerativeModel(y=y, TR=TR, 
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
        with open(os.path.join(config['save_path'], 'neural_noise', name[:-4] + '.pkl'), 'wb') as f:
            pickle.dump(result, f)
        del DCM
        print('--------------SUBJECT ' + str(i) + '--------------')
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run')
    parser.add_argument('--data_path', type=str, 
                        default=r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_GNN_data/HCP')
    parser.add_argument('--save_path', type=str, 
                        default=r'/home/student1/Desktop/Charalampos_Lamprou/VarCoNet_GNN_results')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--finish', type=int, default=10)
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
            
