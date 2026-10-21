import torch
import torch.nn.functional as F
from rdcm import tapas_rdcm_estimate, tapas_rdcm_model_specification
from reverse_dcm import GenerativeModel
import numpy as np
from tqdm import tqdm
import os
import pickle
import argparse
import matplotlib.pyplot as plt

def main(config):
    device = torch.device(config['device']) if torch.cuda.is_available() else torch.device("cpu")
    data_path = os.path.join(config['data_path'], 'ROIsignals_SCH_' + config['dataset'])
    data_dir = os.listdir(data_path)
    data_dir = data_dir[config['start']:config['finish']]
    for i, name in enumerate(data_dir):
        if not os.path.exists(os.path.join(config['save_path'], 'neural_noise', config['dataset'], name + '.pkl')):
            y = np.load(os.path.join(data_path, name, 'ROISignals.npy'))
            TR = np.load(os.path.join(data_path, name, 'TR.npy'))
            dur = y.shape[0]*TR
            if True: # TR <= 2 and dur >= 419:
                y = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
                R = y.shape[-1]
                Y = {
                     'y': y,
                     'dt': TR
                     }

                DCM = tapas_rdcm_model_specification(Y)
                with torch.no_grad():
                    out = tapas_rdcm_estimate(DCM, 'r', None, 1)
                
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
                        model = GenerativeModel(y=y, TR=TR).to(device)
                    else:
                        model = GenerativeModel(y=y, TR=TR, 
                                                neural_init=neural_noise, 
                                                beta_init=beta).to(device)
                    
                    opt = torch.optim.Adam([
                        {"params": model.parameters(), "lr": lr}
                    ])
                    losses = []
                    with tqdm(total=config['epochs'], desc='(T)') as pbar:
                        for step in range(1,config['epochs']+1):
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
                    #plt.show()
                    
                    if flag_stop:
                        break
                    if rep == 4 and loss.item() > 1:
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
                if not os.path.exists(os.path.join(config['save_path'], 'neural_noise', config['dataset'])):
                    os.makedirs(os.path.join(config['save_path'], 'neural_noise', config['dataset']), exist_ok=True)
                with open(os.path.join(config['save_path'], 'neural_noise', config['dataset'], name + '.pkl'), 'wb') as f:
                    pickle.dump(result, f)
                del DCM
                print('--------------SUBJECT ' + str(i) + '--------------')
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run')
    parser.add_argument('--data_path', type=str, 
                        default=r'/media/student1/ExtremePro/rDCM-GEN/data')
    parser.add_argument('--dataset', type=str, 
                        default='ABIDEI')
    parser.add_argument('--save_path', type=str, 
                        default=r'/media/student1/ExtremePro/rDCM-GEN/rDCM-GEN_results')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--finish', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda:2')
    args = parser.parse_args()
    config = {
        'data_path': args.data_path,
        'dataset': args.dataset,
        'save_path': args.save_path,
        'start': args.start,
        'finish': args.finish,
        'epochs': args.epochs,
        'device': args.device
        }
    main(config)
            
