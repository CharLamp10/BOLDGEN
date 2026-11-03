"""Evaluate preservation of individual functional-connectivity fingerprints."""
import argparse

import os
import numpy as np
import torch
import pickle


def fingerprinting(x1, x2):
    corr_coeffs = torch.corrcoef(torch.cat([x1,x2],dim=0))[:x1.shape[0],x2.shape[0]:]
    lower_indices = torch.tril_indices(corr_coeffs.shape[0],corr_coeffs.shape[1], offset=-1)
    upper_indices = torch.triu_indices(corr_coeffs.shape[0],corr_coeffs.shape[1], offset=1)
    corr_coeffs1 = corr_coeffs.clone()
    corr_coeffs2 = corr_coeffs.clone()
    corr_coeffs1[lower_indices[0],lower_indices[1]] = -2
    corr_coeffs2[upper_indices[0],upper_indices[1]] = -2
    counter1 = 0
    counter2 = 0
    for j in range(corr_coeffs1.shape[0]):
        if torch.argmax(corr_coeffs1[j, :]) == j:
            counter1 += 1
    for j in range(corr_coeffs2.shape[1]):
        if torch.argmax(corr_coeffs2[:, j]) == j:
            counter2 += 1
    
    total_samples = x1.shape[0] + x2.shape[0]
    accuracy = ((counter1 + counter2) / total_samples)
    return accuracy





def main(config):
    names = []
    with open(os.path.join(config['path_data'],'names_train.txt'), 'r') as f:
        for line in f:
            temp = line.strip()
            names.append(temp)
            
    data = np.load(os.path.join(config['path_data'],'train_data_HCP_SCH.npz'))
    train_data = []
    for key in data:
        train_data.append(data[key][186:-187])
        
    names_rdcmgen = []
    with open(os.path.join(config['path_augmented_rdcm'],'names_rDCM-GEN_' + config['case'] + '.txt'), 'r') as f:
        for line in f:
            temp = line.strip()
            names_rdcmgen.append(temp)
            
    names_crvae = []
    with open(os.path.join(config['path_augmented_crvae'],'names_CR-VAE_' + config['case'] + '.txt'), 'r') as f:
        for line in f:
            temp = line.strip()
            names_crvae.append(temp)
            
    augmented_rdcm = torch.load(os.path.join(config['path_augmented_rdcm'], 'augmented_rDCM-GEN_' + config['case'] + '.pt'))
    augmented_crvae = torch.load(os.path.join(config['path_augmented_crvae'], 'augmented_CR-VAE_' + config['case'] + '.pt'))
    
    names_SF = []
    for name in names:
        if name in names_rdcmgen and name in names_crvae and 'REST1_LR' in name:
            name2 = name
            name2 = name2.replace('REST1_LR', 'REST2_LR')
            if name2 in names and name2 in names_rdcmgen and name2 in names_crvae:
                names_SF.append(name[:6])
    
    names_SF = names_SF[:128]
    
    ids = names.copy()
    for i,idd in enumerate(ids):
        ids[i] = idd[:6]
        
    data1 = []
    data2 = []
    data1_rDCM = []
    data2_rDCM = []
    data1_CRVAE = []
    data2_CRVAE = []
    for name in names_SF:
        pos = np.where(np.array(ids) == name)[0]
        for p in pos:
            if "REST1_LR" in names[p]:
                data1.append(train_data[p])
            elif "REST2_LR" in names[p]:
                data2.append(train_data[p])
                
                
    ids = names_rdcmgen.copy()
    for i,idd in enumerate(ids):
        ids[i] = idd[:6]
    for name in names_SF:
        pos = np.where(np.array(ids) == name)[0]
        for p in pos:
            if "REST1_LR" in names_rdcmgen[p]:
                data1_rDCM.append(augmented_rdcm[p])
            elif "REST2_LR" in names_rdcmgen[p]:
                data2_rDCM.append(augmented_rdcm[p])
                
    ids = names_crvae.copy()
    for i,idd in enumerate(ids):
        ids[i] = idd[:6]
    for name in names_SF:
        pos = np.where(np.array(ids) == name)[0]
        for p in pos:
            if "REST1_LR" in names_crvae[p]:
                data1_CRVAE.append(augmented_crvae[p])
            elif "REST2_LR" in names_crvae[p]:
                data2_CRVAE.append(augmented_crvae[p])
                
    data1 = torch.from_numpy(np.stack(data1))
    data2 = torch.from_numpy(np.stack(data2))
    data1 = (data1 - torch.mean(data1, dim=1, keepdim=True))/torch.std(data1, dim=1, keepdim=True)
    data2 = (data2 - torch.mean(data2, dim=1, keepdim=True))/torch.std(data2, dim=1, keepdim=True)
    
    accuracies = []
    accuracies_rdcm = []
    accuracies_crvae = []
    for j in range(15):
        data1_rDCM_j = torch.stack(data1_rDCM)[:,j,:,:]
        data2_rDCM_j = torch.stack(data2_rDCM)[:,j,:,:]
        data1_CRVAE_j = torch.stack(data1_CRVAE)[:,j,:,:]
        data2_CRVAE_j = torch.stack(data2_CRVAE)[:,j,:,:]

        
        R = 100
        FC_test1 = torch.zeros(data1.shape[0], R*(R-1)//2)
        FC_test2 = torch.zeros(data1.shape[0], R*(R-1)//2)
        FC_rdcm_test1 = torch.zeros(data1.shape[0], R*(R-1)//2)
        FC_rdcm_test2 = torch.zeros(data1.shape[0], R*(R-1)//2)
        FC_crvae_test1 = torch.zeros(data1.shape[0], R*(R-1)//2)
        FC_crvae_test2 = torch.zeros(data1.shape[0], R*(R-1)//2)
        triu_indices = torch.triu_indices(R, R, offset=1)
        for i, (y1, y2, y1_rdcm, y2_rdcm, y1_crvae, y2_crvae) in enumerate(zip(data1, data2, data1_rDCM_j, data2_rDCM_j, data1_CRVAE_j, data2_CRVAE_j)):
            FC1 = torch.corrcoef(y1.T)
            FC_test1[i] = FC1[triu_indices[0], triu_indices[1]]
            FC2 = torch.corrcoef(y2.T)
            FC_test2[i] = FC2[triu_indices[0], triu_indices[1]]
            FC1_gen = torch.corrcoef(y1_rdcm.T)
            FC_rdcm_test1[i] = FC1_gen[triu_indices[0], triu_indices[1]]
            FC2_gen = torch.corrcoef(y2_rdcm.T)
            FC_rdcm_test2[i] = FC2_gen[triu_indices[0], triu_indices[1]]
            FC1_gen = torch.corrcoef(y1_crvae.T)
            FC_crvae_test1[i] = FC1_gen[triu_indices[0], triu_indices[1]]
            FC2_gen = torch.corrcoef(y2_crvae.T)
            FC_crvae_test2[i] = FC2_gen[triu_indices[0], triu_indices[1]]
            
        accuracy = fingerprinting(FC_test1, FC_test2)
            
        accuracy_rdcm_1 = fingerprinting(FC_test1, FC_rdcm_test2)
        accuracy_rdcm_2 = fingerprinting(FC_rdcm_test1, FC_test2)
        accuracy_rdcm_both = (accuracy_rdcm_1 + accuracy_rdcm_2)/2
        
        accuracy_crvae_1 = fingerprinting(FC_test1, FC_crvae_test2)
        accuracy_crvae_2 = fingerprinting(FC_crvae_test1, FC_test2)
        accuracy_crvae_both = (accuracy_crvae_1 + accuracy_crvae_2)/2
        
        accuracies.append(accuracy)
        accuracies_rdcm.append(accuracy_rdcm_both)
        accuracies_crvae.append(accuracy_crvae_both)
    
    result = {}
    result['orig'] = accuracies
    result['rDCM-GEN'] = accuracies_rdcm
    result['CR-VAE'] = accuracies_crvae
    
    os.makedirs(os.path.join(config['path_save'], 'SF', config['case']), exist_ok=True)
    with open(os.path.join(config['path_save'], 'SF', config['case'], 'results.pkl'), 'wb') as f:
        pickle.dump(result, f)
    
    return result
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate preservation of individual functional-connectivity fingerprints.')
    parser.add_argument('--path_data', type=str, required=True)
    parser.add_argument('--path_save', type=str, required=True)
    parser.add_argument('--path_augmented_rdcm', type=str, required=True)
    parser.add_argument('--path_augmented_crvae', type=str, required=True)
    parser.add_argument('--case', type=str, default="full_subsampled")
    args = parser.parse_args()
    config = {}
    config['path_data'] = args.path_data
    config['path_save'] = args.path_save
    config['path_augmented_rdcm'] = args.path_augmented_rdcm
    config['path_augmented_crvae'] = args.path_augmented_crvae
    config['case'] = args.case
    result = main(config)