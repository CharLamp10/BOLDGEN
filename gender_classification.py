"""Evaluate contrastively pretrained representations through gender classification."""
import argparse

import numpy as np
import torch
import os
import pickle
import pandas as pd
from contrastive_learning import Transformer
from classifier import LREvaluator
from scipy.signal import butter, filtfilt, resample


def preprocess_HCP(data, cat):
    for i, y in enumerate(data):
        if cat == "full_original":
            y_train = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
            order = 4
            cutoff = 0.1
            fs = 1/config['TR']
            b, a = butter(order, cutoff, btype='low', fs=fs)
            y = filtfilt(b, a, y.T).T.astype('float32')
        elif cat == "half_original":
            y = (y[125:708] - np.mean(y[125:708], axis=0, keepdims=True))/np.std(y[125:708], axis=0, keepdims=True)
            order = 4
            cutoff = 0.1
            fs = 1/config['TR']
            b, a = butter(order, cutoff, btype='low', fs=fs)
            y = filtfilt(b, a, y.T).T.astype('float32')
        elif cat == "full_subsampled":
            y_train = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
            order = 4
            cutoff = 0.1
            fs = 1/config['TR']
            b, a = butter(order, cutoff, btype='low', fs=fs)
            y = filtfilt(b, a, y.T).T.astype('float32')
            rate = config['newTR']/config['TR']
            new_samples = int(np.round(y_train.shape[0] / rate))
            y = resample(y, new_samples)
            y = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
        elif cat == "half_subsampled":
            y = (y[125:708] - np.mean(y[125:708], axis=0, keepdims=True))/np.std(y[125:708], axis=0, keepdims=True)
            order = 4
            cutoff = 0.1
            fs = 1/config['TR']
            b, a = butter(order, cutoff, btype='low', fs=fs)
            y = filtfilt(b, a, y.T).T.astype('float32')
            rate = config['newTR']/config['TR']
            new_samples = int(np.round(y.shape[0] / rate))
            y = resample(y, new_samples)
            y = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
        data[i] = y
    return data


def test(encoder_model,train_data,val_data,test_data,age_train,age_val,age_test,batch_size,device):
    train_data = torch.from_numpy(np.stack(train_data))
    val_data = torch.from_numpy(np.stack(val_data))
    test_data = torch.from_numpy(np.stack(test_data))
    encoder_model.eval()
    with torch.no_grad():
        z_train = []
        for i in range(0, train_data.size(0), batch_size):
            x = train_data[i:i+batch_size].to(device)
            z_train.append(encoder_model(x))
        z_train = torch.cat(z_train, dim=0)  
        
        z_val = []
        for i in range(0, val_data.size(0), batch_size):
            x = val_data[i:i+batch_size].to(device)
            z_val.append(encoder_model(x))
        z_val = torch.cat(z_val, dim=0)  
        
        z_test = []
        for i in range(0, test_data.size(0), batch_size):
            x = test_data[i:i+batch_size].to(device)
            z_test.append(encoder_model(x))
        z_test = torch.cat(z_test, dim=0) 
        
        age_train = torch.tensor(age_train).long().to(device)
        age_val = torch.tensor(age_val).long().to(device)
        age_test = torch.tensor(age_test).long().to(device)
        
    res = LREvaluator(num_epochs=200,learning_rate=1e-4).evaluate(z_train, age_train, z_val, age_val, z_test, age_test, 2, device)
        
    return res


def main(config):

    path = config['path_data']
    phenotypic = pd.read_csv(os.path.join(path, 'phenotypic.csv'))

    names_train = []
    with open(os.path.join(path,'names_train.txt'), 'r') as f:
        for line in f:
            temp = line.strip()
            names_train.append(temp[:-4])
            
    names_val = []
    with open(os.path.join(path,'names_val.txt'), 'r') as f:
        for line in f:
            temp = line.strip()
            names_val.append(temp[:-4])
            
    names_test = []
    with open(os.path.join(path,'names_test.txt'), 'r') as f:
        for line in f:
            temp = line.strip()
            names_test.append(temp[:-4])
            
    gender = phenotypic['Gender'].values
    subject_id = phenotypic['Subject'].values
    
    gender_train = []
    for name in names_train:
        pos = np.where(subject_id == int(name[:6]))[0][0]
        g = gender[pos]
        if g == 'M':
            g = 1
        elif g == 'F':
            g = 0
        gender_train.append(g)

    gender_val = []
    for name in names_val:
        pos = np.where(subject_id == int(name[:6]))[0][0]
        g = gender[pos]
        if g == 'M':
            g = 1
        elif g == 'F':
            g = 0
        gender_val.append(g)
     
    gender_test = []
    for name in names_test:
        pos = np.where(subject_id == int(name[:6]))[0][0]
        g = gender[pos]
        if g == 'M':
            g = 1
        elif g == 'F':
            g = 0
        gender_test.append(g)
    
    data = np.load(os.path.join(path,'train_data_HCP_' + config['atlas'] + '.npz'))
    train_data = []
    pos_train = []
    for i,key in enumerate(data):
        if data[key].shape[0] != 1200 or np.all(data[key] == 0):
            pos_train.append(i)
        train_data.append(data[key][183:-184])
    
    data = np.load(os.path.join(path,'val_data_HCP_' + config['atlas'] + '.npz'))
    val_data = []
    pos_val = []
    for i,key in enumerate(data):
        if data[key].shape[0] != 1200 or np.all(data[key] == 0):
            pos_val.append(i)
        val_data.append(data[key][183:-184])
        
    data = np.load(os.path.join(path,'test_data_HCP_' + config['atlas'] + '.npz'))
    test_data = []
    pos_test = []
    for i,key in enumerate(data):
        if data[key].shape[0] != 1200 or np.all(data[key] == 0):
            pos_test.append(i)
        test_data.append(data[key][183:-184])
     
    pos_train = pos_train[::-1]
    pos_val = pos_val[::-1]
    pos_test = pos_test[::-1]
    
    for p in pos_train:
        gender_train.pop(p)
        train_data.pop(p)
        
    for p in pos_val:
        gender_val.pop(p)
        val_data.pop(p)
        
    for p in pos_test:
        gender_test.pop(p)
        test_data.pop(p)
    
    train_data = preprocess_HCP(train_data, config['case'])
    val_data = preprocess_HCP(val_data, config['case'])
    test_data = preprocess_HCP(test_data, config['case'])
    
    T,R = test_data[0].shape
    
    device = torch.device(config['device']) if torch.cuda.is_available() else torch.device("cpu")
    batch_size = config['batch_size']
    save_results = config['save_results']
    
    for tau in config['taus']:
        for method in config['methods']:   
            result = []
            for i in range(10):
                path_pretrained = os.path.join(config['path_save'],'models', 'SimCLR', config['case'], 'model_' + method + '_tau' + str(tau) + '.pth') 
                state_dict = torch.load(path_pretrained)  
                model_config = config['model_config']
                encoder_model = Transformer(R,
                                    model_config['D'],
                                    model_config['out_dim'],
                                    model_config['num_layers'],
                                    model_config['num_heads'],
                                    model_config['dim_feedforward']).to(device)
                encoder_model.load_state_dict(state_dict)
            
                res,_ = test(encoder_model, train_data, val_data, test_data, gender_train, gender_val, gender_test, batch_size, device)
                result.append(res)
            results = {}
            results['result'] = result
            
            if save_results:
                if not os.path.exists(os.path.join(config['path_save'],'Gender','SimCLR', config['case'], 'tau_' + str(tau))):
                    os.makedirs(os.path.join(config['path_save'],'Gender','SimCLR', config['case'], 'tau_' + str(tau)), exist_ok=True)   
                with open(os.path.join(config['path_save'],'Gender','SimCLR', config['case'], 'tau_' + str(tau),'results_' + method + '.pkl'), 'wb') as f:
                    pickle.dump(results,f)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate contrastively pretrained representations through gender classification.')
    parser.add_argument('--path_data', type=str, required=True)
    parser.add_argument('--path_save', type=str, required=True)
    parser.add_argument('--methods', nargs="+", type=str, default=['CR-VAE', 'BOLDGEN'])
    parser.add_argument('--atlas', type=str, default='SCH')
    parser.add_argument('--case', type=str, default="full_subsampled")
    parser.add_argument('--TR', type=float, default=0.72)
    parser.add_argument('--newTR', type=float, default=2.0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--shuffle', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--taus', nargs="+", type=float, default=[0.05, 0.1, 0.15])
    parser.add_argument('--model_num_layers', type=int, default=4)
    parser.add_argument('--model_num_heads', type=int, default=4)
    parser.add_argument('--model_dim_feedforward', type=int, default=1024)
    parser.add_argument('--model_out_dim', type=int, default=4096)
    parser.add_argument('--model_D', type=int, default=512)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--save_results', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    config = {}
    config['path_data'] = args.path_data
    config['path_save'] = args.path_save
    config['methods'] = args.methods
    config['atlas'] = args.atlas
    config['case'] = args.case
    config['TR'] = args.TR
    config['newTR'] = args.newTR
    config['batch_size'] = args.batch_size
    config['shuffle'] = args.shuffle
    config['taus'] = args.taus
    config['model_config'] = {}
    config['model_config']['num_layers'] = args.model_num_layers
    config['model_config']['num_heads'] = args.model_num_heads
    config['model_config']['dim_feedforward'] = args.model_dim_feedforward
    config['model_config']['out_dim'] = args.model_out_dim
    config['model_config']['D'] = args.model_D
    config['device'] = args.device
    config['save_results'] = args.save_results
    results = main(config)
