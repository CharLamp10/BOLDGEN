import torch
import os
import numpy as np
from contrastive_learning import Transformer
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


def forward_Transformer(encoder_model,test_data1,test_data2,batch_size,device):
    test_data1 = torch.from_numpy(np.stack(test_data1))
    test_data2 = torch.from_numpy(np.stack(test_data2))
    encoder_model.eval()
    with torch.no_grad():
        outputs1 = []
        for i in range(0, test_data1.size(0), batch_size):
            data = test_data1[i:i+batch_size].to(device)
            z1 = encoder_model(data)
            outputs1.append(z1)
        outputs1 = torch.cat(outputs1, dim=0)
        
        outputs2 = []
        for i in range(0, test_data2.size(0), batch_size):
            data = test_data2[i:i+batch_size].to(device)
            z2 = encoder_model(data)
            outputs2.append(z2)
        outputs2 = torch.cat(outputs2, dim=0)
    return outputs1,outputs2


def main(config):
    os.makedirs(config['path_save'], exist_ok=True)
    device = torch.device(config['device'])
    model_config = config['model_config']
        
    for method in config['methods']:
        for tau in config['taus']:
            names = []
            with open(os.path.join(config['path_data'],'names_test.txt'), 'r') as f:
                for line in f:
                    temp = line.strip()
                    names.append(temp)
                
            data = np.load(os.path.join(config['path_data'],'test_data_HCP_' + config['atlas'] + '.npz'))
            test_data1 = []
            names_temp = []
            for (key, name) in zip(data, names):
                if 'LR' in name and 'REST1' in name and data[key].shape[0] == 1200 and not np.all(data[key] == 0):
                    temp = data[key][183:-184]
                    test_data1.append(temp)
                    names_temp.append(name[:6])
            
            data = np.load(os.path.join(config['path_data'],'test_data_HCP_' + config['atlas'] + '.npz'))
            test_data2 = []
            names_test = []
            for (key, name) in zip(data, names):
                if 'LR' in name and 'REST2' in name and name[:6] in names_temp and data[key].shape[0] == 1200 and not np.all(data[key] == 0):
                    temp = data[key][183:-184]
                    test_data2.append(temp)
                    names_test.append(name[:6])
            
            remove = []
            for (i,name) in enumerate(names_temp):
                if name not in names_test:
                    remove.append(i)
            remove = remove[::-1]
            
            for index in remove:
                test_data1.pop(index)
                
            test_data1 = preprocess_HCP(test_data1, config['case'])
            test_data2 = preprocess_HCP(test_data2, config['case'])
            
            R = test_data1[0].shape[1]
            state_dict = torch.load(os.path.join(config['path_models'], 'models', 'SimCLR', config['case'], 'model_' + method + '_tau' + str(tau) + '.pth'))
            model = Transformer(R,
                                model_config['D'],
                                model_config['out_dim'],
                                model_config['num_layers'],
                                model_config['num_heads'],
                                model_config['dim_feedforward']).to(device)
            model.load_state_dict(state_dict)
            model = model.to(device)
            outputs1, outputs2 = forward_Transformer(model, test_data1, test_data2, 512, device)
            outputs1 = outputs1.cpu()
            outputs2 = outputs2.cpu()
            path_save_new = os.path.join(config['path_save'], 'SimCLR_embeddings', config['case'], method, 'tau' + str(tau))
            os.makedirs(path_save_new, exist_ok=True)
            torch.save(outputs1, os.path.join(path_save_new, 'outputs1.pt'))
            torch.save(outputs2, os.path.join(path_save_new, 'outputs2.pt'))


if __name__ == '__main__':
    config = {}
    config['path_data'] = r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_GNN_data/HCP_2026'
    config['path_models'] = r'/home/student1/Desktop/Charalampos_Lamprou/rDCM-GEN_results_HCP_2026'
    config['path_save'] = r'/home/student1/Desktop/Charalampos_Lamprou/rDCM-GEN_results_HCP_2026'
    config['methods'] = ['CR-VAE','rDCM-GEN']
    config['atlas'] = 'SCH'
    config['case'] = "full_subsampled"
    config['TR'] = 0.72
    config['newTR'] = 2.0
    config['taus'] = [0.05, 0.1, 0.15]
    config['model_config'] = {}
    config['model_config']['num_layers'] = 4
    config['model_config']['num_heads'] = 4
    config['model_config']['dim_feedforward'] = 1024
    config['model_config']['out_dim'] = 4096
    config['model_config']['D'] = 512
    config['device'] = "cuda:0"
    main(config)
