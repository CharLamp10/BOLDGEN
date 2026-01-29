import torch
import os
import numpy as np
from contrastive_learning import Transformer


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
            data = np.load(os.path.join(config['path_data'],'test_data_HCP_' + config['atlas'] + '_1_resampled.npz'))
            test_data1 = []
            for key in data:
                test_data1.append(data[key])
    
            data = np.load(os.path.join(config['path_data'],'test_data_HCP_' + config['atlas'] + '_2_resampled.npz'))
            test_data2 = []
            for key in data:
                test_data2.append(data[key])
            
            R = test_data1[0].shape[1]
            state_dict = torch.load(os.path.join(config['path_models'], 'models', 'SimCLR', 'model_' + method + '_tau' + str(tau) + '.pth'))
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
            path_save_new = os.path.join(config['path_save'], 'SimCLR', method, 'tau' + str(tau))
            os.makedirs(path_save_new, exist_ok=True)
            torch.save(outputs1, os.path.join(path_save_new, 'outputs1.pt'))
            torch.save(outputs2, os.path.join(path_save_new, 'outputs2.pt'))


if __name__ == '__main__':
    config = {}
    config['path_data'] = r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_GNN_data/HCP'
    config['path_models'] = r'/home/student1/Desktop/Charalampos_Lamprou/rDCM-GEN_results'
    config['path_save'] = r'/home/student1/Desktop/Charalampos_Lamprou/rDCM-GEN_results'
    config['methods'] = ['CRVAE','Noise-6dB', 'Noise-3dB', 'rDCM-GEN']
    config['atlas'] = 'SCH'
    config['taus'] = [0.05, 0.1, 0.15]
    config['model_config'] = {}
    config['model_config']['num_layers'] = 4
    config['model_config']['num_heads'] = 4
    config['model_config']['dim_feedforward'] = 1024
    config['model_config']['out_dim'] = 4096
    config['model_config']['D'] = 512
    config['device'] = "cuda:1"
    main(config)
