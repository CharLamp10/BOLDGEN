import numpy as np
import torch
import os
import pickle
import pandas as pd
from classifier import LREvaluator
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, filtfilt, resample


def preprocess_HCP(data, cat):
    for i, y in enumerate(data):
        if cat == "full_original":
            y_train = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
            order = 4
            cutoff = 0.1
            fs = 1/config['TR']
            b, a = butter(order, cutoff, btype='low', fs=fs)
            y = filtfilt(b, a, y.T).astype('float32')
        elif cat == "half_original":
            y = (y[125:708] - np.mean(y[125:708], axis=0, keepdims=True))/np.std(y[125:708], axis=0, keepdims=True)
            order = 4
            cutoff = 0.1
            fs = 1/config['TR']
            b, a = butter(order, cutoff, btype='low', fs=fs)
            y = filtfilt(b, a, y.T).astype('float32')
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
            y = y.T
        elif cat == "half_subsampled":
            y = (y[125:708] - np.mean(y[125:708], axis=0, keepdims=True))/np.std(y[125:708], axis=0, keepdims=True)
            order = 4
            cutoff = 0.1
            fs = 1/config['TR']
            b, a = butter(order, cutoff, btype='low', fs=fs)
            y = filtfilt(b, a, y.T).T.astype('float32')
            rate = config['newTR']/config['TR']
            new_samples = int(np.round(y_train.shape[0] / rate))
            y = resample(y, new_samples)
            y = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
            y = y.T
        data[i] = y
    return data


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerEncoder(nn.Module):
    """
    Encoder-only transformer + CLS token for binary classification.

    Expects inputs shaped [B, F, T] (same as your original model),
    with F == num_rois.
    """
    def __init__(self,
                 num_rois,
                 nhead,
                 num_encoder_layers,
                 dim_feedforward,
                 dropout,
                 max_timepoints):
        super().__init__()
        self.num_rois = num_rois

        # This matches your original encoder config:
        encoder_layer = nn.TransformerEncoderLayer(d_model=num_rois,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation='gelu',
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.pos_encoder = PositionalEncoding(num_rois, max_len=max_timepoints)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, F, T]
        returns:
            logits: [B] (raw logits for BCEWithLogitsLoss)
        """
        B, F, T = x.shape
        assert F == self.num_rois

        # [B, F, T] -> [B, T, F]
        x_seq = x.permute(0, 2, 1).contiguous()  # [B, T, F]

        x_seq = self.pos_encoder(x_seq)  # [B, T, F]

        # Add positional encoding
        x_seq = self.pos_encoder(x_seq)  # [B, T, F]

        # Encode
        encoded = self.encoder(x_seq)    # [B, T, F]
        return encoded.view(B,-1)


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
    
    for method in config['methods']:   
        result = []
        for i in range(10):
            path_pretrained = os.path.join(config['path_save'],'models', 'MSM', 'model_' + method + '.pth') 
            state_dict = torch.load(path_pretrained)  
            model_config = config['model_config']
            encoder_model = TransformerEncoder(num_rois=config['R'],
                                         nhead=model_config['nheads'], #4
                                         num_encoder_layers=model_config['n_enc_layers'],
                                         dim_feedforward=4 * config['R'],
                                         dropout=model_config['dropout'],
                                         max_timepoints=config['T']).to(device)
            encoder_model.load_state_dict(state_dict, strict=False)
        
            res,_ = test(encoder_model, train_data, val_data, test_data, gender_train, gender_val, gender_test, batch_size, device)
            result.append(res)
        results = {}
        results['result'] = result
        
        if save_results:
            if not os.path.exists(os.path.join(config['path_save'],'Gender','MSM')):
                os.makedirs(os.path.join(config['path_save'],'Gender','MSM'), exist_ok=True)   
            with open(os.path.join(config['path_save'],'Gender','MSM','results_' + method + '.pkl'), 'wb') as f:
                pickle.dump(results,f)


if __name__ == '__main__':
    config = {}
    config['path_data'] = r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_GNN_data/HCP_2026'
    config['path_save'] = r'/home/student1/Desktop/Charalampos_Lamprou/rDCM-GEN_results_HCP_2026'
    config['methods'] = ['CRVAE', 'TimeVAE', 'TimeGAN','T2IDIFF', 'DSFM','rDCM-GEN']
    config['atlas'] = 'SCH' #AICHA, AAL
    config['case'] = "full_original"
    config['TR'] = 0.72
    config['newTR'] = 2.0
    if config['case'] == "full_original":
        config['T'] = 833
    config['R'] = 100
    config['batch_size'] = 128
    config['shuffle'] = True
    config['model_config'] = {}
    config['model_config']['n_enc_layers'] = 5
    config['model_config']['nheads'] = 4
    config['model_config']['dropout'] = 0.1
    config['device'] = "cuda:1"
    config['save_results'] = True
    main(config)
