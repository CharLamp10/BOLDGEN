import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from tqdm import tqdm
from torch.optim import Adam
from utils import InfoNCE, MaskedInstanceNorm1d, DualBranchContrast, BatchSampler
import copy
import os
import pickle
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


class Transformer(nn.Module):
    def __init__(
        self,
        R,
        D,
        out_dim,
        num_layers,
        num_heads,
        dim_feedforward
    ):
        super().__init__()
        self.R = R
        self.D = D
        
        self.in_norm = MaskedInstanceNorm1d()
        
        # 1. Linear projection R -> D
        self.input_proj = nn.Linear(R, D)

        # 2. Learned mask token
        self.mask_token = nn.Parameter(torch.randn(D))

        # 3. Learned CLS token
        self.cls_token = nn.Parameter(torch.randn(D))

        # 4. Learned positional embedding for sequence of length (T+1)
        #    (we will dynamically expand to batch size)
        self.pos_emb = nn.Parameter(torch.randn(1, 1, D))  

        # 5. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(D,out_dim)


    def forward(self, x):
        """
        x: (B, T, R)
        returns:
            recon: (B, T, R)
            mask: boolean mask of masked positions
            cls_logits: (B, num_classes)
        """
        x = x.float()
        B, T, R = x.shape
        
        mask = (x[:,:,0] != 0)
        x = self.in_norm(x, mask)
        
        x_mask = (x[:, :, 0] == 0).bool()

        # --- Step 1: project to embeddings ---
        x = self.input_proj(x)   # (B, T, D)

        # --- Step 3: prepend CLS token ---
        cls_tok = self.cls_token.unsqueeze(0).unsqueeze(1)   # (1, 1, D)
        cls_tok = cls_tok.repeat(B, 1, 1)                    # (B, 1, D)

        x_seq = torch.cat([cls_tok, x], dim=1)        # (B, T+1, D)
        cls_mask = torch.zeros(x.shape[0], 1, dtype=torch.bool, device=x.device)
        x_mask = torch.cat([cls_mask, x_mask], dim=1)

        # --- Step 4: add positional embeddings ---
        x_seq = x_seq + self.pos_emb
        x_seq[x_mask] = 0

        # --- Step 5: transformer ---
        z = self.transformer(x_seq,src_key_padding_mask=x_mask) # (B, T+1, D)

        # --- Step 6: extract CLS output for classification ---
        cls_out = z[:, 0, :]                                 # (B, D)
        cls_out = self.out_proj(cls_out)

        return cls_out
    

def train(x, encoder_model, contrast_model, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z1 = encoder_model(x[0])
    z2 = encoder_model(x[1])
    loss = contrast_model(z1, z2)
    loss.backward()
    optimizer.step()
    return loss.item()


def fingerprinting(encoder_model,test_data1,test_data2,batch_size,device):
    length = len(test_data1)
    N,T,R = test_data1[0].shape
    test_data1 = torch.cat(test_data1, dim=0)
    test_data2 = torch.cat(test_data2, dim=0)
    encoder_model.eval()
    with torch.no_grad():
        outputs1 = []
        for i in range(0, test_data1.size(0), batch_size):
            data = test_data1[i:i+batch_size].to(device)
            z1 = encoder_model(data)
            outputs1.append(z1)
        outputs1 = torch.cat(outputs1, dim=0)
        outputs1 = outputs1.view(length, N, -1)
        
        outputs2 = []
        for i in range(0, test_data2.size(0), batch_size):
            data = test_data2[i:i+batch_size].to(device)
            z2 = encoder_model(data)
            outputs2.append(z2)
        outputs2 = torch.cat(outputs2, dim=0)
        outputs2 = outputs2.view(length, N, -1)
        
        mean_accs = []
        std_accs= []
        corr_coeffs_all = []
        mean_metrics = []
        num_real = int(outputs1.shape[1])
        print('')
        accuracies = []
        metrics = []
        for j in range(num_real):
            corr_coeffs = torch.corrcoef(torch.cat([outputs1[:, j, :],outputs2[:, j, :]],dim=0))[0:outputs1.shape[0],outputs1.shape[0]:]
            corr_coeffs_all.append(corr_coeffs.cpu().numpy())
            lower_indices = torch.tril_indices(corr_coeffs.shape[0],corr_coeffs.shape[1], offset=-1)
            upper_indices = torch.triu_indices(corr_coeffs.shape[0],corr_coeffs.shape[1], offset=1)
            inter = (torch.mean(corr_coeffs[lower_indices[0], lower_indices[1]]) + torch.mean(corr_coeffs[upper_indices[0], upper_indices[1]]))/2
            intra = torch.mean(torch.diag(corr_coeffs))
            metric = (intra/inter).item()
            metrics.append(metric)
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

            total_samples = outputs1.shape[0] + outputs2.shape[0]
            accuracies.append((counter1 + counter2) / total_samples)
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        mean_metric = np.mean(metrics)
        print(f'meanAcc: {mean_acc:.2f}, meanMetric: {mean_metric:.3f}')
        mean_accs.append(mean_acc)
        std_accs.append(std_acc)
        mean_metrics.append(mean_metric)
        print('')
    return mean_metrics, mean_accs, std_accs, corr_coeffs_all


def main(config):
    path = config['path_data']

    names = []
    with open(os.path.join(path,'names_train.txt'), 'r') as f:
        for line in f:
            temp = line.strip()
            names.append(temp)
            
    data = np.load(os.path.join(path,'train_data_HCP_' + config['atlas'] + '.npz'))
    train_data = []
    names_train = []
    for (key,name) in zip(data,names):
        if data[key].shape[0] == 1200:
            temp = data[key][183:-184]
            train_data.append(temp)
            names_train.append(name)
    
    
    names = []
    with open(os.path.join(path,'names_val.txt'), 'r') as f:
        for line in f:
            temp = line.strip()
            names.append(temp)
    
    data = np.load(os.path.join(path,'val_data_HCP_' + config['atlas'] + '.npz'))
    val_data1 = []
    names_temp = []
    for (key, name) in zip(data, names):
        if 'LR' in name and 'REST1' in name and data[key].shape[0] == 1200:
            temp = data[key][183:-184]
            val_data1.append(temp)
            names_temp.append(name[:6])
    
    data = np.load(os.path.join(path,'val_data_HCP_' + config['atlas'] + '.npz'))
    val_data2 = []
    names_val = []
    for (key, name) in zip(data, names):
        if 'LR' in name and 'REST2' in name and name[:6] in names_temp and data[key].shape[0] == 1200:
            temp = data[key][183:-184]
            val_data2.append(temp)
            names_val.append(name[:6])
            
    remove = []
    for (i,name) in enumerate(names_temp):
        if name not in names_val:
            remove.append(i)
    remove = remove[::-1]
    
    for index in remove:
        val_data1.pop(index)
            
    names = []
    with open(os.path.join(path,'names_test.txt'), 'r') as f:
        for line in f:
            temp = line.strip()
            names.append(temp)
        
    data = np.load(os.path.join(path,'test_data_HCP_' + config['atlas'] + '.npz'))
    test_data1 = []
    names_temp = []
    for (key, name) in zip(data, names):
        if 'LR' in name and 'REST1' in name and data[key].shape[0] == 1200:
            temp = data[key][183:-184]
            test_data1.append(temp)
            names_temp.append(name[:6])
    
    data = np.load(os.path.join(path,'test_data_HCP_' + config['atlas'] + '.npz'))
    test_data2 = []
    names_test = []
    for (key, name) in zip(data, names):
        if 'LR' in name and 'REST2' in name and name[:6] in names_temp and data[key].shape[0] == 1200:
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
            
    train_data = preprocess_HCP(train_data, config['case'])
    val_data1 = preprocess_HCP(val_data1, config['case'])
    val_data2 = preprocess_HCP(val_data2, config['case'])
    test_data1 = preprocess_HCP(test_data1, config['case'])
    test_data2 = preprocess_HCP(test_data2, config['case'])
        
    names_rdcm = []
    with open(os.path.join(config['path_augmented_rdcm'],'names_rDCM-GEN_' + config['case'] + '.txt'), 'r') as f:
        for line in f:
            temp = line.strip()
            names_rdcm.append(temp)
            
    names_crvae = []
    with open(os.path.join(config['path_augmented_crvae'],'names_CR-VAE_' + config['case'] + '.txt'), 'r') as f:
        for line in f:
            temp = line.strip()
            names_crvae.append(temp)
            
    augmented_rdcm = torch.load(os.path.join(config['path_augmented_rdcm'], 'augmented_rDCM-GEN_' + config['case'] + '.pt'))
    augmented_crvae = torch.load(os.path.join(config['path_augmented_crvae'], 'augmented_CR-VAE_' + config['case'] + '.pt'))


    T,R = test_data1[0].shape

    device = torch.device(config['device']) if torch.cuda.is_available() else torch.device("cpu")
    batch_size = config['batch_size']
    shuffle = config['shuffle']
    epochs = config['epochs']
    lr = config['lr']
    model_config = config['model_config']
    save_results = config['save_results']
    save_models = config['save_models']
    
    for i,data in enumerate(val_data1):
        data = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)
        val_data1[i] = data
        
    for i,data in enumerate(val_data2):
        data = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)
        val_data2[i] = data
        
    for i,data in enumerate(test_data1):
        data = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)
        test_data1[i] = data
        
    for i,data in enumerate(test_data2):
        data = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)
        test_data2[i] = data
    
    for tau in config['taus']:
        for method in config['methods']:
            if method == 'rDCM-GEN':
                batch_sampler = BatchSampler(names_rdcm.copy(), batch_size=batch_size,shuffle=shuffle,drop_last=True)
            elif method == 'CRVAE':
                batch_sampler = BatchSampler(names_crvae.copy(), batch_size=batch_size,shuffle=shuffle,drop_last=True)
            else:
                batch_sampler = BatchSampler(names_train.copy(), batch_size=batch_size,shuffle=shuffle,drop_last=True)
            train_loader = DataLoader(train_data, batch_sampler=batch_sampler) 

            model = Transformer(R,
                                model_config['D'],
                                model_config['out_dim'],
                                model_config['num_layers'],
                                model_config['num_heads'],
                                model_config['dim_feedforward']).to(device)
            contrast_model = DualBranchContrast(loss=InfoNCE(tau=tau),mode='L2L').to(device)   
            optimizer = Adam(model.parameters(), lr=lr)
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer=optimizer,
                warmup_start_lr = 1e-5,
                warmup_epochs=config['warm_up_epochs'],
                max_epochs=epochs)
            
            max_val_acc = 0
            losses = []
            res_all = []
            with tqdm(total=epochs, desc='(T)') as pbar:
                for epoch in range(1,epochs+1):
                    total_loss = 0.0
                    batch_count = 0                              
                    for batch_idx, sample_inds in enumerate(train_loader.batch_sampler):
                        if 'rDCM-GEN' in method:
                            batch_data_rdcm = augmented_rdcm[sample_inds]
                            idx = torch.randint(0, 8, (len(sample_inds),))
                            batch_data1 = batch_data_rdcm[torch.arange(len(sample_inds)), idx]
                            idx = torch.randint(8, 15, (len(sample_inds),))
                            batch_data2 = batch_data_rdcm[torch.arange(len(sample_inds)), idx]
                            batch_data = [batch_data1.to(device), batch_data2.to(device)]
                        elif 'CRVAE' in method:
                            batch_data_crvae = augmented_crvae[sample_inds]
                            idx = torch.randint(0, 8, (len(sample_inds),))
                            batch_data1 = batch_data_crvae[torch.arange(len(sample_inds)), idx]
                            idx = torch.randint(8, 15, (len(sample_inds),))
                            batch_data2 = batch_data_crvae[torch.arange(len(sample_inds)), idx]
                            batch_data = [batch_data1.to(device), batch_data2.to(device)]
                        loss = train(batch_data,model,contrast_model,optimizer)
                        total_loss += loss
                        batch_count += 1
                    scheduler.step()
                    average_loss = total_loss / batch_count if batch_count > 0 else float('nan')
                    losses.append(average_loss)
                    pbar.set_postfix({'loss': average_loss})
                    pbar.update()        
                    
                    res = fingerprinting(model,
                                         val_data1,
                                         val_data2,
                                         batch_size,
                                         device)
                    res_all.append(res)
                    if np.mean(res[0]) + np.min(res[0]) > max_val_acc:
                        max_val_acc = np.mean(res[0]) + np.min(res[0])
                        max_val_acc_model = copy.deepcopy(model.state_dict())
                    print('')
                    
            model = Transformer(R,
                                model_config['D'],
                                model_config['out_dim'],
                                model_config['num_layers'],
                                model_config['num_heads'],
                                model_config['dim_feedforward']).to(device)
            model.load_state_dict(max_val_acc_model)
            val_result = fingerprinting(model,
                                        val_data1,
                                        val_data2,
                                        batch_size,
                                        device)
            test_result = fingerprinting(model,
                                        test_data1,
                                        test_data2,
                                        batch_size,
                                        device)
            
            results = {}
            results['losses'] = losses
            results['val_result'] = val_result
            results['test_result'] = test_result
            results['val_results_all'] = res_all
            
            if save_models:
                if not os.path.exists(os.path.join(config['path_save'],'models','SimCLR', config['case'])):
                    os.makedirs(os.path.join(config['path_save'],'models','SimCLR', config['case']),exist_ok=True)   
                torch.save(max_val_acc_model, os.path.join(config['path_save'],'models','SimCLR', config['case'],'model_' + method + '_tau' + str(tau) + '.pth'))
            
            if save_results:
                if not os.path.exists(os.path.join(config['path_save'],'SimCLR', config['case'])):
                    os.makedirs(os.path.join(config['path_save'],'SimCLR', config['case']),exist_ok=True)   
                with open(os.path.join(config['path_save'],'SimCLR', config['case'],'results_' + method + '_tau' + str(tau) + '.pkl'), 'wb') as f:
                    pickle.dump(results,f)


if __name__ == '__main__':
    config = {}
    config['path_data'] = r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_GNN_data/HCP_2026'
    config['path_save'] = r'/home/student1/Desktop/Charalampos_Lamprou/rDCM-GEN_results_HCP_2026'
    config['path_augmented_rdcm'] = r'/media/student1/ExtremePro/rDCM-GEN/rDCM-GEN_results/augmented_rDCM-GEN_noFCloss'
    config['path_augmented_crvae'] = r'/media/student1/ExtremePro/CRVAE_augmented'
    config['methods'] = ['CRVAE','rDCM-GEN']  
    config['atlas'] = 'SCH'
    config['case'] = "full_subsampled"
    config['TR'] = 0.72
    config['newTR'] = 2.0
    config['batch_size'] = 128
    config['shuffle'] = True
    config['epochs'] = 200
    config['taus'] = [0.05, 0.1, 0.15]
    config['lr'] = 1e-4
    config['warm_up_epochs'] = 10
    config['eval_epochs'] = list(range(1,config['epochs']+1))
    config['model_config'] = {}
    config['model_config']['num_layers'] = 4
    config['model_config']['num_heads'] = 4
    config['model_config']['dim_feedforward'] = 1024
    config['model_config']['out_dim'] = 4096
    config['model_config']['D'] = 512
    config['device'] = "cuda:0"
    config['save_results'] = True
    config['save_models'] = True
    main(config)


