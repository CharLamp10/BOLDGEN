import numpy as np
import torch
import os
import pickle
import pandas as pd
import math
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
import copy


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerCLSEncoder(nn.Module):
    def __init__(self,
                 num_rois,
                 d_model,
                 nhead,
                 num_encoder_layers,
                 dim_feedforward,
                 dropout,
                 max_timepoints):
        super().__init__()
        self.num_rois = num_rois
        self.d_model = d_model

        # same as original
        self.input_proj = nn.Linear(num_rois, d_model)

        # encoder works in num_rois space (as in your original model)
        self.pos_encoder = PositionalEncoding(num_rois, max_len=max_timepoints)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_rois,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # NEW: CLS token (only new trainable piece vs original)
        self.cls_token = nn.Parameter(torch.randn(1, 1, num_rois) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, num_rois]
        returns: CLS representation [B, num_rois]
        """
        B, T, _ = x.shape

        cls_tokens = self.cls_token.expand(B, 1, -1)    # [B, 1, num_rois]
        x = self.pos_encoder(x)                         # [B, 1+T, num_rois]
        x = torch.cat([cls_tokens, x], dim=1)           # [B, 1+T, num_rois]
        enc = self.encoder(x)                           # [B, 1+T, num_rois]

        cls_rep = enc[:, 0, :]                          # [B, num_rois]
        return cls_rep


def load_pretrained_encoder_with_cls(new_model, ckpt_path):
    state_dict = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = new_model.load_state_dict(state_dict, strict=False)
    print("Missing keys (e.g. cls_token):", missing)
    print("Unexpected keys (e.g. decoder, mask_token):", unexpected)



def test(encoder_model,
         classifier,
         optimizer,
         criterion,
         train_data,
         val_data,
         test_data,
         gender_train,
         gender_val,
         gender_test,
         batch_size,
         device):

    # ----- prepare data as tensors -----
    # assume train_data/val_data/test_data are lists/arrays of [T, num_rois]
    train_data = torch.from_numpy(np.stack(train_data)).float()
    val_data   = torch.from_numpy(np.stack(val_data)).float()
    test_data  = torch.from_numpy(np.stack(test_data)).float()

    # labels -> tensors on device
    gender_train = torch.as_tensor(gender_train, dtype=torch.long, device=device)
    gender_val   = torch.as_tensor(gender_val,   dtype=torch.long, device=device)
    gender_test  = torch.as_tensor(gender_test,  dtype=torch.long, device=device)

    train_data = train_data.to(device)
    val_data   = val_data.to(device)
    test_data  = test_data.to(device)

    num_classes = 2

    # ----- tracking -----
    train_losses, val_losses, test_losses = [], [], []
    train_aucs,   val_aucs,   test_aucs   = [], [], []
    train_f1s,    val_f1s,    test_f1s    = [], [], []

    best_val_loss = float("inf")
    best_train_loss = best_test_loss = None
    best_train_auc = best_val_auc = best_test_auc = None
    best_train_f1 = best_val_f1 = best_test_f1 = None
    linear_state_dict = copy.deepcopy(classifier.state_dict())

    # ----- helper: run model over a dataset in batches -----
    def run_epoch(data_tensor):
        """Return logits for the whole dataset in batches."""
        all_logits = []
        for i in range(0, data_tensor.size(0), batch_size):
            x_batch = data_tensor[i:i + batch_size]
            z = encoder_model(x_batch)
            logits = classifier(z)
            all_logits.append(logits)
        return torch.cat(all_logits, dim=0)  # [N, num_classes]

    # ----- training loop -----
    for epoch in range(400):
        encoder_model.train()
        classifier.train()

        # training step (mini-batches)
        for i in range(0, train_data.size(0), batch_size):
            x_batch = train_data[i:i + batch_size]
            y_batch = gender_train[i:i + batch_size]

            optimizer.zero_grad()
            z = encoder_model(x_batch)
            logits = classifier(z)  # [B, num_classes]

            loss = criterion(
                logits,
                F.one_hot(y_batch, num_classes=num_classes).float()
            )
            loss.backward()
            optimizer.step()

        # ----- evaluation -----
        encoder_model.eval()
        classifier.eval()

        with torch.no_grad():
            # get logits for entire train/val/test sets (batched)
            output_train = run_epoch(train_data)  # [N_train, num_classes]
            output_val   = run_epoch(val_data)    # [N_val, num_classes]
            output_test  = run_epoch(test_data)   # [N_test, num_classes]

            # losses
            class_loss_train = criterion(
                output_train,
                F.one_hot(gender_train, num_classes=num_classes).float()
            ).item()

            class_loss_val = criterion(
                output_val,
                F.one_hot(gender_val, num_classes=num_classes).float()
            ).item()

            class_loss_test = criterion(
                output_test,
                F.one_hot(gender_test, num_classes=num_classes).float()
            ).item()

            # probabilities for positive class
            # (assuming logits as raw scores; adjust if you already use sigmoid/softmax)
            train_prob = output_train.softmax(dim=-1)[:, -1].cpu().numpy()
            val_prob   = output_val.softmax(dim=-1)[:, -1].cpu().numpy()
            test_prob  = output_test.softmax(dim=-1)[:, -1].cpu().numpy()

            y_train_np = gender_train.cpu().numpy()
            y_val_np   = gender_val.cpu().numpy()
            y_test_np  = gender_test.cpu().numpy()

            train_auc = roc_auc_score(y_train_np, train_prob)
            val_auc   = roc_auc_score(y_val_np,   val_prob)
            test_auc  = roc_auc_score(y_test_np,  test_prob)

            # --- F1 with Youden's J threshold for each split ---

            # train threshold
            fpr, tpr, thresholds = roc_curve(y_train_np, train_prob)
            youden_j = tpr - fpr
            best_index = np.argmax(youden_j)
            best_threshold_train = thresholds[best_index]
            train_pred = (train_prob >= best_threshold_train).astype(int)
            train_f1 = f1_score(y_train_np, train_pred)

            # val threshold
            fpr, tpr, thresholds = roc_curve(y_val_np, val_prob)
            youden_j = tpr - fpr
            best_index = np.argmax(youden_j)
            best_threshold_val = thresholds[best_index]
            val_pred = (val_prob >= best_threshold_val).astype(int)
            val_f1 = f1_score(y_val_np, val_pred)

            # test threshold
            fpr, tpr, thresholds = roc_curve(y_test_np, test_prob)
            youden_j = tpr - fpr
            best_index = np.argmax(youden_j)
            best_threshold_test = thresholds[best_index]
            test_pred = (test_prob >= best_threshold_test).astype(int)
            test_f1 = f1_score(y_test_np, test_pred)

            # log histories
            train_losses.append(class_loss_train)
            val_losses.append(class_loss_val)
            test_losses.append(class_loss_test)

            train_aucs.append(train_auc)
            val_aucs.append(val_auc)
            test_aucs.append(test_auc)

            train_f1s.append(train_f1)
            val_f1s.append(val_f1)
            test_f1s.append(test_f1)

            # track best model by validation loss
            if class_loss_val < best_val_loss:
                best_val_loss = class_loss_val
                best_train_loss = class_loss_train
                best_test_loss = class_loss_test

                best_train_auc = train_auc
                best_val_auc = val_auc
                best_test_auc = test_auc

                best_train_f1 = train_f1
                best_val_f1 = val_f1
                best_test_f1 = test_f1
                
                encoder_state_dict = copy.deepcopy(encoder_model.state_dict())
                linear_state_dict = copy.deepcopy(classifier.state_dict())
    print('')
    print(f'train_auc: {best_train_auc:.2f} | train_loss: {best_train_loss:.2f}')
    print(f'val_auc  : {best_val_auc:.2f} | val_loss  : {best_val_loss:.3f}')
    print(f'test_auc : {best_test_auc:.2f} | test_loss : {best_test_loss:.2f}')
    # you can return whatever you need; example:
    return {
        'best_train_auc': best_train_auc,
        'best_val_auc': best_val_auc,
        'best_test_auc': best_test_auc,
        'best_train_loss': best_train_loss,
        'best_val_loss': best_val_loss,
        'best_test_loss': best_test_loss,
        'best_train_f1': best_train_f1,
        'best_val_f1': best_val_f1,
        'best_test_f1': best_test_f1,
        'y_train': y_train_np,
        'y_val': y_val_np,
        'y_test': y_test_np,
    }, encoder_state_dict, linear_state_dict



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
    
    data = np.load(os.path.join(path,'train_data_HCP_' + config['atlas'] + '_resampled.npz'))
    train_data = []
    for key in data:
        temp = data[key][88:-88]
        temp = (temp - np.mean(temp, axis=0, keepdims=True))/np.std(temp, axis=0, keepdims=True)
        train_data.append(temp)
    
    data = np.load(os.path.join(path,'val_data_HCP_' + config['atlas'] + '_1_resampled.npz'))
    val_data1 = []
    for key in data:
        temp = data[key][88:-88]
        temp = (temp - np.mean(temp, axis=0, keepdims=True))/np.std(temp, axis=0, keepdims=True)
        val_data1.append(temp)
    
    data = np.load(os.path.join(path,'val_data_HCP_' + config['atlas'] + '_2_resampled.npz'))
    val_data2 = []
    for key in data:
        temp = data[key][88:-88]
        temp = (temp - np.mean(temp, axis=0, keepdims=True))/np.std(temp, axis=0, keepdims=True)
        val_data2.append(temp)
        
    data = np.load(os.path.join(path,'test_data_HCP_' + config['atlas'] + '_1_resampled.npz'))
    test_data1 = []
    for key in data:
        temp = data[key][88:-88]
        temp = (temp - np.mean(temp, axis=0, keepdims=True))/np.std(temp, axis=0, keepdims=True)
        test_data1.append(temp)
    
    data = np.load(os.path.join(path,'test_data_HCP_' + config['atlas'] + '_2_resampled.npz'))
    test_data2 = []
    for key in data:
        temp = data[key][88:-88]
        temp = (temp - np.mean(temp, axis=0, keepdims=True))/np.std(temp, axis=0, keepdims=True)
        test_data2.append(temp)
    

    T,R = test_data1[0].shape
    
    device = torch.device(config['device']) if torch.cuda.is_available() else torch.device("cpu")
    batch_size = config['batch_size']
    save_results = config['save_results']
    
    for method in config['methods']:            
        path_pretrained = os.path.join(config['path_save'],'models', 'MSM', 'model_' + method + '.pth') 
        model_config = config['model_config']
        encoder_model = TransformerCLSEncoder(num_rois=R,
                                     d_model=R,
                                     nhead=model_config['nheads'], #4
                                     num_encoder_layers=model_config['n_enc_layers'],
                                     dim_feedforward=4 * R,
                                     dropout=model_config['dropout'],
                                     max_timepoints=T).to(device)
        load_pretrained_encoder_with_cls(encoder_model, path_pretrained)
        
        for name, p in encoder_model.named_parameters():
            p.requires_grad = (name == "cls_token")

        # binary head on top of CLS
        classifier = nn.Sequential(nn.Linear(R, 2), nn.Softmax(dim=-1)).to(device)
        params_to_optimize = [p for p in encoder_model.parameters() if p.requires_grad] + \
                             list(classifier.parameters())

        optimizer = torch.optim.Adam(params_to_optimize, lr=1e-4)
        criterion = nn.BCELoss()
        
        result,_,_ = test(encoder_model, classifier, optimizer, criterion, train_data, val_data1+val_data2, test_data1+test_data2, gender_train, gender_val+gender_val, gender_test+gender_test, batch_size, device)
                        
        results = {}
        results['result'] = result
        
        if save_results:
            if not os.path.exists(os.path.join(config['path_save'],'Gender','MSM')):
                os.makedirs(os.path.join(config['path_save'],'Gender','MSM'), exist_ok=True)   
            with open(os.path.join(config['path_save'],'Gender','MSM','results_' + method + '.pkl'), 'wb') as f:
                pickle.dump(results,f)


if __name__ == '__main__':
    config = {}
    config['path_data'] = r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_GNN_data/HCP'
    config['path_save'] = r'/home/student1/Desktop/Charalampos_Lamprou/rDCM-GEN_results'
    config['methods'] = ['TimeGAN', 'TimeVAE','CRVAE','Noise_6dB', 'Noise_3dB', 'rDCM']
    config['atlas'] = 'SCH' #AICHA, AAL
    config['batch_size'] = 128
    config['shuffle'] = True
    config['model_config'] = {}
    config['model_config']['n_enc_layers'] = 3
    config['model_config']['n_dec_layers'] = 3
    config['model_config']['nheads'] = 4
    config['model_config']['dropout'] = 0.1
    config['device'] = "cuda:1"
    config['save_results'] = True
    main(config)
