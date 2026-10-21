parent_path = r'/home/student1/Desktop/Charalampos_Lamprou/VarCoNet_extTest_extras'
import sys
sys.path.append(parent_path)
from BrainMass3.ddp_byol import train 
import torch
import numpy as np
import os
import pickle
from torch.utils.data import Dataset, DataLoader
from nilearn.connectome import ConnectivityMeasure
from BrainMass3.network_dataset import mask_timeseries, mask_timeseries_per, random_timeseries
from sklearn.model_selection import train_test_split
from classifier import LREvaluator
from BrainMass3.byol_trainer import BYOLTrainer



def forward_pass(encoder_model, loader, device):
    encoder_model.eval()
    with torch.no_grad():
        outputs = []
        y = []
        for data,label in loader:
            y.append(label)
            data = data.to(device)
            outputs.append(encoder_model.target_network(data))
        outputs = torch.cat(outputs, dim=0).clone().detach()
        y = torch.cat(y,dim=0).to(device)
    return outputs,y


class BrainMassDataset(Dataset):

    def __init__(self, signals = None, mask_way='mask',mask_len=10, time_len=30):
        self.signals = signals
        self.mask_way = mask_way
        self.mask_len = mask_len
        self.time_len = time_len
        self.correlation_measure = ConnectivityMeasure(kind='correlation')

    def __getitem__(self,index):
        img = self.signals[index].T
        if self.mask_way == 'mask':
            slices = [mask_timeseries(img,mask=self.mask_len).T, mask_timeseries(img,mask=self.mask_len).T]
        elif self.mask_way == 'mask_per':
            slices = [mask_timeseries_per(img,mask=self.mask_len).T, mask_timeseries_per(img,mask=self.mask_len).T]
        elif self.mask_way == 'random':
            slices = [random_timeseries(img,sample_len=self.time_len).T, random_timeseries(img,sample_len=self.time_len).T]
        else:
            raise KeyError(f"mask way error, your input is {self.mask_way}")
        correlation_matrix = self.correlation_measure.fit_transform(slices)
        correlation_matrix[correlation_matrix!=correlation_matrix]=0
        return correlation_matrix[0], correlation_matrix[1]

    def __len__(self):
        return len(self.signals)
    
    
class BrainMassDataset2(Dataset):

    def __init__(self, signals = None, labels = None):
        self.signals = signals
        self.labels = labels
        self.correlation_measure = ConnectivityMeasure(kind='correlation')

    def __getitem__(self,index):
        img = self.signals[index]
        label = self.labels[index]
        label = torch.squeeze(torch.tensor(label, dtype=torch.long))
        correlation_matrix = self.correlation_measure.fit_transform(np.expand_dims(img,axis=0))
        correlation_matrix[correlation_matrix!=correlation_matrix]=0
        return torch.from_numpy(correlation_matrix[0]).float(), label

    def __len__(self):
        return len(self.signals)


def main(config):
    device = torch.device(config['device']) if torch.cuda.is_available() else torch.device("cpu")
    temp1 = np.load(os.path.join(config['path_data'], 'ABIDEI_ASD_nilearn.npy'))
    temp0 = np.load(os.path.join(config['path_data'], 'ABIDEI_NC_nilearn.npy'))
    labels1 = np.ones((temp1.shape[0],1))
    labels0 = np.zeros((temp0.shape[0],1))
    train_data_abidei = np.concatenate((temp1, temp0), axis=0)
    labels = np.concatenate((labels1, labels0), axis=0)
    for method in config['methods']:
        if method == "ABIDEI":
            temp1 = np.load(os.path.join(config['path_data'], 'ABIDEI_ASD_nilearn.npy'))
            temp0 = np.load(os.path.join(config['path_data'], 'ABIDEI_NC_nilearn.npy'))
            train_data = np.concatenate((temp1, temp0), axis=0)
        elif method == "CRVAE":
            train_data = torch.load(os.path.join(config['path_augmented_crvae'], 'augmented_CR-VAE_ABIDEI.pt'))
            B, N, T, R = train_data.shape
            train_data = train_data.reshape(B * N, T, R)
            train_data = train_data.numpy()
        elif method == "rDCM-GEN":
            train_data = torch.load(os.path.join(config['path_augmented_rdcmgen'], 'augmented_rDCM-GEN_ABIDEI.pt'))
            B, N, T, R = train_data.shape
            train_data = train_data.reshape(B * N, T, R)
            train_data = train_data.numpy()
        elif method == "TimeVAE":
            train_data_asd = np.load(os.path.join(config['path_augmented_timevae'], 'ABIDEI_ASD', 'timeVAE_ABIDEI_ASD_prior_samples.npy'))
            train_data_nc = np.load(os.path.join(config['path_augmented_timevae'], 'ABIDEI_NC', 'timeVAE_ABIDEI_NC_prior_samples.npy'))
            train_data = np.concatenate((train_data_asd, train_data_nc), axis=0)
            train_data = torch.from_numpy(train_data)
            train_data = (train_data - train_data.mean(dim=1, keepdim=True))/train_data.std(dim=1, keepdim=True)
            train_data = train_data.numpy()
        elif method == "TimeGAN":
            train_data_asd = np.load(os.path.join(config['path_augmented_timegan'], 'augmented_TimeGAN_ABIDEI_ASD.npy'))
            train_data_nc = np.load(os.path.join(config['path_augmented_timegan'], 'augmented_TimeGAN_ABIDEI_NC.npy'))
            train_data = np.concatenate((train_data_asd, train_data_nc), axis=0)
            train_data = torch.from_numpy(train_data)
            train_data = (train_data - train_data.mean(dim=1, keepdim=True))/train_data.std(dim=1, keepdim=True)
            train_data = train_data.numpy()
        elif method == "T2IDIFF":
            train_data = np.load(os.path.join(config['path_augmented_t2idiff'], 'ABIDEI', 'gen_sig_ABIDEI_fold1.npy'))
            train_data = torch.from_numpy(train_data)
            train_data = (train_data - train_data.mean(dim=1, keepdim=True))/train_data.std(dim=1, keepdim=True)
            train_data = train_data.numpy()
        elif method == "DSFM":
            train_data_asd = np.load(os.path.join(config['path_augmented_dsfm'], 'ABIDEI', 'samples', 'asd_bold.npy'))
            train_data_nc = np.load(os.path.join(config['path_augmented_dsfm'], 'ABIDEI', 'samples', 'control_bold.npy'))
            train_data = np.concatenate((train_data_asd, train_data_nc), axis=0)
            train_data = torch.from_numpy(train_data)
            train_data = (train_data - train_data.mean(dim=1, keepdim=True))/train_data.std(dim=1, keepdim=True)
            train_data = train_data.numpy()

        config['roi_num'] = train_data[0].shape[-1]    
        time_len = int(config['time_len'])
        mask_len = int(config['time_mask'])
        mask_way = str(config['mask_way'])
        dataset = BrainMassDataset(train_data, mask_way=mask_way,mask_len=mask_len, time_len=time_len)
        final_epoch_state_dict,best_epoch_state_dict, losses = train(config,
                                                                     dataset)
        
        model = BYOLTrainer(config['roi_num'],
                            config['depth'],
                            config['heads'],
                            config['m'],
                            config['feature_dim'],
                            config['dim_feedforward'],
                            mm=config['mm'],
                            clf_mask = config['clf_mask'],
                            mse_mask = config['mse_mask']).to(device)
        model.target_network.load_state_dict(final_epoch_state_dict)
        dataset = BrainMassDataset2(train_data_abidei, labels)
        loader = DataLoader(dataset, batch_size=config['batch_size'])
        train_data, y_train = forward_pass(model, loader, device)
     
        test_aucs = []
        y_trues = []
        y_preds = []
        for i in range(10):            
            test1 = np.load(os.path.join(config['path_data'], 'ABIDEII_ASD_nilearn.npy'))
            test0 = np.load(os.path.join(config['path_data'], 'ABIDEII_NC_nilearn.npy'))
            test_labels1 = np.ones((test1.shape[0],1))
            test_labels0 = np.zeros((test0.shape[0],1))
            test_data_all = np.concatenate((test1, test0), axis=0)
            test_labels_all = np.concatenate((test_labels1, test_labels0), axis=0)
            
            test_data, val_data, test_labels, val_labels = train_test_split(test_data_all, test_labels_all, test_size=0.2, shuffle=True, stratify=test_labels_all)
            
            test_dataset = BrainMassDataset2(test_data, test_labels)
            val_dataset = BrainMassDataset2(val_data, val_labels)
            test_loader = DataLoader(test_dataset, batch_size=128)
            val_loader = DataLoader(val_dataset, batch_size=128)
            
            val_data, y_val = forward_pass(model, val_loader, device)
            test_data, y_test = forward_pass(model, test_loader, device)
            
            res,_ = LREvaluator(num_epochs=config['epochs_cls'],learning_rate=config['lr_cls']).evaluate(train_data, y_train, val_data, y_val, test_data, y_test, config['num_classes'], device)
            y_trues.append(res['y_test'])
            #y_preds.append(y_pred)
            test_auc = res['best_test_auc']
            test_aucs.append(test_auc)
            print('AUC = ' + str(np.round(test_auc, 3)))
                
        print('mean AUC = ' + str(np.round(np.mean(test_aucs), 3)) + ' | std AUC = ' + str(np.round(np.std(test_aucs), 3)))
                
        results = {}
        results['auc_scores'] = test_aucs
        results['y_true'] = y_trues
        results['y_pred'] = y_preds
    
        if config['save_results']:
            os.makedirs(os.path.join(config['path_save'], method), exist_ok=True)
            with open(os.path.join(config['path_save'], method, 'results.pkl'), 'wb') as f:
                pickle.dump(results,f)

    return results


if __name__ == '__main__':
    config = {
        'path_data': r'/media/student1/ExtremePro/rDCM-GEN/data',
        'path_save': r'/home/student1/Desktop/Charalampos_Lamprou/rDCM-GEN_results_HCP_2026/ASD_CLC/BrainMass',
        'path_augmented_rdcmgen': r'/media/student1/ExtremePro/rDCM-GEN/rDCM-GEN_results/augmented_rDCM-GEN_noFCloss',
        'path_augmented_crvae': r'/media/student1/ExtremePro/CRVAE_augmented',
        'path_augmented_timevae': r'/media/student1/ExtremePro/TimeVAE_results/outputs/gen_data',
        'path_augmented_timegan': r'/media/student1/ExtremePro/TimeGAN_results',
        'path_augmented_t2idiff': r'/media/student1/ExtremePro/T2IDIFF',
        'path_augmented_dsfm': r'/media/student1/ExtremePro/DSFM_results',
        'methods': ['ABIDEI'], #['T2IDIFF','DSFM','rDCM-GEN','TimeVAE', 'TimeGAN','CRVAE','ABIDEI'],
        'atlas': 'SCH',
        'mm': 'byol+clf+mse',
        'feature_dim': 1024,
        'heads': 10,
        'depth': 16,
        'dim_feedforward': 2048,
        'clf_mask': 10,
        'mse_mask': 5,
        'm': 0.996,
        'acc_lambda': 0.1,
        'mse_lambda': 2.0,
        'mask_way': 'mask_per',
        'time_mask': 20,
        'time_len': 30,
        'lr': 0.0003,
        'weight_decay': 0.00005,
        'warmup_epochs': 10,
        'n_epochs': 100,
        'max_epochs': 100,
        'batch_size': 128,
        'resume': False,
        'epochs_cls': 1000,
        'lr_cls': 1e-3,
        'num_classes': 2,
        'device': "cuda:0",
        'save_results': True
        }
    results = main(config)
    
    
    


