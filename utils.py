parent_path = r'/home/student1/Desktop/Charalampos_Lamprou/VarCoNetV2_extras'
import sys
sys.path.append(parent_path)
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from torch_geometric.data import Dataset as Dataset_geo
from torch_geometric.data import Data as Data_geo
from torch_geometric.data import Data,InMemoryDataset
from collections import defaultdict
import numpy as np
import random
import os
from random import randint
from scipy.signal import detrend
import scipy.sparse as sp
import torch.utils.data as utils
from nilearn import connectome
from torch_geometric.utils import remove_self_loops
from sklearn.model_selection import StratifiedShuffleSplit
from rdcm import tapas_rdcm_model_specification, tapas_rdcm_set_options,tapas_rdcm_get_convolution_bm,tapas_dcm_euler_make_indices
#from reverse_dcm import tapas_rdcm_generate
import torch.fft as fft
import copy



class Loss(ABC):
    @abstractmethod
    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs) -> torch.FloatTensor:
        pass

    def __call__(self, anchor, sample, pos_mask=None, neg_mask=None, *args, **kwargs) -> torch.FloatTensor:
        loss = self.compute(anchor, sample, pos_mask, neg_mask, *args, **kwargs)
        return loss

def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()

class InfoNCE(Loss):
    def __init__(self, tau):
        super(InfoNCE, self).__init__()
        self.tau = tau

    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
        sim = _similarity(anchor, sample) / self.tau
        exp_sim = torch.exp(sim) * (pos_mask + neg_mask)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob * pos_mask
        loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
        return -loss.mean()

class Sampler(ABC):
    def __init__(self, intraview_negs=False):
        self.intraview_negs = intraview_negs

    def __call__(self, anchor, sample, *args, **kwargs):
        ret = self.sample(anchor, sample, *args, **kwargs)
        if self.intraview_negs:
            ret = self.add_intraview_negs(*ret)
        return ret

def get_sampler(mode: str, intraview_negs: bool) -> Sampler:
    if mode in {'L2L', 'G2G'}:
        return SameScaleSampler(intraview_negs=intraview_negs)
    else:
        raise RuntimeError(f'unsupported mode: {mode}')

class SameScaleSampler(Sampler):
    def __init__(self, *args, **kwargs):
        super(SameScaleSampler, self).__init__(*args, **kwargs)

    def sample(self, anchor, sample, *args, **kwargs):
        assert anchor.size(0) == sample.size(0)
        num_nodes = anchor.size(0)
        device = anchor.device
        pos_mask = torch.eye(num_nodes, dtype=torch.float32, device=device)
        neg_mask = 1. - pos_mask
        return anchor, sample, pos_mask, neg_mask

class BatchSampler(Sampler):
    def __init__(self, names, batch_size, shuffle=True, drop_last=True):
        for i,name in enumerate(names):
            pos = name.find('_')
            if pos != -1:
                name = name[:pos]
                names[i] = name
        self.names = names
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.name_to_indices = defaultdict(list)
        for idx, name in enumerate(names):
            self.name_to_indices[name].append(idx)

        self.unique_names = list(np.unique(names))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.unique_names)

        # Flatten name-grouped indices into batches
        batch = []
        name_in_batch = set()

        for name in self.unique_names:
            indices = self.name_to_indices[name]
            random.shuffle(indices)

            if name in name_in_batch:
                continue

            batch.append(indices[0])
            name_in_batch.add(name)

            if len(batch) == self.batch_size:
                yield batch
                batch = []
                name_in_batch = set()
                
        if not self.drop_last and len(batch) > 0:
            yield batch
            batch = []
            name_in_batch = set()

    def __len__(self):
        return (len(self.unique_names) + self.batch_size - 1) // self.batch_size

def add_extra_mask(pos_mask, neg_mask=None, extra_pos_mask=None, extra_neg_mask=None):
    if extra_pos_mask is not None:
        pos_mask = torch.bitwise_or(pos_mask.bool(), extra_pos_mask.bool()).float()
    if extra_neg_mask is not None:
        neg_mask = torch.bitwise_and(neg_mask.bool(), extra_neg_mask.bool()).float()
    else:
        neg_mask = 1. - pos_mask
    return pos_mask, neg_mask

    
class DualBranchContrast(torch.nn.Module):
    def __init__(self, loss: Loss, mode: str, intraview_negs: bool = False, **kwargs):
        super(DualBranchContrast, self).__init__()
        self.loss = loss
        self.mode = mode
        self.sampler = get_sampler(mode, intraview_negs=intraview_negs)
        self.kwargs = kwargs

    def forward(self, h1=None, h2=None, g1=None, g2=None, batch=None, h3=None, h4=None,
                extra_pos_mask=None, extra_neg_mask=None):
        if self.mode == 'L2L':
            assert h1 is not None and h2 is not None
            anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=h1, sample=h2)
            anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=h2, sample=h1)
        elif self.mode == 'G2G':
            assert g1 is not None and g2 is not None
            anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=g2)
            anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=g1)

        pos_mask1, neg_mask1 = add_extra_mask(pos_mask1, neg_mask1, extra_pos_mask, extra_neg_mask)
        pos_mask2, neg_mask2 = add_extra_mask(pos_mask2, neg_mask2, extra_pos_mask, extra_neg_mask)
        l1 = self.loss(anchor=anchor1, sample=sample1, pos_mask=pos_mask1, neg_mask=neg_mask1, **self.kwargs)
        l2 = self.loss(anchor=anchor2, sample=sample2, pos_mask=pos_mask2, neg_mask=neg_mask2, **self.kwargs)

        return (l1 + l2) * 0.5
    
class ABIDEDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not torch.is_tensor(self.data):
            x = torch.tensor(self.data[idx], dtype=torch.float32)
            y = torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            x = self.data[idx].clone().detach().float()
            y = self.labels[idx].clone().detach().long()
        return x, y
    
    
class ABIDEDataset_BAnD(Dataset):
    def __init__(self, data, labels, path, max_length, mode):
        self.data = data
        self.labels = labels
        self.path = path
        self.max_length = max_length
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = np.load(os.path.join(self.path,self.data[idx]))
        if self.mode == 'train':
            if x.shape[-1] < self.max_length:
                repeats = (self.max_length // x.shape[-1]) + 1
                x = np.tile(x, repeats)
                x = x[:,:,:,:self.max_length]
            else:
                start_idx = np.random.randint(0, x.shape[-1] - self.max_length + 1)
                x = x[:, :, :, start_idx:start_idx + self.max_length]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
    
def upper_triangular_cosine_similarity(x):
    N, M, D = x.shape
    x_norm = F.normalize(x, p=2, dim=-1)
    cosine_similarity = torch.matmul(x_norm, x_norm.transpose(1, 2))
    triu_indices = torch.triu_indices(M, M, offset=1)
    upper_triangular_values = cosine_similarity[:, triu_indices[0], triu_indices[1]]
    return upper_triangular_values, cosine_similarity

    
class MaskedInstanceNorm1d(torch.nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.register_parameter("gamma", None)
        self.register_parameter("beta", None)

    def forward(self, x, mask):
        """
        x: [B, L, C]  (batch, length, channels)
        mask: [B, L]  (1 for valid positions, 0 for padding)
        """
        B, L, C = x.shape
        mask = mask.unsqueeze(-1).to(x.dtype)  # [B, L, 1]
        x_masked = x * mask
        valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1, 1]
        mean = x_masked.sum(dim=1, keepdim=True) / valid_counts
        var = ((x_masked - mean) * mask).pow(2).sum(dim=1, keepdim=True) / (valid_counts-1)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        x_norm = x_norm * mask

        return x_norm

def removeDuplicates(names,inds):
    names_batch = []
    for ind in inds:
        names_batch.append(names[ind])
    names_unique,counts = np.unique(names_batch,return_counts=True)
    if len(names_unique) == len(names_batch):
        return inds
    else:
        non_common = list(set(names).symmetric_difference(set(names_batch)))
        positions = np.where(counts>1)[0]
        for pos in positions:
            name_dupl = names_unique[pos]
            pos_name = np.where(np.array(names_batch) == name_dupl)[0][1]
            names_batch[pos_name] = non_common[random.randint(0, len(non_common)-1)]
            possible_pos = np.where(np.array(names) == names_batch[pos_name])[0]
            inds[pos_name] = possible_pos[random.randint(0,len(possible_pos)-1)]
            non_common = list(set(non_common).symmetric_difference(set(names_batch)))
        return inds

def test_augment(data,wind_sizes,num_winds,max_length):
    windows_all = []
    for wind_size in wind_sizes:
        windows = []
        step_size = (data.shape[0] - wind_size) // (num_winds - 1)
        for i in range(0, step_size*num_winds,step_size):
            temp = torch.zeros(max_length,data.shape[1])
            temp[:wind_size] = data[i:i+wind_size]
            windows.append(temp)
        windows_all.append(torch.stack(windows))
    return torch.cat(windows_all,dim=0)



def test_augment_AE(data,wind_sizes,num_winds,max_length):
    for wind_size in wind_sizes:
        windows = []
        step_size = (data.shape[0] - wind_size) // (num_winds - 1)
        for i in range(0, step_size*num_winds,step_size):
            temp = torch.zeros(max_length,data.shape[1])
            temp[:wind_size] = data[i:i+wind_size]
            windows.append(temp)
    return torch.stack(windows)

"""
def augment_hcp(data, train_length_limits, device):
    max_length = train_length_limits[-1]   
    data1 = []
    data2 = []
    for dat in data:
        length1 = random.randint(train_length_limits[0],train_length_limits[-1]-1)
        length2 = random.randint(train_length_limits[0],train_length_limits[-1]-1)

        a11 = int(length1/2+1)
        a12 = dat.shape[0]-int(length1/2+1)
        a21 = int(length2/2+1)
        a22 = dat.shape[0]-int(length2/2+1)
        if a12 > a11:
            c1 = random.randint(a11,a12)
        else:
            c1 = a11
        if a22 > a21:
            c2 = random.randint(a21,a22)   
        else:
            c2 = a21  
        counter = 0            
        c1s = [c1]
        c2s = [c2]
        dists = [abs(c1 - c2)]
        while counter < 100:
            if a12 > a11:
                c1 = random.randint(a11,a12)
            else:
                c1 = a11
            if a22 > a21:
                c2 = random.randint(a21,a22)   
            else:
                c2 = a21
            c1s.append(c1)
            c2s.append(c2)
            dists.append(abs(c1 - c2))
            counter = counter + 1
        max_dist = max(dists)
        max_pos = dists.index(max_dist)
        c1 = c1s[max_pos]
        c2 = c2s[max_pos]
        temp = torch.zeros(max_length,dat.shape[1])
        if length1 % 2 == 0:
            temp[:length1,:] = dat[c1-int(length1/2):c1+int(length1/2),:]
        else:
            temp[:length1,:] = dat[c1-int(length1/2):c1+int(length1/2)+1,:]
        data1.append(temp) 
        temp = torch.zeros(max_length,dat.shape[1])
        if length2 % 2 == 0:
            temp[:length2,:] = dat[c2-int(length2/2):c2+int(length2/2),:]
        else:
            temp[:length2,:] = dat[c2-int(length2/2):c2+int(length2/2)+1,:]
        data2.append(temp)
    data1 = torch.stack(data1).to(device)
    data2 = torch.stack(data2).to(device)
    return [data1,data2]
"""

def augment(data, train_length_limits, device):
    data = data.to(device)  
    max_length = train_length_limits[-1]
    batch_size, _, feat_dim = data.shape
    data1 = torch.zeros((batch_size, max_length, feat_dim), device=device)
    data2 = torch.zeros((batch_size, max_length, feat_dim), device=device)

    for i, dat in enumerate(data):
        zero_rows = torch.all(dat == 0, dim=1)
        zero_row_indices = torch.where(zero_rows)[0]
        if len(zero_row_indices) > 0:
            dat = dat[:torch.min(zero_row_indices), :]

        low_limit = train_length_limits[0]
        up_limit = min(dat.shape[0], train_length_limits[1])

        length1 = torch.randint(low_limit, up_limit, (1,), device=device).item()
        length2 = torch.randint(low_limit, up_limit, (1,), device=device).item()

        a11, a12 = length1 // 2 + 1, dat.shape[0] - length1 // 2 - 2
        a21, a22 = length2 // 2 + 1, dat.shape[0] - length2 // 2 - 2

        # Compute 100 alternative distances on GPU
        c1_candidates = torch.randint(a11, max(a12, a11 + 1), (100,), device=device)
        c2_candidates = torch.randint(a21, max(a22, a21 + 1), (100,), device=device)
        dists = torch.abs(c1_candidates - c2_candidates)

        # Select the maximum distance
        max_pos = torch.argmax(dists).item()
        c1, c2 = c1_candidates[max_pos].item(), c2_candidates[max_pos].item()

        # Slice & store in preallocated tensors
        if length1 % 2 == 0:
            data1[i, :length1, :] = dat[c1 - length1 // 2 : c1 + length1 // 2, :]
        else:
            data1[i, :length1, :] = dat[c1 - length1 // 2 : c1 + length1 // 2 + 1, :]

        if length2 % 2 == 0:
            data2[i, :length2, :] = dat[c2 - length2 // 2 : c2 + length2 // 2, :]
        else:
            data2[i, :length2, :] = dat[c2 - length2 // 2 : c2 + length2 // 2 + 1, :]

    return [data1, data2]


def other_augment1(data, train_length_limits, device):
    data = data.to(device)  
    max_length = train_length_limits[-1]
    batch_size, _, feat_dim = data.shape
    data1 = torch.zeros((batch_size, max_length, feat_dim), device=device)
    data2 = torch.zeros((batch_size, max_length, feat_dim), device=device)
    data3 = torch.zeros((batch_size, max_length, feat_dim), device=device)
    data4 = torch.zeros((batch_size, max_length, feat_dim), device=device)
    for i, dat in enumerate(data):
        zero_rows = torch.all(dat == 0, dim=1)
        zero_row_indices = torch.where(zero_rows)[0]
        if len(zero_row_indices) > 0:
            dat = dat[:torch.min(zero_row_indices), :]
        windowsize = int(dat.shape[0]/3)
        strite = int(windowsize/2)
        t = int((dat.shape[0]-windowsize)/strite)+1
        k1 = randint(0, t-1)*strite
        k2 = randint(0, t-1)*strite
        while k1 == k2:
            k2 = randint(0, t-1)*strite
        data1[i, :windowsize, :] = dat[k1 : k1+windowsize, :]
        data2[i, :windowsize, :] = dat[k2 : k2+windowsize, :]
        
        windowsizes = [int(dat.shape[0]/5),int(dat.shape[0]/4),int(dat.shape[0]/3),int(dat.shape[0]/2),dat.shape[0]]
        windowsizes = [x for x in windowsizes if x <= max_length]
        windowsize1 = windowsizes[randint(0,len(windowsizes)-1)]
        windowsize2 = windowsizes[randint(0,len(windowsizes)-1)]
        while windowsize2 == windowsize1:
            windowsize2 = windowsizes[randint(0,len(windowsizes)-1)]
        if windowsize1 > windowsize2:
            strite = int(windowsize1/2)
            t = int((dat.shape[0]-windowsize1)/strite)+1
        else:
            strite = int(windowsize2/2)
            t = int((dat.shape[0]-windowsize2)/strite)+1
        if t > 1:
            k = randint(1, t-1)*strite
        else:
            k=strite
        if windowsize1 % 2 == 0:
            data3[i, :windowsize1, :] = dat[k - windowsize1 // 2 : k + windowsize1 // 2, :]
        else:
            data3[i, :windowsize1, :] = dat[k - windowsize1 // 2 : k + windowsize1 // 2 + 1, :]

        if windowsize2 % 2 == 0:
            data4[i, :windowsize2, :] = dat[k - windowsize2 // 2 : k + windowsize2 // 2, :]
        else:
            data4[i, :windowsize2, :] = dat[k - windowsize2 // 2 : k + windowsize2 // 2 + 1, :]
        
    return [data1, data2, data3, data4]

def other_augment2(data, train_length_limits, device):
    data = data.to(device)  
    max_length = train_length_limits[-1]
    batch_size, _, feat_dim = data.shape
    data1 = torch.zeros((batch_size, max_length, feat_dim), device=device)
    data2 = torch.zeros((batch_size, max_length, feat_dim), device=device)
    for i, dat in enumerate(data):
        zero_rows = torch.all(dat == 0, dim=1)
        zero_row_indices = torch.where(zero_rows)[0]
        if len(zero_row_indices) > 0:
            dat = dat[:torch.min(zero_row_indices), :]
        if dat.shape[0] <= max_length:
            windowsize = int(0.9*dat.shape[0])
        else:
            dat = dat[:max_length,:]
            windowsize = int(0.9*dat.shape[0])
        data1[i, :windowsize, :] = dat[:windowsize, :]
        data2[i, :windowsize, :] = dat[-windowsize:, :] 
        
    return [data1, data2]


def test_augment_overlap(data, wind_sizes, overlap, min_length, max_length, device):
    data = data.to(device)  

    zero_rows = torch.all(data == 0, dim=1)
    zero_row_indices = torch.where(zero_rows)[0]
    if len(zero_row_indices) > 0:
        data = data[:torch.min(zero_row_indices), :]

    windows_all = []
    data_len = data.shape[0]

    for i in range(len(wind_sizes) - 1):
        if data_len // wind_sizes[i+1] >= min_length:
            wind_size_min = max(min_length, wind_sizes[i])
            wind_size_max = min(data_len // wind_sizes[i+1], data_len)
            wind_size = torch.randint(wind_size_min, wind_size_max + 1, (1,), device=device).item()

            step_size = int(wind_size * (1 - overlap))
            num_windows = max(1, (data_len - wind_size) // step_size + 1)

            indices = torch.arange(0, num_windows * step_size, step_size, device=device).unsqueeze(1)
            range_indices = torch.arange(wind_size, device=device).unsqueeze(0)
            window_indices = indices + range_indices  # Shape: (num_windows, wind_size)

            windows = data[window_indices]  # Gather windows using tensor indexing

            # Pad windows to max_length (if needed)
            pad_size = max_length - wind_size
            if pad_size > 0:
                pad_tensor = torch.zeros((windows.shape[0], pad_size, data.shape[1]), device=device)
                windows = torch.cat((windows, pad_tensor), dim=1)

            windows_all.append(windows)

    return torch.cat(windows_all, dim=0) if windows_all else torch.empty(0, max_length, data.shape[1], device=device)


def repeat_to_fill(x):
    batch_size, seq_len, num_channels = x.shape
    new_x = torch.zeros_like(x)
    for i in range(batch_size):
        nonzero_rows = (x[i] != 0).any(dim=1)
        N = nonzero_rows.sum().item()
        if N == 0:
            continue
        signal = x[i, :N,:]
        if seq_len // N > 1:
            for i in range(seq_len // N):
                if i == 0:
                    new_signal = torch.concat((signal,signal),dim=0)
                elif i < seq_len // N - 1:
                    new_signal = torch.concat((new_signal,signal),dim=0)
                else:
                    new_signal = torch.concat((new_signal,signal[:seq_len-new_signal.shape[0],:]),dim=0)
        elif seq_len // N == 1: 
            new_signal = torch.concat((signal,signal[:seq_len-signal.shape[0],:]),dim=0)
        else:
            new_signal = signal
        new_x[i] = new_signal
    return new_x


def PCC(X):
    M, N = X.shape
    mean = X.mean(dim=1, keepdim=True)
    X_centered = X - mean
    std = X_centered.std(dim=1, unbiased=True, keepdim=True)
    std[std == 0] = 1
    Z = X_centered / std
    corr_matrix = torch.matmul(Z, Z.T) / (N - 1)
    return corr_matrix

def augment_VAE(data,a,b,device):
    x = []
    for dat in data:
        length = random.randint(a, b)
        max_start = dat.shape[0] - length
        start = random.randint(0, max_start)
        dat = dat[start:start + length,:]
        corr = PCC(dat.T)
        triu_indices = torch.triu_indices(corr.shape[0], corr.shape[0], offset=1)
        upper_triangular_values = corr[triu_indices[0], triu_indices[1]]
        x.append(upper_triangular_values)
    return torch.stack(x).float().to(device)


def test_augment_PCC(data,wind_sizes,num_winds):
    new_data = []
    for wind_size in wind_sizes:
        step_size = (data.shape[0] - wind_size) // (num_winds - 1)
        for i in range(0, step_size*num_winds,step_size):
            segment = data[i:i+wind_size]
            corr = np.corrcoef(segment.T)
            triu_indices = np.triu_indices(corr.shape[0], k=1)
            upper_triangular_values = corr[triu_indices[0], triu_indices[1]]
            new_data.append(upper_triangular_values)
    return np.stack(new_data)


def compute_alff_falff(AllVolume, ASamplePeriod, HighCutoff, LowCutoff):

    sampleFreq = 1.0 / ASamplePeriod
    sampleLength = AllVolume.shape[0]
    paddedLength = 2**int(np.ceil(np.log2(sampleLength)))

    if LowCutoff >= sampleFreq / 2:
        idx_LowCutoff = paddedLength // 2 + 1
    else:
        idx_LowCutoff = int(np.ceil(LowCutoff * paddedLength * ASamplePeriod + 1))

    if (HighCutoff >= sampleFreq / 2) or (HighCutoff == 0):
        idx_HighCutoff = paddedLength // 2 + 1
    else:
        idx_HighCutoff = int(np.fix(HighCutoff * paddedLength * ASamplePeriod + 1))
        
    AllVolume = detrend(AllVolume, axis=0)


    # Zero padding
    padding = paddedLength - sampleLength
    if padding > 0:
        AllVolume = np.vstack([AllVolume, np.zeros((padding, AllVolume.shape[1]))])

    AllVolume = 2 * np.abs(np.fft.fft(AllVolume, axis=0)) / sampleLength

    ALFF_2D = AllVolume[idx_LowCutoff:idx_HighCutoff, :].mean(axis=0)

    return ALFF_2D

class AGCLDataset(InMemoryDataset):
    def __init__(self, root, data, y, atlas, transform=None, pre_transform=None):
        self.root = root
        self.data = data
        self.atlas = atlas
        self.y = y

        super(AGCLDataset, self).__init__(root,transform, pre_transform)
        path = os.path.join(self.processed_dir, 'data_' + atlas + '.pt')
        self.data, self.slices = torch.load(path)

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')
    
    @property
    def processed_file_names(self):
        return  'data_' + self.atlas + '.pt'


    def process(self):
        atlas = self.atlas
        data_list = []
        for signal,y in zip(self.data, self.y):  
            label = torch.squeeze(torch.tensor(y, dtype=torch.long))
            if np.any(np.all(signal == 0, axis=1)):
                min_index = np.min(np.where(np.all(signal == 0, axis=1)))
                signal = signal[:min_index,:]
                signal = (signal - np.mean(signal, axis=0, keepdims=True))/np.std(signal, axis=0, keepdims=True)
            pcc = np.corrcoef(signal.T)
            alff_s5 = compute_alff_falff(signal, 1.5, 0.027, 0.01)
            alff_s4 = compute_alff_falff(signal, 1.5, 0.073, 0.027)
            alff_classic = compute_alff_falff(signal, 1.5, 0.08, 0.01)
            alffs = np.stack([alff_s5, alff_s4, alff_classic])
            alffs = (alffs - np.min(alffs)/np.max(alffs) - np.min(alffs))
            alffs = alffs.T
            pcc -= np.eye(pcc.shape[0])
            pcc /= np.max(np.abs(pcc))
            pcc += np.eye(pcc.shape[0])  
            x = np.nan_to_num(alffs)
            x = torch.Tensor(x)

            edge_index = pcc

            edge_index = np.nan_to_num(edge_index)
            edge_index_temp = sp.coo_matrix(edge_index)
            edge_weight = torch.Tensor(edge_index_temp.data)

            edge_index = torch.Tensor(edge_index)  
            edge_index = edge_index.nonzero(as_tuple=False).t().contiguous()
            num_nodes = int(edge_index.max()) + 1

            data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight,y=label)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
        
        torch.save(self.collate(data_list),
                os.path.join(self.processed_dir, 'data_' + atlas + '.pt'))

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

class AGCLDataset2(Dataset_geo):
    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        label = torch.squeeze(torch.tensor(label, dtype=torch.long))
        if np.any(np.all(signal == 0, axis=1)):
            min_index = np.min(np.where(np.all(signal == 0, axis=1)))
            signal = signal[:min_index,:]
            signal = (signal - np.mean(signal, axis=0, keepdims=True))/np.std(signal, axis=0, keepdims=True)
        pcc = np.corrcoef(signal.T)
        alff_s5 = compute_alff_falff(signal, 1.5, 0.027, 0.01)
        alff_s4 = compute_alff_falff(signal, 1.5, 0.073, 0.027)
        alff_classic = compute_alff_falff(signal, 1.5, 0.08, 0.01)
        alffs = np.stack([alff_s5, alff_s4, alff_classic])
        alffs = (alffs - np.min(alffs)/np.max(alffs) - np.min(alffs))
        alffs = alffs.T
        pcc -= np.eye(pcc.shape[0])
        pcc /= np.max(np.abs(pcc))
        pcc += np.eye(pcc.shape[0])  
        x = np.nan_to_num(alffs)
        x = torch.Tensor(x)
        edge_index = pcc
        edge_index = np.nan_to_num(edge_index)
        edge_index_temp = sp.coo_matrix(edge_index)
        edge_weight = torch.Tensor(edge_index_temp.data)
        edge_index = torch.Tensor(edge_index)  
        edge_index = edge_index.nonzero(as_tuple=False).t().contiguous()
        data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight,y=label)
        return data
    
    
class UCGLDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        T = signal.shape[0]
        S = int(0.9*T)
        data1 = signal[:S,:]
        data2 = signal[-S:,:]
        data1 = torch.from_numpy(data1).unsqueeze(0).unsqueeze(-1)
        data2 = torch.from_numpy(data2).unsqueeze(0).unsqueeze(-1)
        y = torch.squeeze(torch.tensor(label, dtype=torch.long))
        return data1, data2, y
    
class UCGLDataset2(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        data = torch.from_numpy(signal).unsqueeze(0).unsqueeze(-1)
        y = torch.squeeze(torch.tensor(label, dtype=torch.long))
        return data, y
    
def LoadAllUCGL(signals):
    data_alla1 = None
    data_alla2 = None
    
    for i in range(len(signals)): #subjects
        signal = signals[i]
        if np.any(np.all(signal == 0, axis=1)):
            min_index = np.min(np.where(np.all(signal == 0, axis=1)))
            signal = signal[:min_index,:] 
        T = signal.shape[0]
        S = int(0.9*T)
        X1=signal[:S,:]
        X2=signal[-S:,:]
        
        if data_alla1 is None:              
           data_alla1 = X1
        else:
           data_alla1 = np.concatenate((data_alla1, X1), axis=0)
           
        if data_alla2 is None:              
           data_alla2 = X2
        else:
           data_alla2 = np.concatenate((data_alla2, X2), axis=0) 
    return data_alla1, data_alla2


class GCDADataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        if np.any(np.all(signal == 0, axis=1)):
            min_index = np.min(np.where(np.all(signal == 0, axis=1)))
            signal = signal[:min_index,:] 
        pc = np.corrcoef(signal.T)
        pc = np.nan_to_num(pc)
        pc = abs(pc)              
        E = torch.from_numpy(pc)
        E = torch.where(E >= 0.3, torch.tensor(1.0), torch.tensor(0.0)).long()
        E = F.one_hot(E, num_classes=2)  
        X = torch.from_numpy(pc)
        y_0 = torch.zeros([0]).float()
        y = torch.from_numpy(y_0.numpy())
        label = torch.squeeze(torch.tensor(label, dtype=torch.long))
        return X, E, y, label
    
class GCDADataset2(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        if np.any(np.all(signal == 0, axis=1)):
            min_index = np.min(np.where(np.all(signal == 0, axis=1)))
            signal = signal[:min_index,:] 
        pc = np.corrcoef(signal.T)
        pc = np.nan_to_num(pc)
        pc = abs(pc)              
        E = torch.from_numpy(pc)
        E = torch.where(E >= 0.3, torch.tensor(1.0), torch.tensor(0.0)).long()
        X = torch.from_numpy(pc)
        E = E.unsqueeze(0)
        X = X.unsqueeze(0)
        label = torch.squeeze(torch.tensor(label, dtype=torch.long))
        return X, E, label
    
def GCDA_dims(atlas):
    if atlas == 'AAL':
        roi_num = 166
    elif atlas == 'AICHA':
        roi_num = 384
    A20 = np.random.rand(10, 320, roi_num)
    pc_l = []
    y_list = []      
    for i in range(len(A20)): 
        pc = np.corrcoef(A20[i].T)
        pc = np.nan_to_num(pc)
        pc = abs(pc)         
        pc_l.append(pc)
        y_0 = torch.zeros([0]).float()
        y_np = y_0.numpy()
        y_list.append(y_np)
    X = np.array(pc_l) 
    E = torch.from_numpy(X)
    E = torch.where(E >= 0.3, torch.tensor(1.0), torch.tensor(0.0)).long()
    E = F.one_hot(E, num_classes=2)
    y = np.array(y_list) 
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    return  X,E,y


class BrainLMDataset(Dataset):

    def __init__(self, signals, labels, ids, data_median_per_voxel,IQR):
        self.signals = signals
        self.labels = labels
        self.data_median_per_voxel = data_median_per_voxel
        self.IQR = IQR
        self.ids = ids
        
    def __getitem__(self,index):
        signal = self.signals[index]
        label = self.labels[index]
        data_median_per_voxel = self.data_median_per_voxel
        IQR = self.IQR
        ID = self.ids[index]
        dataset_dict = {
            "Raw_Recording": [],
            "Voxelwise_RobustScaler_Normalized_Recording": [],
            "Filename": [],
            "Patient ID": [],
            "class": []
        }
        signal_norm = signal - np.mean(signal, axis=0, keepdims=True)
        signal_norm = (signal_norm - data_median_per_voxel)/IQR
        dataset_dict["Raw_Recording"].append(signal)
        dataset_dict["Voxelwise_RobustScaler_Normalized_Recording"].append(signal_norm)
        dataset_dict["Filename"].append(ID)
        dataset_dict["Patient ID"].append(ID)
        dataset_dict["class"].append(label)
        return dataset_dict

    def __len__(self):
        return len(self.signals)


def BrainIBDataset(signals, labels):
    graphs = []
    for signal, label in zip(signals, labels):
        if np.any(np.all(signal == 0, axis=1)):
            min_index = np.min(np.where(np.all(signal == 0, axis=1)))
            signal = signal[:min_index,:] 
        corr = np.corrcoef(signal.T)
        corr = np.nan_to_num(corr)
        abs_corr = np.abs(corr)
        threshold = np.percentile(abs_corr, 80)
        adj = (abs_corr > threshold).astype(int)
        np.fill_diagonal(adj, 0)
        corr_triu = np.triu(corr, k=1)
        r, c = np.where(abs_corr > threshold)
        r = r.astype(int)
        c = c.astype(int)
        rows, columns, vals = [], [], []
        neighbors = {}
        for j in range(corr.shape[0]):
            pos = np.where(r == j)[0]
            rows.extend(r[pos])
            columns.extend(c[pos])
            linear_idx = np.ravel_multi_index((r[pos], c[pos]), corr_triu.shape)
            vals.extend(linear_idx)
            neighbors[j] = list(c[pos])
        
        edge = np.vstack((rows, columns, vals)).T
        corr_no_diag = corr.copy()
        np.fill_diagonal(corr_no_diag, 0)
        node_tags = np.arange(corr.shape[0])
    
        edge = torch.Tensor(edge)
        ROI = torch.Tensor(corr_no_diag)
        node_tags = torch.Tensor(node_tags)
        adj = torch.Tensor(adj)
        y = torch.squeeze(torch.tensor(label, dtype=torch.long))
        A = torch.sparse_coo_tensor(
            indices = edge[:, :2].t().long(),
            values = edge[:, -1].reshape(-1,).float(),
            size = (corr.shape[0], corr.shape[0])
            )
        G = (A.t() + A).coalesce()
        graph = Data_geo(x=ROI.reshape(-1,corr.shape[0]).float(),
                     edge_index=G.indices().reshape(2,-1).long(),
                     edge_attr=G.values().reshape(-1,1).float(),
                     y=y.long())
        graphs.append(graph)
    return graphs




class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def FBNETGENLoader(final_fc, labels, batch_size):
    final_pearson = []
    for signal in final_fc:
        corr = np.corrcoef(signal)
        final_pearson.append(corr)
    final_pearson = np.stack(final_pearson)

    timeseries = signal.shape[1]

    _, node_size, node_feature_size = final_pearson.shape

    scaler = StandardScaler(mean=np.mean(
        final_fc), std=np.std(final_fc))
    
    final_fc = scaler.transform(final_fc)

    final_fc, final_pearson, labels = [torch.from_numpy(
        data).float() for data in (final_fc, final_pearson, labels)]


    dataset = utils.TensorDataset(
        final_fc,
        final_pearson,
        labels
    )

    dataloader = utils.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    return dataloader, node_size, node_feature_size, timeseries

'''
def extract_windows(data, lengths, device):
    data = data.to(device)  
    batch_size, _, R = data.shape
    data1_batch = []
    data2_batch = []

    for i, dat in enumerate(data):
        zero_rows = torch.all(dat == 0, dim=1)
        zero_row_indices = torch.where(zero_rows)[0]
        if len(zero_row_indices) > 0:
            dat = dat[:torch.min(zero_row_indices), :]
        T = dat.shape[0]
        dat1 = dat[:T//2,:]
        dat2 = dat[T//2:,:]
        
        count = 0
        for length in lengths:
            for pos in range(0, dat1.shape[0], length):
                count += 1
        data1 = torch.zeros((count, R*(R-1)//2), device=device)
        
        count = 0
        for length in lengths:
            for pos in range(0, dat2.shape[0], length):
                count += 1
        data2 = torch.zeros((count, R*(R-1)//2), device=device)
        
        count = 0
        for length in lengths:
            for pos in range(0, dat1.shape[0], length):
                corr = torch.corrcoef(dat1[pos:pos+length].T)
                triu_indices = torch.triu_indices(R, R, offset=1)
                corr = corr[triu_indices[0], triu_indices[1]]
                data1[count,:] = corr
                count += 1
        data1_batch.append(data1)
        
        count = 0
        for length in lengths:
            for pos in range(0, dat2.shape[0], length):
                corr = torch.corrcoef(dat2[pos:pos+length].T)
                triu_indices = torch.triu_indices(R, R, offset=1)
                corr = corr[triu_indices[0], triu_indices[1]]
                data2[count,:] = corr
                count += 1   
        data2_batch.append(data2)
    
    data1_batch = torch.stack(data1_batch)
    data2_batch = torch.stack(data2_batch)
    return [data1_batch, data2_batch]
'''

def batch_corrcoef(X):
    # X: [B, T, R]
    X = X - X.mean(dim=1, keepdim=True)
    cov = X.transpose(1, 2) @ X / (X.shape[1] - 1)
    std = X.std(dim=1, keepdim=True)
    corr = cov / (std.transpose(2, 1) @ std + 1e-8)
    return corr


def extract_corr_windows(dat, lengths, overlap=0.75):
    R = dat.shape[1]
    results = []
    triu_indices = torch.triu_indices(R, R, offset=1, device=dat.device)
    for length in lengths:
        step = int(length * (1 - overlap))
        if length <= dat.shape[0] // 2:
            windows = dat.unfold(0, length, step).permute(0, 2, 1)
        elif length > dat.shape[0] // 2 and length < dat.shape[0]:
            windows = torch.stack([dat[:length,:], dat[dat.shape[0]//2-length//2:dat.shape[0]//2+length//2], dat[-length:,:]])
        else:
            windows = dat.unsqueeze(0)
        # batch corr computation
        corr = batch_corrcoef(windows)
        # extract upper triangles
        corr = corr[:, triu_indices[0], triu_indices[1]]
        results.append(corr)
    return torch.cat(results, dim=0)


def extract_temporal_windows(dat, lengths, overlap=0.5):
    R = dat.shape[1]
    results = []
    for length in lengths:
        step = int(length * (1 - overlap))
        # compute number of windows
        #n = dat.shape[0] // length
        # reshape into [n, length, R]
        #windows = dat[:n*length].reshape(n, length, R)
        if length <= dat.shape[0] // 2:
            nwin = np.floor(2*(dat.shape[0]-length)/length) + 1
            windows = torch.zeros((int(nwin), dat.shape[0], R), device=dat.device)
            windows[:,:length,:] = dat.unfold(0, length, step).permute(0, 2, 1)
        elif length > dat.shape[0] // 2 and length < dat.shape[0]:
            nwin = 3
            windows = torch.zeros((int(nwin), dat.shape[0], R), device=dat.device)
            windows[:,:length,:] = torch.stack([dat[:length,:], dat[dat.shape[0]//2-length//2:dat.shape[0]//2+length//2], dat[-length:,:]])
        elif length == dat.shape[0]:
            windows = dat.unsqueeze(0)
        results.append(windows)
    return torch.cat(results, dim=0)


def extract_windows_old(data, data_rdcm, lengths, device):
    B, T, R = data.shape
    data = data.to(device)
    data_rdcm = data_rdcm.to(device)
    #data = time_augment(data, [120,T//2], device)
    #data_rdcm = time_augment(data_rdcm, [120,T//2], device)
    data1_batch, data2_batch = [], []

    for (dat,dat_rdcm) in zip(data,data_rdcm):
        data1 = extract_corr_windows(dat, lengths)
        data2 = extract_corr_windows(dat_rdcm, lengths)
        data1_batch.append(data1)
        data2_batch.append(data2)

    # pad to equal length if needed
    max_len1 = max([d.shape[0] for d in data1_batch])
    max_len2 = max([d.shape[0] for d in data2_batch])
    data1_batch = torch.stack([F.pad(d, (0,0,0,max_len1-d.shape[0])) for d in data1_batch])
    data2_batch = torch.stack([F.pad(d, (0,0,0,max_len2-d.shape[0])) for d in data2_batch])
    return [data1_batch, data2_batch]


def extract_windows(data, lengths, device):
    #data = data.to(device)
    B, T, R = data.shape
    data1, data2 = data[:,:T//2,:], data[:,T//2:,:]
    data1 = time_augment(data1, [120,T//2], device)
    data2 = time_augment(data2, [120,T//2], device)
    data1_batch, data2_batch = [], []

    for dat1,dat2 in zip(data1, data2):
        #zero_rows = torch.all(dat == 0, dim=1)
        #if zero_rows.any():
        #    dat = dat[:torch.min(torch.where(zero_rows)[0])]

        data1_t = extract_temporal_windows(dat1, lengths, R, device)
        data2_t = extract_temporal_windows(dat2, lengths, R, device)
        data1_batch.append(data1_t)
        data2_batch.append(data2_t)

    # pad to equal length if needed
    max_len1 = max([d.shape[0] for d in data1_batch])
    max_len2 = max([d.shape[0] for d in data2_batch])
    data1_batch = torch.stack([F.pad(d, (0,0,0,max_len1-d.shape[0])) for d in data1_batch])
    data2_batch = torch.stack([F.pad(d, (0,0,0,max_len2-d.shape[0])) for d in data2_batch])
    return [data1_batch, data2_batch]


def time_augment(data, limits, device):
    data = data.to(device)  
    max_length = limits[-1]
    batch_size, _, feat_dim = data.shape
    data1 = torch.zeros((batch_size, max_length, feat_dim), device=device)

    for i, dat in enumerate(data):
        zero_rows = torch.all(dat == 0, dim=1)
        zero_row_indices = torch.where(zero_rows)[0]
        if len(zero_row_indices) > 0:
            dat = dat[:torch.min(zero_row_indices), :]

        low_limit = limits[0]
        up_limit = min(dat.shape[0], limits[1])

        length = torch.randint(low_limit, up_limit, (1,), device=device).item()

        a1, a2 = length // 2 + 1, dat.shape[0] - length // 2 - 2

        # Compute 100 alternative distances on GPU
        c = torch.randint(a1, max(a2, a1 + 1), (1,), device=device)

        # Slice & store in preallocated tensors
        if length % 2 == 0:
            data1[i, :length, :] = dat[c - length // 2 : c + length // 2, :]
        else:
            data1[i, :length, :] = dat[c - length // 2 : c + length // 2 + 1, :]

    return data1


def extract_windows_rdcm(data, data_rdcm, lengths, device):
    #data = data.to(device)
    #data_rdcm = data_rdcm.to(device)
    data = time_augment(data, [120,400], device)
    data_rdcm = time_augment(data_rdcm, [120,400], device)
    B, _, R = data.shape
    data1_batch, data2_batch = [], []

    for (dat, dat_rdcm) in zip(data, data_rdcm):
        #zero_rows = torch.all(dat == 0, dim=1)
        #if zero_rows.any():
        #    dat = dat[:torch.min(torch.where(zero_rows)[0])]
        #zero_rows = torch.all(dat_rdcm == 0, dim=1)
        #if zero_rows.any():
        #    dat_rdcm = dat_rdcm[:torch.min(torch.where(zero_rows)[0])]

        data1 = extract_temporal_windows(dat, lengths)
        data2 = extract_temporal_windows(dat_rdcm, lengths)
        data1_batch.append(data1)
        data2_batch.append(data2)

    # pad to equal length if needed
    max_len1 = max([d.shape[0] for d in data1_batch])
    max_len2 = max([d.shape[0] for d in data2_batch])
    data1_batch = torch.stack([F.pad(d, (0,0,0,max_len1-d.shape[0])) for d in data1_batch])
    data2_batch = torch.stack([F.pad(d, (0,0,0,max_len2-d.shape[0])) for d in data2_batch])
    return [data1_batch, data2_batch]


def test_augment_windows_old(data, length, device):
    data = data.to(device)
    _, R = data.shape
    zero_rows = torch.all(data == 0, dim=1)
    if zero_rows.any():
        data = data[:torch.min(torch.where(zero_rows)[0])]
    data = data[:400]
    data = extract_corr_windows(data, length)
    return data.unsqueeze(0)


def test_augment_windows(data, lengths, device):
    data = data.to(device)
    _, R = data.shape
    zero_rows = torch.all(data == 0, dim=1)
    if zero_rows.any():
        data = data[:torch.min(torch.where(zero_rows)[0])]
    T = data.shape[0]
    data1 = data[:T//2]
    data2 = data[T//2:]
    data3 = data[T//4:3*T//4]
    data4 = data[T//4:3*T//4]
    data5 = data[:T//4]
    data6 = data[T//4:T//2]
    data7 = data[T//2:3*T//4]
    data8 = data[3*T//4:]
    lengths = np.array(lengths)
    lengths1 = lengths[np.where(np.array(lengths) <= data1.shape[0])[0]]
    lengths2 = lengths[np.where(np.array(lengths) <= data2.shape[0])[0]]
    lengths3 = lengths[np.where(np.array(lengths) <= data3.shape[0])[0]]
    lengths4 = lengths[np.where(np.array(lengths) <= data4.shape[0])[0]]
    lengths5 = lengths[np.where(np.array(lengths) <= data5.shape[0])[0]]
    lengths6 = lengths[np.where(np.array(lengths) <= data6.shape[0])[0]]
    lengths7 = lengths[np.where(np.array(lengths) <= data7.shape[0])[0]]
    lengths8 = lengths[np.where(np.array(lengths) <= data8.shape[0])[0]]
    data1 = extract_corr_windows(data1, lengths1)
    data2 = extract_corr_windows(data2, lengths2)
    data3 = extract_corr_windows(data3, lengths3)
    data4 = extract_corr_windows(data4, lengths4)
    data5 = extract_corr_windows(data5, lengths5)
    data6 = extract_corr_windows(data6, lengths6)
    data7 = extract_corr_windows(data7, lengths7)
    data8 = extract_corr_windows(data8, lengths8)
    max_shape = np.max([data1.shape[0],data2.shape[0],data3.shape[0],
                        data4.shape[0],data5.shape[0],data6.shape[0],
                        data7.shape[0],data8.shape[0]])
    data = [data1, data2, data3, data4, data5, data6, data7, data8]
    for i,dat in enumerate(data):
        if dat.shape[0] < max_shape:
            temp = torch.zeros((max_shape, dat.shape[1]), dtype=data1.dtype, device=data1.device)
            temp[:dat.shape[0],:] = dat
            data[i] = temp
    data = torch.stack(data)
    return data


def test_augment_temporal_windows(data, lengths, device):
    data = data.to(device)
    _, R = data.shape
    zero_rows = torch.all(data == 0, dim=1)
    if zero_rows.any():
        data = data[:torch.min(torch.where(zero_rows)[0])]
    T = data.shape[0]
    data1 = data[:T//2]
    data2 = data[T//2:]
    data3 = data[T//4:3*T//4]
    data4 = data[T//4:3*T//4]
    data5 = data[:T//4]
    data6 = data[T//4:T//2]
    data7 = data[T//2:3*T//4]
    data8 = data[3*T//4:]
    lengths = np.array(lengths)
    lengths1 = lengths[np.where(np.array(lengths) <= data1.shape[0])[0]]
    lengths2 = lengths[np.where(np.array(lengths) <= data2.shape[0])[0]]
    lengths3 = lengths[np.where(np.array(lengths) <= data3.shape[0])[0]]
    lengths4 = lengths[np.where(np.array(lengths) <= data4.shape[0])[0]]
    lengths5 = lengths[np.where(np.array(lengths) <= data5.shape[0])[0]]
    lengths6 = lengths[np.where(np.array(lengths) <= data6.shape[0])[0]]
    lengths7 = lengths[np.where(np.array(lengths) <= data7.shape[0])[0]]
    lengths8 = lengths[np.where(np.array(lengths) <= data8.shape[0])[0]]
    data1 = extract_temporal_windows(data1, lengths1, R, device)
    data2 = extract_temporal_windows(data2, lengths2, R, device)
    data3 = extract_temporal_windows(data3, lengths3, R, device)
    data4 = extract_temporal_windows(data4, lengths4, R, device)
    data5 = extract_temporal_windows(data5, lengths5, R, device)
    data6 = extract_temporal_windows(data6, lengths6, R, device)
    data7 = extract_temporal_windows(data7, lengths7, R, device)
    data8 = extract_temporal_windows(data8, lengths8, R, device)
    max_shape1 = np.max([data1.shape[0],data2.shape[0],data3.shape[0],
                        data4.shape[0],data5.shape[0],data6.shape[0],
                        data7.shape[0],data8.shape[0]])
    max_shape2 = np.max([data1.shape[1],data2.shape[1],data3.shape[1],
                        data4.shape[1],data5.shape[1],data6.shape[1],
                        data7.shape[1],data8.shape[1]])
    data = [data1, data2, data3, data4, data5, data6, data7, data8]
    for i,dat in enumerate(data):
        if dat.shape[0] < max_shape1:
            temp = torch.zeros((max_shape1, max_shape2, R), dtype=data1.dtype, device=data1.device)
            temp[:dat.shape[0],:dat.shape[1],:] = dat
            data[i] = temp
    data = torch.stack(data)
    return data

def test_augment_temporal_windows_rdcm(data, lengths, device):
    data = data.to(device)
    _, R = data.shape
    zero_rows = torch.all(data == 0, dim=1)
    if zero_rows.any():
        data = data[:torch.min(torch.where(zero_rows)[0])]
    T = data.shape[0]
    data1 = data[:400]
    data2 = data[-400:]
    data3 = data[:400]
    data4 = data[-400:]
    data5 = data[:T//2]
    data6 = data[T//2:]
    data7 = data[T//4:3*T//4]
    data8 = data[T//4:3*T//4]
    data9 = data[:T//4]
    data10 = data[T//4:T//2]
    data11 = data[T//2:3*T//4]
    data12 = data[3*T//4:]
    lengths = np.array(lengths)
    lengths1 = lengths[np.where(np.array(lengths) <= data1.shape[0])[0]]
    lengths2 = lengths[np.where(np.array(lengths) <= data2.shape[0])[0]]
    lengths3 = lengths[np.where(np.array(lengths) <= data3.shape[0])[0]]
    lengths4 = lengths[np.where(np.array(lengths) <= data4.shape[0])[0]]
    lengths5 = lengths[np.where(np.array(lengths) <= data5.shape[0])[0]]
    lengths6 = lengths[np.where(np.array(lengths) <= data6.shape[0])[0]]
    lengths7 = lengths[np.where(np.array(lengths) <= data7.shape[0])[0]]
    lengths8 = lengths[np.where(np.array(lengths) <= data8.shape[0])[0]]
    lengths9 = lengths[np.where(np.array(lengths) <= data9.shape[0])[0]]
    lengths10 = lengths[np.where(np.array(lengths) <= data10.shape[0])[0]]
    lengths11 = lengths[np.where(np.array(lengths) <= data11.shape[0])[0]]
    lengths12 = lengths[np.where(np.array(lengths) <= data12.shape[0])[0]]
    data1 = extract_temporal_windows(data1, lengths)
    data2 = extract_temporal_windows(data2, lengths2)
    data3 = extract_temporal_windows(data3, lengths3)
    data4 = extract_temporal_windows(data4, lengths4)
    data5 = extract_temporal_windows(data5, lengths5)
    data6 = extract_temporal_windows(data6, lengths6)
    data7 = extract_temporal_windows(data7, lengths7)
    data8 = extract_temporal_windows(data8, lengths8)
    data9 = extract_temporal_windows(data9, lengths9)
    data10 = extract_temporal_windows(data10, lengths10)
    data11 = extract_temporal_windows(data11, lengths11)
    data12 = extract_temporal_windows(data12, lengths12)
    max_shape1 = np.max([data1.shape[0],data2.shape[0],data3.shape[0],
                        data4.shape[0],data5.shape[0],data6.shape[0],
                        data7.shape[0],data8.shape[0],data9.shape[0],
                        data10.shape[0],data11.shape[0],data12.shape[0]])
    max_shape2 = np.max([data1.shape[1],data2.shape[1],data3.shape[1],
                        data4.shape[1],data5.shape[1],data6.shape[1],
                        data7.shape[1],data8.shape[1],data9.shape[1],
                        data10.shape[1], data11.shape[1], data12.shape[1]])
    data = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12]
    for i,dat in enumerate(data):
        if dat.shape[0] < max_shape1:
            temp = torch.zeros((max_shape1, max_shape2, R), dtype=data1.dtype, device=data1.device)
            temp[:dat.shape[0],:dat.shape[1],:] = dat
            data[i] = temp
    data = torch.stack(data)
    return data


class BrainGNNDataset(Dataset_geo):
    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        if torch.is_tensor(self.signals[idx]):
            signal = self.signals[idx].cpu().numpy()
        else:
            signal = self.signals[idx]
        label = self.labels[idx]
        if np.any(np.all(signal == 0, axis=1)):
            min_index = np.min(np.where(np.all(signal == 0, axis=1)))
            signal = signal[:min_index,:] 
        PCC = connectome.ConnectivityMeasure(kind='correlation')
        pcc = torch.from_numpy(PCC.fit_transform(np.expand_dims(signal, axis=0)).squeeze()).float()
        
        PC = connectome.ConnectivityMeasure(kind='partial correlation')
        partial_corr = PC.fit_transform(np.expand_dims(signal, axis=0)).squeeze()
        
        N = partial_corr.shape[0]
        flat = partial_corr[np.triu_indices(N, k=1)]
        positive = flat[flat > 0]
        
        k = int(len(positive) * 0.1)  # top 10%
        threshold = np.partition(positive, -k)[-k] if k > 0 else np.max(positive)
        
        # Build adjacency mask
        adj = (partial_corr >= threshold).astype(np.float32)
        #np.fill_diagonal(adj, 0.0)
        
        edge_indices = np.array(np.nonzero(adj))
        edge_attrs = partial_corr[edge_indices[0], edge_indices[1]]
        
        # Convert to torch tensors
        edge_index = torch.tensor(edge_indices, dtype=torch.long)
        edge_att = torch.tensor(edge_attrs, dtype=torch.float32)
        
        edge_index, edge_att = remove_self_loops(edge_index, edge_att)
        
        pseudo = torch.diag(torch.ones(116)).float()
        label = torch.tensor(label).long()
        graph = Data(x=pcc, edge_index=edge_index, edge_attr=edge_att, y=label, pos=pseudo)
        return graph


def stratified_group_train_test_split(X, y, groups, test_size, random_state=None):
    group_to_indices = defaultdict(list)
    group_to_label = {}
    
    for i,g in enumerate(groups):
        group_to_indices[g].append(i)
        if g not in group_to_label:
            group_to_label[g] = y[i]
        
    unique_groups = np.array(list(group_to_indices.keys()))
    group_labels = np.array([group_to_label[g] for g in unique_groups])
    
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    group_train_idx, group_test_idx = next(splitter.split(unique_groups, group_labels))
    
    train_indices = []
    test_indices = []
    
    for g in unique_groups[group_train_idx]:
        train_indices.extend(group_to_indices[g])
    for g in unique_groups[group_test_idx]:
        test_indices.extend(group_to_indices[g])
    
    return (
        np.array(X)[train_indices],
        np.array(X)[test_indices],
        np.array(y)[train_indices],
        np.array(y)[test_indices],
        train_indices,
        test_indices
        )


def safe_clone(obj):
    """
    Recursively clones a DCM-like structure containing nested dicts, lists, and tensors.
    Works even when copy.deepcopy fails (e.g., GPU tensors, lambda objects).
    """
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    elif isinstance(obj, dict):
        return {k: safe_clone(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_clone(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(safe_clone(v) for v in obj)
    else:
        # For non-tensor leaf nodes (numbers, strings, None, etc.)
        return obj



def tapas_dcm_euler_gen(DCM, Ep):
    """
    Batched version of tapas_dcm_euler_gen.

    Args:
        DCM (dict): Model structure.
        Ep (dict): Data-generating parameters.

    Returns:
        y (torch.Tensor): Generated BOLD signal (batch × time × regions).
        x (torch.Tensor): Generated neuronal signal (batch × time × regions).
    """

    # --------------------------------------------------------
    # Number of regions
    nr = Ep['A'].shape[0]
    A = Ep['A'].T.clone()  # (nr, nr)

    # --------------------------------------------------------
    # Prepare empty B and D arrays
    mArrayB = torch.zeros((nr, nr, nr), dtype=torch.float32, device=A.device).permute(1, 0, 2).clone()
    Ep['D'] = torch.zeros((nr, nr, nr), dtype=torch.float32, device=A.device)
    mArrayD = Ep['D'].permute(1, 0, 2).clone()

    # --------------------------------------------------------
    # Driving inputs
    # U: (N, T, R)
    U = DCM['U']['u']
    if U.ndim == 2:
        U = U.unsqueeze(0)  # convert (T,R) -> (1,T,R)
    N, T, R = U.shape

    C = U @ (Ep['C'].T / 16.0).float()  # (N, T, nr)

    # --------------------------------------------------------
    # Hemodynamic constants
    H = torch.tensor([0.64, 0.32, 2.00, 0.32, 0.32], dtype=torch.float32, device=A.device)

    # --------------------------------------------------------
    # Constants for hemodynamic model
    oxygenExtractionFraction = torch.full((nr, 1), 0.32, dtype=torch.float32, device=A.device)
    alphainv = 1.0 / H[3]
    tau = H[2] * torch.exp(Ep['transit'])
    gamma = H[1]
    kappa = H[0] * torch.exp(Ep['decay'])
    epsilon = torch.exp(Ep['epsilon']) * torch.ones((nr, 1), dtype=torch.float32, device=A.device)

    # -------------------------------------------------------
    paramList = [
        DCM['U']['dt'],
        T,
        nr,
        R,
        0,
        1,
        1
    ]

    # -------------------------------------------------------
    # Batched Euler integration
    x = dcm_euler_integration_vectorized_batch(
        A, C, U, mArrayB, mArrayD,
        oxygenExtractionFraction, alphainv, tau, gamma, kappa, paramList
    )

    return x


def dcm_euler_integration_vectorized_batch(A, C, U, B, D, rho, alphainv, tau, gamma, kappa, paramList):
    """
    Batched Euler integration for multiple input sequences.

    Args:
        A (nr, nr)
        C (N, T, nr)
        U (N, T, R)
        B, D: connectivity tensors
        ...
        paramList: same as before

    Returns:
        x_out (N, T, nr)
    """
    # Unpack parameters
    timeStep = float(paramList[0])
    nTime = int(paramList[1])
    nStates = int(paramList[2])
    nInputs = int(paramList[3])
    dcmTypeB = 0
    dcmTypeD = 0

    N = U.shape[0]
    device = A.device

    # Allocate output
    x_out = torch.zeros((N, nTime, nStates), dtype=torch.float64, device=device)

    # ------------------------------------------------------------
    # Main Euler integration loop (batched)
    # ------------------------------------------------------------
    for iStep in range(nTime - 1):
        x_t = x_out[:, iStep, :]  # (N, nr)

        # --------------------------------------------------------
        # Update x (neuronal activity)
        # A-term
        Ax = x_t @ A.double()  # (N, nr)
        x_new = x_t + timeStep * (Ax + C[:, iStep, :].double())

        # B-term (bilinear)
        if dcmTypeB != 0:
            UB_effect = torch.zeros_like(x_t)
            for kIter in range(nInputs):
                UB_effect += (U[:, iStep, kIter].unsqueeze(1)) * (x_t @ B[:, :, kIter].double())
            x_new += timeStep * UB_effect

        # D-term (nonlinear)
        if dcmTypeD != 0:
            temp_D = torch.stack([x_t @ D[:, :, k].double() for k in range(nStates)], dim=-1)  # (N, nr, nr)
            x_new += timeStep * torch.sum(x_t.unsqueeze(2) * temp_D, dim=1)

        x_out[:, iStep + 1, :] = x_new

    return x_out




def tapas_rdcm_generate(DCM, options, SNR):
    """
    tapas_rdcm_generate(DCM, options, SNR)

    Generates synthetic fMRI data under a given signal-to-noise ratio (SNR)
    with the fixed hemodynamic convolution kernel.

    Args:
        DCM (dict): Model structure.
        options (dict): Estimation options.
        SNR (float): Signal-to-noise ratio (Inf for noise-free simulation).

    Returns:
        DCM (dict): Updated model structure with synthetic data (Y.y, y, x).
    """

    # --------------------------------------------------------
    # Set dt if specified in options
    if options and 'y_dt' in options and options['y_dt']:
        DCM['Y']['dt'] = options['y_dt']

    r_dt = DCM['Y']['dt'] / DCM['U']['dt']
    N = DCM['U']['u'].shape[1]
    nr = DCM['a'].shape[0]


    # --------------------------------------------------------
    # Generate or use stored HRF
    if ('h' not in options) or (len(options['h']) != DCM['U']['u'].shape[1]):
        options['DCM'] = DCM
        options['conv_dt'] = DCM['U']['dt']
        options['conv_length'] = DCM['U']['u'].shape[0]
        options['conv_full'] = True
        options['h'] = tapas_rdcm_get_convolution_bm(options)

    h = options['h']

    # --------------------------------------------------------
    # Generate neuronal using Euler integration
    DCM_dummy = safe_clone(DCM)
    DCM_dummy['U']['u'] = DCM_dummy['U']['u'][0]
    DCM = tapas_dcm_euler_make_indices(DCM)
    x = tapas_dcm_euler_gen(DCM, DCM['Tp']) #_,x = ...
    # Triplicate output for convolution padding
    x = torch.cat([x,x,x], dim=1)
    
    # --------------------------------------------------------
    # Allocate data array
    B = x.shape[0]
    y = torch.zeros((B, N, nr), dtype=torch.float32, device=x.device)

    # --------------------------------------------------------
    # Convolve neuronal signal with HRF
    H_pad = torch.cat([h.to(x.device), torch.zeros((N * 3 - len(h),), dtype=torch.float32, device=x.device)])
    H_fft = torch.fft.fft(H_pad)
    xi_fft = torch.fft.fft(x.permute(0,2,1).contiguous().view(-1, x.shape[1]), dim=1)
    tmp = torch.fft.ifft(xi_fft * H_fft).real
    tmp = tmp.view(B, nr, N*3).permute(0,2,1)
    y = tmp[:,N:2 * N,:]

    # --------------------------------------------------------
    # Downsample to match acquisition rate
    step = int(round(r_dt))
    y = y[:,::step, :]

    # --------------------------------------------------------
    # Add Gaussian noise according to SNR
    if torch.isfinite(torch.tensor(SNR)) and SNR > 0:
        std_per_region = torch.std(y, dim=0)
        noise = torch.randn_like(y) * std_per_region / SNR
        y_noise = y + noise
    else:
        y_noise = y.clone()

    # --------------------------------------------------------
    # Save outputs
    #DCM['Y']['y'] = y_noise
    #DCM['y'] = y
    #DCM['x'] = x[:,N:2 * N, :]

    return y_noise, y, x[:,N:2 * N, :]



import torch
import torch.fft as fft

def pink_from_normal(x, beta):
    """
    Batched differentiable pink-noise transform.

    Args:
        x (torch.Tensor): Input tensor of shape (N, L, R) — typically standard normal noise.
        beta (float or torch.Tensor): Pinkness exponent β (can be scalar or tensor on same device).

    Returns:
        pink (torch.Tensor): Pink noise of shape (N, L, R), zero-mean and unit-variance per (N,R).
    """
    # Unpack shape
    if len(x.shape) < 3:
        x = x.unsqueeze(0)
    N, L, R = x.shape

    # FFT along time dimension (L)
    Xf = fft.rfft(x, dim=1)  # (N, F, R)

    # Frequency vector for scaling (F bins)
    F = Xf.shape[1]
    freqs = torch.linspace(0.001, 0.5, F, dtype=x.dtype, device=x.device)

    # Compute 1/f^(β/2) scaling factor
    # Shape (1, F, 1) for broadcasting across N and R
    scaling = (1.0 / (freqs ** (beta / 2.0))).view(1, F, 1)

    # Apply scaling in frequency domain
    Xf_scaled = Xf * scaling

    # Inverse FFT to time domain
    pink = fft.irfft(Xf_scaled, n=L, dim=1)

    # Normalize per batch & channel (over time)
    mean = pink.mean(dim=1, keepdim=True)
    std = pink.std(dim=1, keepdim=True) + 1e-8
    pink = (pink - mean) / std

    return pink  # (N, L, R)



def rDCM_generate(T,R,TR,beta,neural_noise,A,scale,transit,decay,epsilon):
    #with torch.no_grad():
    Y = {
         'y': torch.randn(T,R),
         'dt': TR
         }
    
    U = {
         'u': torch.zeros((T*16,R))
         }
    
    args = {
            'a': np.ones((R,R)),
            'c': np.eye(R,R)
            }
    DCM = tapas_rdcm_model_specification(Y, U, args)
    options = tapas_rdcm_set_options(DCM)
    #std =  torch.std(neural_noise, dim=0, keepdim=True)
    #random_noise = torch.randn_like(neural_noise)*(scale*std)
    #random_noise = random_noise*(scale*std)
    #neural_noise = neural_noise + random_noise
    #neural_noise = (neural_noise - torch.mean(neural_noise, dim=0, keepdims=True))/torch.std(neural_noise, dim=0, keepdims=True)
    neural_pink = pink_from_normal(neural_noise, beta)# + random_noise)
    if len(neural_pink.shape) < 3:
        neural_pink = neural_pink.unsqueeze(0)
    DCM['U']['u'] = neural_pink
    DCM['Tp'] = {}
    DCM['Tp']['A'] = A.to(neural_pink.device)
    DCM['Tp']['C'] = 0.05*torch.eye(A.shape[0]).to(neural_pink.device)
    DCM['Tp']['transit'] = transit.to(neural_pink.device)
    DCM['Tp']['decay'] = decay.to(neural_pink.device)
    DCM['Tp']['epsilon'] = epsilon.to(neural_pink.device)
    _, y_gen, x_gen = tapas_rdcm_generate(DCM, options, float('inf'))
    del DCM
    #y_gen.requires_grad = False
        
    return y_gen.squeeze(), x_gen.squeeze()



