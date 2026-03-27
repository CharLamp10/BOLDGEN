import numpy as np
import os


save_path = r'C:\Users\100063082\Desktop\SSL_FC_matrix_GNN_data\HCP_2026'
os.makedirs(save_path, exist_ok=True)

path_data_SCH = r'E:\HCP_2026_nilearn\ROISignals_SCH'

data_dir = os.listdir(path_data_SCH)
Times_AICHA = []
Times_AAL3 = []
Times_BN = []
Times_SCH = []
Times_AAL = []
Times_A424 = []
names_train = []
for data in data_dir:
    feature_dir_sch = os.path.join(path_data_SCH, data, 'ROISignals.npy')
    feature_sch = np.load(feature_dir_sch)
    Times_SCH.append(feature_sch)
    names_train.append(data)

np.savez(os.path.join(save_path,'data_HCP_SCH'),*Times_SCH)
with open(os.path.join(save_path,'names.txt'), 'w') as f:
    for item in names_train:
        f.write("%s\n" % item)    
