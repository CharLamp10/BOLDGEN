import os
from nilearn.input_data import  NiftiLabelsMasker
import pandas as pd
import numpy as np
import nibabel as nib

atlas = 'SCH'

if atlas == "AAL3":
    path_atlas = r'C:\Users\100063082\Desktop\atlases\AAL3v1.nii'
elif atlas == 'AICHA':
    path_atlas = r'C:\Users\100063082\Desktop\atlases\AICHA.nii'
elif atlas == 'BN':
    path_atlas = r'C:\Users\100063082\Desktop\atlases\BN_Atlas_246_2mm.nii'
elif atlas == "SCH":
    path_atlas = r'C:\Users\100063082\Desktop\atlases\Schaefer2018_100Parcels_17Networks_order_FSLMNI152_2mm.nii'
elif atlas == "A424":
    path_atlas = r'C:\Users\100063082\Desktop\atlases\A424+2mm.nii'
elif atlas == "AAL":
    path_atlas = r'C:\Users\100063082\Desktop\atlases\ROI_MNI_V4.nii'


path = r'D:\ADHD200-fmriprep\fmriprep'
path_phenotypic = r'C:\Users\100063082\Desktop\prepare_ADHD200_ABIDE_phenotypics\phenotypic_ADHD200_fmriprep.csv'
phenotypic = pd.read_csv(path_phenotypic,encoding="latin-1")
ids = phenotypic["ScanDirID"].values

path_save = r'E:\ADHD200_nilearn_rDCM-GEN\fmriprep\ROIsignals_' + atlas
if not os.path.exists(path_save):
    os.makedirs(path_save,exist_ok=True)

dire = os.listdir(path);
for i in range(len(dire)):
    if 'sub' in dire[i] and os.path.isdir(os.path.join(path,dire[i])):
        path_subject = os.path.join(path,dire[i])
        dir_sucject = os.listdir(path_subject)
        for ses in dir_sucject:
            if 'ses' in ses:
                dir_session = os.listdir(os.path.join(path_subject, ses))
                path_func = os.path.join(path_subject, ses, 'func')
                if os.path.exists(path_func):
                    dir_func = os.listdir(path_func)
                    for j in range(len(dir_func)):
                        file = dir_func[j]
                        if 'smoothAROMAnonaggr' in file and '.gz' in file:
                            pos_space = file.find("_space")   
                            pos_underscore = file.find("_")
                            path_data = os.path.join(path_func,file)
                            path_confounds = os.path.join(path_func,file[:pos_space] + '_desc-confounds_regressors.tsv')
                            try:
                                confounds = pd.read_table(path_confounds).iloc[:,:2]
                            except:
                                continue
                            if pos_underscore == -1:
                                raise ValueError("Invalid filename format")
                            
                            try:
                                id_val = int(file[4:pos_underscore])
                            except ValueError:
                                raise ValueError("Invalid ID extraction")
                        
                            path_save_new = os.path.join(path_save,file[:pos_space])
                            imgs = nib.load(path_data)
                            TR = imgs.header.get_zooms()[-1]
                            if os.path.isfile(path_data) and not os.path.exists(os.path.join(path_save_new,'ROISignals.npy')):
                                try:
                                    masker = NiftiLabelsMasker(
                                            labels_img=path_atlas,
                                            standardize="zscore",
                                            detrend="True",
                                            memory="nilearn_cache",
                                            verbose=0,
                                        )
                                    time_series = masker.fit_transform(path_data, confounds=confounds)
                                    if not os.path.exists(path_save_new):
                                        os.mkdir(path_save_new)
                                    np.save(os.path.join(path_save_new,'ROISignals.npy'), time_series)
                                    np.save(os.path.join(path_save_new,'TR.npy'), TR)
                                except:
                                    continue
                                
