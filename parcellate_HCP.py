import os
from nilearn.input_data import  NiftiLabelsMasker
import numpy as np

#FIX IF CONFOUNDS IS MISSING
atlas = 'SCH'

if atlas == "AAL3":
    path_atlas = r'C:\Users\100063082\Desktop\atlases\AAL3v1.nii'
elif atlas == "AICHA":
    path_atlas = r'C:\Users\100063082\Desktop\atlases\AICHA.nii'
elif atlas == "BN":
    path_atlas = r'C:\Users\100063082\Desktop\atlases\BN_Atlas_246_2mm.nii'
elif atlas == "SCH":
    path_atlas = r'C:\Users\100063082\Desktop\atlases\Schaefer2018_100Parcels_17Networks_order_FSLMNI152_2mm.nii'
elif atlas == "A424":
    path_atlas = r'C:\Users\100063082\Desktop\atlases\A424+2mm.nii'
elif atlas == "AAL":
    path_atlas = r'C:\Users\100063082\Desktop\atlases\ROI_MNI_V4.nii'
    


path = r'D:\HCP_new'
directory = os.listdir(path)
directory = directory[::-1]

path_save = r'E:\HCP_2026_nilearn\ROIsignals_' + atlas
if not os.path.exists(path_save):
    os.makedirs(path_save,exist_ok=True)

for file in directory:
    if os.path.isdir(os.path.join(path, file)):
        path_new = os.path.join(path, file, 'MNINonLinear', 'Results')
        directory_new = os.listdir(path_new)
        for file_new in directory_new:
            if 'LR' in file_new or 'RL' in file_new:
                fmri_path = os.path.join(path_new, file_new, file_new + '_hp2000_clean_rclean_tclean.nii.gz') 
                path_save_new = os.path.join(path_save, file + '_' + file_new)
                if not os.path.exists(os.path.join(path_save_new,'ROISignals.npy')):
                    try:
                        masker = NiftiLabelsMasker(
                                labels_img=path_atlas,
                                standardize="zscore",
                                detrend="True",
                                memory="nilearn_cache",
                                verbose=0,
                            )
                        time_series = masker.fit_transform(fmri_path)
                        if not os.path.exists(path_save_new):
                            os.mkdir(path_save_new)
                        np.save(os.path.join(path_save_new,'ROISignals.npy'), time_series)
                    except:
                        continue
                    print(file)


