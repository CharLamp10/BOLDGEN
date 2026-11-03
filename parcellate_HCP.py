"""Parcellate preprocessed HCP fMRI data into regional time series."""
import argparse

import os
from nilearn.input_data import  NiftiLabelsMasker
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Parcellate preprocessed HCP fMRI data into regional time series.')
    parser.add_argument("--path_atlas", type=str, required=True)
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--path_save_prefix", type=str, required=True)
    parser.add_argument("--atlas", default='SCH')
    args = parser.parse_args()

    #FIX IF CONFOUNDS IS MISSING
    atlas = args.atlas
    path_atlas = args.path_atlas
    


    path = args.path
    directory = os.listdir(path)
    directory = directory[::-1]

    path_save = args.path_save_prefix + atlas
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


if __name__ == "__main__":
    main()
