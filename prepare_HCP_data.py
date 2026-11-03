"""Collect parcellated HCP subject arrays into a consolidated dataset."""
import argparse

import numpy as np
import os



def main():
    parser = argparse.ArgumentParser(description='Collect parcellated HCP subject arrays into a consolidated dataset.')
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    path_data_SCH = args.data_path

    data_dir = os.listdir(path_data_SCH)
    Times_SCH = []
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


if __name__ == "__main__":
    main()
