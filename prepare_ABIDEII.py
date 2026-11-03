"""Harmonize and separate ABIDE II rs-fMRI time series by diagnostic label."""
import argparse

import os
import numpy as np
from scipy.signal import resample


def main():
    parser = argparse.ArgumentParser(description='Harmonize and separate ABIDE II rs-fMRI time series by diagnostic label.')
    parser.add_argument("--roi_path", type=str, required=True)
    parser.add_argument("--labels_path", type=str, required=True)
    parser.add_argument("--names_path", type=str, required=True)
    parser.add_argument("--asd_output", type=str, required=True)
    parser.add_argument("--nc_output", type=str, required=True)
    parser.add_argument("--newTR", type=float, default=1.5)
    args = parser.parse_args()

    path = args.roi_path
    dire = os.listdir(path)

    labels = np.load(args.labels_path)
    names = []
    with open(args.names_path, 'r') as f:
        for line in f:
            names.append(line.strip())

    newTR = args.newTR

    T_min = 10000
    y_all_1 = np.zeros((len(dire), 1000, 100))
    y_all_0 = np.zeros((len(dire), 1000, 100))
    c0 = 0
    c1 = 0
    for file in dire:
        try:
            y = np.load(os.path.join(path, file, 'ROISignals.npy'))
            TR = np.load(os.path.join(path, file, 'TR.npy'))
            rate = newTR/TR
            if rate != 1:
                new_samples = int(np.round(y.shape[0] / rate))
                y = resample(y, new_samples)
            y = (y - np.mean(y, axis=0, keepdims=True))/np.std(y, axis=0, keepdims=True)
            if y.shape[0] < T_min and y.shape[0] >= 200:
                T_min = y.shape[0]
        
            pos = np.where(np.array(names) == file[:9])[0]
            try:
                label = labels[pos[0]]
            except:
                label = labels[pos]
            if label == 1 and y.shape[0] >= 200:
                y_all_1[c1,:y.shape[0],:] = y
                c1 = c1 + 1
            elif label ==0 and y.shape[0] >= 200:
                y_all_0[c0,:y.shape[0],:] = y
                c0 = c0 + 1
        except:
            continue
    
    y_all_1 = y_all_1[:,:T_min,:]
    y_all_0 = y_all_0[:,:T_min,:]

    pos = np.where(np.all(y_all_1 == 0, axis=1))
    y_all_1 = y_all_1[:pos[0][0]]

    pos = np.where(np.all(y_all_0 == 0, axis=1))
    y_all_0 = y_all_0[:pos[0][0]]

    np.save(args.asd_output, y_all_1)
    np.save(args.nc_output, y_all_0)

    

    


if __name__ == "__main__":
    main()
