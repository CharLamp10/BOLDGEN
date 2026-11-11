"""Export subject-level contrastive embeddings to ReX-compatible CSV files."""
import argparse

import numpy as np
import os
import torch
import pandas as pd



def main(config):
    for method in config['methods']:
        for tau in config['taus']:
            out1 = torch.load(os.path.join(config['path'], 'SimCLR_embeddings', config['case'], method, 'tau'+str(tau), 'outputs1.pt')).numpy()
            out2 = torch.load(os.path.join(config['path'], 'SimCLR_embeddings', config['case'], method, 'tau'+str(tau), 'outputs2.pt')).numpy()
            out = np.zeros((2*out1.shape[0], out1.shape[1]))
            subIDs = []
            visits = []
            count = 0
            for i in range(0,out1.shape[0]):
                out[count] = out1[i,:]
                count += 1
                out[count] = out2[i,:]
                count += 1
                if i < 10:
                    subIDs.append('sub00' + str(i))
                    subIDs.append('sub00' + str(i))
                elif i >= 10 and i < 100:
                    subIDs.append('sub0' + str(i))
                    subIDs.append('sub0' + str(i))
                else:
                    subIDs.append('sub' + str(i))
                    subIDs.append('sub' + str(i))
                visits.append('time1')
                visits.append('time2')
            
            columns = []
            for i in range(out1.shape[1]):
                columns.append('ROI.' + str(i))
                
            subIDs = np.expand_dims(np.array(subIDs), axis=1)
            visits = np.expand_dims(np.array(visits), axis=1)
            subIDs = pd.DataFrame(subIDs, columns=['subID'])
            visits = pd.DataFrame(subIDs, columns=['visit'])
            out = pd.DataFrame(out, columns=columns)
            out = pd.concat((subIDs,visits,out), axis=1)
            os.makedirs(os.path.join(config['path'], 'ReX_files', config['case'], 'tau' + str(tau)), exist_ok=True)
            out.to_csv(os.path.join(config['path'], 'ReX_files', config['case'], 'tau' + str(tau),'rex_' + method + '.csv'))

                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export subject-level contrastive embeddings to ReX-compatible CSV files.')
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--methods', nargs="+", type=str, default=['CR-VAE','BOLDGEN'])
    parser.add_argument('--taus', nargs="+", type=float, default=[0.05, 0.1, 0.15])
    parser.add_argument('--case', type=str, default="full_subsampled")
    args = parser.parse_args()
    config = {}
    config['path'] = args.path
    config['methods'] = args.methods
    config['taus'] = args.taus
    config['case'] = args.case
    main(config)


