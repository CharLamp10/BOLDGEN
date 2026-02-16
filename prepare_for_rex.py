import numpy as np
import os
import torch
import pandas as pd



def main(config):
    for method in config['methods']:
        for tau in config['taus']:
            out1 = torch.load(os.path.join(config['path'], 'SimCLR2', method, 'tau'+str(tau), 'outputs1.pt')).numpy()
            out2 = torch.load(os.path.join(config['path'], 'SimCLR2', method, 'tau'+str(tau), 'outputs2.pt')).numpy()
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
            os.makedirs(os.path.join(config['path'], 'ReX_files2', 'tau' + str(tau)), exist_ok=True)
            out.to_csv(os.path.join(config['path'], 'ReX_files2', 'tau' + str(tau),'rex_' + method + '.csv'))

                    
if __name__ == "__main__":
    config = {}
    config['path'] = r'/home/student1/Desktop/Charalampos_Lamprou/rDCM-GEN_results'
    config['methods'] = ['CRVAE', 'Noise-3dB', 'Noise-6dB', 'rDCM-GEN']
    config['length'] = 200
    config['taus'] = [0.05, 0.1, 0.15]
    main(config)


