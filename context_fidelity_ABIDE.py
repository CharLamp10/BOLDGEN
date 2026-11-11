"""Compute Context-FID between empirical and generated ABIDE time series."""
import argparse

parent_path = r'/home/student1/Desktop/Charalampos_Lamprou/Diffusion-TS'
import sys
sys.path.append(parent_path)
import os
import torch
import numpy as np
from Models.ts2vec.ts2vec import TS2Vec
from Utils.context_fid import calculate_fid
import pickle



def main(config):
    for method in config['methods']:               
        path = config['path_data']

        data_asd = np.load(os.path.join(path, 'ABIDEI_ASD_nilearn.npy'))
        data_nc = np.load(os.path.join(path, 'ABIDEI_NC_nilearn.npy'))
                      
        train_real = np.concatenate((data_asd[:300], data_nc[:300]), axis=0)
        test_real = np.concatenate((data_asd[300:], data_nc[300:]), axis=0)
        
        
        if method == "CR-VAE":
            test_gen = torch.load(os.path.join(config['path_augmented_crvae'], 'augmented_CR-VAE_ABIDEI.pt'))
            test_gen = test_gen.view(-1, test_gen.shape[2], test_gen.shape[3])
            test_gen = test_gen.numpy()
        elif method == "BOLDGEN":
            test_gen = torch.load(os.path.join(config['path_augmented_boldgen'], 'augmented_BOLDGEN_ABIDEI.pt'))
            test_gen = test_gen.view(-1, test_gen.shape[2], test_gen.shape[3])
            test_gen = test_gen.numpy()
        elif method == "TimeVAE":
            test_gen_asd = np.load(os.path.join(config['path_augmented_timevae'], 'ABIDEI_ASD', 'timeVAE_ABIDEI_ASD_prior_samples.npy'))
            test_gen_nc = np.load(os.path.join(config['path_augmented_timevae'], 'ABIDEI_NC', 'timeVAE_ABIDEI_NC_prior_samples.npy'))
            test_gen = np.concatenate((test_gen_asd, test_gen_nc), axis=0)
            test_gen = (test_gen - test_gen.mean(axis=2, keepdims=True))/test_gen.std(axis=2, keepdims=True)
        elif method == "TimeGAN":
            test_gen_asd = np.load(os.path.join(config['path_augmented_timegan'], 'augmented_TimeGAN_ABIDEI_ASD.npy'))
            test_gen_nc = np.load(os.path.join(config['path_augmented_timegan'], 'augmented_TimeGAN_ABIDEI_NC.npy'))
            test_gen = np.concatenate((test_gen_asd, test_gen_nc), axis=0)
            test_gen = (test_gen - test_gen.mean(axis=2, keepdims=True))/test_gen.std(axis=2, keepdims=True)
        elif method == "T2I-Diff":
            test_gen = np.load(os.path.join(config['path_augmented_t2idiff'], 'ABIDEI', 'gen_sig_ABIDEI_fold1.npy'))
            test_gen = (test_gen - test_gen.mean(axis=2, keepdims=True))/test_gen.std(axis=2, keepdims=True)
        elif method == "DSFM":
            test_gen_asd = np.load(os.path.join(config['path_augmented_dsfm'], 'ABIDEI', 'samples', 'asd_bold.npy'))
            test_gen_nc = np.load(os.path.join(config['path_augmented_dsfm'], 'ABIDEI', 'samples', 'control_bold.npy'))
            test_gen = np.concatenate((test_gen_asd, test_gen_nc), axis=0)
            test_gen = (test_gen - test_gen.mean(axis=2, keepdims=True))/test_gen.std(axis=2, keepdims=True)
        
        scores = []
        for i in range(10):
            rng = np.random.default_rng()
            test_gen = test_gen[rng.permutation(test_gen.shape[0])]
            test_gen = test_gen[:test_real.shape[0],:,:]
            
            encoder = TS2Vec(
                input_dims=train_real.shape[-1],
                output_dims=320,
                device=3)  
            encoder.fit(train_real)
            z_real = encoder.encode(test_real, encoding_window="full_series")
            z_generated = encoder.encode(test_gen, encoding_window="full_series")
            
            score = calculate_fid(z_real, z_generated)
            scores.append(score)
    
        results = {}
        results['context_FID'] = scores
        
        if config['save_results']:
            os.makedirs(os.path.join(config['path_save'], 'cFID', 'ABIDEI', method), exist_ok=True)
            with open(os.path.join(config['path_save'], 'cFID', 'ABIDEI', method, 'results.pkl'), 'wb') as f:
                pickle.dump(results,f)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute Context-FID between empirical and generated ABIDE time series.')
    parser.add_argument('--path_data', type=str, required=True)
    parser.add_argument('--path_save', type=str, required=True)
    parser.add_argument('--path_augmented_boldgen', type=str, required=True)
    parser.add_argument('--path_augmented_crvae', type=str, required=True)
    parser.add_argument('--path_augmented_timevae', type=str, required=True)
    parser.add_argument('--path_augmented_timegan', type=str, required=True)
    parser.add_argument('--path_augmented_t2idiff', type=str, required=True)
    parser.add_argument('--path_augmented_dsfm', type=str, required=True)
    parser.add_argument('--methods', nargs="+", type=str, default=['CR-VAE', 'TimeVAE', 'TimeGAN','T2I-Diff','DSFM','BOLDGEN'])
    parser.add_argument('--TR', type=float, default=0.72)
    parser.add_argument('--newTR', type=float, default=2.0)
    parser.add_argument('--R', type=int, default=100)
    parser.add_argument('--atlas', type=str, default='SCH')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--save_results', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    config = {}
    config['path_data'] = args.path_data
    config['path_save'] = args.path_save
    config['path_augmented_boldgen'] = args.path_augmented_boldgen
    config['path_augmented_crvae'] = args.path_augmented_crvae
    config['path_augmented_timevae'] = args.path_augmented_timevae
    config['path_augmented_timegan'] = args.path_augmented_timegan
    config['path_augmented_t2idiff'] = args.path_augmented_t2idiff
    config['path_augmented_dsfm'] = args.path_augmented_dsfm
    config['methods'] = args.methods
    config['TR'] = args.TR
    config['newTR'] = args.newTR
    config['R'] = args.R
    config['atlas'] = args.atlas
    config['batch_size'] = args.batch_size
    config['weight_decay'] = args.weight_decay
    config['save_results'] = args.save_results
    main(config)