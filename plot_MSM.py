import argparse
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os

SETTINGS = [('full_original','10 min, TR = 0.72 s'),('full_subsampled','10 min, TR = 2.0 s'),('half_original','7 min, TR = 0.72 s'),('half_subsampled','7 min, TR = 2.0 s')]
METHODS = ['TimeVAE','TimeGAN','T2IDIFF','DSFM','CRVAE','rDCM-GEN']
NAMES = ['TimeVAE','TimeGAN','T2I-Diff','DSFM','CR-VAE','rDCM-GEN']
COLORS = ['#708197','#9270B4','#D49343','#7DA25B','#C36B65','#177D86']
FOLDERS = {m:m for m in METHODS}  # Edit if your directory names differ.
HCP_FOLDER = 'HCP'

def read_values(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Missing: {path}')
    with open(path, 'rb') as f:
        payload = pickle.load(f)  # Only load trusted pickle files.
    if not isinstance(payload,dict) or 'test_masked_mse' not in payload:
        raise KeyError(f'Missing test_masked_mse in {path}')
    a = np.asarray(payload['test_masked_mse'],dtype=float).ravel()
    if not len(a) or not np.all(np.isfinite(a)):
        raise ValueError(f'Invalid MSE values: {path}')
    return a

def summarize(a):
    return (float(a.mean()), float(a.std(ddof=1)) if len(a)>1 else 0.)

def load(parent,setting,CONFIGS):
    d={}
    for cfg,_ in CONFIGS:
        for method in METHODS:
            file=os.path.join(parent,'MSM',cfg,setting,FOLDERS[method],'results.pkl')
            a=read_values(file)
            d[cfg,method]=summarize(a)
            print(f'{setting} {cfg} {method}: {d[cfg,method][0]:.4f} +/- {d[cfg,method][1]:.4f} (n={len(a)})')
    for cfg in ['heavyweight_small','lightweight_small']:
        file=os.path.join(parent,'MSM',cfg,setting,HCP_FOLDER,'results.pkl')
        if os.path.exists(file):
            d[cfg,'HCP']=summarize(read_values(file))
        else:
            print('Optional HCP baseline not found:',file)
    return d

def plot(d, title, out, stem, CONFIGS):
    """One grouped bar chart; HCP is a seventh method only for small-data settings."""
    plt.rcParams.update({
        'font.family':'DejaVu Sans','font.size':13,
        'axes.labelsize':16,'xtick.labelsize':12,
        'ytick.labelsize':13,'legend.fontsize':12,
        'pdf.fonttype':42,'svg.fonttype':'none'
    })
    fig, ax = plt.subplots(figsize=(15.5, 6.4))
    x = np.arange(len(CONFIGS))
    w = .115
    methods = METHODS + ['HCP']
    names = NAMES + ['Original HCP']
    colors = COLORS + ['#364A60']
    for j, method in enumerate(methods):
        for i, (cfg, _) in enumerate(CONFIGS):
            if (cfg, method) not in d:
                continue  # HCP exists only for heavyweight_small/lightweight_small.
            mean, sd = d[cfg, method]
            pos = x[i] + (j - 3)*w
            ax.bar(pos, mean, w*.94, yerr=sd, capsize=2.5,
                   color=colors[j], edgecolor='white', linewidth=.5,
                   label=names[j] if i == 0 or (method == 'HCP' and cfg == 'heavyweight_small') else None)
    ax.set_xticks(x, [name.replace(' / ', chr(10))+' samples' for _,name in CONFIGS])
    ax.set_ylabel('Masked reconstruction MSE')
    ax.set_ylim(bottom=0)
    ax.set_title(title, fontsize=17, weight='bold', pad=18)
    ax.grid(axis='y',alpha=.2)
    ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(axis='x',length=0,pad=12)
    ax.tick_params(axis='y',length=0)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.29),ncol=7,frameon=False)
    fig.subplots_adjust(top=.73,bottom=.20,left=.085,right=.985)
    os.makedirs(out,exist_ok=True)
    fig.savefig(os.path.join(out, f'{stem}.png'),dpi=600,bbox_inches='tight')
    plt.close(fig)
    
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--parent_path", default=r'/home/student1/Desktop/Charalampos_Lamprou/rDCM-GEN_results_HCP_2026')
    parser.add_argument("--output-dir", default=r'/home/student1/Desktop/Charalampos_Lamprou/rDCM-GEN/plots/MSM')
    args=parser.parse_args()
    for setting,title in SETTINGS:
        if setting == "full_original" or setting == "full_subsampled":
            CONFIGS = [('heavyweight_large','5 layers / 18,375'),('lightweight_large','2 layers / 18,375'),('heavyweight_small','5 layers / 5,000'),('lightweight_small','2 layers / 5,000')]
        elif setting == "half_original":
            CONFIGS = [('heavyweight_large','5 layers / 17,955'),('lightweight_large','2 layers / 17,955'),('heavyweight_small','5 layers / 5,000'),('lightweight_small','2 layers / 5,000')]
        elif setting == "half_subsampled":
            CONFIGS = [('heavyweight_large','5 layers / 18,345'),('lightweight_large','2 layers / 18,345'),('heavyweight_small','5 layers / 5,000'),('lightweight_small','2 layers / 5,000')]
        plot(load(args.parent_path,setting,CONFIGS),title,args.output_dir,'msm_'+setting,CONFIGS)

if __name__=='__main__':
    main()
