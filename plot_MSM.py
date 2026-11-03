import argparse
import csv
import itertools
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel

SETTINGS = [
    ('full_original', '10 min, TR = 0.72 s'),
    ('full_subsampled', '10 min, TR = 2.0 s'),
    ('half_original', '7 min, TR = 0.72 s'),
    ('half_subsampled', '7 min, TR = 2.0 s'),
]
METHODS = ['TimeVAE', 'TimeGAN', 'T2I-DIFF', 'DSFM', 'CR-VAE', 'rDCM-GEN']
NAMES = ['TimeVAE', 'TimeGAN', 'T2I-Diff', 'DSFM', 'CR-VAE', 'rDCM-GEN']
COLORS = ['#708197', '#9270B4', '#D49343', '#7DA25B', '#C36B65', '#177D86']
FOLDERS = {m: m for m in METHODS}
HCP_FOLDER = 'HCP'
DISPLAY = dict(zip(METHODS + ['HCP'], NAMES + ['HCP']))


def configs_for(setting):
    if setting in ('full_original', 'full_subsampled'):
        large = '18,375'
    elif setting == 'half_original':
        large = '17,955'
    else:
        large = '18,345'
    return [
        ('heavyweight_large', f'5 layers / {large}'),
        ('lightweight_large', f'2 layers / {large}'),
        ('heavyweight_small', '5 layers / 5,000'),
        ('lightweight_small', '2 layers / 5,000'),
    ]


def read_values(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing: {path}')
    with path.open('rb') as f:
        payload = pickle.load(f)  # Only load trusted pickle files.
    if not isinstance(payload, dict) or 'test_masked_mse' not in payload:
        raise KeyError(f'Missing test_masked_mse in {path}')
    a = np.asarray(payload['test_masked_mse'], dtype=float).ravel()
    if not len(a) or not np.all(np.isfinite(a)):
        raise ValueError(f'Invalid MSE values: {path}')
    return a


def summarize(a):
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.


def load(parent, setting, configs):
    """Retain original per-run values so paired t-tests remain possible."""
    values = {}
    for cfg, _ in configs:
        for method in METHODS:
            path = Path(parent) / 'MSM' / cfg / setting / FOLDERS[method] / 'results.pkl'
            a = read_values(path)
            values[cfg, method] = a
            avg, sd = summarize(a)
            print(f'{setting} {cfg} {method}: {avg:.4f} +/- {sd:.4f} (n={len(a)})')
    for cfg in ('heavyweight_small', 'lightweight_small'):
        path = Path(parent) / 'MSM' / cfg / setting / HCP_FOLDER / 'results.pkl'
        if path.is_file():
            a = read_values(path)
            values[cfg, 'HCP'] = a
            avg, sd = summarize(a)
            print(f'{setting} {cfg} HCP: {avg:.4f} +/- {sd:.4f} (n={len(a)})')
        else:
            print('Optional HCP baseline not found:', path)
    return values


def effective_key(cfg, method):
    if method == 'HCP':
        return ('heavyweight_small' if cfg.startswith('heavyweight') else 'lightweight_small', 'HCP')
    return cfg, method


def plot_bars(values, out, stem, configs, dpi=600):
    """Original grouped MSE bar plot; HCP reference reused at matched depth."""
    with plt.rc_context({
        'font.family': 'DejaVu Sans', 'font.size': 21, 'axes.labelsize': 25,
        'xtick.labelsize': 21, 'ytick.labelsize': 21, 'legend.fontsize': 20,
        'pdf.fonttype': 42, 'svg.fonttype': 'none',
    }):
        fig, ax = plt.subplots(figsize=(20, 9))
        x, w = np.arange(len(configs)), .115
        methods = METHODS + ['HCP']
        colors = COLORS + ['#364A60']
        for j, method in enumerate(methods):
            label_added = False
            for i, (cfg, _) in enumerate(configs):
                key = effective_key(cfg, method)
                if key not in values:
                    continue
                mean, sd = summarize(values[key])
                ax.bar(x[i] + (j-3)*w, mean, w*.94, yerr=sd, capsize=3.5,
                       color=colors[j], edgecolor='white', linewidth=.5,
                       label=DISPLAY[method] if not label_added else None)
                label_added = True
        ax.set_xticks(x, [name.replace(' / ', '\n') + ' samples' for _, name in configs])
        ax.tick_params(axis='x', labelsize=21, length=0, pad=17)
        ax.tick_params(axis='y', labelsize=21, length=0)
        ax.set_ylabel('MSE', fontsize=25)
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.15), ncol=7,
                  frameon=False, fontsize=20, columnspacing=.8,
                  handlelength=1.2, handletextpad=.4)
        fig.subplots_adjust(top=.74, bottom=.22, left=.09, right=.98)
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f'{stem}.png', dpi=dpi, bbox_inches='tight')
        fig.savefig(out / f'{stem}.pdf', bbox_inches='tight')
        plt.close(fig)


def paired_test(a, b):
    """Uncorrected, two-sided paired t-test. No silent truncation of arrays."""
    if len(a) != len(b):
        return np.nan, np.nan, 'unequal_repetition_counts'
    if len(a) < 2:
        return np.nan, np.nan, 'insufficient_pairs'
    d = a - b
    # In the constant-difference case, the standard paired t-test is undefined.
    tolerance = 16 * np.finfo(float).eps * max(1., float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    if np.ptp(d) <= tolerance:
        return np.nan, np.nan, 'zero_variance_paired_differences'
    result = ttest_rel(a, b, alternative='two-sided')
    return float(result.statistic), float(result.pvalue), 'ok'


def compute_tests(values, setting, configs):
    tests = []
    for cfg, _ in configs:
        methods = [m for m in METHODS + ['HCP'] if effective_key(cfg, m) in values]
        for a, b in itertools.combinations(methods, 2):
            key_a, key_b = effective_key(cfg, a), effective_key(cfg, b)
            x, y = values[key_a], values[key_b]
            t, p, status = paired_test(x, y)
            tests.append({
                'setting': setting, 'configuration': cfg,
                'method_a': a, 'method_b': b,
                'n_a': len(x), 'n_b': len(y),
                'mean_difference_a_minus_b': float(x.mean() - y.mean()),
                't_stat': t, 'p_raw': p, 'status': status,
            })
            print(f'  {setting} | {cfg} | {a} vs {b}: '
                  f'p={p:.4g}, t={t:.4g} ({status}; n={len(x)}/{len(y)})')
    return tests


def pvalue_label(p, alpha):
    if not np.isfinite(p):
        return 'NA'
    number = f'{p:.2g}' if p >= .001 else f'{p:.1e}'
    stars = '***' if p < .001 else '**' if p < .01 else '*' if p < alpha else ''
    return stars#f'{number}\n{stars}' if stars else number


def plot_pvalue_matrices(tests, values, setting, configs, output_dir, alpha=.05, dpi=600):
    """One symmetric pairwise p-value heatmap per configuration (four per setting)."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for cfg, _ in configs:
        methods = [m for m in METHODS + ['HCP'] if effective_key(cfg, m) in values]
        n = len(methods)
        indices = {m: i for i, m in enumerate(methods)}
        matrix = np.full((n, n), np.nan)
        for row in tests:
            if row['setting'] == setting and row['configuration'] == cfg:
                i, j = indices[row['method_a']], indices[row['method_b']]
                matrix[i, j] = matrix[j, i] = row['p_raw']

        cmap = plt.get_cmap('Blues_r').copy()
        cmap.set_bad('#e9edf2')
        masked = np.ma.masked_invalid(matrix)
        fig_size = max(8., 1.2*n + 2.5)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size*.86))
        im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=alpha, interpolation='nearest')
        names = [DISPLAY[m] for m in methods]
        ax.set_xticks(range(n), names, rotation=45, ha='right', fontsize=24)
        ax.set_yticks(range(n), names, fontsize=24)
        ax.set_xticks(np.arange(-.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-.5, n, 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=2)
        ax.tick_params(which='minor', bottom=False, left=False)
        ax.tick_params(which='major', length=0)
        for i in range(n):
            for j in range(n):
                p = matrix[i, j]
                label = '—' if i == j else pvalue_label(p, alpha)
                ax.text(j, i, label, ha='center', va='center', fontsize=24,
                        color='white' if np.isfinite(p) and p < alpha*.30 else '#172536')
        cbar = fig.colorbar(im, ax=ax, fraction=.045, pad=.035)
        cbar.set_label('p-value', fontsize=24)
        cbar.ax.tick_params(labelsize=20)
        fig.tight_layout()
        stem = f'msm_{setting}_{cfg}_pvalues'
        fig.savefig(output / f'{stem}.png', dpi=dpi, bbox_inches='tight')
        fig.savefig(output / f'{stem}.pdf', bbox_inches='tight')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Plot MSM error and uncorrected paired t-test p-value matrices')
    parser.add_argument('--parent_path', default='/home/student1/Desktop/Charalampos_Lamprou/rDCM-GEN_results_HCP_2026')
    parser.add_argument('--output-dir', default='/home/student1/Desktop/Charalampos_Lamprou/rDCM-GEN/plots/MSM')
    parser.add_argument('--alpha', type=float, default=.05)
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()
    if not 0 < args.alpha < 1 or args.dpi < 1:
        parser.error('Require 0 < alpha < 1 and dpi > 0')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_tests = []
    for setting, _ in SETTINGS:
        configs = configs_for(setting)
        values = load(args.parent_path, setting, configs)
        plot_bars(values, out, f'msm_{setting}', configs, args.dpi)
        tests = compute_tests(values, setting, configs)
        all_tests.extend(tests)
        plot_pvalue_matrices(tests, values, setting, configs, out, args.alpha, args.dpi)
    with (out / 'paired_t_tests.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_tests[0]))
        writer.writeheader()
        writer.writerows(all_tests)
    (out / 'statistical_notes.txt').write_text(
        'Two-sided paired t-tests on test_masked_mse repetition values; raw p-values, no multiple-testing correction.\n'
        'Values must correspond to identical runs/seeds in identical order across methods; equal lengths do not establish valid pairing.\n'
        'Undefined tests (n<2 or constant paired differences) and unequal-length tests are reported as NA.\n'
        'HCP baseline for each large configuration is drawn from the matched-depth small setting, as in the original bar plots.\n'
        'Each matrix uses the six synthesis methods and HCP when available.\n'
        'Repeated or dependent runs can make ordinary paired t-test p-values exploratory.\n', encoding='utf-8'
    )
    print(f'\nSaved 4 bar charts, 16 p-value heatmaps (PNG + PDF), and paired_t_tests.csv to {out}')


if __name__ == '__main__':
    main()