"""Plot Context-FID means/SD and raw paired t-test p-value heatmaps.

Pairing assumes the same repetition/seed ordering across methods within each
setting. Equal array lengths do not establish paired experimental designs.
Only load trusted results.pkl files.
"""
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
    ('full_original', '10 min\nTR = 0.72 s'),
    ('full_subsampled', '10 min\nTR = 2.0 s'),
    ('half_original', '7 min\nTR = 0.72 s'),
    ('half_subsampled', '7 min\nTR = 2.0 s'),
    ('ABIDEI', 'ABIDE I'),
]
METHODS = ['TimeVAE', 'TimeGAN', 'T2I-Diff', 'DSFM', 'CR-VAE', 'BOLDGEN']
COLORS = {
    'TimeVAE': '#718096', 'TimeGAN': '#8C6BB1', 'T2I-Diff': '#D59443',
    'DSFM': '#7CA35A', 'CR-VAE': '#C46763', 'BOLDGEN': '#167B84',
}
FOLDER_NAMES = {method: method for method in METHODS}


def load_stats(parent):
    """Return means and SDs for barplot plus raw arrays for paired comparisons."""
    means = {m: [] for m in METHODS}
    stds = {m: [] for m in METHODS}
    raw = {}
    for folder, _ in SETTINGS:
        for method in METHODS:
            file = Path(parent) / 'cFID' / folder / FOLDER_NAMES[method] / 'results.pkl'
            if not file.is_file():
                raise FileNotFoundError(f'Missing file: {file}')
            with file.open('rb') as f:
                result = pickle.load(f)  # Trusted files only.
            if not isinstance(result, dict) or 'context_FID' not in result:
                raise ValueError(f"Expected dict with 'context_FID' in {file}")
            values = np.asarray(result['context_FID'], dtype=float).ravel()
            if values.size == 0 or not np.all(np.isfinite(values)):
                raise ValueError(f'Empty/non-finite context_FID values in {file}')
            raw[folder, method] = values
            means[method].append(float(values.mean()))
            stds[method].append(float(values.std(ddof=1)) if len(values) > 1 else 0.)
            print(f'{folder:17s} {method:9s}: {means[method][-1]:8.3f} +/- '
                  f'{stds[method][-1]:.3f} (n={len(values)})')
    return means, stds, raw


def paired_tests(raw):
    rows = []
    for folder, _ in SETTINGS:
        print(f'\n{folder}: raw two-sided paired t-tests')
        for a, b in itertools.combinations(METHODS, 2):
            x, y = raw[folder, a], raw[folder, b]
            row = dict(setting=folder, method_a=a, method_b=b,
                       n_a=len(x), n_b=len(y), mean_difference=np.nan,
                       t_stat=np.nan, p_raw=np.nan, status='ok')
            if len(x) != len(y):
                row['status'] = 'unequal_repetition_counts'
            elif len(x) < 2:
                row['status'] = 'insufficient_pairs'
            else:
                diff = x - y
                row['mean_difference'] = float(diff.mean())
                # An exactly constant paired difference makes the usual t-test undefined.
                tol = 16 * np.finfo(float).eps * max(1., np.max(np.abs(x)), np.max(np.abs(y)))
                if np.ptp(diff) <= tol:
                    row['status'] = 'zero_variance_paired_differences'
                else:
                    res = ttest_rel(x, y, alternative='two-sided')
                    row['t_stat'], row['p_raw'] = float(res.statistic), float(res.pvalue)
            rows.append(row)
            print(f'  {a:9s} vs {b:9s}: p={row["p_raw"]:.4g} [{row["status"]}]')
    return rows


def plot(means, stds, output_dir, prefix='context_fid', dpi=600):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({
        'font.family': 'DejaVu Sans', 'font.size': 15,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    }):
        fig, ax = plt.subplots(figsize=(11.8, 5.4))
        x = np.arange(len(SETTINGS))
        width = 0.13
        for j, method in enumerate(METHODS):
            positions = x + (j - (len(METHODS) - 1) / 2) * width
            ax.bar(positions, means[method], width * 0.94, yerr=stds[method],
                   capsize=2.3, color=COLORS[method], label=method,
                   edgecolor='white', linewidth=0.5,
                   error_kw={'elinewidth': 0.9, 'capthick': 0.9})
        ax.set_yscale('log')
        ax.set_ylim(1, 150)
        ax.set_yticks([1, 2, 5, 10, 20, 50, 100])
        ax.set_yticklabels(['1', '2', '5', '10', '20', '50', '100'])
        ax.set_ylabel('Context-FID')
        ax.set_xticks(x, [lab for _, lab in SETTINGS])
        ax.tick_params(axis='x', length=0, pad=10, labelsize=14)
        ax.tick_params(axis='y', length=0, labelsize=14)
        ax.grid(axis='y', which='major', alpha=0.20, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14),
                  ncol=6, frameon=False, fontsize=14)
        fig.subplots_adjust(left=0.08, right=0.99, bottom=0.15, top=0.82)
        fig.savefig(output / f'{prefix}.png', dpi=dpi, bbox_inches='tight')
        plt.close(fig)


def plot_pvalue_matrices(tests, output_dir, alpha=.05, dpi=600):
    """Five symmetric matrices of raw paired t-test p-values, one per setting."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    n = len(METHODS)
    indices = {method: i for i, method in enumerate(METHODS)}
    with plt.rc_context({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'svg.fonttype': 'none'}):
        for setting, _ in SETTINGS:
            matrix = np.full((n, n), np.nan)
            for row in tests:
                if row['setting'] == setting:
                    i, j = indices[row['method_a']], indices[row['method_b']]
                    matrix[i, j] = matrix[j, i] = row['p_raw']
            cmap = plt.get_cmap('Blues_r').copy()
            cmap.set_bad('#e9edf2')
            fig_size = max(7.2, 1.2 * n + 2.5)
            fig, ax = plt.subplots(figsize=(fig_size, fig_size * .86))
            im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0,
                           vmax=alpha, interpolation='nearest')
            ax.set_xticks(range(n), METHODS, rotation=45, ha='right', fontsize=24)
            ax.set_yticks(range(n), METHODS, fontsize=24)
            ax.set_xticks(np.arange(-.5, n, 1), minor=True)
            ax.set_yticks(np.arange(-.5, n, 1), minor=True)
            ax.grid(which='minor', color='white', linewidth=2)
            ax.tick_params(which='minor', bottom=False, left=False)
            ax.tick_params(which='major', length=0)
            for i in range(n):
                for j in range(n):
                    p = matrix[i, j]
                    if i == j:
                        label = '—'
                    elif not np.isfinite(p):
                        label = 'NA'
                    else:
                        numeric = f'{p:.2g}' if p >= .001 else f'{p:.1e}'
                        stars = '***' if p < .001 else '**' if p < .01 else '*' if p < alpha else ''
                        #label = f'{numeric}\n{stars}' if stars else numeric
                        label = stars
                    ax.text(j, i, label, ha='center', va='center', fontsize=24,
                            color='white' if np.isfinite(p) and p < alpha * .30 else '#172536')
            cbar = fig.colorbar(im, ax=ax, fraction=.045, pad=.035)
            cbar.set_label('p-value', fontsize=24)
            cbar.ax.tick_params(labelsize=20)
            fig.tight_layout(rect=(0, .035, 1, 1))
            filename = f'context_fid_{setting}_pvalues'
            fig.savefig(output / f'{filename}.png', dpi=dpi, bbox_inches='tight')
            fig.savefig(output / f'{filename}.pdf', bbox_inches='tight')
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Context-FID bar plot and paired p-value matrices')
    parser.add_argument('--parent', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--alpha', type=float, default=.05)
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()
    if not 0 < args.alpha < 1 or args.dpi < 1:
        parser.error('Require 0 < --alpha < 1 and --dpi >= 1')
    means, stds, raw = load_stats(args.parent)
    tests = paired_tests(raw)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'paired_t_tests.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(tests[0]))
        writer.writeheader()
        writer.writerows(tests)
    plot(means, stds, output, dpi=args.dpi)
    plot_pvalue_matrices(tests, output, args.alpha, args.dpi)
    print(f'\nSaved original bar chart, five p-value heatmaps (PNG/PDF), and paired_t_tests.csv to {output}')
    print('Reminder: pairing by array index is only valid for matched seeds/repetitions.')


if __name__ == '__main__':
    main()