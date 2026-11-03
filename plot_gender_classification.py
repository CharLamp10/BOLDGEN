"""Plot gender SimCLR AUC and F1 with uncorrected paired t-test annotations.

Pairing is by the position of each score in the concatenated, sorted file order.
This is valid only if CR-VAE and rDCM-GEN use matching seeds/splits and run order.
"""
import argparse
import csv
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
TAUS = [('tau_0.05', .05), ('tau_0.1', .1), ('tau_0.15', .15)]
METHODS = ['CR-VAE', 'rDCM-GEN']
COLORS = ['#D18A45', '#167B84']
METRICS = [('AUC', 'best_test_auc'), ('F1', 'best_test_f1')]


def get_values(payload, key):
    result = payload['result']
    if isinstance(result, dict):
        values = np.asarray(result[key], dtype=float).ravel()
    elif isinstance(result, (list, tuple)):
        values = np.concatenate([np.asarray(r[key], dtype=float).ravel() for r in result])
    else:
        raise ValueError("'result' must be a dictionary or list of dictionaries")
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError(f'Invalid {key} values')
    if np.all((values >= 0) & (values <= 1)):
        values = values * 100
    return values


def load(parent, patterns):
    data = {}
    for folder, _ in SETTINGS:
        for tau_dir, tau in TAUS:
            for method in METHODS:
                base = Path(parent) / 'Gender' / 'SimCLR' / folder / tau_dir
                files = sorted(base.glob(patterns[method]))
                if not files:
                    raise FileNotFoundError(f'No matching files in {base}: {patterns[method]}')
                payloads = []
                for file in files:
                    with file.open('rb') as f:
                        payloads.append(pickle.load(f))  # Load trusted pickles only.
                for metric, key in METRICS:
                    values = np.concatenate([get_values(payload, key) for payload in payloads])
                    data[folder, tau, method, metric] = values
                    sd = np.std(values, ddof=1) if len(values) > 1 else 0.
                    print(f'{folder} tau={tau:.2f} {method} {metric}: '
                          f'{values.mean():.2f} +/- {sd:.2f} (n={len(values)})')
    return data


def significance(p):
    if not np.isfinite(p):
        return 'NA'
    if p < .001:
        return '***'
    if p < .01:
        return '**'
    if p < .05:
        return '*'
    return 'ns'


def paired_tests(data):
    rows = []
    for folder, _ in SETTINGS:
        for _, tau in TAUS:
            for metric, _ in METRICS:
                a = data[folder, tau, 'CR-VAE', metric]
                b = data[folder, tau, 'rDCM-GEN', metric]
                if len(a) != len(b):
                    raise ValueError(f'Cannot pair {folder} tau={tau} {metric}: '
                                     f'CR-VAE has {len(a)} scores, rDCM-GEN has {len(b)}. '
                                     'Align matching runs before testing.')
                n = len(a)
                difference = a - b
                if n < 2:
                    t, p, status = np.nan, np.nan, 'insufficient_pairs'
                elif np.ptp(difference) <= (16 * np.finfo(float).eps * max(1., np.max(np.abs(a)), np.max(np.abs(b)))):
                    t, p, status = np.nan, np.nan, 'zero_variance_differences'
                else:
                    res = ttest_rel(a, b, alternative='two-sided')
                    t, p, status = float(res.statistic), float(res.pvalue), 'ok'
                row = dict(setting=folder, tau=tau, metric=metric,
                           method_a='CR-VAE', method_b='rDCM-GEN', n_pairs=n,
                           mean_difference_pp=float(difference.mean()),
                           t_stat=t, p_raw=p, significance=significance(p), status=status)
                rows.append(row)
                print(f'{folder:17} tau={tau:.2f} {metric:3} paired t-test: '
                      f't={t:.4g}, p={p:.4g}, {row["significance"]} [{status}]')
    return rows


def plot(data, tests, output_dir, prefix='gender_simclr', dpi=600, show_ns=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lookup = {(r['setting'], r['tau'], r['metric']): r for r in tests}
    style = {'font.family': 'DejaVu Sans', 'font.size': 28,
             'axes.labelsize': 34, 'xtick.labelsize': 30,
             'ytick.labelsize': 30, 'legend.fontsize': 28,
             'pdf.fonttype': 42, 'svg.fonttype': 'none'}
    x = np.arange(len(TAUS))
    width = .35
    with plt.rc_context(style):
        for folder, label in SETTINGS:
            for metric, _ in METRICS:
                fig, ax = plt.subplots(figsize=(12, 8))
                means_by_method = []
                sds_by_method = []
                for j, method in enumerate(METHODS):
                    means = np.array([data[folder, tau, method, metric].mean() for _, tau in TAUS])
                    sds = np.array([data[folder, tau, method, metric].std(ddof=1)
                                    if len(data[folder, tau, method, metric]) > 1 else 0.
                                    for _, tau in TAUS])
                    means_by_method.append(means)
                    sds_by_method.append(sds)
                    ax.bar(x + (j - .5) * width, means, width * .94,
                           yerr=sds, capsize=5, color=COLORS[j], edgecolor='white',
                           linewidth=.8, error_kw={'elinewidth': 1.5, 'capthick': 1.5},
                           label='CR-VAE' if j == 0 else 'rDCM-GEN')

                # Brackets for the paired comparisons, one per temperature.
                occupied_top = max(float(np.max(means_by_method[j] + sds_by_method[j]))
                                   for j in range(len(METHODS)))
                tops = []
                for k, (_, tau) in enumerate(TAUS):
                    test = lookup[folder, tau, metric]
                    stars = test['significance']
                    if stars in ('NA', 'ns') and not show_ns:
                        continue
                    left, right = x[k] - width / 2, x[k] + width / 2
                    bar_top = max(means_by_method[j][k] + sds_by_method[j][k]
                                  for j in range(len(METHODS)))
                    y = max(bar_top + 3., 4.)
                    h = 1.2
                    ax.plot([left, left, right, right], [y, y+h, y+h, y],
                            color='black', linewidth=1.35, clip_on=False)
                    ax.text(x[k], y+h+0.7, stars, ha='center', va='bottom',
                            fontsize=23, clip_on=False)
                    tops.append(y+h+4.)

                ax.set_xticks(x, ['0.05', '0.10', '0.15'])
                ax.set_ylim(0, max(100., max(tops, default=0.) + 2.))
                ax.set_ylabel(f'{metric} (%)' if metric == 'AUC' else 'F1 score (%)')
                ax.set_xlabel('SimCLR temperature (τ)', labelpad=15)
                ax.grid(axis='y', alpha=.2)
                ax.set_axisbelow(True)
                ax.spines[['top', 'right']].set_visible(False)
                ax.tick_params(axis='x', labelsize=30, length=0, pad=12)
                ax.tick_params(axis='y', labelsize=30, length=0, pad=10)
                ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.15), ncol=2,
                          frameon=False, fontsize=28, columnspacing=1.2, handlelength=1.4)
                fig.tight_layout()
                suffix = '' if metric == 'AUC' else '_F1'
                filename = f'{prefix}_{folder}{suffix}'
                fig.savefig(output_dir / f'{filename}.png', dpi=dpi, bbox_inches='tight')
                fig.savefig(output_dir / f'{filename}.pdf', bbox_inches='tight')
                plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent_path', default='/home/student1/Desktop/Charalampos_Lamprou/rDCM-GEN_results_HCP_2026')
    parser.add_argument('--output-dir', default='/home/student1/Desktop/Charalampos_Lamprou/rDCM-GEN/plots/Gender_CLS')
    parser.add_argument('--crvae-pattern', default='results_CR-VAE*.pkl')
    parser.add_argument('--rdcm-pattern', default='results_rDCM-GEN*.pkl')
    parser.add_argument('--show-ns', action='store_true', help='Annotate nonsignificant/undefined comparisons too')
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()
    data = load(args.parent_path, {'CR-VAE': args.crvae_pattern, 'rDCM-GEN': args.rdcm_pattern})
    tests = paired_tests(data)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'paired_t_tests.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(tests[0]))
        writer.writeheader()
        writer.writerows(tests)
    plot(data, tests, out, dpi=args.dpi, show_ns=args.show_ns)
    print(f'Wrote 8 plots (PNG + PDF) and paired_t_tests.csv to {out}')


if __name__ == '__main__':
    main()