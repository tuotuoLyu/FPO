"""Latent-space evolution analysis (paper Fig. 5).

Reads the exploration latents written by a run with ``--save_latents 1``
(``<outputdir>/latents/latent_actions_step_<i>.npy``; each file holds every
exploration latent collected up to gradient step i), groups the latents by the
step at which they were collected, embeds the three groups jointly with t-SNE and
draws (a)-(c) density maps and (d) the exploration-range bar chart.

    python analysis/latent_evolution.py --latent_dir <outputdir>/latents --out fig5.png

Requires numpy, scipy, scikit-learn and matplotlib.
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

# phase name -> (first step, last step, max points, colour map); defaults follow Fig. 5
PHASES = {
    'Early training (0-10K steps)': (0, 10_000, 2000, 'Greys'),
    'Breakthrough phase (80-100K steps)': (80_000, 100_000, 1500, 'Oranges'),
    'Late-training phase (130K+ steps)': (130_000, None, 1500, 'Greens'),
}


def load_by_collection_step(latent_dir):
    """{step: latents collected since the previous file} from cumulative files."""
    files = glob.glob(os.path.join(latent_dir, 'latent_actions_step_*.npy'))
    steps = sorted((int(re.findall(r'_(\d+)\.npy$', f)[0]), f) for f in files)
    out, seen = {}, 0
    for step, f in steps:
        data = np.load(f)
        data = data.reshape(len(data), -1)
        out[step] = data[seen:]
        seen = len(data)
    return out


def group_phases(per_step, rng):
    groups = {}
    for name, (lo, hi, cap, _) in PHASES.items():
        chunks = [v for s, v in per_step.items() if s >= lo and (hi is None or s <= hi) and len(v)]
        if not chunks:
            continue
        data = np.vstack(chunks)
        if len(data) > cap:
            data = data[rng.choice(len(data), cap, replace=False)]
        groups[name] = data
    return groups


def spread_metrics(points):
    try:
        area = ConvexHull(points).volume
    except Exception:
        area = np.ptp(points[:, 0]) * np.ptp(points[:, 1])
    return np.mean(np.std(points, axis=0)), area, np.std(points.ravel())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--latent_dir', required=True)
    p.add_argument('--out', default='latent_evolution.png')
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    groups = group_phases(load_by_collection_step(a.latent_dir), rng)
    if len(groups) < 2:
        raise SystemExit('need latents from at least two phases')
    names = list(groups)
    stacked = np.vstack([groups[n] for n in names])
    emb = TSNE(n_components=2, perplexity=min(50, len(stacked) // 6), init='pca',
               random_state=a.seed).fit_transform(stacked)
    splits = np.cumsum([len(groups[n]) for n in names])[:-1]
    embedded = dict(zip(names, np.split(emb, splits)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, name in zip(axes.ravel()[:3], names):
        pts = embedded[name]
        hb = ax.hexbin(pts[:, 0], pts[:, 1], gridsize=25, cmap=PHASES[name][3], mincnt=1, alpha=0.85)
        fig.colorbar(hb, ax=ax, shrink=0.8, label='Point density')
        xx, yy = np.mgrid[pts[:, 0].min() - 10:pts[:, 0].max() + 10:50j, pts[:, 1].min() - 10:pts[:, 1].max() + 10:50j]
        dens = gaussian_kde(pts.T)(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        ax.contour(xx, yy, dens, colors='black', alpha=0.6, linewidths=1.0)
        ax.set(title=name, xlabel='t-SNE component 1', ylabel='t-SNE component 2')
        ax.grid(alpha=0.3)

    ax = axes[1, 1]
    metrics = np.array([spread_metrics(embedded[n]) for n in names])          # [phases, 3]
    norm = metrics / metrics.max(axis=0, keepdims=True)
    labels = ['Exploring range (mean std)', 'Covering area (convex hull)', 'Overall dispersion']
    x = np.arange(len(names))
    for j, lab in enumerate(labels):
        bars = ax.bar(x + (j - 1) * 0.25, norm[:, j], 0.25, label=lab)
        for b, v in zip(bars, metrics[:, j]):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f'{v:.4g}', ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(' (', '\n(') for n in names], fontsize=9)
    ax.set(ylim=(0, 1.2), ylabel='Normalised value', title='Exploration range comparison')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out, dpi=200)
    print('saved', a.out)


if __name__ == '__main__':
    main()
