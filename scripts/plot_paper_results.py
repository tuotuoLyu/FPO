"""Re-plot the paper's reported numbers (Table I average column and Table II).

    python scripts/plot_paper_results.py --out assets/paper_results.png

All values are copied from the ICRA 2026 paper; nothing here is re-measured.
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# Table I, "Average" column (success rate, %)
LIBERO_AVG = [
    ('Diffusion Policy', 72.4), ('Octo (SFT)', 75.1), ('OpenVLA (SFT)', 76.5), ('GRAPE (DPO)', 79.2),
    ('VLA-RL', 81.0), ('π0-FAST', 85.5), ('π0-FPO (ours)', 87.2),
]
# Table II, LIBERO-90 "pick up the butter and put it in the tray" (final success rate, %)
ABLATIONS = [
    ('FPO (complete)', 78.5), ('single critic (no ensemble)', 71.2), ('single-step integration (K=1)', 61.7),
    ('without PPO clipping', 45.1), ('without CFM ratio proxy (SAC)', 32.4),
]
HIGHLIGHT, NEUTRAL, INK, MUTED = '#2a78d6', '#c9c8c2', '#0b0b0b', '#52514e'


def hbar(ax, rows, highlight_label, title, xmax):
    labels = [r[0] for r in rows][::-1]
    values = [r[1] for r in rows][::-1]
    colors = [HIGHLIGHT if lab == highlight_label else NEUTRAL for lab in labels]
    bars = ax.barh(labels, values, color=colors, height=0.62, edgecolor='white', linewidth=2)
    for b, v, lab in zip(bars, values, labels):
        ax.text(v + xmax * 0.012, b.get_y() + b.get_height() / 2, f'{v:.1f}', va='center', fontsize=11,
                color=INK, fontweight='bold' if lab == highlight_label else 'normal')
    ax.set_xlim(0, xmax)
    ax.set_title(title, loc='left', fontsize=13, color=INK, pad=10)
    ax.set_xlabel('Success rate (%)', color=MUTED, fontsize=11)
    ax.tick_params(axis='y', labelsize=11, colors=INK, length=0)
    ax.tick_params(axis='x', labelsize=10, colors=MUTED)
    ax.grid(axis='x', color='#e6e5e0', linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#d6d5cf')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='assets/paper_results.png')
    a = p.parse_args()
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={'width_ratios': [1, 1.05]})
    fig.patch.set_facecolor('white')
    hbar(ax1, LIBERO_AVG, 'π0-FPO (ours)', 'LIBERO average over four suites (Table I)', 100)
    hbar(ax2, ABLATIONS, 'FPO (complete)', 'Ablations on a LIBERO-90 task (Table II)', 100)
    fig.text(0.01, 0.01, 'Numbers as reported in the paper.', fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(a.out, dpi=160, facecolor='white')
    print('saved', a.out)


if __name__ == '__main__':
    main()
