from itertools import combinations_with_replacement

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec


def assemble_global_sm(ts_list, lccs, global_offsets):
    n = len(ts_list)
    total_length = global_offsets[-1]

    global_sm = np.full((total_length, total_length), np.nan)

    pair_to_lcc = {}
    idx = 0
    for i, j in combinations_with_replacement(range(n), 2):
        pair_to_lcc[(i, j)] = lccs[idx]
        idx += 1

    for i in range(n):
        for j in range(i, n):
            lcc = pair_to_lcc[(i, j)]
            sm = lcc.get_sm()

            row_start, row_end = global_offsets[i], global_offsets[i + 1]
            col_start, col_end = global_offsets[j], global_offsets[j + 1]

            global_sm[row_start:row_end, col_start:col_end] = sm

    return global_sm


def plot_global_sm(global_sm, global_offsets, ts_list, figsize=(12, 12)):
    total_length = global_offsets[-1]
    ts_list = np.concatenate(ts_list)

    height_ratios = [0.9, 5]
    width_ratios = [0.9, 5]

    fig = plt.figure(figsize=figsize, frameon=True)
    gs = gridspec.GridSpec(
        2, 2, wspace=5, hspace=5, height_ratios=height_ratios, width_ratios=width_ratios
    )

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_axis_off()

    ax_top = fig.add_subplot(gs[0, 1])
    ax_top.set_axis_off()
    ax_top.plot(range(total_length), ts_list, linewidth=1.5, ls='-')
    ax_top.set_xlim([-0.5, total_length - 0.5])

    for offset in global_offsets[1:-1]:
        ax_top.axvline(offset - 0.5, color='black', linewidth=1)

    ax_left = fig.add_subplot(gs[1, 0])
    ax_left.set_axis_off()
    ax_left.plot(-ts_list, range(total_length, 0, -1), linewidth=1.5, ls='-')
    ax_left.set_ylim([0.5, total_length + 0.5])

    for offset in global_offsets[1:-1]:
        ax_left.axhline(total_length - offset + 0.5, color='black', linewidth=1)

    ax_sm = fig.add_subplot(gs[1, 1])
    ax_sm.set_aspect(1)
    ax_sm.tick_params(
        axis='both',
        which='both',
        labeltop=False,
        labelleft=False,
        labelright=False,
        labelbottom=False,
    )

    cmap = plt.cm.viridis.copy()
    cmap.set_bad('white', 1.0)

    ax_sm.matshow(global_sm, cmap=cmap)

    for offset in global_offsets[1:-1]:
        ax_sm.axhline(offset - 0.5, color='black', linewidth=1)
        ax_sm.axvline(offset - 0.5, color='black', linewidth=1)

    ax_sm.set_xlim([-0.5, total_length - 0.5])
    ax_sm.set_ylim([total_length - 0.5, -0.5])

    labels = [1, 3]
    label_fontsize = 30
    label_fontweight = 'bold'

    for i in range(len(labels)):
        start = global_offsets[i]
        end = global_offsets[i + 1]
        center = (start + end - 1) / 2
        ax_sm.text(
            center,
            center,
            str(labels[i]),
            color='red',
            fontsize=label_fontsize,
            fontweight=label_fontweight,
            ha='center',
            va='center',
        )

    upper_centers = []
    start_i = global_offsets[1]
    end_i = global_offsets[-1]
    start_j = global_offsets[0]
    end_j = global_offsets[1]
    center_x = (start_i + end_i - 1) / 2
    center_y = (start_j + end_j - 1) / 2
    upper_centers.append((center_x, center_y))

    if upper_centers:
        ut_x = np.mean([x for x, y in upper_centers])
        ut_y = np.mean([y for x, y in upper_centers])

        ax_sm.text(
            ut_x,
            ut_y,
            '2',
            color='red',
            fontsize=label_fontsize,
            fontweight=label_fontweight,
            ha='center',
            va='center',
        )

    gs.tight_layout(fig)

    return fig, (ax0, ax_top, ax_left, ax_sm)
