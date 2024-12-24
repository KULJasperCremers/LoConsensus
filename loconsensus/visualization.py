from itertools import combinations_with_replacement

import matplotlib.patches as patches
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

    ps = []
    for tup in [(0, 0), (0, 1), (1, 1)]:
        lcc = pair_to_lcc[tup]
        paths = lcc.get_paths()
        ps.append(paths)
        """
        if tup == (0, 1):
            mps = []
            for path in paths:
                mpath = np.zeros(path.shape)
                mpath[:, 0] = path[:, 1]
                mpath[:, 1] = path[:, 0]
                mps.append(mpath)
            ps.append(mps)
        """

    for i in range(n):
        for j in range(i, n):
            if i == j:
                continue
            lcc = pair_to_lcc[(i, j)]
            sm = lcc.get_sm()

            row_start, row_end = global_offsets[i], global_offsets[i + 1]
            col_start, col_end = global_offsets[j], global_offsets[j + 1]

            global_sm[row_start:row_end, col_start:col_end] = sm

    return global_sm, ps


def plot_global_sm(global_sm, global_offsets, ts_list, figsize=(12, 12)):
    total_length = global_offsets[-1]
    num_ts = len(ts_list)
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
        ax_top.axvline(x=offset - 0.5, color='magenta', linewidth=5, ls='-')

    ax_left = fig.add_subplot(gs[1, 0])
    ax_left.set_axis_off()
    ax_left.plot(-ts_list, range(total_length, 0, -1), linewidth=1.5, ls='-')
    ax_left.set_ylim([0.5, total_length + 0.5])

    ax_left.axhline(
        y=global_offsets[2] - global_offsets[1] - 0.5,
        color='magenta',
        linewidth=5,
        ls='-',
    )

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

    label_fontsize = 50
    label_fontweight = 'bold'

    real_labels = ['(0,0)', '(0,1)', '(1,0)', '(1,1)']
    labels = [1, 2, 3, 4]
    for i in range(num_ts):
        for j in range(num_ts):
            row_start = global_offsets[i]
            row_end = global_offsets[i + 1]
            col_start = global_offsets[j]
            col_end = global_offsets[j + 1]

            center_x = (col_start + col_end - 1) / 2
            center_y = (row_start + row_end - 1) / 2

            label = labels[i + j] if i == 0 else labels[i + j + 1]

            ax_sm.text(
                center_x,
                center_y,
                real_labels[label - 1],
                color='magenta',
                fontsize=label_fontsize,
                fontweight=label_fontweight,
                ha='center',
                va='center',
            )
    gs.tight_layout(fig)

    return fig, (ax0, ax_top, ax_left, ax_sm)


def plot_ut_with_lwp(global_sm, global_offsets, ts_list, ps, figsize=(12, 12)):
    total_length = global_offsets[-1]
    num_ts = len(ts_list)
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
        ax_top.axvline(x=offset - 0.5, color='magenta', linewidth=5, ls='-')

    ax_left = fig.add_subplot(gs[1, 0])
    ax_left.set_axis_off()
    ax_left.plot(-ts_list, range(total_length, 0, -1), linewidth=1.5, ls='-')
    ax_left.set_ylim([0.5, total_length + 0.5])

    ax_left.axhline(
        y=global_offsets[2] - global_offsets[1] - 0.5,
        color='magenta',
        linewidth=5,
        ls='-',
    )

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
    ax = fig.axes
    for i, p in enumerate(ps):
        plot_local_warping_paths(ax, p, i, global_offsets)

    for offset in global_offsets[1:-1]:
        ax_sm.axhline(offset - 0.5, color='black', linewidth=1)
        ax_sm.axvline(offset - 0.5, color='black', linewidth=1)

    ax_sm.set_xlim([-0.5, total_length - 0.5])
    ax_sm.set_ylim([total_length - 0.5, -0.5])

    label_fontsize = 50
    label_fontweight = 'bold'

    # """
    real_labels = ['(0,0)', '(0,1)', '(1,0)', '(1,1)']
    labels = [1, 2, 3, 4]
    for i in range(num_ts):
        for j in range(num_ts):
            row_start = global_offsets[i]
            row_end = global_offsets[i + 1]
            col_start = global_offsets[j]
            col_end = global_offsets[j + 1]

            center_x = (col_start + col_end - 1) / 2
            center_y = (row_start + row_end - 1) / 2

            label = labels[i + j] if i == 0 else labels[i + j + 1]
            if label in [1, 4]:
                continue

            ax_sm.text(
                center_x,
                center_y,
                real_labels[label - 1],
                color='magenta',
                fontsize=label_fontsize,
                fontweight=label_fontweight,
                ha='center',
                va='center',
            )
    # """
    """
    # first column outline
    col_start = global_offsets[0]
    col_end = global_offsets[1]
    col = patches.Rectangle(
        (col_start - 0.5, -0.5),
        col_end - col_start,
        total_length,
        linewidth=10,
        edgecolor='cyan',
        facecolor='none',
    )

    labels_to_add = ['1', '2']
    for idx, (start, end) in enumerate(zip(global_offsets[:-1], global_offsets[1:])):
        label = labels_to_add[idx]
        col_center_x = (start + end - 1) / 2
        col_center_y = (total_length - 1) / 2
        ax_sm.text(
            col_center_x,
            col_center_y,
            label,
            color='cyan',
            fontsize=label_fontsize,
            fontweight=label_fontweight,
            ha='center',
            va='center',
        )

    ax_sm.add_patch(col)
    """
    gs.tight_layout(fig)

    return fig, (ax0, ax_top, ax_left, ax_sm)


def plot_local_warping_paths(axs, paths, i, offsets, **kwargs):
    for p in paths:
        """
        axs[3].plot(p[:, 1] + offsets[i], p[:, 0] + offsets[i], 'r', **kwargs)
        """
        if i == 0:
            axs[3].plot(p[:, 1] + offsets[0], p[:, 0] + offsets[0], 'r', **kwargs)
        if i == 1:
            axs[3].plot(p[:, 1] + offsets[0], p[:, 0] + offsets[0], 'r', **kwargs)
        if i == 2:
            axs[3].plot(p[:, 1] + offsets[0], p[:, 0] + offsets[0], 'r', **kwargs)

    return axs
