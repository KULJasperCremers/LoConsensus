import matplotlib.pyplot as plt
import numpy as np


def plot_motif_sets(ts1, ts2, column_motif_sets, row_motif_sets):
    FONT_SIZE = 8
    # get the amount of columns required for the plot
    plot_columns = 0
    for _, motif_set in column_motif_sets + row_motif_sets:
        if len(motif_set) > plot_columns:
            plot_columns = len(motif_set)

    plot_rows = len(column_motif_sets) + len(row_motif_sets)
    fig, axs = plt.subplots(plot_rows, plot_columns + 1, sharey=True, squeeze=False)

    for i, motif_set in enumerate(column_motif_sets + row_motif_sets, start=0):
        (representative_start, representative_end), motifs = motif_set
        # plot the representative from the main POV
        axs[i][0].set_prop_cycle(None)
        axs[i][0].set_xticks([])
        axs[i][0].set_xticklabels([])
        (main_ts, title) = (
            (ts1, 'Column POV') if i < len(column_motif_sets) else (ts2, 'Row POV')
        )
        axs[i][0].set_title(title, size=FONT_SIZE)
        axs[i][0].plot(
            range(representative_start, representative_end),
            main_ts[representative_start:representative_end, :],
            alpha=1,
            lw=1,
        )
        for j in range(1, plot_columns + 1):
            # plot the motif from the time-warped POV
            axs[i][j].set_prop_cycle(None)
            axs[i][j].set_xticks([])
            axs[i][j].set_xticklabels([])
            if j > len(motifs):
                axs[i][j].set_visible(False)
                continue

            start, end = motifs[j - 1]
            projected_ts = ts2 if i < len(column_motif_sets) else ts1
            axs[i][j].set_title(f'Motif {j}', size=FONT_SIZE)
            axs[i][j].plot(
                range(start, end),
                projected_ts[start:end, :],
                alpha=1,
                lw=1,
            )

    plt.tight_layout()
    return fig, axs


def plot_sm(
    s1,
    s2,
    sm,
    path=None,
    figsize=(5, 5),
    colorbar=False,
    matshow_kwargs=None,
    ts_kwargs={'linewidth': 1.5, 'ls': '-'},
):
    from cycler import cycler
    from matplotlib import gridspec

    width_ratios = [0.9, 5]
    if colorbar:
        height_ratios = [0.8, 5, 0.15]
    else:
        height_ratios = width_ratios

    fig = plt.figure(figsize=figsize, frameon=True)
    gs = gridspec.GridSpec(
        2 + colorbar,
        2,
        wspace=5,
        hspace=5,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
    )

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_axis_off()

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_prop_cycle(None)
    ax1.set_axis_off()
    ax1.plot(range(len(s2)), s2, **ts_kwargs)
    ax1.set_xlim([-0.5, len(s2) - 0.5])

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_prop_cycle(None)
    ax2.set_axis_off()
    ax2.plot(-s1, range(len(s1), 0, -1), **ts_kwargs)
    ax2.set_ylim([0.5, len(s1) + 0.5])

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_aspect(1)
    ax3.tick_params(
        axis='both',
        which='both',
        labeltop=False,
        labelleft=False,
        labelright=False,
        labelbottom=False,
    )

    kwargs = {} if matshow_kwargs is None else matshow_kwargs
    img = ax3.matshow(sm, **kwargs)

    cax = None
    if colorbar:
        cax = fig.add_subplot(gs[2, 1])
        fig.colorbar(img, cax=cax, orientation='horizontal')

    gs.tight_layout(fig)

    # Align the subplots:
    ax1pos = ax1.get_position().bounds
    ax2pos = ax2.get_position().bounds
    ax3pos = ax3.get_position().bounds
    ax2.set_position(
        (ax2pos[0], ax2pos[1] + ax2pos[3] - ax3pos[3], ax2pos[2], ax3pos[3])
    )  # adjust the time series on the left vertically
    if len(s1) < len(s2):
        ax3.set_position(
            (ax3pos[0], ax2pos[1] + ax2pos[3] - ax3pos[3], ax3pos[2], ax3pos[3])
        )  # move the time series on the left and the distance matrix upwards
    if len(s1) > len(s2):
        ax3.set_position(
            (ax1pos[0], ax3pos[1], ax3pos[2], ax3pos[3])
        )  # move the time series at the top and the distance matrix to the left
        ax1.set_position(
            (ax1pos[0], ax1pos[1], ax3pos[2], ax1pos[3])
        )  # adjust the time series at the top horizontally
    if len(s1) == len(s2):
        ax1.set_position(
            (ax3pos[0], ax1pos[1], ax3pos[2], ax1pos[3])
        )  # adjust the time series at the top horizontally

    ax = fig.axes
    return fig, ax, cax


def plot_local_warping_paths(axs, paths, direction=None, **kwargs):
    for p in paths:
        if direction == 'column':
            axs[3].plot(p[:, 1], p[:, 0], 'r', **kwargs)
        elif direction == 'row':
            axs[3].plot(p[:, 1], p[:, 0], 'g', **kwargs)

    return axs
