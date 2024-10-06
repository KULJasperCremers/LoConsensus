import matplotlib.pyplot as plt
import numpy as np


def plot_similarity_matrix(
    ts1,
    ts2,
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
    # univariate plots
    ax1.plot(range(len(ts2)), ts2, **ts_kwargs)
    ax1.set_xlim([-0.5, len(ts2) - 0.5])

    # multivariate plots
    # ax1.plot(ts2[:,0], ts2[:,1], **ts_kwargs)
    # ax1.set_xlim([min(ts2[:,0])-0.5, max(ts2[:,0])+0.5])
    # ax1.set_ylim([min(ts2[:,1])-0.5, max(ts2[:,1])+0.5])

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_prop_cycle(None)
    ax2.set_axis_off()
    # univariate plots
    ax2.plot(-ts1, range(len(ts1), 0, -1), **ts_kwargs)
    ax2.set_ylim([0.5, len(ts1) + 0.5])

    # multivariate plots
    # ax2.plot(ts1[:,0], ts1[:,1], **ts_kwargs)
    # ax2.set_xlim([min(ts1[:,0])-0.5, max(ts1[:,0])+0.5])
    # ax2.set_ylim([min(ts1[:,1])-0.5, max(ts1[:,1])+0.5])

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
    if len(ts1) < len(ts2):
        ax3.set_position(
            (ax3pos[0], ax2pos[1] + ax2pos[3] - ax3pos[3], ax3pos[2], ax3pos[3])
        )  # move the time series on the left and the distance matrix upwards
    if len(ts1) > len(ts2):
        ax3.set_position(
            (ax1pos[0], ax3pos[1], ax3pos[2], ax3pos[3])
        )  # move the time series at the top and the distance matrix to the left
        ax1.set_position(
            (ax1pos[0], ax1pos[1], ax3pos[2], ax1pos[3])
        )  # adjust the time series at the top horizontally
    if len(ts1) == len(ts2):
        ax1.set_position(
            (ax3pos[0], ax1pos[1], ax3pos[2], ax1pos[3])
        )  # adjust the time series at the top horizontally

    ax = fig.axes
    return (fig, ax, cax)


def plot_local_warping_paths(axs, paths, **kwargs):
    for p in paths:
        axs[3].plot(p[:, 1], p[:, 0], 'r', **kwargs)
    return axs
