import matplotlib.pyplot as plt


def plot_consensus_motifs(consensus_motifs):
    FONT_SIZE = 8
    COLORS = ['lime', 'cyan', 'magenta']

    for i, cm in enumerate(consensus_motifs):
        consensus_cols = sum(len(cmsl) for cmsl in cm.consensus_motif_set_list)
        plot_columns = len(cm.motif_set) + consensus_cols

        fig, axs = plt.subplots(
            1, plot_columns + 1, sharey=True, squeeze=True, figsize=(15, 5)
        )

        # plot the representative timeseries
        (representative_start, representative_end) = cm.representative
        ax = axs[0]
        ax.set_prop_cycle(None)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_title(f'Fit: {cm.fitness:.2f}', size=FONT_SIZE)
        ax.plot(
            range(representative_start, representative_end),
            cm.representative_ts[representative_start:representative_end, :],
            alpha=1,
            lw=1,
        )
        for spine in ax.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(1)

        # plot the motif sets
        for j, (start, end) in enumerate(cm.motif_set):
            ax = axs[j + 1]
            ax.set_prop_cycle(None)
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_title(f'Motif {j+1}', size=FONT_SIZE)
            ax.plot(range(start, end), cm.motif_ts[start:end, :], alpha=1, lw=1)
            for spine in ax.spines.values():
                spine.set_edgecolor('blue')
                spine.set_linewidth(1)

        # plot consensus motifs
        consensus_col_start = 1 + len(cm.motif_set)
        current_col = consensus_col_start

        for k, (consensus_set, consensus_ts) in enumerate(
            zip(cm.consensus_motif_set_list, cm.consensus_ts_list)
        ):
            color = COLORS[k % len(COLORS)]  # color based on consensus index

            for l, (start, end) in enumerate(consensus_set):
                ax = axs[current_col]
                ax.set_prop_cycle(None)
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.set_title(f'Cmotif {k+1}.{l+1}', size=FONT_SIZE)
                ax.plot(
                    range(start, end),
                    consensus_ts[start:end, :],
                    alpha=1,
                    lw=1,
                )
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(1)

                current_col += 1

        plt.tight_layout()
        fig.savefig(f'./plots/consensus_motifs/consensus_motifs{i+1}.png')
        plt.close()
