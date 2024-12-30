import multiprocessing
import pickle
import time
from itertools import combinations_with_replacement
from pathlib import Path

import locomotif.visualize as visualize
import loconsensus.lococonsensus as lococonsensus
import loconsensus.motifconsensus as motifconsensus
import loconsensus.visualization as visualization
import matplotlib.pyplot as plt
import numpy as np
import utils
from constants import (
    L_MAX,
    L_MIN,
    RHO,
)
from joblib import Parallel, delayed

if __name__ == '__main__':
    data_file = Path('./data/subjects.pkl')
    with data_file.open('rb') as f:
        subjects = pickle.load(f)  # downsampled by factor 10 to 10 hz

    subject101 = subjects.get('subject101')
    subject102 = subjects.get('subject102')
    subject103 = subjects.get('subject103')

    ts1 = subject101.get('walking')[250:750]
    ts11 = subject101.get('walking')[1250:1750]
    ts2 = subject101.get('running')[250:750]
    ts21 = subject101.get('running')[1250:1750]
    ts3 = subject101.get('cycling')[250:750]
    ts31 = subject101.get('cycling')[1250:1750]
    ts4 = subject102.get('walking')[250:750]
    ts41 = subject102.get('walking')[1250:1750]
    ts5 = subject102.get('cycling')[250:750]
    ts51 = subject102.get('cycling')[1250:1750]
    ts6 = subject103.get('walking')[250:750]
    ts61 = subject103.get('walking')[1250:1750]

    # ts_list = [ts1, ts2]
    # ts_list = [ts1, ts2, ts3, ts4, ts5, ts6]
    ts_list = [ts1, ts2, ts3, ts4, ts5, ts6, ts11, ts21, ts31, ts41, ts51, ts61]

    vis = False

    series_file = Path('./data/series.pkl')
    with series_file.open('wb') as f:
        pickle.dump(np.concatenate(ts_list), f)

    ts_lengths = [len(ts) for ts in ts_list]
    n = len(ts_list)
    offset_indices = utils.offset_indexer(n)
    # creates a np.array /w for each global index the cutoff point
    # e.g. [0, 1735, 2722, 3955] for n=3
    global_offsets = np.cumsum([0] + ts_lengths, dtype=np.int32)

    total_comparisons = n * (n + 1) // 2
    print(f'Performing {total_comparisons} total comparisons.')

    lccs = []
    args_list = []
    # combinations_with_replacements returns self comparisons, e.g. (ts1, ts1)
    for cindex, (ts1, ts2) in enumerate(combinations_with_replacement(ts_list, 2)):
        lcc = lococonsensus.get_lococonsensus_instance(
            ts1, ts2, global_offsets, offset_indices[cindex], L_MIN, RHO, cindex, n
        )
        lccs.append(lcc)
        args_list.append(lcc)

    num_threads = multiprocessing.cpu_count()

    def process_comparison(lcc):
        lcc.apply_loco()

    outer_start_time = time.perf_counter()
    Parallel(n_jobs=num_threads, backend='threading')(
        delayed(process_comparison)(args) for args in args_list
    )
    outer_end_time = time.perf_counter()
    print(f'Time LoCo: {outer_end_time - outer_start_time:.2f} seconds.')

    if vis:
        for c, lcc in enumerate(lccs):
            fig, ax, _ = visualize.plot_sm(lcc.ts1, lcc.ts2, lcc.get_sm())
            ax = visualize.plot_local_warping_paths(ax, lcc.get_paths())
            plt.savefig(f'./plots/smlwp{c}.png')
            plt.close()

    mc = motifconsensus.get_motifconsensus_instance(
        n, global_offsets, L_MIN, L_MAX, lccs
    )

    outer_start_time = time.perf_counter()
    overlap = 0.0
    nb = None
    motif_sets2 = []
    inner_start_time = time.perf_counter()
    for motif in mc.apply_motif(nb, overlap):
        inner_end_time = time.perf_counter()
        print(f'Time: {inner_end_time - inner_start_time:.2f} seconds.')
        motif_sets2.append(motif)
        inner_start_time = time.perf_counter()

    outer_end_time = time.perf_counter()
    print(f'Time Motif: {outer_end_time - outer_start_time:.2f} seconds.')
    print(f'Motifs found: {len(motif_sets2)}')

    goffsets_file = Path('./data/goffsets.pkl')
    with goffsets_file.open('wb') as f:
        pickle.dump(global_offsets, f)

    motifs_file = Path('./data/motifs.pkl')
    with motifs_file.open('wb') as f:
        pickle.dump(motif_sets2, f)

    if vis:
        for i, motif_set in enumerate(motif_sets2):
            (b, e), motifs, _, ips, _ = motif_set
            if len(motifs) < 50:
                fig, ax0, ax1_0 = visualize.plot_motifs(
                    np.concatenate(ts_list), motifs, ips, b, e
                )
                plt.savefig(f'./plots/lcc_motifs_{i}.png')
                plt.close()
