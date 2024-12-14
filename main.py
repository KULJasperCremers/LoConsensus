import multiprocessing
import pickle
from itertools import combinations_with_replacement
from pathlib import Path

import locomotif.locomotif as locomotif
import locomotif.visualize as visualize
import loconsensus.lococonsensus as lococonsensus
import loconsensus.motifconsensus as motifconsensus
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
    ts1 = np.concatenate([subject101.get('walking'), subject101.get('cycling')])
    subject105 = subjects.get('subject105')
    ts5 = np.concatenate([subject105.get('walking'), subject105.get('running')])
    subject106 = subjects.get('subject106')
    ts6 = np.concatenate([subject106.get('walking'), subject106.get('running')])
    subject102 = subjects.get('subject102')
    ts2 = np.concatenate([subject102.get('walking'), subject101.get('cycling')])

    # ts = np.concatenate([ts1, ts2])
    # to run LoCoMotif for comparison
    loco = False
    if loco:
        l1 = locomotif.get_locomotif_instance(
            ts1, l_min=L_MIN, l_max=L_MAX, rho=RHO, warping=True, ts2=ts2
        )
        l1.align()

        p1 = l1.find_best_paths(vwidth=L_MIN // 2)
        print(f'LoCoMotif: {len(p1)}')

        fig, ax, _ = visualize.plot_sm(ts1, ts2, l1.get_ssm())
        visualize.plot_local_warping_paths(ax, l1.get_paths())
        plt.savefig('./plots/ts1ts2.png')
        plt.close()
        l2 = locomotif.get_locomotif_instance(
            ts2, l_min=L_MIN, l_max=L_MAX, rho=RHO, warping=True, ts2=ts1
        )
        l2.align()

        p2 = l2.find_best_paths(vwidth=L_MIN // 2)
        print(f'LoCoMotif: {len(p2)}')

        fig, ax, _ = visualize.plot_sm(ts1, ts2, l2.get_ssm())
        visualize.plot_local_warping_paths(ax, l2.get_paths())
        plt.savefig('./plots/ts2ts1.png')
        plt.close()

    # ts_list = [ts1, ts2]  # 30 motifs ~107s
    # ts_list = [ts2, ts1]  # 30 motifs ~107s
    ts_list = [ts5, ts1, ts6, ts2]
    series_file = Path('./data/series.pkl')
    with series_file.open('wb') as f:
        pickle.dump(np.concatenate(ts_list), f)
    ts_lengths = [len(ts) for ts in ts_list]
    n = len(ts_list)
    offset_indices = utils.offset_indexer(n)
    # creates a np.array /w for each global index the cutoff point
    # e.g. [0, 1735, 2722, 3955] for n=3
    global_offsets = np.cumsum([0] + ts_lengths)

    print(f'matrix n: {global_offsets[-1]}')

    # total_comparisons = n * (n + 1) // 2

    vis = True

    lccs = []
    args_list = []
    # combinations_with_replacements returns self comparisons, e.g. (ts1, ts1)
    for cindex, (ts1, ts2) in enumerate(combinations_with_replacement(ts_list, 2)):
        lcc = lococonsensus.get_lococonsensus_instance(
            ts1, ts2, global_offsets, offset_indices[cindex], L_MIN, L_MAX, RHO
        )
        lccs.append(lcc)
        args_list.append(lcc)

    num_threads = multiprocessing.cpu_count()

    def process_comparison(lcc):
        lcc.apply_loco()

    Parallel(n_jobs=num_threads, backend='threading')(
        delayed(process_comparison)(args) for args in args_list
    )

    if vis:
        for comparison, lcc in enumerate(lccs):
            fig, ax, _ = visualize.plot_sm(lcc.ts1, lcc.ts2, lcc.get_sm())
            visualize.plot_local_warping_paths(ax, lcc.get_paths())
            plt.savefig(f'./plots/lwp_{comparison}.png')
            plt.close()

    mc = motifconsensus.get_motifconsensus_instance(
        n, global_offsets, L_MIN, L_MAX, lccs
    )

    import time

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
    print(f'LoConsensus: {len(motif_sets2)}')
    print(f'Time: {outer_end_time - outer_start_time:.2f} seconds.')

    goffsets_file = Path('./data/goffsets.pkl')
    with goffsets_file.open('wb') as f:
        pickle.dump(global_offsets, f)

    motifs_file = Path('./data/motifs.pkl')
    with motifs_file.open('wb') as f:
        pickle.dump(motif_sets2, f)
