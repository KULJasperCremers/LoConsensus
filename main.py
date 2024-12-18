import multiprocessing
import pickle
import time
from itertools import combinations_with_replacement
from pathlib import Path

import locomotif.locomotif as locomotif
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
    ts1 = subject101.get('walking')
    ts2 = subject102.get('walking')
    vis = True
    # to run LoCoMotif for comparison
    loco = False
    if loco:
        l1 = locomotif.get_locomotif_instance(
            ts1, l_min=L_MIN, l_max=L_MAX, rho=RHO, warping=True
        )
        l1.align()
        p1 = l1.find_best_paths(vwidth=L_MIN // 2)
        print(f'fp lcm: {len(p1)}')
        fig, ax, _ = visualize.plot_sm(ts1, ts1, l1.get_ssm())
        visualize.plot_local_warping_paths(ax, l1.get_paths())
        plt.savefig('./plots/lmts1ts1.png')
        plt.close()
        motif = 0
        motif_sets1 = []
        for (b, e), motif_set, ips, _ in l1.find_best_motif_sets(nb=None, overlap=0.0):
            motif_sets1.append(((b, e), motif_set, ips))
            if vis:
                fig, ax0, ax1_0 = visualize.plot_motifs(ts1, motif_set, ips, b, e)
                plt.savefig(f'./plots/lcm_motifs_{motif}.png')
                plt.close()
            motif += 1
        print(f'LoCoMotif: {len(motif_sets1)}')

    ts_list = [ts1, ts2]
    series_file = Path('./data/series.pkl')
    with series_file.open('wb') as f:
        pickle.dump(np.concatenate(ts_list), f)
    ts_lengths = [len(ts) for ts in ts_list]
    n = len(ts_list)
    offset_indices = utils.offset_indexer(n)
    # creates a np.array /w for each global index the cutoff point
    # e.g. [0, 1735, 2722, 3955] for n=3
    global_offsets = np.cumsum([0] + ts_lengths)

    total_comparisons = n * (n + 1) // 2
    print(f'Performing {total_comparisons} total comparisons.')

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

    outer_start_time = time.perf_counter()
    Parallel(n_jobs=num_threads, backend='threading')(
        delayed(process_comparison)(args) for args in args_list
    )
    outer_end_time = time.perf_counter()
    print(f'Time LoCo: {outer_end_time - outer_start_time:.2f} seconds.')

    if vis:
        for comparison, lcc in enumerate(lccs):
            fig, ax, _ = visualize.plot_sm(lcc.ts1, lcc.ts2, lcc.get_sm())
            plt.savefig(f'./plots/sm{comparison}.png')
            plt.close()
            global_sm = visualization.assemble_global_sm(ts_list, lccs, global_offsets)
            fig, ax = visualization.plot_global_sm(global_sm, global_offsets, ts_list)
            plt.savefig('./plots/gsm.png')
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
            fig, ax0, ax1_0 = visualize.plot_motifs(
                np.concatenate(ts_list), motifs, ips, b, e
            )
            plt.savefig(f'./plots/lcc_motifs_{i}.png')
            plt.close()
