import multiprocessing
import pickle
from itertools import combinations_with_replacement
from pathlib import Path

import locomotif.locomotif as locomotif
import loconsensus.lococonsensus as lococonsensus
import loconsensus.motifconsensus as motifconsensus
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
    ts2 = np.concatenate([subject105.get('walking'), subject105.get('running')])

    # motif_sets1 = locomotif.apply_locomotif(
    # ts1, l_min=L_MIN, l_max=L_MAX, rho=RHO, warping=True
    # )
    # print(f'LoCoMotif: {len(motif_sets1)}')

    # ts_list = [ts1]
    # TODO: self comparison bug!
    # ts_list = [ts1, ts1]
    # ts_list = [ts1, ts2]
    ts_list = [ts2, ts1]
    ts_lengths = [len(ts) for ts in ts_list]
    n = len(ts_list)
    offset_indices = utils.offset_indexer(n)
    # creates a np.array /w for each global index the cutoff point
    # e.g. [0, 1735, 2722, 3955] for n=3
    global_offsets = np.cumsum([0] + ts_lengths)

    print(f'matrix n: {global_offsets[-1]}')

    # total_comparisons = n * (n + 1) // 2

    lccs = []
    args_list = []
    # combinations_with_replacements returns self comparisons, e.g. (ts1, ts1)
    for _, (ts1, ts2) in enumerate(combinations_with_replacement(ts_list, 2)):
        lcc = lococonsensus.get_lococonsensus_instance(
            ts1, ts2, l_min=L_MIN, l_max=L_MAX, rho=RHO
        )
        lccs.append(lcc)
        args_list.append(lcc)

    num_threads = multiprocessing.cpu_count()

    def process_comparison(lcc):
        lcc.apply_loco()

    Parallel(n_jobs=num_threads, backend='threading')(
        delayed(process_comparison)(args) for args in args_list
    )

    mc = motifconsensus.get_motifconsensus_instance(
        n, global_offsets, L_MIN, L_MAX, lccs, offset_indices
    )

    overlap = 0.0
    nb = None
    motif_sets2 = []
    for motif in mc.apply_motif(nb, overlap):
        motif_sets2.append(motif)

    print(f'LoConsensus: {len(motif_sets2)}')
