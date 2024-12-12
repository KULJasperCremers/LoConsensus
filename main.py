import multiprocessing
import pickle
from itertools import combinations_with_replacement
from pathlib import Path

import locomotif.locomotif as locomotif
import loconsensus.consensuscolumn as lcc
import loconsensus.loconsensus as loconsensus
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
    """
    motif_sets = locomotif.apply_locomotif(
        ts1, l_min=L_MIN, l_max=L_MAX, rho=RHO, warping=True
    )
    """
    # ts_list = [ts1, ts1]
    ts_list = [ts1, ts2]
    ts_lengths = [len(ts) for ts in ts_list]
    n = len(ts_list)
    offset_indices = utils.offset_indexer(n)
    # creates a np.array /w for each global index the cutoff point
    # e.g. [0, 1735, 2722, 3955] for n=3
    global_offsets = np.cumsum([0] + ts_lengths)

    # total_comparisons = n * (n + 1) // 2

    _lcs = []
    args_list = []
    # combinations_with_replacements returns self comparisons, e.g. (ts1, ts1)
    for _, (ts1, ts2) in enumerate(combinations_with_replacement(ts_list, 2)):
        lcs = loconsensus.get_loconsensus_instance(
            ts1, ts2, l_min=L_MIN, l_max=L_MAX, rho=RHO
        )
        _lcs.append(lcs)
        args_list.append(lcs)

    num_threads = multiprocessing.cpu_count()

    def process_comparison(lcs):
        lcs.apply_loco()

    Parallel(n_jobs=num_threads, backend='threading')(
        delayed(process_comparison)(args) for args in args_list
    )

    """
    for lcs in _lcs:
        # Non-threading for debugging
        # lcs.apply_loco()
    """

    global_consensus_columns = []
    for column_index in range(n):
        gcc = lcc.ConsensusColumn(column_index, global_offsets, L_MIN, L_MAX)
        global_consensus_columns.append(gcc)

    # use threading here???
    gcolumn = 0
    for comparison_index, lcs in enumerate(_lcs):
        gcolumn += 0 if lcs.is_diagonal else 1
        global_consensus_columns[gcolumn].append_paths(
            lcs._paths, offset_indices[comparison_index]
        )
        if not lcs.is_diagonal:
            global_consensus_columns[gcolumn - 1].append_mpaths(
                lcs._mirrored_paths, offset_indices[comparison_index]
            )

    # TODO: overlap???
    nb = None
    motif_sets = []
    smask = np.full(global_offsets[-1], True)
    emask = np.full(global_offsets[-1], True)
    while nb is None:
        if not np.any(smask) or not np.any(emask):
            break

        for gcc in global_consensus_columns:
            mask = gcc.mask
            if np.all(mask):
                break

            s = gcc.start_offset
            e = gcc.end_offset
            m_slice = mask[s:e]

            smask[s:e][m_slice] = False
            emask[s:e][m_slice] = False

            # csmask = smask[gcc.start_offset : gcc.end_offset]
            # cemask = emask[gcc.start_offset : gcc.end_offset]
            # threading???
            gcc.candidate_finder(
                smask[s:e], emask[s:e], overlap=np.float64(0.0), keep_fitnesses=False
            )
