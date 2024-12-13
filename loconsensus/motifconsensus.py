import multiprocessing

import numpy as np
from joblib import Parallel, delayed
from loconsensus.global_column import GlobalColumn


def get_motifconsensus_instance(n, global_offsets, l_min, l_max, lccs):
    gcs = []
    for column_index in range(n):
        gcs.append(GlobalColumn(column_index, global_offsets, l_min, l_max))

    gcolumn = 0
    for lcc in lccs:
        gcolumn += 0 if lcc.is_diagonal else 1
        gcs[gcolumn].append_paths(lcc._paths)
        if not lcc.is_diagonal:
            gcs[gcolumn - 1].append_mpaths(lcc._mirrored_paths)
    return MotifConsensus(global_offsets, gcs)


class MotifConsensus:
    def __init__(self, global_offsets, global_columns):
        self.global_offsets = global_offsets
        self.global_columns = global_columns

    def apply_motif(self, nb, overlap):
        smask = np.full(self.global_offsets[-1], True)
        emask = np.full(self.global_offsets[-1], True)
        while nb is None:
            if not np.any(smask) or not np.any(emask):
                break

            best_fitness = 0.0
            best_candidate = None
            best_cindex = None
            num_threads = multiprocessing.cpu_count()
            args_list = []
            for cindex, gc in enumerate(self.global_columns):
                mask = gc.mask
                if np.all(mask):
                    break

                s, e = gc.start_offset, gc.end_offset
                m = mask[s:e]

                smask[s:e][m] = False
                emask[s:e][m] = False

                args = (
                    cindex,
                    gc,
                    smask[s:e],
                    emask[s:e],
                    overlap,
                    False,
                )
                args_list.append(args)

            results = Parallel(n_jobs=num_threads, backend='threading')(
                delayed(process_candidate)(args) for args in args_list
            )

            for cindex, candidate, fitness, _ in results:
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_candidate = candidate
                    best_cindex = cindex

            if best_fitness == 0.0:
                print('eureka!')
                break

            b, e = best_candidate
            print(f'({b},{e})')
            gc = self.global_columns[best_cindex]
            motif_set = vertical_projections(gc.induced_paths(b, e))
            for bm, em in motif_set:
                gc.update_mask(bm, em, overlap)

            yield (b, e), motif_set, _


def process_candidate(args):
    (cindex, gc, smask, emask, overlap, keep_fitnesses) = args
    candidate, best_fitness, _ = gc.candidate_finder(
        smask, emask, overlap, keep_fitnesses
    )

    return cindex, candidate, best_fitness, _


def vertical_projections(paths):
    return [(p[0][0], p[len(p) - 1][0] + 1) for p in paths]
