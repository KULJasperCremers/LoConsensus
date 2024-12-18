import multiprocessing

import numpy as np
from joblib import Parallel, delayed
from loconsensus.global_column import GlobalColumn


def get_motifconsensus_instance(n, global_offsets, l_min, l_max, lccs):
    gcs = []
    for column_index in range(n):
        gcs.append(GlobalColumn(column_index, global_offsets, l_min, l_max))

    coffset = -1
    for lcc in lccs:
        if lcc.is_diagonal:
            coffset += 1
            gcs[coffset].append_paths(lcc._paths)
            gcolumn = coffset
        else:
            gcs[gcolumn].append_mpaths(lcc._mirrored_paths)
            gcolumn += 1
            gcs[gcolumn].append_paths(lcc._paths)
    return MotifConsensus(global_offsets, gcs)


class MotifConsensus:
    def __init__(self, global_offsets, global_columns):
        self.global_offsets = global_offsets
        self.global_columns = global_columns
        self.ccs = [None] * len(global_columns)

    def apply_motif(self, nb, overlap):
        smask = np.full(self.global_offsets[-1], True)
        emask = np.full(self.global_offsets[-1], True)
        mask = np.full(self.global_offsets[-1], False)

        num_threads = multiprocessing.cpu_count()
        while nb is None:
            if np.all(mask) and not np.any(smask) or not np.any(emask):
                break

            smask[mask] = False
            emask[mask] = False

            best_fitness = 0.0
            best_candidate = None
            best_cindex = None

            args_list = []
            for cindex, gc in enumerate(self.global_columns):
                if not self.ccs[cindex]:
                    s, e = gc.start_offset, gc.end_offset

                    args = (
                        cindex,
                        gc,
                        smask[s:e],
                        emask[s:e],
                        mask,
                        overlap,
                        False,
                    )
                    args_list.append(args)

            results = Parallel(n_jobs=num_threads, backend='threading')(
                delayed(_process_candidate)(args) for args in args_list
            )

            for cindex, candidate, fitness, _ in results:
                if fitness > 0.0:
                    self.ccs[cindex] = (candidate, fitness, _)
                else:
                    self.ccs[cindex] = None

            for cindex, cc in enumerate(self.ccs):
                if not cc:
                    continue
                candidate, fitness, _ = cc
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_candidate = candidate
                    best_cindex = cindex

            if best_fitness == 0.0:
                print('eureka!')
                break

            b, e = best_candidate
            print(f'({b},{e}), bf: {best_fitness}')
            gc = self.global_columns[best_cindex]
            ips, csims = gc.induced_paths(b, e, mask)
            motif_set = vertical_projections(ips)
            for bm, em in motif_set:
                ml = em - bm
                mask[bm + int(overlap * ml) - 1 : em - int(overlap * ml)] = True

            args_list = []
            for cindex, cc in enumerate(self.ccs):
                if cindex == best_cindex or not cc:
                    continue
                candidate, _, _ = cc
                args = (cindex, candidate, mask)
                args_list.append(args)

            results = Parallel(n_jobs=num_threads, backend='threading')(
                delayed(_process_motifs)(args) for args in args_list
            )

            # set candidates with overlapping motifs to None
            for cindex, masked in results:
                if masked:
                    self.ccs[cindex] = None

            # set candidate to None
            self.ccs[best_cindex] = None

            """

            for cindex, cc in enumerate(self.ccs):
                if not cc:
                    continue
                if cindex == best_cindex:
                    self.ccs[best_cindex] = None
                candidate, _, _ = cc
                b, e = candidate
                if mask[b] or mask[e]:
                    self.ccs[cindex] = None

            """
            yield (b, e), motif_set, csims, ips, _


def _process_candidate(args):
    (cindex, gc, smask, emask, mask, overlap, keep_fitnesses) = args
    candidate, best_fitness, _ = gc.candidate_finder(
        smask, emask, mask, overlap, keep_fitnesses
    )

    return cindex, candidate, best_fitness, _


def _process_motifs(args):
    (cindex, candidate, mask) = args
    b, e = candidate
    masked = mask[b] or mask[e]
    return cindex, masked


def vertical_projections(paths):
    return [(p[0][0], p[len(p) - 1][0] + 1) for p in paths]
