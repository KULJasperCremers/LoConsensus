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

        self._calculate_candidates(smask, emask, overlap)

        while nb is None:
            if not np.any(smask) or not np.any(emask):
                break

            best_fitness = 0.0
            best_candidate = None
            best_cindex = None

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
            print(f'({b},{e})')
            gc = self.global_columns[best_cindex]
            ips, csims = gc.induced_paths(b, e)
            motif_set = vertical_projections(ips)
            for bm, em in motif_set:
                gc.update_mask(bm, em, overlap)

            yield (b, e), motif_set, csims, _

            self.ccs[best_cindex] = self._calculate_candidate(
                best_cindex, smask, emask, overlap
            )

    def _calculate_candidate(self, cindex, smask, emask, overlap):
        gc = self.global_columns[cindex]
        ssmask, semask = self._masking(gc, smask, emask)

        candidate, fitness, _ = gc.candidate_finder(ssmask, semask, overlap, False)
        if fitness > 0.0:
            return candidate, fitness, _
        return None

    def _calculate_candidates(self, smask, emask, overlap):
        args_list = []
        for cindex, gc in enumerate(self.global_columns):
            ssmask, semask = self._masking(gc, smask, emask)

            args = (
                cindex,
                gc,
                ssmask,
                semask,
                overlap,
                False,
            )
            args_list.append(args)

        num_threads = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_threads, backend='threading')(
            delayed(process_candidate)(args) for args in args_list
        )
        for cindex, candidate, fitness, _ in results:
            if fitness > 0.0:
                self.ccs[cindex] = (candidate, fitness, _)
            else:
                self.ccs[cindex] = None

    def _masking(self, c, smask, emask):
        mask = c.mask
        s, e = c.start_offset, c.end_offset
        m = mask[s:e]
        smask[s:e][m] = False
        emask[s:e][m] = False

        return smask[s:e], emask[s:e]


def process_candidate(args):
    (cindex, gc, smask, emask, overlap, keep_fitnesses) = args
    candidate, best_fitness, _ = gc.candidate_finder(
        smask, emask, overlap, keep_fitnesses
    )

    return cindex, candidate, best_fitness, _


def vertical_projections(paths):
    return [(p[0][0], p[len(p) - 1][0] + 1) for p in paths]
