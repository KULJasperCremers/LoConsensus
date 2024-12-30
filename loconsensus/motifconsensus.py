import multiprocessing

import numpy as np
from joblib import Parallel, delayed
from loconsensus.global_column import GlobalColumn


def get_motifconsensus_instance(n, global_offsets, l_min, l_max, lccs):
    gcs = []
    for column_index in range(n):
        gcs.append(GlobalColumn(column_index, global_offsets, l_min, l_max))

    i = 0
    for r in range(n):
        for c in range(r, n):
            lcc = lccs[i]
            i += 1

            if r == c:
                gcs[r].append_paths(lcc._paths)
            else:
                gcs[c].append_paths(lcc._paths)
                if lcc._mirrored_paths:
                    gcs[r].append_paths(lcc._mirrored_paths)

    return MotifConsensus(global_offsets, gcs)


class MotifConsensus:
    def __init__(self, global_offsets, global_columns):
        self.global_offsets = global_offsets
        self.global_columns = global_columns
        self.ccs = [None] * len(global_columns)

    def apply_motif(self, nb, overlap):
        for i, c in enumerate(self.global_columns):
            paths = c._column_paths
            print(f'c{i}: len paths {len(paths)}')
            # for p in paths:
            # print(f'gi1: {p.gi1} // gil: {p.gil}')
            # print(f'gi1: {p.gj1} // gil: {p.gjl}')
        print()
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
                    args = (cindex, gc, smask, emask, mask, overlap, False)
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
            gc = self.global_columns[best_cindex]
            ips, csims = gc.induced_paths(b, e, mask)
            motif_set = vertical_projections(ips)
            for bm, em in motif_set:
                l = em - bm
                mask[bm + int(overlap * l) - 1 : em - int(overlap * l)] = True

            for cindex, cc in enumerate(self.ccs):
                if cindex == best_cindex or not cc:
                    continue
                (b2, e2), _, _ = cc
                gc2 = self.global_columns[cindex]
                ips2, _ = gc2.induced_paths(b2, e2, mask)
                if np.any(mask[b2:e2]) or len(ips2) < 2:
                    self.ccs[cindex] = None
            self.ccs[best_cindex] = None

            yield (b, e), motif_set, csims, ips, _


def _process_candidate(args):
    (cindex, gc, smask, emask, mask, overlap, keep_fitnesses) = args
    candidate, best_fitness, _ = gc.candidate_finder(
        smask, emask, mask, overlap, keep_fitnesses
    )

    return cindex, candidate, best_fitness, _


def vertical_projections(paths):
    return [(p[0][0], p[len(p) - 1][0] + 1) for p in paths]


def vertical_projection(path):
    return (path[0][0], path[len(path) - 1][0] + 1)
