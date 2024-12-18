import loconsensus.global_path as gpath_class
import numpy as np
from numba import boolean, float32, float64, int32, njit, prange, typed, types


class GlobalColumn:
    def __init__(self, cindex, global_offsets, l_min, l_max):
        self.global_offsets = global_offsets
        self.global_n = global_offsets[-1]
        self.l_min = l_min
        self.l_max = l_max
        self._column_paths = None

        self.start_offset = global_offsets[cindex]
        self.end_offset = global_offsets[cindex + 1]

    def candidate_finder(self, smask, emask, mask, overlap, keep_fitnesses):
        (b, e), best_fitness, fitnesses = _find_best_candidate(
            smask,
            emask,
            mask,
            self._column_paths,
            self.l_min,
            self.l_max,
            self.start_offset,
            overlap,
            keep_fitnesses,
        )
        return (b, e), best_fitness, fitnesses

    def append_paths(self, paths):
        if self._column_paths is None:
            self._column_paths = typed.List()
        for path in paths:
            self._column_paths.append(path)

    def append_mpaths(self, mpaths):
        if self._column_paths is None:
            self._column_paths = typed.List()
        for mpath in mpaths:
            self._column_paths.append(mpath)

    def induced_paths(self, b, e, mask):
        induced_paths = []
        csims = []
        for p in self._column_paths:
            if p.gj1 <= b and e <= p.gjl:
                kb, ke = p.find_gj(b), p.find_gj(e - 1)
                bm, em = p[kb][0], p[ke][0] + 1
                if not np.any(mask[bm:em]):
                    induced_path = np.copy(p.path[kb : ke + 1])
                    induced_paths.append(induced_path)
                    csims.append(p.cumsim[ke + 1] - p.cumsim[kb])

        return induced_paths, csims


@njit(
    types.Tuple((types.UniTuple(int32, 2), float32, float32[:, :]))(
        boolean[:],
        boolean[:],
        boolean[:],
        types.ListType(gpath_class.GlobalPath.class_type.instance_type),  # type:ignore
        int32,
        int32,
        int32,
        float64,
        boolean,
    )
)
def _find_best_candidate(
    start_mask,
    end_mask,
    mask,
    paths,
    l_min,
    l_max,
    start_offset,
    overlap=0.0,
    keep_fitnesses=False,
):
    fitnesses = []
    n = len(mask)

    # j1s and jls respectively contain the column index of the first and last position of all paths
    j1s = np.array([path.gj1 for path in paths])  # global???
    jls = np.array([path.gjl for path in paths])  # global???

    nbp = len(paths)

    # bs and es will respectively contain the start and end indices of the motifs in the  candidate motif set of the current candidate [b : e].
    bs = np.zeros(nbp, dtype=np.int32)
    es = np.zeros(nbp, dtype=np.int32)

    # kbs and kes will respectively contain the index on the path (\in [0 : len(path)]) where the path crosses the vertical line through b and e.
    kbs = np.zeros(nbp, dtype=np.int32)
    kes = np.zeros(nbp, dtype=np.int32)

    best_fitness = 0.0
    best_candidate = (0, n)

    # for b in range(n - l_min + 1): # total time for 27 motifs: ~141s
    for b in prange(n - l_min + 1):  # total time for 27 motifs: ~121s
        if not start_mask[b]:
            continue

        # global mapping required!
        gb = b + start_offset
        smask = j1s <= gb

        for e in range(b + l_min, min(n + 1, b + l_max + 1)):
            if not end_mask[e - 1]:
                continue

            # global mapping required!
            ge = e + start_offset
            if np.any(mask[gb:ge]):
                break

            emask = jls >= ge
            pmask = smask & emask

            # If there are not paths that cross both the vertical line through b and e, skip the candidate.
            if not np.sum(pmask) > 1:
                break

            for p in np.flatnonzero(pmask):
                path = paths[p]
                kbs[p] = pi = path.find_gj(gb)
                kes[p] = pj = path.find_gj(ge - 1)
                bs[p] = path[pi][0]
                es[p] = path[pj][0] + 1
                # Check overlap with previously found motifs.
                if np.any(
                    mask[bs[p] : es[p]]
                ):  # or es[p] - bs[p] < l_min or es[p] - bs[p] > l_max:
                    pmask[p] = False

            # If the candidate only matches with itself, skip it.
            if not np.sum(pmask) > 1:
                break

            # Sort bs and es on bs such that overlaps can be calculated efficiently
            bs_ = bs[pmask]
            es_ = es[pmask]

            perm = np.argsort(bs_)
            bs_ = bs_[perm]
            es_ = es_[perm]

            # Calculate the overlaps
            len_ = es_ - bs_
            len_[:-1] = np.minimum(len_[:-1], len_[1:])
            overlaps = np.maximum(es_[:-1] - bs_[1:], 0)

            # Overlap check within motif set
            if np.any(overlaps > overlap * len_[:-1]):
                continue

            # TODO: coverage nodig of niet???
            # Calculate normalized coverage
            coverage = np.sum(es_ - bs_) - np.sum(overlaps)
            n_coverage = coverage / float(n)

            # Calculate normalized score
            score = 0
            for p in np.flatnonzero(pmask):
                score += paths[p].cumsim[kes[p] + 1] - paths[p].cumsim[kbs[p]]
            n_score = score / float(np.sum(kes[pmask] - kbs[pmask] + 1))

            # Calculate the fitness value
            fit = 0.0
            if n_coverage != 0 or n_score != 0:
                fit = 2 * (n_coverage * n_score) / (n_coverage + n_score)

            if fit == 0.0:
                continue

            # Update best fitness
            if fit > best_fitness:
                best_candidate = (gb, ge)
                best_fitness = fit

            # Store fitness if necessary
            if keep_fitnesses:
                fitnesses.append((gb, ge, fit, n_coverage, n_score))

    fitnesses = (
        np.array(fitnesses, dtype=np.float32)
        if fitnesses
        else np.empty((0, 5), dtype=np.float32)
    )
    return best_candidate, best_fitness, fitnesses
