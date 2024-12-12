import loconsensus.path as path_class
import numpy as np
from numba import boolean, float32, float64, int32, njit, typed, types


class ConsensusColumn:
    def __init__(self, global_offsets, l_min, l_max):
        self.global_offsets = global_offsets
        self.global_n = global_offsets[-1]
        self.l_min = l_min
        self.l_max = l_max
        self._column_paths = None

    def append_paths(self, paths, offset_indices):
        if self._column_paths is None:
            self._column_paths = typed.List()
        # local mapping to global mapping
        row_start = self.global_offsets[offset_indices[0][0]]
        col_start = self.global_offsets[offset_indices[1][0]]
        for path in paths:
            path._construct_global_index(path.path + [row_start, col_start])
            self._column_paths.append(path)

    def append_mpaths(self, mpaths, offset_indices):
        if self._column_paths is None:
            self._column_paths = typed.List()
        mrow_start = self.global_offsets[offset_indices[1][0]]
        mcol_start = self.global_offsets[offset_indices[0][0]]
        for mpath in mpaths:
            # TODO: Daan???
            mpath._construct_global_index(mpath.path + [mrow_start, mcol_start])
            self._column_paths.append(mpath)

    # TODO: local paths?
    def induced_paths(self, b, e, mask=None):
        if mask is None:
            # TODO: local n has to be used here???
            mask = np.full(len(self.ts), False)

        induced_paths = []
        for p in self._paths:
            if p.j1 <= b and e <= p.jl:
                kb, ke = p.find_j(b), p.find_j(e - 1)
                bm, em = p[kb][0], p[ke][0] + 1
                if not np.any(mask[bm:em]):
                    induced_path = np.copy(p.path[kb : ke + 1])
                    induced_paths.append(induced_path)

        return induced_paths

    def find_best_motif_sets(
        self, nb=None, start_mask=None, end_mask=None, overlap=0.0
    ):
        n = self.global_n
        # handle masks
        if start_mask is None:
            start_mask = np.full(n, True)
        if end_mask is None:
            end_mask = np.full(n, True)

        assert 0.0 <= overlap and overlap <= 0.5
        assert start_mask.shape == (n,)
        assert end_mask.shape == (n,)

        # iteratively find best motif sets
        current_nb = 0
        mask = np.full(n, False)
        while nb is None or current_nb < nb:
            if np.all(mask) or not np.any(start_mask) or not np.any(end_mask):
                break

            start_mask[mask] = False
            end_mask[mask] = False

            (b, e), best_fitness, fitnesses = _find_best_candidate(
                start_mask,
                end_mask,
                mask,
                paths=self._paths,
                l_min=self.l_min,
                l_max=self.l_max,
                overlap=overlap,
                keep_fitnesses=False,
            )

            if best_fitness == 0.0:
                break

            motif_set = vertical_projections(self.induced_paths(b, e, mask))
            for bm, em in motif_set:
                l = em - bm
                mask[bm + int(overlap * l) - 1 : em - int(overlap * l)] = True

            current_nb += 1
            yield (b, e), motif_set, fitnesses


@njit(
    types.Tuple((types.UniTuple(int32, 2), float32, float32[:, :]))(
        boolean[:],
        boolean[:],
        boolean[:],
        types.ListType(path_class.Path.class_type.instance_type),  # type:ignore
        int32,
        int32,
        float64,
        boolean,
    )
)
def _find_best_candidate(
    start_mask, end_mask, mask, paths, l_min, l_max, overlap=0.0, keep_fitnesses=False
):
    fitnesses = []
    n = len(start_mask)

    # j1s and jls respectively contain the column index of the first and last position of all paths
    j1s = np.array([path.j1 for path in paths])
    jls = np.array([path.jl for path in paths])

    nbp = len(paths)

    # bs and es will respectively contain the start and end indices of the motifs in the  candidate motif set of the current candidate [b : e].
    bs = np.zeros(nbp, dtype=np.int32)
    es = np.zeros(nbp, dtype=np.int32)

    # kbs and kes will respectively contain the index on the path (\in [0 : len(path)]) where the path crosses the vertical line through b and e.
    kbs = np.zeros(nbp, dtype=np.int32)
    kes = np.zeros(nbp, dtype=np.int32)

    best_fitness = 0.0
    best_candidate = (0, n)

    for b in range(n - l_min + 1):
        if not start_mask[b]:
            continue

        smask = j1s <= b

        for e in range(b + l_min, min(n + 1, b + l_max + 1)):
            if not end_mask[e - 1]:
                continue

            if np.any(mask[b:e]):
                break

            emask = jls >= e
            pmask = smask & emask

            # If there are not paths that cross both the vertical line through b and e, skip the candidate.
            if not np.any(pmask[1:]):
                break

            for p in np.flatnonzero(pmask):
                path = paths[p]
                kbs[p] = pi = path.find_j(b)
                kes[p] = pj = path.find_j(e - 1)
                bs[p] = path[pi][0]
                es[p] = path[pj][0] + 1
                # Check overlap with previously found motifs.
                if np.any(
                    mask[bs[p] : es[p]]
                ):  # or es[p] - bs[p] < l_min or es[p] - bs[p] > l_max:
                    pmask[p] = False

            # If the candidate only matches with itself, skip it.
            if not np.any(pmask[1:]):
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
            overlaps = np.maximum(es_[:-1] - bs_[1:] - 1, 0)

            # Overlap check within motif set
            if np.any(overlaps > overlap * len_[:-1]):
                continue

            # Calculate normalized coverage
            coverage = np.sum(es_ - bs_) - np.sum(overlaps)
            n_coverage = (coverage - (e - b)) / float(n)

            # Calculate normalized score
            score = 0
            for p in np.flatnonzero(pmask):
                score += paths[p].cumsim[kes[p] + 1] - paths[p].cumsim[kbs[p]]
            n_score = (score - (e - b)) / float(np.sum(kes[pmask] - kbs[pmask] + 1))

            # Calculate the fitness value
            fit = 0.0
            if n_coverage != 0 or n_score != 0:
                fit = 2 * (n_coverage * n_score) / (n_coverage + n_score)

            if fit == 0.0:
                continue

            # Update best fitness
            if fit > best_fitness:
                best_candidate = (b, e)
                best_fitness = fit

            # Store fitness if necessary
            if keep_fitnesses:
                fitnesses.append((b, e, fit, n_coverage, n_score))

    fitnesses = (
        np.array(fitnesses, dtype=np.float32)
        if fitnesses
        else np.empty((0, 5), dtype=np.float32)
    )
    return best_candidate, best_fitness, fitnesses


def vertical_projections(paths):
    return [(p[0][0], p[len(p) - 1][0] + 1) for p in paths]


def horizontal_projections(paths):
    return [(p[0][1], p[len(p) - 1][1] + 1) for p in paths]
