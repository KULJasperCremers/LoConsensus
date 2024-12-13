import loconsensus.global_path as gpath_class
import numpy as np
from numba import boolean, float32, float64, int32, njit, typed, types


def get_lococonsensus_instance(
    ts1, ts2, global_offsets, offset_indices, l_min, l_max, rho
):
    is_diagonal = False
    if np.array_equal(ts1, ts2):
        is_diagonal = True
    ts1 = np.array(ts1, dtype=np.float32)
    ts2 = np.array(ts2, dtype=np.float32)

    gamma = 1
    sm, ut_sm = None, None
    if is_diagonal:
        sm = calculate_similarity_matrix(ts1, ts2, gamma, only_triu=is_diagonal)
        tau = estimate_tau_symmetric(sm, rho)
    else:
        ut_sm = calculate_similarity_matrix(ts1, ts2, gamma, only_triu=is_diagonal)
        # redundant!
        # lt_sm = calculate_similarity_matrix(ts2, ts1, gamma, only_triu=is_diagonal)
        tau = estimate_tau_assymmetric(ut_sm, rho)

    delta_a = 2 * tau
    delta_m = 0.5
    step_sizes = np.array([(1, 1), (2, 1), (1, 2)])
    lcs = LoCoConsensus(
        ts1=ts1,
        ts2=ts2,
        is_diagonal=is_diagonal,
        l_min=l_min,
        l_max=l_max,
        gamma=gamma,
        tau=tau,
        delta_a=delta_a,
        delta_m=delta_m,
        step_sizes=step_sizes,
        global_offsets=global_offsets,
        offset_indices=offset_indices,
    )
    lcs._sm = (sm, ut_sm)
    return lcs


class LoCoConsensus:
    def __init__(
        self,
        ts1,
        ts2,
        is_diagonal,
        l_min,
        l_max,
        gamma,
        tau,
        delta_a,
        delta_m,
        step_sizes,
        global_offsets,
        offset_indices,
    ):
        self.ts1 = ts1
        self.is_diagonal = is_diagonal
        self.ts2 = ts2
        self.l_min = np.int32(l_min)
        self.l_max = np.int32(l_max)
        self.step_sizes = step_sizes.astype(np.int32)

        self.gamma = gamma
        self.tau = tau
        self.delta_a = delta_a
        self.delta_m = delta_m

        self._sm = (None, None)
        self._csm = None
        self._paths = None
        self._mirrored_paths = None

        # global path logic
        self.rstart = self.cstart_mir = global_offsets[offset_indices[0][0]]
        self.cstart = self.rstart_mir = global_offsets[offset_indices[1][0]]

    def apply_loco(self):
        # apply LoCo
        self.align()
        self.find_best_paths(vwidth=self.l_min // 2)

    def align(self):
        if self.is_diagonal:
            # pass sm for diagonal comparisons
            sm = self._sm[0]
        else:
            # pass ut_sm for non diagonal comparisons
            sm = self._sm[1]
        self._csm = calculate_cumulative_similarity_matrix(
            sm,
            tau=self.tau,
            delta_a=self.delta_a,
            delta_m=self.delta_m,
            step_sizes=self.step_sizes,
            only_triu=self.is_diagonal,
        )

    def find_best_paths(self, vwidth):
        if self.is_diagonal:
            mask = np.full(self._csm.shape, True)
            mask[np.triu_indices(len(mask), k=vwidth)] = False
            diagonal = np.vstack(np.diag_indices(len(self.ts1))).T
            # TODO: check gdiagonal requirement???
            gdiagonal = diagonal + [self.rstart, self.cstart]
        else:
            mask = np.full(self._csm.shape, False)
            self._mirrored_paths = typed.List()
        found_paths = _find_best_paths(
            self._csm, mask, l_min=self.l_min, vwidth=vwidth, step_sizes=self.step_sizes
        )

        self._paths = typed.List()

        if self.is_diagonal:
            # TODO: check gdiagonal requirement???
            self._paths.append(
                gpath_class.GlobalPath(
                    gdiagonal.astype(np.int32),
                    np.ones(len(diagonal)).astype(np.float32),
                )
            )

        for path in found_paths:
            i, j = path[:, 0], path[:, 1]
            # global path logic
            gpath = np.zeros(path.shape, dtype=np.int32)
            gpath[:, 0] = np.copy(path[:, 0]) + self.rstart
            gpath[:, 1] = np.copy(path[:, 1]) + self.cstart
            gpath_mir = np.zeros(path.shape, dtype=np.int32)
            gpath_mir[:, 0] = np.copy(path[:, 1]) + self.rstart_mir
            gpath_mir[:, 1] = np.copy(path[:, 0]) + self.cstart_mir

            if self.is_diagonal:
                path_sims = self._sm[0][i, j]
                self._paths.append(gpath_class.GlobalPath(gpath, path_sims))
                self._paths.append(gpath_class.GlobalPath(gpath_mir, path_sims))
            else:
                path_sims = self._sm[1][i, j]
                self._paths.append(gpath_class.GlobalPath(gpath, path_sims))
                mir_path_sims = self._sm[1].T[j, i]
                self._mirrored_paths.append(
                    gpath_class.GlobalPath(gpath_mir, mir_path_sims)
                )


@njit(float32[:, :](float32[:, :], float32[:, :], int32, boolean))
def calculate_similarity_matrix(ts1, ts2, gamma, only_triu):
    n, m = len(ts1), len(ts2)
    similarity_matrix = np.full((n, m), -np.inf, dtype=np.float32)
    for i in range(n):
        j_start = i if only_triu else 0
        j_end = m
        similarities = np.exp(
            -gamma * np.sum(np.power(ts1[i, :] - ts2[j_start:j_end, :], 2), axis=1)
        )
        similarity_matrix[i, j_start:j_end] = similarities
    return similarity_matrix


@njit(float32[:, :](float32[:, :], float64, float64, float64, int32[:, :], boolean))
def calculate_cumulative_similarity_matrix(
    sm, tau, delta_a, delta_m, step_sizes, only_triu
):
    n, m = sm.shape
    max_v = np.amax(step_sizes[:, 0])
    max_h = np.amax(step_sizes[:, 1])

    csm = np.zeros((n + max_v, m + max_h), dtype=np.float32)
    for i in range(n):
        j_start = i if only_triu else 0
        j_end = m
        for j in range(j_start, j_end):
            sim = sm[i, j]

            indices = np.array([i + max_v, j + max_h]) - step_sizes
            max_cumsim = np.amax(np.array([csm[i_, j_] for (i_, j_) in indices]))

            if sim < tau:
                csm[i + max_v, j + max_h] = max(0, delta_m * max_cumsim - delta_a)
            else:
                csm[i + max_v, j + max_h] = max(0, sim + max_cumsim)
    return csm


@njit(int32[:, :](float32[:, :], boolean[:, :], int32, int32, int32[:, :]))
def max_warping_path(csm, mask, i, j, step_sizes):
    max_v = max(step_sizes[:, 0])
    max_h = max(step_sizes[:, 1])
    path = []
    while i >= max_v and j >= max_h:
        path.append((i - max_v, j - max_h))
        indices = np.array([i, j], dtype=np.int32) - step_sizes
        values = np.array([csm[i_, j_] for (i_, j_) in indices])
        masked = np.array([mask[i_, j_] for (i_, j_) in indices])
        argmax = np.argmax(values)

        if masked[argmax]:
            break

        i, j = i - step_sizes[argmax, 0], j - step_sizes[argmax, 1]

    path.reverse()
    return np.array(path, dtype=np.int32)


@njit(boolean[:, :](int32[:, :], boolean[:, :], int32, int32))
def mask_path(path, mask, v, h):
    for x, y in path:
        mask[x + h, y + v] = True
    return mask


@njit(boolean[:, :](int32[:, :], boolean[:, :], int32, int32, int32))
def mask_vicinity(path, mask, v, h, vwidth):
    (xc, yc) = path[0] + np.array((v, h))
    for xt, yt in path[1:] + np.array([v, h]):
        dx = xt - xc
        dy = yc - yt
        err = dx + dy
        while xc != xt or yc != yt:
            mask[xc - vwidth : xc + vwidth + 1, yc] = True
            mask[xc, yc - vwidth : yc + vwidth + 1] = True
            e = 2 * err
            if e > dy:
                err += dy
                xc += 1
            if e < dx:
                err += dx
                yc += 1
    mask[xt - vwidth : xt + vwidth + 1, yt] = True
    mask[xt, yt - vwidth : yt + vwidth + 1] = True
    return mask


@njit(types.List(int32[:, :])(float32[:, :], boolean[:, :], int32, int32, int32[:, :]))
def _find_best_paths(csm, mask, l_min, vwidth, step_sizes):
    max_v = max(step_sizes[:, 0])
    max_h = max(step_sizes[:, 1])

    is_, js_ = np.nonzero(csm <= 0)
    for index_best in range(len(is_)):
        mask[is_[index_best], js_[index_best]] = True

    is_, js_ = np.nonzero(csm)
    values = np.array([csm[is_[i], js_[i]] for i in range(len(is_))])
    perm = np.argsort(values)
    is_ = is_[perm]
    js_ = js_[perm]

    index_best = len(is_) - 1
    paths = []

    while index_best >= 0:
        path = np.empty((0, 0), dtype=np.int32)
        path_found = False
        while not path_found:
            while mask[is_[index_best], js_[index_best]]:
                index_best -= 1
                if index_best < 0:
                    return paths

            i_best, j_best = is_[index_best], js_[index_best]

            if i_best < max_v or j_best < max_h:
                return paths

            path = max_warping_path(csm, mask, i_best, j_best, step_sizes)
            mask = mask_path(path, mask, max_v, max_h)

            if (path[-1][0] - path[0][0] + 1) >= l_min or (
                path[-1][1] - path[0][1] + 1
            ) >= l_min:
                path_found = True

        mask = mask_vicinity(path, mask, max_v, max_h, vwidth)
        paths.append(path)

    return paths


def estimate_tau_symmetric(sm, rho):
    tau = np.quantile(sm[np.triu_indices(len(sm))], rho, axis=None)
    return tau


def estimate_tau_assymmetric(sm, rho):
    tau = np.quantile(sm, rho, axis=None)
    return tau
