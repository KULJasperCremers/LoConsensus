import numpy as np
from numba import float32, int32
from numba.experimental import jitclass

spec = [
    ('path', int32[:, :]),
    ('sim', float32[:]),
    ('cumsim', float32[:]),
    ('index_gi', int32[:]),
    ('index_gj', int32[:]),
    ('gi1', int32),
    ('gil', int32),
    ('gj1', int32),
    ('gjl', int32),
]


@jitclass(spec)
class GlobalPath:
    def __init__(self, path, sim):
        assert len(path) == len(sim)
        self.path = path
        self.sim = sim.astype(np.float32)
        self.cumsim = np.concatenate(
            (np.array([0.0], dtype=np.float32), np.cumsum(sim))
        )
        self.gi1 = path[0][0]
        self.gil = path[len(path) - 1][0] + 1
        self.gj1 = path[0][1]
        self.gjl = path[len(path) - 1][1] + 1
        self._construct_global_index(path)

    def __getitem__(self, i):
        return self.path[i, :]

    def __len__(self):
        return len(self.path)

    def _construct_global_index(self, path):
        i_curr = path[0][0]
        j_curr = path[0][1]

        index_gi = np.zeros(self.gil - self.gi1, dtype=np.int32)
        index_gj = np.zeros(self.gjl - self.gj1, dtype=np.int32)

        for i in range(1, len(path)):
            if path[i][0] != i_curr:
                index_gi[i_curr - self.gi1 + 1 : path[i][0] - self.gi1 + 1] = i
                i_curr = path[i][0]

            if path[i][1] != j_curr:
                index_gj[j_curr - self.gj1 + 1 : path[i][1] - self.gj1 + 1] = i
                j_curr = path[i][1]

        self.index_gi = index_gi
        self.index_gj = index_gj

    def find_gi(self, i):
        assert i - self.gi1 >= 0 and i - self.gi1 < len(self.index_gi)
        return self.index_gi[i - self.gi1]

    def find_gj(self, j):
        assert j - self.gj1 >= 0 and j - self.gj1 < len(self.index_gj)
        return self.index_gj[j - self.gj1]
