import numpy as np
from numba import float32, int32
from numba.experimental import jitclass

spec = [
    ('path', int32[:, :]),
    ('sim', float32[:]),
    ('cumsim', float32[:]),
    ('index_i', int32[:]),
    ('index_j', int32[:]),
    ('i1', int32),
    ('il', int32),
    ('j1', int32),
    ('jl', int32),
    ('gindex_i', int32[:]),
    ('gindex_j', int32[:]),
    ('gi1', int32),
    ('gil', int32),
    ('gj1', int32),
    ('gjl', int32),
]


@jitclass(spec)
class Path:
    def __init__(self, path, sim):
        assert len(path) == len(sim)
        self.path = path
        self.sim = sim.astype(np.float32)
        self.cumsim = np.concatenate(
            (np.array([0.0], dtype=np.float32), np.cumsum(sim))
        )
        self.i1 = path[0][0]
        self.il = path[len(path) - 1][0] + 1
        self.j1 = path[0][1]
        self.jl = path[len(path) - 1][1] + 1
        self._construct_index(path)

    def __getitem__(self, i):
        return self.path[i, :]

    def __len__(self):
        return len(self.path)

    def _construct_index(self, path):
        i_curr = path[0][0]
        j_curr = path[0][1]

        index_i = np.zeros(self.il - self.i1, dtype=np.int32)
        index_j = np.zeros(self.jl - self.j1, dtype=np.int32)

        for i in range(1, len(path)):
            if path[i][0] != i_curr:
                index_i[i_curr - self.i1 + 1 : path[i][0] - self.i1 + 1] = i
                i_curr = path[i][0]

            if path[i][1] != j_curr:
                index_j[j_curr - self.j1 + 1 : path[i][1] - self.j1 + 1] = i
                j_curr = path[i][1]

        self.index_i = index_i
        self.index_j = index_j

    def find_i(self, i):
        assert i - self.i1 >= 0 and i - self.i1 < len(self.index_i)
        return self.index_i[i - self.i1]

    def find_j(self, j):
        assert j - self.j1 >= 0 and j - self.j1 < len(self.index_j)
        return self.index_j[j - self.j1]

    def _construct_global_index(self, path):
        self.gi1 = path[0][0]
        self.gil = path[len(path) - 1][0] + 1
        self.gj1 = path[0][1]
        self.gjl = path[len(path) - 1][1] + 1

        i_curr = path[0][0]
        j_curr = path[0][1]

        gindex_i = np.zeros(self.il - self.i1, dtype=np.int32)
        gindex_j = np.zeros(self.jl - self.j1, dtype=np.int32)

        for i in range(1, len(path)):
            if path[i][0] != i_curr:
                gindex_i[i_curr - self.i1 + 1 : path[i][0] - self.i1 + 1] = i
                i_curr = path[i][0]

            if path[i][1] != j_curr:
                gindex_j[j_curr - self.j1 + 1 : path[i][1] - self.j1 + 1] = i
                j_curr = path[i][1]

        self.gindex_i = gindex_i
        self.gindex_j = gindex_j
