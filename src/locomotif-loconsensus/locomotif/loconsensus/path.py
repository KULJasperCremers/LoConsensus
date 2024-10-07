import numpy as np


class Path:
    def __init__(self, path, path_similarities):
        self.path = path
        self.path_similarities = path_similarities
        self.cumulative_path_similarity = np.concatenate(
            (np.array([0, 0]), np.cumsum(self.path_similarities))
        )

    def __getiten__(self, i):
        return self.path[i, :]
