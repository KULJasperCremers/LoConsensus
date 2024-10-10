import numpy as np


class Path:
    def __init__(self, path: np.ndarray, path_similarities: np.ndarray):
        self.path = path
        self.path_similarities = path_similarities
        self.cumulative_path_similarity = np.concatenate(
            (np.array([0, 0]), np.cumsum(self.path_similarities))
        )
        self.row_start = path[0][0]
        self.row_end = path[len(path) - 1][0] + 1
        self.column_start = path[0][1]
        self.column_end = path[len(path) - 1][1] + 1
        self.construct_indices(path)

    def construct_indices(self, path: np.ndarray) -> None:
        current_row = self.row_start
        current_column = self.column_start

        row_indices = np.zeros(self.row_end - self.row_start, dtype=np.int32)
        column_indices = np.zeros(self.column_end - self.column_start, dtype=np.int32)

        for path_index in range(1, len(path)):
            if path[path_index][0] != current_row:
                row_indices[
                    current_row - self.row_start + 1 : path[path_index][0]
                    - self.row_start
                    + 1
                ] = path_index
                current_row = path[path_index][0]

            if path[path_index][1] != current_column:
                column_indices[
                    current_column - self.column_start + 1 : path[path_index][1]
                    - self.column_start
                    + 1
                ] = path_index
                current_column = path[path_index][1]

        self.row_indices = row_indices
        self.column_indices = column_indices

    def find_row(self, row):
        return self.row_indices[row - self.row_start]

    def find_column(self, column):
        return self.column_indices[column - self.column_start]

    def __getitem__(self, row):
        return self.path[row, :]
