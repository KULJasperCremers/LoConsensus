import logging

import numpy as np

LOGGER = logging.getLogger(__name__)
STEP_SIZES = np.array([[1, 1], [2, 1], [1, 2]])


# TODO: this has to be a list of timeseries
def calculate_similarity_matrix(
    timeseries1: np.ndarray, timeseries2: np.ndarray
) -> np.ndarray:
    n = len(timeseries1)
    m = len(timeseries2)
    similarity_matrix: np.ndarray = np.full((n, m), -np.inf)
    for row in range(n):
        column_start = 0
        column_end = m
        similarities: list[int] = np.exp(
            -1.0
            * np.sum(
                np.power(
                    timeseries1[row, :] - timeseries2[column_start:column_end, :], 2
                ),
                axis=1,
            )
        )
        similarity_matrix[row, column_start:column_end] = similarities
    return similarity_matrix


def calculate_cumulative_similarity_matrix(similarity_matrix: np.ndarray) -> np.ndarray:
    pass
