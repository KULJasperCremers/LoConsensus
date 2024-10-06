import logging

import numpy as np

LOGGER = logging.getLogger(__name__)


# TODO: this has to be a list of timeseries as input
def calculate_similarity_matrix(
    timeseries1: np.ndarray, timeseries2: np.ndarray, GAMMA: int
) -> np.ndarray:
    n = len(timeseries1)
    m = len(timeseries2)
    similarity_matrix = np.full((n, m), -np.inf)
    for row in range(n):
        column_start = 0
        column_end = m
        similarities: list[int] = np.exp(
            -GAMMA
            * np.sum(
                np.power(
                    # use column_start:column_end because of assymetry
                    timeseries1[row, :] - timeseries2[column_start:column_end, :],
                    2,
                ),
                axis=1,
            )
        )
        similarity_matrix[row, column_start:column_end] = similarities
    return similarity_matrix


def calculate_cumulative_similarity_matrix(
    similarity_matrix: np.ndarray,
    STEP_SIZES: np.ndarray,
    tau: float,
    delta_a: float,
    delta_m: float,
) -> np.ndarray:
    max_vertical_step = np.max(STEP_SIZES[:, 0])
    max_horizontal_step = np.max(STEP_SIZES[:, 1])
    n = similarity_matrix.shape[0]
    m = similarity_matrix.shape[1]
    cumulative_similarity_matrix = np.zeros(
        (
            n + max_vertical_step,
            m + max_horizontal_step,
        )
    )

    for row in range(n):
        column_start = 0
        column_end = m
        for column in range(column_start, column_end):
            similarity = similarity_matrix[row, column]
            indices: np.ndarray = (
                np.array([row + max_vertical_step, column + max_horizontal_step])
                - STEP_SIZES
            )
            # look at the 3 possible steps in indices and take the max value
            max_cumulative_similarity: float = np.amax(
                np.array(
                    [
                        cumulative_similarity_matrix[_row, _column]
                        for (_row, _column) in indices
                    ]
                )
            )
            if similarity < tau:
                cumulative_similarity_matrix[
                    row + max_vertical_step, column + max_horizontal_step
                ] = max(0, delta_m * max_cumulative_similarity - delta_a)
            else:
                cumulative_similarity_matrix[
                    row + max_vertical_step, column + max_vertical_step
                ] = max(0, max_cumulative_similarity + similarity)

    return cumulative_similarity_matrix
