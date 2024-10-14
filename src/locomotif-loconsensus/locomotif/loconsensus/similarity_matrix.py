import logging

import numpy as np

LOGGER = logging.getLogger(__name__)


# TODO: this has to be a list of timeseries as input
def calculate_similarity_matrixV1(
    timeseries1: np.ndarray, timeseries2: np.ndarray, GAMMA: int
) -> np.ndarray:
    """Calculate the similarity matrix between two timeseries using specified GAMMA value."""
    n = len(timeseries1)
    m = len(timeseries2)
    similarity_matrix = np.full((n, m), -np.inf)
    for row in range(n):
        squared_difference = np.sum(
            np.power(timeseries1[row, np.newaxis, :] - timeseries2, 2), axis=1
        )
        similarities = np.exp(-GAMMA * squared_difference)
        similarity_matrix[row, :] = similarities
    return similarity_matrix


def calculate_cumulative_similarity_matrixV1(
    similarity_matrix: np.ndarray,
    STEP_SIZES: np.ndarray,
    tau: float,
    delta_a: float,
    delta_m: float,
) -> np.ndarray:
    """Calculate the cumulative similarity matrix from the similarity matrix."""
    max_vertical_step = np.max(STEP_SIZES[:, 0])
    max_horizontal_step = np.max(STEP_SIZES[:, 1])
    n, m = similarity_matrix.shape
    cumulative_similarity_matrix = np.zeros(
        (
            n + max_vertical_step,
            m + max_horizontal_step,
        )
    )

    for row in range(n):
        for column in range(m):
            similarity: float = similarity_matrix[row, column]
            indices = (
                np.array([row + max_vertical_step, column + max_horizontal_step])
                - STEP_SIZES
            )
            # look at the 3 possible steps in indices and take the max value
            max_cumulative_similarity = np.amax(
                cumulative_similarity_matrix[indices[:, 0], indices[:, 1]]
            )
            if similarity < tau:
                cumulative_similarity_matrix[
                    row + max_vertical_step, column + max_horizontal_step
                ] = max(0, delta_m * max_cumulative_similarity - delta_a)
            else:
                cumulative_similarity_matrix[
                    row + max_vertical_step, column + max_horizontal_step
                ] = max(0, max_cumulative_similarity + similarity)

    return cumulative_similarity_matrix
