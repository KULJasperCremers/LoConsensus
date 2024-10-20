import numpy as np


def calculate_similarity_matrixV1(
    timeseries1: np.ndarray, timeseries2: np.ndarray, GAMMA: int
) -> np.ndarray:
    """Calculate the similarity matrix between two timeseries using specified GAMMA value."""
    squared_differences = np.sum(
        (timeseries1[:, np.newaxis, :] - timeseries2[np.newaxis, :, :]) ** 2, axis=2
    )
    similarities = np.exp(-GAMMA * squared_differences)
    return similarities


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
