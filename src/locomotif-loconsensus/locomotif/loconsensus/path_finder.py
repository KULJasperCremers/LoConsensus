import logging

import numpy as np
from locomotif.loconsensus import path as path_class

LOGGER = logging.getLogger(__name__)
# TODO: what value for V_WIDTH?
V_WIDTH = 2


def find_pathsV1(
    cumulative_similarity_matrix: np.ndarray, STEP_SIZES: np.ndarray, L_MIN: int
) -> list[np.ndarray]:
    """Identify the best warping paths in the cumulative similarity matrix."""
    max_vertical_step = np.max(STEP_SIZES[:, 0])
    max_horizontal_step = np.max(STEP_SIZES[:, 1])

    mask = np.full(cumulative_similarity_matrix.shape, False)
    # set mask to true where csm <= 0
    mask[cumulative_similarity_matrix <= 0] = True

    # extract non-zero indices and sort based on their values
    row_indices, column_indices = np.nonzero(cumulative_similarity_matrix)
    non_zero_values = cumulative_similarity_matrix[row_indices, column_indices]
    sorting_indices = np.argsort(non_zero_values)
    sorted_row_indices: np.ndarray = row_indices[sorting_indices]
    sorted_column_indices: np.ndarray = column_indices[sorting_indices]

    best_index: int = len(sorted_row_indices) - 1
    paths: list[np.ndarray] = []

    while best_index >= 0:
        path = np.empty((0, 0))
        path_found = False
        while not path_found:
            # skip indices already marked in the mask
            while mask[
                sorted_row_indices[best_index], sorted_column_indices[best_index]
            ]:
                best_index -= 1
                if best_index < 0:
                    return paths

            best_row_index: int = sorted_row_indices[best_index]
            best_column_index: int = sorted_column_indices[best_index]
            # check if the current index is within the allowable step range
            if (
                best_row_index < max_vertical_step
                or best_column_index < max_horizontal_step
            ):
                return paths

            # find the maximum warping path starting from the current index
            path: np.ndarray = max_warping_pathV1(
                cumulative_similarity_matrix,
                mask,
                best_row_index,
                best_column_index,
                STEP_SIZES,
                max_horizontal_step,
                max_vertical_step,
            )

            # updat ethe mask to include the found path
            mask: np.ndarray = mask_pathV1(
                path, mask, max_horizontal_step, max_vertical_step
            )

            # check if the path length is greater than or equel to the L_MIN
            if (path[-1][0] - path[0][0] + 1) >= L_MIN or (
                path[-1][1] - path[0][1] + 1
            ) >= L_MIN:
                path_found = True

        # update the mask to include the vicnity of the fond path
        mask = mask_vicinityV0(path, mask, max_horizontal_step, max_vertical_step)
        paths.append(path)

    return paths


def max_warping_pathV1(
    cumulative_similarity_matrix: np.ndarray,
    mask: np.ndarray,
    best_row_index: int,
    best_column_index: int,
    STEP_SIZES: np.ndarray,
    max_horizontal_step: int,
    max_vertical_step: int,
) -> np.ndarray:
    """Trace back the maximum warping path from a given position in the csm."""
    path = []
    # continue as long as the indices are within bounds
    while (
        best_row_index >= max_vertical_step and best_column_index >= max_horizontal_step
    ):
        # insert the current position at the beginning of the path to avoid reversing
        path.insert(
            0,
            (
                best_row_index - max_vertical_step,
                best_column_index - max_horizontal_step,
            ),
        )
        # calculate new indices by substracting the step sizes
        indices = np.array([best_row_index, best_column_index]) - STEP_SIZES
        # extract the values from the csm at new indices
        values = np.array(
            [cumulative_similarity_matrix[_row, _column] for (_row, _column) in indices]
        )
        # extract the values from the mask at new indices
        masked = np.array([mask[_row, _column] for (_row, _column) in indices])
        # find the index of the maximum extracted value, if masked break
        argmax = np.argmax(values)
        if masked[argmax]:
            break

        best_row_index = best_row_index - STEP_SIZES[argmax, 0]
        best_column_index = best_column_index - STEP_SIZES[argmax, 1]

    return np.array(path)


def mask_pathV1(
    path: np.ndarray, mask: np.ndarray, max_horizontal_step: int, max_vertical_step: int
) -> np.ndarray:
    """Update the mask to include the positions covered by the given path."""
    rows, columns = path[:, 0], path[:, 1]
    mask[rows + max_horizontal_step, columns + max_vertical_step] = True
    return mask


def mask_vicinityV0(
    path: np.ndarray, mask: np.ndarray, max_horizontal_step: int, max_vertical_step: int
) -> np.ndarray:
    """Update the mask to include the vicinity around the given path."""
    (row_start, column_start) = path[0] + np.array(
        (max_vertical_step, max_horizontal_step)
    )
    for row_next, column_next in path[1:] + np.array(
        [max_vertical_step, max_horizontal_step]
    ):
        row_difference = row_next - row_start
        column_difference = column_start - column_next
        error = row_difference + column_difference
        # loop until reaching the next point
        while row_start != row_next or column_start != column_next:
            # update vertical vicinity
            mask[row_start - V_WIDTH : row_start + V_WIDTH + 1, column_start] = True
            # update horizontal vicinity
            mask[row_start, column_start - V_WIDTH : column_start + V_WIDTH + 1] = True

            step = 2 * error
            if step > column_difference:
                error += column_difference
                row_start += 1
            if step < row_difference:
                error += row_difference
                column_start += 1

    # final update for last point, ensuring indices are within bounds
    mask[row_next - V_WIDTH : row_next + V_WIDTH + 1, column_next] = True
    mask[row_next, column_next - V_WIDTH : column_next + V_WIDTH + 1] = True
    return mask


def find_induced_paths(
    start_index: int, end_index: int, paths: list[path_class.Path], mask: np.ndarray
) -> list[path_class.Path]:
    """Find paths induced by given start and end indices that do not overlap with masked
    positions.
    """
    induced_paths = []
    for path in paths:
        # check if the path includes the specified index range
        if path.column_start <= start_index and end_index <= path.column_end:
            start_column = path.find_column(start_index)
            end_column = path.find_column(end_index - 1)
            motif_start = path[start_column][0]
            motif_end = path[end_column][0] + 1

            # check if the motif region is not already masked
            if not np.any(mask[motif_start:motif_end]):
                # copy the relevant path segment and add it to the induced paths
                induced_path = np.copy(path.path[start_column : end_column + 1])
                induced_paths.append(induced_path)
    return induced_paths
