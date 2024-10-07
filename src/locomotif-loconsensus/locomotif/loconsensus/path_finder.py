import logging

import numpy as np

LOGGER = logging.getLogger(__name__)
# TODO: what value for V_WIDTH?
V_WIDTH = 2


def find_paths(
    cumulative_similarity_matrix: np.ndarray, STEP_SIZES: np.ndarray, L_MIN: int
) -> list[np.ndarray]:
    max_vertical_step = np.max(STEP_SIZES[:, 0])
    max_horizontal_step = np.max(STEP_SIZES[:, 1])

    mask = np.full(cumulative_similarity_matrix.shape, False)
    row_indices, column_indices = np.nonzero(cumulative_similarity_matrix <= 0)
    for index in range(len(row_indices)):
        # set the positions of each non zero index to True in the mask
        mask[row_indices[index], column_indices[index]] = True

    # sort indices based on the values in D
    row_indices, column_indices = np.nonzero(cumulative_similarity_matrix)
    non_zero_values = np.array(
        [
            cumulative_similarity_matrix[row_indices[index], column_indices[index]]
            for index in range(len(row_indices))
        ]
    )
    sorting_indices = np.argsort(non_zero_values)
    sorted_row_indices: np.ndarray = row_indices[sorting_indices]
    sorted_column_indices: np.ndarray = column_indices[sorting_indices]

    best_index: int = len(sorted_row_indices) - 1
    paths: list[np.ndarray] = []
    while best_index >= 0:
        path = np.empty((0, 0))
        path_found = False
        while not path_found:
            while mask[
                sorted_row_indices[best_index], sorted_column_indices[best_index]
            ]:
                best_index -= 1
                if best_index < 0:
                    return paths

            best_row_index: int = sorted_row_indices[best_index]
            best_column_index: int = sorted_column_indices[best_index]
            if (
                best_row_index < max_vertical_step
                or best_column_index < max_horizontal_step
            ):
                return paths

            path: np.ndarray = max_warping_path(
                cumulative_similarity_matrix,
                mask,
                best_row_index,
                best_column_index,
                STEP_SIZES,
                max_horizontal_step,
                max_vertical_step,
            )

            mask: np.ndarray = mask_path(
                path, mask, max_horizontal_step, max_vertical_step
            )

            if (path[-1][0] - path[0][0] + 1) >= L_MIN or (
                path[-1][1] - path[0][1] + 1
            ) >= L_MIN:
                path_found = True

        mask = mask_vicinity(path, mask, max_horizontal_step, max_vertical_step)
        paths.append(path)

    return paths


def max_warping_path(
    cumulative_similarity_matrix: np.ndarray,
    mask: np.ndarray,
    best_row_index: int,
    best_column_index: int,
    STEP_SIZES: np.ndarray,
    max_horizontal_step: int,
    max_vertical_step: int,
) -> np.ndarray:
    path = []
    while (
        best_row_index >= max_vertical_step and best_column_index >= max_horizontal_step
    ):
        path.append(
            (
                best_row_index - max_vertical_step,
                best_column_index - max_horizontal_step,
            )
        )
        indices = np.array([best_row_index, best_column_index]) - STEP_SIZES
        values = np.array(
            [cumulative_similarity_matrix[_row, _column] for (_row, _column) in indices]
        )
        masked = np.array([mask[_row, _column] for (_row, _column) in indices])
        argmax = np.argmax(values)

        if masked[argmax]:
            break

        best_row_index = best_row_index - STEP_SIZES[argmax, 0]
        best_column_index = best_column_index - STEP_SIZES[argmax, 1]

    path.reverse()
    return np.array(path)


def mask_path(
    path: np.ndarray, mask: np.ndarray, max_horizontal_step: int, max_vertical_step: int
) -> np.ndarray:
    for row, column in path:
        mask[row + max_horizontal_step, column + max_vertical_step] = True
    return mask


def mask_vicinity(
    path: np.ndarray, mask: np.ndarray, max_horizontal_step: int, max_vertical_step: int
) -> np.ndarray:
    (row_start, column_start) = path[0] + np.array(
        (max_vertical_step, max_horizontal_step)
    )
    for row_next, column_next in path[1:] + np.array(
        [max_vertical_step, max_horizontal_step]
    ):
        row_difference = row_next - row_start
        column_difference = column_start - column_next
        error = row_difference + column_difference
        while row_start != row_next or column_start != column_next:
            mask[row_start - V_WIDTH : row_start + V_WIDTH + 1, column_start] = True
            mask[row_start, column_start - V_WIDTH : column_start + V_WIDTH + 1] = True
            step = 2 * error
            if step > column_difference:
                error += column_difference
                row_start += 1
            if step < row_difference:
                error += row_difference
                column_start += 1
    mask[row_next - V_WIDTH : row_next + V_WIDTH + 1, column_next] = True
    mask[row_next, column_next - V_WIDTH : column_next + V_WIDTH + 1] = True
    return mask
