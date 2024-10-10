import logging

import numpy as np
from locomotif.loconsensus import path as path_class

LOGGER = logging.getLogger(__name__)


# TODO: keep_fitness?
def find_candidates(
    start_mask: np.ndarray,
    end_mask: np.ndarray,
    mask: np.ndarray,
    paths: list[path_class.Path],
    L_MIN: int,
    L_MAX: int,
):
    n = len(start_mask)

    column_start_indices = np.array([path.column_start for path in paths])
    column_end_indicdes = np.array([path.column_end for path in paths])

    best_start_indices = np.zeros(len(paths), dtype=np.int32)
    best_end_indices = np.zeros(len(paths), dtype=np.int32)

    crossing_start_indices = np.zeros(len(paths), dtype=np.int32)
    crossing_end_indices = np.zeros(len(paths), dtype=np.int32)

    best_fitness = 0.0
    best_candidate = (0, n)

    for start_index in range(n - L_MIN + 1):
        if not start_mask[start_index]:
            continue

        start_indices_mask = column_start_indices <= start_index

        for end_index in range(
            start_index + L_MIN, min(n + 1, start_index + L_MAX + 1)
        ):
            if not end_mask[end_index - 1]:
                continue
            if np.any(mask[start_index:end_index]):
                break

            end_indices_mask = column_end_indicdes >= end_index
            path_mask = start_indices_mask & end_indices_mask

            if not np.any(path_mask[1:]):
                break

            for path_index in np.flatnonzero(path_mask):
                path = paths[path_index]
                # TODO: twee keer find_column???
                crossing_start_indices[path_index] = path_row = path.find_column(
                    start_index
                )
                crossing_end_indices[path_index] = path_column = path.find_column(
                    end_index - 1
                )
                best_start_indices[path_index] = path[path_row][0]
                best_end_indices[path_index] = path[path_column][0] + 1

                if np.any(
                    mask[best_start_indices[path_index] : best_end_indices[path_index]]
                ):
                    path_mask[path_index] = False

            if not np.any(path_mask[1:]):
                break

            sorted_best_start_indices = best_start_indices[path_mask]
            sorted_best_end_indices = best_end_indices[path_mask]
            sorted_best_indices = np.argsort(sorted_best_start_indices)
            sorted_best_start_indices = sorted_best_start_indices[sorted_best_indices]
            sorted_best_end_indices = sorted_best_end_indices[sorted_best_indices]

            overlap = sorted_best_end_indices - sorted_best_start_indices
            overlap[:-1] = np.minimum(overlap[:-1], overlap[1:])
            overlaps = np.maximum(
                sorted_best_end_indices[:-1] - sorted_best_start_indices[1:] - 1, 0
            )

            # TODO: overlap param
            if np.any(overlaps > 0.0 * overlap[:-1]):
                continue

            coverage = np.sum(
                sorted_best_end_indices - sorted_best_start_indices
            ) - np.sum(overlaps)
            coverage_amount = (coverage - (end_index - start_index)) / float(n)

            score = 0
            for path_index in np.flatnonzero(path_mask):
                score += (
                    paths[path_index].cumulative_path_similarity[
                        crossing_end_indices[path_index] + 1
                    ]
                    - paths[path_index].cumulative_path_similarity[
                        crossing_start_indices[path_index]
                    ]
                )
            score_amount = (score - (end_index - start_index)) / float(
                np.sum(
                    crossing_end_indices[path_mask]
                    - crossing_start_indices[path_mask]
                    + 1
                )
            )

            fit = 0.0
            if coverage_amount != 0 or score_amount != 0:
                fit = (
                    2
                    * (coverage_amount * score_amount)
                    / (coverage_amount + score_amount)
                )

            if fit == 0.0:
                continue

            if fit > best_fitness:
                best_candidate = (start_index, end_index)
                best_fitness = fit

    return best_candidate, best_fitness
