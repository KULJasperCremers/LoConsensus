import logging

import numpy as np
from locomotif.loconsensus import path as path_class

LOGGER = logging.getLogger(__name__)


# TODO: keep_fitness?
def find_candidatesV1(
    start_mask: np.ndarray,
    end_mask: np.ndarray,
    mask: np.ndarray,
    paths: list[path_class.Path],
    L_MIN: int,
    L_MAX: int,
    OVERLAP: float,
):
    """Identify the best candidate motif within specified masks and paths using fitness
    evaluation.
    """
    mask_length = len(start_mask)
    # extract the start and end indices of the paths
    column_start_indices = np.array([path.column_start for path in paths])
    column_end_indices = np.array([path.column_end for path in paths])
    best_start_indices = np.zeros(len(paths), dtype=np.int32)
    best_end_indices = np.zeros(len(paths), dtype=np.int32)
    crossing_start_indices = np.zeros(len(paths), dtype=np.int32)
    crossing_end_indices = np.zeros(len(paths), dtype=np.int32)
    best_fitness = 0.0
    best_candidate = (0, mask_length)

    # compute the cumulative sum of the mask
    mask_cumulative_sum = np.concatenate(([0], np.cumsum(mask)))

    for start_index in range(mask_length - L_MIN + 1):
        # skip if start index is masked
        if not start_mask[start_index]:
            continue

        # identify the paths that start before or at the start index
        start_indices_mask = column_start_indices <= start_index

        for end_index in range(
            start_index + L_MIN, min(mask_length + 1, start_index + L_MAX + 1)
        ):
            # skip if end index is masked
            if not end_mask[end_index - 1]:
                continue

            # check if the current range overlaps with any masked positions
            if np.any(
                mask_cumulative_sum[end_index] - mask_cumulative_sum[start_index] > 0
            ):
                break

            # identify the paths that end after or at the end index
            end_indices_mask = column_end_indices >= end_index

            # continue with paths that have full coverage
            path_mask = start_indices_mask & end_indices_mask
            if not np.any(path_mask):
                break

            for path_index in np.flatnonzero(path_mask):
                path = paths[path_index]
                # find columns in the path corresponding to the start and end indices
                crossing_start_indices[path_index] = path_row = path.find_column(
                    start_index
                )
                crossing_end_indices[path_index] = path_column = path.find_column(
                    end_index - 1
                )
                # get the actual positions in the path
                best_start_indices[path_index] = path[path_row][0]
                best_end_indices[path_index] = path[path_column][0] + 1

                # check overlaps with masked positions in the path segment and exlude
                if np.any(
                    mask[best_start_indices[path_index] : best_end_indices[path_index]]
                ):
                    path_mask[path_index] = False

            # break if no valid paths remain
            if not np.any(path_mask):
                break

            sorted_best_start_indices = best_start_indices[path_mask]
            sorted_best_end_indices = best_end_indices[path_mask]
            sorted_best_indices = np.argsort(sorted_best_start_indices)
            sorted_best_start_indices = sorted_best_start_indices[sorted_best_indices]
            sorted_best_end_indices = sorted_best_end_indices[sorted_best_indices]

            # calculate the overlap between consecutive paths
            overlaps = np.maximum(
                sorted_best_end_indices[:-1] - sorted_best_start_indices[1:] - 1, 0
            )

            # check overlaps exceed allowed overlap
            if np.any(
                overlaps
                > OVERLAP
                * (sorted_best_end_indices[:-1] - sorted_best_start_indices[:-1])
            ):
                continue

            # compute total coverage of motif
            coverage = np.sum(
                sorted_best_end_indices - sorted_best_start_indices
            ) - np.sum(overlaps)
            coverage_amount = (coverage - (end_index - start_index)) / float(
                mask_length
            )
            # ensure non-negative coverage
            coverage_amount = max(coverage_amount, 0.0)

            path_indices = np.flatnonzero(path_mask)
            crossing_starts = crossing_start_indices[path_indices]
            crossing_ends = crossing_end_indices[path_indices]

            # compute cumulative similarity scores at start and end of path segments
            cumulative_sums_at_starts = np.array(
                [
                    paths[path_index].cumulative_path_similarity[crossing_start]
                    for path_index, crossing_start in zip(path_indices, crossing_starts)
                ]
            )
            cumulative_sums_at_ends = np.array(
                [
                    paths[path_index].cumulative_path_similarity[crossing_end + 1]
                    for path_index, crossing_end in zip(path_indices, crossing_ends)
                ]
            )

            # calculate score differences for the motif
            score_differences = cumulative_sums_at_ends - cumulative_sums_at_starts
            score = np.sum(score_differences)

            total_length = np.sum(crossing_ends - crossing_starts + 1)
            score_amount = (score - (end_index - start_index)) / float(total_length)
            # ensure non-negative score
            score_amount = max(score_amount, 0.0)

            # compute fitness score using haronic mean of coverage and score
            denominator = coverage_amount + score_amount
            if denominator != 0:
                fit = 2 * (coverage_amount * score_amount) / denominator
            else:
                fit = 0.0

            if fit == 0.0:
                continue

            if fit > best_fitness:
                best_candidate = (start_index, end_index)
                best_fitness = fit

    return best_candidate, best_fitness
