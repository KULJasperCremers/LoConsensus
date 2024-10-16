import logging
from typing import Generator

import numpy as np
from locomotif.loconsensus import candidate_finder as cf
from locomotif.loconsensus import path as path_class
from locomotif.loconsensus import path_finder as pf

LOGGER = logging.getLogger(__name__)


OVERLAP = 0.0


def find_motifsV1(
    max_amount: int,
    n: int,
    m: int,
    paths: list[path_class.Path],
    L_MIN: int,
    L_MAX: int,
) -> Generator[
    tuple[tuple[int, int], list[path_class.Path], list[tuple[int, int]]], None, None
]:
    """Generate motifs by finding and masking the best candidates based on fitness scores."""
    # determine the max length since the two timeseries are no longer equal
    max_length = max(n, m)
    start_mask = np.full(max_length, True)
    end_mask = np.full(max_length, True)
    mask = np.full(max_length, False)
    amount = 0

    while max_amount is None or amount < max_amount:
        # break fi all positions are masked or no valid start/end positions are left
        if np.all(mask) or not np.any(start_mask) or not np.any(end_mask):
            break

        # update masks to exclude already masked positions
        start_mask &= ~mask
        end_mask &= ~mask

        best_candidate, best_fitness = cf.find_candidatesV1(
            start_mask, end_mask, mask, paths, L_MIN, L_MAX, OVERLAP
        )

        LOGGER.debug(
            msg=f'Best candidate: {best_candidate}\nBest fitness: {best_fitness}'
        )

        # break if no better candidate can be found
        if best_fitness == 0.0:
            break

        (start_index, end_index) = best_candidate
        induced_paths = pf.find_induced_paths(start_index, end_index, paths, mask)
        motif_set = [(path[0][0], path[-1][0] + 1) for path in induced_paths]

        for motif_start, motif_end in motif_set:
            motif_length = motif_end - motif_start
            overlap = int(OVERLAP * motif_length)

            # ensure valid adjusted indices that account for overlap
            start_index = motif_start + overlap
            end_index = motif_end - overlap
            start_index = max(0, start_index)
            end_index = min(max_length, end_index)

            # skip if adjusted indices are invalid
            if start_index >= end_index:
                continue

            mask[start_index:end_index] = True

        amount += 1
        yield best_candidate, induced_paths, motif_set
