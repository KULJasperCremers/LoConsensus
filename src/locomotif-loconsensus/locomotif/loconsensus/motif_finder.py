import logging
from typing import Generator

import numpy as np
from locomotif.loconsensus import candidate_finder as cf
from locomotif.loconsensus import path as path_class
from locomotif.loconsensus import path_finder as pf

LOGGER = logging.getLogger(__name__)


def find_motifs(
    max_amount: int,
    n: int,
    m: int,
    paths: list[path_class.Path],
    L_MIN: int,
    L_MAX: int,
) -> Generator[tuple[tuple[int, int], list[tuple[int, int]]], None, None]:
    max_length = max(n, m)
    start_mask = np.full(max_length, True)
    end_mask = np.full(max_length, True)
    mask = np.full(max_length, False)
    amount = 0
    while max_amount is None or amount < max_amount:
        if np.all(mask) or not np.any(start_mask) or not np.any(end_mask):
            break

        start_mask[mask] = False
        end_mask[mask] = False

        best_candidate, best_fitness = cf.find_candidates(
            start_mask, end_mask, mask, paths, L_MIN, L_MAX
        )

        LOGGER.debug(
            msg=f'Best candidate: {best_candidate}\nBest fitness: {best_fitness}'
        )

        if best_fitness == 0.0:
            break

        (start_index, end_index) = best_candidate
        motif_set = [
            (path[0][0], path[len(path) - 1][0] + 1)
            for path in pf.find_induced_paths(start_index, end_index, paths, mask)
        ]

        # TODO: overlap parameter?
        for motif_start, motif_end in motif_set:
            motif_length = motif_end - motif_start
            mask[
                motif_start + int(0.0 * motif_length) - 1 : motif_end
                - int(0.0 * motif_length)
            ] = True

        amount += 1
        yield best_candidate, motif_set
