import logging
from typing import Generator

import numpy as np
from locomotif.loconsensus import candidate_finder as cf
from locomotif.loconsensus import path as path_class

LOGGER = logging.getLogger(__name__)


def find_motifs(
    max_amount: int,
    n: int,
    m: int,
    paths: list[path_class.Path],
    L_MIN: int,
    L_MAX: int,
) -> Generator[tuple[tuple[int, int], list[path_class.Path], np.ndarray], None, None]:
    max_length = max(n, m)
    start_mask = np.full(max_length, True)
    end_mask = np.full(max_length, True)
    mask = np.full(max_length, False)
    amount = 0
    while amount < max_amount:
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
