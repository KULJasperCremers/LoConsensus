import logging
from typing import Generator

import numpy as np
from locomotif.loconsensus import candidate_finder as cf
from locomotif.loconsensus import path as path_class

LOGGER = logging.getLogger(__name__)


def find_motifs(
    max_amount: int, n: int, m: int
) -> Generator[tuple[tuple[int, int], list[path_class.Path], np.ndarray], None, None]:
    start_mask = np.full((n, m), True)
    end_mask = np.full((n, m), True)
    mask = np.full((n, m), True)
    amount = 0
    while amount < max_amount:
        if np.all(mask) or not np.any(start_mask) or not np.any(mask):
            break

        start_mask[mask] = False
        end_mask[mask] = False

    (b, e), best_fitness, fitness = cf.find_candidates()
