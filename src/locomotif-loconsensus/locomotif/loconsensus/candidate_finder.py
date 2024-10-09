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
    pass
