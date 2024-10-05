import logging

import numpy as np

LOGGER = logging.getLogger(__name__)


def generate_timeseries(length: int, pattern: list[int], n_patterns: int) -> np.ndarray:
    timeseries: np.ndarray = np.random.randint(0, 101, length)
    for _ in range(n_patterns):
        start = np.random.randint(0, length - len(pattern))
        timeseries[start : start + len(pattern)] = pattern

    timeseries: np.ndarray = np.expand_dims(timeseries, axis=1)
    LOGGER.info(msg=f'Timeseries with shape {timeseries.shape} created.')
    return timeseries
