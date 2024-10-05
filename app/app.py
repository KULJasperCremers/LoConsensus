import logging

import numpy as np
from locomotif.loconsensus import similarity_matrix as sm
from locomotif.loconsensus import timeseries_generator as tsg

logging.basicConfig(
    # level=logging.INFO,
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

if __name__ == '__main__':
    LOGGER = logging.getLogger(__name__)

    # generate two basic timeseries for testing
    pattern: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    ts1: np.ndarray = tsg.generate_timeseries(100, pattern, 3)
    ts2: np.ndarray = tsg.generate_timeseries(100, pattern, 3)

    # generate the similarity matrix
    sm1 = sm.generate_similarity_matrix(ts1, ts2)
    print(sm1)
