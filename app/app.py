import logging

import numpy as np
import utils
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
    pattern: list[int] = [i for i in range(1, 11)]
    ts1: np.ndarray = tsg.generate_timeseries(100, pattern, 3)
    ts1 = utils.z_normalize(ts1)
    LOGGER.debug(msg=f'timeseries 1: {ts1.flatten()}')
    ts2: np.ndarray = tsg.generate_timeseries(100, pattern, 3)
    ts2 = utils.z_normalize(ts2)
    LOGGER.debug(msg=f'timeseries 2: {ts2.flatten()}')

    # calculate the similarity matrix
    gamma = 1
    sm1 = sm.calculate_similarity_matrix(ts1, ts2, gamma)
    LOGGER.info(msg='Similarity matrix calculated.')

    # calculate the cumulative similarity matrix
    tau = utils.estimate_tau_assymetric(sm1)
    delta_a = 2 * tau
    LOGGER.debug(msg=f'Calculated tau: {tau} and calculated delta_a: {delta_a}.')
    delta_m = 0.5
    csm = sm.calculate_cumulative_similarity_matrix(sm1, tau, delta_a, delta_m)
    LOGGER.info(msg='Cumulative similarity matrix calculated.')

    print(csm.shape)
