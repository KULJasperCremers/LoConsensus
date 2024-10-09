import logging

import numpy as np
import utils
from locomotif.loconsensus import motif_finder as mf
from locomotif.loconsensus import path as path_class
from locomotif.loconsensus import path_finder, visualize
from locomotif.loconsensus import similarity_matrix as sm
from locomotif.loconsensus import timeseries_generator as tsg

logging.basicConfig(
    level=logging.INFO,
    # level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

GAMMA = 1
STEP_SIZES = np.array([[1, 1], [2, 1], [1, 2]])
# dependent on the length of the pattern!
L_MIN = 5
L_MAX = 10


if __name__ == '__main__':
    LOGGER = logging.getLogger(__name__)

    # generate two basic timeseries for testing
    pattern = [i for i in range(1, 101)]
    ts1: np.ndarray = tsg.generate_timeseries(1000, pattern, 5)
    ts1: np.ndarray = utils.z_normalize(ts1)
    LOGGER.debug(msg=f'timeseries 1: {ts1.flatten()}')
    ts2: np.ndarray = tsg.generate_timeseries(1000, pattern, 5)
    ts2: np.ndarray = utils.z_normalize(ts2)
    LOGGER.debug(msg=f'timeseries 2: {ts2.flatten()}')

    # calculate the similarity matrix
    sm1: np.ndarray = sm.calculate_similarity_matrix(ts1, ts2, GAMMA)
    LOGGER.info(msg='Similarity matrix calculated.')

    # calculate the cumulative similarity matrix
    tau: float = utils.estimate_tau_assymetric(sm1)
    delta_a = 2 * tau
    LOGGER.debug(msg=f'Calculated tau: {tau} and calculated delta_a: {delta_a}.')
    delta_m = 0.5
    csm1: np.ndarray = sm.calculate_cumulative_similarity_matrix(
        sm1, STEP_SIZES, tau, delta_a, delta_m
    )
    LOGGER.info(msg='Cumulative similarity matrix calculated.')

    fig, axs, _ = visualize.plot_similarity_matrix(ts1, ts2, sm1)
    fig.savefig('sm.png')
    fig, axs, _ = visualize.plot_similarity_matrix(ts1, ts2, csm1)
    fig.savefig('csm.png')

    # find the best paths
    found_paths = path_finder.find_paths(csm1, STEP_SIZES, L_MIN)
    LOGGER.info(msg=f'Found {len(found_paths)} paths.')

    # create a Path() for each path including the similarity information
    paths = []
    for found_path in found_paths:
        rows, columns = found_path[:, 0], found_path[:, 1]
        path_similarities = sm1[rows, columns]
        paths.append(path_class.Path(found_path, path_similarities))

    axs = visualize.plot_local_warping_paths(
        axs, [path_object.path for path_object in paths], lw=1
    )
    fig.savefig('paths.png')
