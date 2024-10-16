import logging
from pathlib import Path

import numpy as np
import pandas as pd
import utils
from locomotif.loconsensus import motif_finder as mf
from locomotif.loconsensus import path as path_class
from locomotif.loconsensus import path_finder, visualize
from locomotif.loconsensus import similarity_matrix as sm

logging.basicConfig(
    level=logging.INFO,
    # level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

GAMMA = 1
STEP_SIZES = np.array([[1, 1], [2, 1], [1, 2]])

if __name__ == '__main__':
    LOGGER = logging.getLogger(__name__)

    data1 = np.load('./id1.npy', allow_pickle=True)
    # data1 = pd.DataFrame(data1, columns=['x', 'y'])
    data2 = np.load('./als1.npy', allow_pickle=True)
    # data2 = pd.DataFrame(data2, columns=['x', 'y'])
    if len(data1) > len(data2):
        ts1, ts2 = data2, data1
    else:
        ts1, ts2 = data1, data2
    sample_frequency = 30
    L_MIN = int(3 * sample_frequency)
    L_MAX = int(10 * sample_frequency)

    # calculate the similarity matrix
    assert len(ts2) > len(ts1)
    sm1: np.ndarray = sm.calculate_similarity_matrixV1(ts2, ts1, GAMMA)
    LOGGER.info(msg='Similarity matrix calculated.')

    # calculate the cumulative similarity matrix
    tau: float = utils.estimate_tau_assymetric(sm1)
    delta_a: float = 2 * tau
    LOGGER.debug(msg=f'Calculated tau: {tau} and calculated delta_a: {delta_a}.')
    delta_m: float = 0.5
    csm1: np.ndarray = sm.calculate_cumulative_similarity_matrixV1(
        sm1, STEP_SIZES, tau, delta_a, delta_m
    )
    LOGGER.info(msg='Cumulative similarity matrix calculated.')

    fig, axs, _ = visualize.plot_sm(ts2, ts1, sm1)
    fig.savefig('sm.png')

    # find the best paths from the cumulative similarity matrix
    found_paths: list[np.ndarray] = path_finder.find_pathsV1(csm1, STEP_SIZES, L_MIN)
    LOGGER.info(msg=f'Found {len(found_paths)} paths.')

    # create a Path() for each path including the similarity information
    paths: list[path_class.Path] = []
    for found_path in found_paths:
        rows, columns = found_path[:, 0], found_path[:, 1]
        path_similarities = sm1[rows, columns]
        paths.append(path_class.Path(found_path, path_similarities))

    # visualize the local warping paths
    axs = visualize.plot_local_warping_paths(axs, [path.path for path in paths], lw=1)

    # find motifs in the paths
    motif_sets: list[tuple[np.ndarray, list[np.ndarray]]] = []
    max_amount = (
        None  # set to a specific number if you want to limit the number of motif sets
    )
    # max_amount = 5
    for representative, induced_paths, motif_set in mf.find_motifsV1(
        max_amount, len(ts2), len(ts1), paths, L_MIN, L_MAX
    ):
        motif_sets.append((representative, motif_set))
        # visualize the representative and the induced paths from the found motif set
        axs = visualize.plot_local_warping_paths(axs, induced_paths, lw=3)
        axs[3].axvline(representative[0], c='k', ls='--')
        axs[3].axvline(representative[1], c='k', ls='--')
    LOGGER.info(msg=f'Found {len(motif_sets)} motif sets.')

    fig.savefig('paths.png')

    # visualize the motif sets TODO: show the left timeseries and motifs next to eachother
    fig, axs = visualize.plot_motif_sets(ts2, ts1, motif_sets)
    fig.savefig('motifs.png')
