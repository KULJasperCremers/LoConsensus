import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

    # data = Path('./data/mitdb_patient214.csv')
    # file1 = open(data)
    # ts1 = np.array([line.split(',') for line in file1.readlines()], dtype=np.double)
    # ts1 = (ts1 - np.mean(ts1, axis=0)) / np.std(ts1, axis=0)
    # ts2 = ts1.copy()
    # fs = 360
    # L_MIN = int(0.6 * fs)
    # L_MAX = int(1 * fs)

    data1 = np.load('./data/cleanlines/id1_scenario1.npy')
    data2 = np.load('./data/cleanlines/id2_scenario1.npy')
    ts1 = data1 if len(data1) < len(data2) else data2
    ts2 = data1 if len(data1) > len(data2) else data1
    sample_frequency = 60
    L_MIN = int(1 * sample_frequency)
    L_MAX = int(3 * sample_frequency)

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

    fig, axs, _ = visualize.plot_sm(ts1, ts2, sm1)
    fig.savefig('sm.png')

    # find the best paths
    found_paths: list[np.ndarray] = path_finder.find_paths(csm1, STEP_SIZES, L_MIN)
    LOGGER.info(msg=f'Found {len(found_paths)} paths.')

    # create a Path() for each path including the similarity information
    paths: list[path_class.Path] = []
    for found_path in found_paths:
        rows, columns = found_path[:, 0], found_path[:, 1]
        path_similarities = sm1[rows, columns]
        paths.append(path_class.Path(found_path, path_similarities))

    motif_sets = []
    # max_amount = None
    max_amount = 5
    for representative, motif_set in mf.find_motifs(
        max_amount, len(ts1), len(ts2), paths, L_MIN, L_MAX
    ):
        motif_sets.append((representative, motif_set))
    LOGGER.info(msg=f'Found {len(motif_sets)} motif sets.')

    fig, axs = visualize.plot_motif_sets(ts1, ts2, motif_sets)
    fig.savefig('motifs.png')
