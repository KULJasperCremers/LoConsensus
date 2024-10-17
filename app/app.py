import logging
from pathlib import Path

import numpy as np
import utils
from locomotif.loconsensus import motif_finder as mf
from locomotif.loconsensus import path as path_class
from locomotif.loconsensus import path_finder, visualize
from locomotif.loconsensus import similarity_matrix as sm
from logger import BASE_LOGGER

logger = BASE_LOGGER
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)

GAMMA = 1
STEP_SIZES = np.array([[1, 1], [2, 1], [1, 2]])

if __name__ == '__main__':
    data1 = np.load('./data/id1.npy', allow_pickle=True)
    data2 = np.load('./data/als1.npy', allow_pickle=True)
    if len(data1) > len(data2):
        ts1, ts2 = data2, data1
    else:
        ts1, ts2 = data1, data2
    sample_frequency = 30
    L_MIN = int(5 * sample_frequency)
    L_MAX = int(25 * sample_frequency)

    # calculate the similarity matrix in both directions
    sm_column: np.ndarray = sm.calculate_similarity_matrixV1(ts2, ts1, GAMMA)
    logger.info(msg='Similarity matrix calculated for the column POV.')
    sm_row: np.ndarray = sm.calculate_similarity_matrixV1(ts1, ts2, GAMMA)
    logger.info(msg='Similarity matrix calculated for the row POV.')

    # calculate the cumulative similarity matrix (identical for both POVs!)
    tau: float = utils.estimate_tau_assymetric(
        sm_column
    )  # or tau: float = utils.estimate_tau_assymetric(sm_row)
    delta_a: float = 2 * tau
    logger.debug(msg=f'Calculated tau: {tau} and calculated delta_a: {delta_a}.')
    delta_m: float = 0.5
    csm: np.ndarray = sm.calculate_cumulative_similarity_matrixV1(
        sm_column,  # or sm_row
        STEP_SIZES,
        tau,
        delta_a,
        delta_m,
    )
    logger.info(msg='Cumulative similarity matrix calculated.')

    # visualize the similarity matrices for both POVs
    fig1, axs1, _ = visualize.plot_sm(ts2, ts1, sm_column)
    fig1.savefig('smcolumn.png')
    fig2, axs2, _ = visualize.plot_sm(ts1, ts2, sm_row)
    fig2.savefig('smrow.png')

    # find the best paths from the cumulative similarity matrix
    found_paths: list[np.ndarray] = path_finder.find_pathsV1(csm, STEP_SIZES, L_MIN)
    logger.info(msg=f'Found {len(found_paths)} paths.')

    # create a Path() for each path including the similarity information
    paths: list[path_class.Path] = []
    mirrored_paths: list[path_class.Path] = []
    for found_path in found_paths:
        rows, columns = found_path[:, 0], found_path[:, 1]
        # column POV: use normal paths and column similarity matrix
        path_similarities = sm_column[rows, columns]
        paths.append(path_class.Path(found_path, path_similarities))

        # row POV: use mirrored paths and row similarity matrix
        mirorred_path = np.array([(column, row) for (row, column) in found_path])
        mirrored_rows, mirrored_columns = mirorred_path[:, 0], mirorred_path[:, 1]
        mirrored_path_similarities = sm_row[mirrored_rows, mirrored_columns]
        mirrored_paths.append(
            path_class.Path(mirorred_path, mirrored_path_similarities)
        )

    # visualize the local warping paths for both POVs
    axs1 = visualize.plot_local_warping_paths(
        axs1, [path.path for path in paths], direction='column', lw=1
    )
    fig1.savefig('pathscolumn.png')
    axs2 = visualize.plot_local_warping_paths(
        axs2, [path.path for path in mirrored_paths], direction='row', lw=1
    )
    fig2.savefig('pathsrow.png')

    # find motifs in the paths
    column_motif_sets: list[tuple[np.ndarray, list[np.ndarray]]] = []
    row_motif_sets: list[tuple[np.ndarray, list[np.ndarray]]] = []
    motif_sets: list[tuple[np.ndarray, list[np.ndarray]]] = []
    max_amount = (
        None  # set to a specific number if you want to limit the number of motif sets
    )
    # max_amount = 5
    for representative, induced_paths, motif_set, direction in mf.find_motifsV1(
        max_amount, len(ts2), len(ts1), paths, mirrored_paths, L_MIN, L_MAX
    ):
        motif_sets.append((representative, motif_set))

        if direction == 'column':
            column_motif_sets.append((representative, motif_set))
            # visualize the representative and the induced paths from the found motif set for the column POV
            axs1 = visualize.plot_local_warping_paths(
                axs1, induced_paths, direction='column', lw=3
            )
            axs1[3].axvline(representative[0], c='k', ls='--')
            axs1[3].axvline(representative[1], c='k', ls='--')
        elif direction == 'row':
            row_motif_sets.append((representative, motif_set))
            # visualize the representative and the induced paths from the found motif set for the row POV
            axs2 = visualize.plot_local_warping_paths(
                axs2, induced_paths, direction='row', lw=3
            )
            axs2[3].axhline(representative[0], c='k', ls='--')
            axs2[3].axhline(representative[1], c='k', ls='--')
    logger.info(msg=f'Found {len(motif_sets)} motif sets.')

    fig1.savefig('motifscolumn.png')
    fig2.savefig('motifsrow.png')

    fig3, axs3 = visualize.plot_motif_sets(ts1, ts2, column_motif_sets, row_motif_sets)
    fig3.savefig('motifs.png')
