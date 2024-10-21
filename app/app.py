import glob
import logging
import os
import pickle
from itertools import combinations

import numpy as np
import utils
from locomotif.loconsensus import (
    ConsensusMotifRepresentative,
    MotifRepresentative,
    PathsRepresentative,
)
from locomotif.loconsensus import consensus_motif_finder as cmf
from locomotif.loconsensus import motif_finder as mf
from locomotif.loconsensus import path as path_class
from locomotif.loconsensus import path_finder as pf
from locomotif.loconsensus import similarity_matrix as sm
from locomotif.loconsensus import visualize as vis
from logger import BASE_LOGGER

logger = BASE_LOGGER
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)

GAMMA = 1
STEP_SIZES = np.array([[1, 1], [2, 1], [1, 2]])

if __name__ == '__main__':
    # with open('./data/scenario_data.pkl', 'rb') as f:
    # scenario_data = pickle.load(f)
    with open('./data/patient_data_scaled.pkl', 'rb') as f:
        patient_data = pickle.load(f)

    data1 = [
        df[['x', 'y', 'z']].to_numpy()
        for df in patient_data['ALS01']
        if df['scenario'].iloc[0] == 'scenario1' and df['time'].iloc[0] == 'time1'
    ][0]

    data2 = [
        df[['x', 'y', 'z']].to_numpy()
        for df in patient_data['ALS02']
        if df['scenario'].iloc[0] == 'scenario1' and df['time'].iloc[0] == 'time1'
    ][0]

    data3 = [
        df[['x', 'y', 'z']].to_numpy()
        for df in patient_data['ALS03']
        if df['scenario'].iloc[0] == 'scenario1' and df['time'].iloc[0] == 'time1'
    ][0]

    data4 = [
        df[['x', 'y', 'z']].to_numpy()
        for df in patient_data['ALS04']
        if df['scenario'].iloc[0] == 'scenario1' and df['time'].iloc[0] == 'time1'
    ][0]

    data5 = [
        df[['x', 'y', 'z']].to_numpy()
        for df in patient_data['ALS05']
        if df['scenario'].iloc[0] == 'scenario1' and df['time'].iloc[0] == 'time1'
    ][0]

    # sample_frequency = 30
    # L_MIN = int(1 * sample_frequency)
    # L_MAX = int(10 * sample_frequency)
    L_MIN = 100
    L_MAX = 1000

    comparison_paths: list[PathsRepresentative] = []
    comparison_representatives: list[list[ConsensusMotifRepresentative]] = []

    # (n * (n - 1)) / 2 comparisons
    # timeseries_list: list[np.ndarray] = [data1, data2, data3]
    timeseries_list: list[np.ndarray] = [data1, data2, data3, data4, data5]
    n = len(timeseries_list)
    logger.info(msg=f'Performing {int(n * (n - 1) / 2)} comparisons in total.\n')
    for comparison, (ts1, ts2) in enumerate(combinations(timeseries_list, 2)):
        logger.info(msg=f'Executing comparison {comparison+1}:')
        # calculate the similarity matrix in both directions = Column POV and Row POV
        sm_column: np.ndarray = sm.calculate_similarity_matrixV1(ts1, ts2, GAMMA)
        logger.info(msg='Similarity matrix calculated for the column POV.')
        sm_row: np.ndarray = sm.calculate_similarity_matrixV1(ts2, ts1, GAMMA)
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

        # find the best paths from the cumulative similarity matrix
        found_paths: list[np.ndarray] = pf.find_warping_pathsV1(csm, STEP_SIZES, L_MIN)
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

        # append the paths and mirrored paths of the comparison to find consensus motifs
        comparison_paths.append(
            PathsRepresentative(
                comparison_pair=(ts1, ts2), paths=paths, mirrored_paths=mirrored_paths
            )
        )
        # find the motif representatives in the paths
        motif_representatives: list[MotifRepresentative] = []
        max_amount = None
        for (
            representative,
            motif_set,
            direction,
            best_fitness,
        ) in mf.find_motifs_representativesV1(
            max_amount, len(ts1), len(ts2), paths, mirrored_paths, L_MIN, L_MAX
        ):
            # create a MotifRepresentative data class for finding consensus motifs
            found_motif_representatitive = MotifRepresentative(
                comparison_pair=(ts1, ts2),
                direction=direction,
                representative=representative,
                motif_set=motif_set,
                fitness=best_fitness,
            )
            motif_representatives.append(found_motif_representatitive)
        logger.info(
            msg=f'Found {len(motif_representatives)} representatives in comparison {comparison + 1}.\n'
        )

        comparison_representatives.append(motif_representatives)

    logger.info(
        msg=f'Found representatives in {len(comparison_representatives)} out of {int((len(timeseries_list) * (len(timeseries_list) - 1)) / 2)} comparisons.'
    )

    logger.info(
        msg=f'For a total of {sum([len(c) for c in comparison_representatives])} representatives found.\n'
    )

    # find consensus motifs with the comparison paths and representatives
    consensus_motifs: list[ConsensusMotifRepresentative] = cmf.find_consensus_motifsV1(
        comparison_paths, comparison_representatives
    )

    logger.info(msg=f'Found {len(consensus_motifs)} consensus motifs.')

    # remove the old plots and replot found consensus motifs
    files = glob.glob('./plots/consensus_motifs/*')
    for file in files:
        os.remove(file)
    vis.plot_consensus_motifs(consensus_motifs)
