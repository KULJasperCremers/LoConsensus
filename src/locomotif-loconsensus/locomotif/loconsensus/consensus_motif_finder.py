import itertools
from collections import defaultdict

import numpy as np
from locomotif.loconsensus import (
    ConsensusMotifRepresentative,
    MotifRepresentative,
    PathsRepresentative,
)
from locomotif.loconsensus import path_finder as pf


def tuple_indexer_comparison_paths(
    comparison_paths: list[PathsRepresentative],
) -> defaultdict[tuple[float, ...], list[PathsRepresentative]]:
    """Index comparison paths based on their comparison pairs for efficient lookup."""
    index = defaultdict(list)
    for comparison_path in comparison_paths:
        # convert comparison pairs into tuples for use as dict keys
        index[tuple(comparison_path.comparison_pair[0].flatten().tolist())].append(
            comparison_path
        )
        index[tuple(comparison_path.comparison_pair[1].flatten().tolist())].append(
            comparison_path
        )
    return index


def process_representative_POV_into_consensus_results(
    comparison_paths_tuple_index: defaultdict[
        tuple[float, ...], list[PathsRepresentative]
    ],
    representative_ts: np.ndarray,
    motif_ts: np.ndarray,
    consensus_ts_tuple_index: int,
    paths_attr: str,
    start_index: int,
    end_index: int,
) -> list[tuple[list[tuple[int, int]], np.ndarray]]:
    """Process a representative based on the POV and return all consensus results."""
    processing_consensus_results = []

    # retrieve the relevant PathRepresentative objects based on the representative
    relevant_paths_representatives = comparison_paths_tuple_index.get(
        tuple(representative_ts.flatten().tolist()), []
    )

    for paths_representative in relevant_paths_representatives:
        # retrieve the consensus timeseries based on the indexer
        consensus_ts = paths_representative.comparison_pair[consensus_ts_tuple_index]
        # ensure we only find consensus induced paths, and not again induced paths
        if not np.array_equal(consensus_ts, motif_ts):
            paths = getattr(paths_representative, paths_attr)
            consensus_induced_paths = pf.find_consensus_induced_pathsV0(
                start_index, end_index, paths
            )
            consensus_motif_set = [
                (path[0][0], path[-1][0] + 1) for path in consensus_induced_paths
            ]
            if consensus_motif_set:
                processing_consensus_results.append((consensus_motif_set, consensus_ts))

    return processing_consensus_results


def find_consensus_motifsV1(
    comparison_paths: list[PathsRepresentative],
    comparison_representatives: list[list[MotifRepresentative]],
) -> list[ConsensusMotifRepresentative]:
    """Find consensus motifs by processing sorted representatives and indexing comparison paths."""
    consensus_motifs: list[ConsensusMotifRepresentative] = []

    # sort the representatives by fitness score
    representatives = list(itertools.chain.from_iterable(comparison_representatives))
    sorted_representatives = sorted(
        representatives, key=lambda r: r.fitness, reverse=True
    )
    # set up indexing for easy access to the comparison paths
    comparison_paths_tuple_index = tuple_indexer_comparison_paths(comparison_paths)

    # TODO: limit the amount of representatives searched for consensus motifs
    # limit = 10
    # for representative in sorted_representatives[:limit]:
    for representative in sorted_representatives:
        (start_index, end_index) = representative.representative
        direction = representative.direction
        # Column POV specific indexing and attributes
        if direction == 'column':
            representative_ts = representative.comparison_pair[1]
            motif_ts = representative.comparison_pair[0]
            consensus_ts_tuple_index = 0
            paths_attr = 'paths'
        # Row POV specific indexing and attributes
        elif direction == 'row':
            representative_ts = representative.comparison_pair[0]
            motif_ts = representative.comparison_pair[1]
            consensus_ts_tuple_index = 1
            paths_attr = 'mirrored_paths'

        consensus_results = process_representative_POV_into_consensus_results(
            comparison_paths_tuple_index,
            representative_ts,
            motif_ts,
            consensus_ts_tuple_index,
            paths_attr,
            start_index,
            end_index,
        )

        # no consensus results for this representative
        if not consensus_results:
            continue

        # aggregate the consensus results
        consensus_motif_set_list = []
        consensus_ts_list = []
        for consensus_motif_set, consensus_ts in consensus_results:
            consensus_motif_set_list.append(consensus_motif_set)
            consensus_ts_list.append(consensus_ts)

        consensus_motif = ConsensusMotifRepresentative(
            representative=representative.representative,
            fitness=representative.fitness,
            representative_ts=representative_ts,
            direction=direction,
            motif_set=representative.motif_set,
            motif_ts=motif_ts,
            consensus_motif_set_list=consensus_motif_set_list,
            consensus_ts_list=consensus_ts_list,
        )
        consensus_motifs.append(consensus_motif)

    return consensus_motifs
