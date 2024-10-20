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


def process_representative_POV_into_consensus_motifs(
    consensus_motifs,
    representative,
    comparison_paths_tuple_index,
    start_index,
    end_index,
    direction,
) -> None:
    """Process a representative based on the POV and create ConsenusMotifRepresentatives."""
    # Column POV specific indexing and attributes
    if direction == 'column':
        representative_ts = representative.comparison_pair[1]
        motif_ts_tuple_index = 0
        consensus_ts_tuple_index = 0
        paths_attr = 'paths'
    # Row POV specific indexing and attributes
    elif direction == 'row':
        representative_ts = representative.comparison_pair[0]
        motif_ts_tuple_index = 1
        consensus_ts_tuple_index = 1
        paths_attr = 'mirrored_paths'

    # retrieve the relevant PathRepresentative objects based on the representative
    relevant_paths_representatives = comparison_paths_tuple_index.get(
        tuple(representative_ts.flatten().tolist()), []
    )

    for paths_representative in relevant_paths_representatives:
        # retrieve the consensus timeseries based on the indexer
        consensus_ts = paths_representative.comparison_pair[consensus_ts_tuple_index]
        # ensure we only find consensus induced paths, and not again induced paths
        if not np.array_equal(
            consensus_ts, representative.comparison_pair[motif_ts_tuple_index]
        ):
            paths = getattr(paths_representative, paths_attr)
            consensus_induced_paths = pf.find_consensus_induced_pathsV0(
                start_index, end_index, paths
            )
            consensus_motif_set = [
                (path[0][0], path[-1][0] + 1) for path in consensus_induced_paths
            ]
            if consensus_motif_set:
                consensus_motif = ConsensusMotifRepresentative(
                    representative=representative.representative,
                    representative_ts=representative_ts,
                    direction=direction,
                    motif_set=representative.motif_set,
                    motif_ts=representative.comparison_pair[motif_ts_tuple_index],
                    consensus_motif_set=consensus_motif_set,
                    consensus_ts=consensus_ts,
                )
                consensus_motifs.append(consensus_motif)


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
    limit = 10
    # for representative in sorted_representatives:
    for representative in sorted_representatives[:limit]:
        (start_index, end_index) = representative.representative
        direction = representative.direction
        process_representative_POV_into_consensus_motifs(
            consensus_motifs,
            representative,
            comparison_paths_tuple_index,
            start_index,
            end_index,
            direction,
        )

    return consensus_motifs
