import itertools

import numpy as np
from locomotif.loconsensus import (
    ConsensusMotifRepresentative,
    MotifRepresentative,
    PathsRepresentative,
)
from locomotif.loconsensus import path_finder as pf


def find_consensus_motifsV0(
    comparison_paths: list[PathsRepresentative],
    comparison_representatives: list[list[MotifRepresentative]],
) -> list[ConsensusMotifRepresentative]:
    representatives = list(itertools.chain.from_iterable(comparison_representatives))
    sorted_representatives = sorted(
        representatives, key=lambda r: r.fitness, reverse=True
    )

    consensus_motifs: list[ConsensusMotifRepresentative] = []
    for representative in sorted_representatives:
        (start_index, end_index) = representative.representative
        if representative.direction == 'column':
            representative_ts = representative.comparison_pair[1]
            for comparison_path in comparison_paths:
                comparison_ts = comparison_path.comparison_pair[0]
                if np.array_equal(
                    representative_ts, comparison_path.comparison_pair[1]
                ) and not np.array_equal(
                    representative.comparison_pair[0],
                    comparison_ts,
                ):
                    mask = np.full(len(comparison_ts), False)
                    consensus_induced_paths = pf.find_induced_paths(
                        start_index, end_index, comparison_path.paths, mask
                    )
                    consensus_motif_set = [
                        (path[0][0], path[-1][0] + 1)
                        for path in consensus_induced_paths
                    ]
                    consensus_motif = ConsensusMotifRepresentative(
                        representative=representative.representative,
                        representative_ts=representative_ts,
                        direction='column',
                        motif_set=representative.motif_set,
                        motif_ts=representative.comparison_pair[0],
                        consensus_motif_set=consensus_motif_set,
                        consensus_ts=comparison_ts,
                    )
                    consensus_motifs.append(consensus_motif)
        elif representative.direction == 'row':
            representative_ts = representative.comparison_pair[0]
            for comparison_path in comparison_paths:
                comparison_ts = comparison_path.comparison_pair[1]
                if np.array_equal(
                    comparison_path.comparison_pair[0], representative_ts
                ) and not np.array_equal(
                    comparison_ts, representative.comparison_pair[1]
                ):
                    mask = np.full(len(comparison_ts), False)
                    consensus_induced_paths = pf.find_induced_paths(
                        start_index, end_index, comparison_path.mirrored_paths, mask
                    )
                    consensus_motif_set = [
                        (path[0][0], path[-1][0] + 1)
                        for path in consensus_induced_paths
                    ]
                    consensus_motif = ConsensusMotifRepresentative(
                        representative=representative.representative,
                        representative_ts=representative_ts,
                        direction='row',
                        motif_set=representative.motif_set,
                        motif_ts=representative.comparison_pair[1],
                        consensus_motif_set=consensus_motif_set,
                        consensus_ts=comparison_ts,
                    )
                    consensus_motifs.append(consensus_motif)

    return consensus_motifs
