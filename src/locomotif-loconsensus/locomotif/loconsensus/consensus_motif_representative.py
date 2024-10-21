from dataclasses import dataclass

import numpy as np


@dataclass
class ConsensusMotifRepresentative:
    """A data class that represents a consensus motif.

    Attributes:
        - representative: a tuple that represents the start and end of the motif in the main timeseries.
        - fitness: a score indicating the quality of the representative.
        - representative_ts: the main timeseries
        - direction : the direction pointing out the POV of the comparison, either 'column' or 'row'
        - motif_set: a list of starts and endings that represent the projection of the representative in the projected timeseries.
        - motif_ts: the projected timeseries.
        - consensus_motif_set_list: a list of a list of starts and endings that represent the projecten of the representative in the consensus timeseries.
        - consensus_ts_list: a list of the consensus timeseries, the index matches the consensus_motif_set_list index for linkage.
    """

    representative: tuple[int, int]
    fitness: float
    representative_ts: np.ndarray
    direction: str
    motif_set: list[tuple[int, int]]
    motif_ts: np.ndarray
    consensus_motif_set_list: list[list[tuple[int, int]]]
    consensus_ts_list: list[np.ndarray]
