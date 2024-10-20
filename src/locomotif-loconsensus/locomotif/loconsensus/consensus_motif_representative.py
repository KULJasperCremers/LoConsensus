from dataclasses import dataclass

import numpy as np


@dataclass
class ConsensusMotifRepresentative:
    """A data class that represents a consensus motif.

    Attributes:
        - representative: a tuple that represents the start and end of the motif in the main timeseries.
        - representative_ts: the main timeseries
        - direction : the direction pointing out the POV of the comparison, either 'column' or 'row'
        - motif_set: a list of starts and endings that represent the projection of the representative in the projected timeseries.
        - motif_ts: the projected timeseries.
        - consensus_motif_set: a list of starts and endings that represent the projecten of the representative in the consensus timeseries.
        - consensus_ts: the consensus timeseries
    """

    representative: tuple[int, int]
    representative_ts: np.ndarray
    direction: str
    motif_set: list[tuple[int, int]]
    motif_ts: np.ndarray
    consensus_motif_set: list[tuple[int, int]]
    consensus_ts: np.ndarray
