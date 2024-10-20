from dataclasses import dataclass

import numpy as np


@dataclass
class MotifRepresentative:
    """A data class to represent a motif.

    Attributes:
        - comparison_pair: a tuple that represents the two timeseries that are being compared.
        - direction : the direction pointing out the POV of the comparison, either 'column' or 'row'
        - representative: a tuple that represents the start and end of the motif in the main timeseries.
        - motif_set: a list of starts and endings that represent the projection of the representative in the projected timeseries.
        - fitness: a score indicating the quality of the representative.
    """

    comparison_pair: tuple[np.ndarray, np.ndarray]
    direction: str
    representative: tuple[int, int]
    motif_set: list[tuple[int, int]]
    fitness: float
