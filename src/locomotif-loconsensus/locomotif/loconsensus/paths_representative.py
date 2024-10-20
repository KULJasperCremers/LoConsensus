from dataclasses import dataclass

import numpy as np
from locomotif.loconsensus import path as path_class


@dataclass
class PathsRepresentative:
    """A data class that represent the paths from a comparison.

    Attributes:
        - comparison_pair: a tuple that represents the two timeseries that are being compared.
        - paths: paths from the Column POV.
        - mirrored_paths: paths from the Row POV.
    """

    comparison_pair: tuple[np.ndarray, np.ndarray]
    paths: list[path_class.Path]
    mirrored_paths: list[path_class.Path]
