import numpy as np

RHO = 0.8


# TODO: invalid way to calculate tau, assumes the similarity matrix is symmetric?
def estimate_tau_symmetric(similarity_matrix: np.ndarray):
    return np.quantile(
        similarity_matrix[np.triu_indices(len(similarity_matrix))], RHO, axis=None
    )


def estimate_tau_assymetric(similarity_matrix: np.ndarray) -> float:
    return np.quantile(similarity_matrix, 0.8, axis=None)


def z_normalize(timeseries: np.array) -> np.array:
    mean = np.mean(timeseries)
    std = np.std(timeseries)
    return (timeseries - mean) / std
