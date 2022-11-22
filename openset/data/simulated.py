#!/usr/bin/env python3

import os

import numpy as np


rng = np.random.default_rng(seed=int(os.getenv('PYTHONHASHSEED', 42)))


def reset_rng(seed: int):
    global rng
    rng = np.random.default_rng(seed)


def cluster(samples: int = 10,
            dimension: int = 1,
            location: int = 0) -> np.ndarray:
    '''Generates a random data cluster of the given specification.

    The cluster consists of data vectors generated from the multivariate
    normal distribution. The identity matrix is used as a covariance matrix.

    Parameters
    ----------
    samples : int
        Number of samples in the generated cluster (default: 10).
    dimension : int
        Number of elements in each data vector (default: 1).
    location : int
        The mean value of normal distribution on each dimension (default: 0).
        For `dimension = 3` and `location = 1` it uses `means = [1, 1, 1]`.

    Returns
    -------
    cluster : np.ndarray
        Generated array of data vectors.

    Examples
    --------
    >>> reset_rng(42)  # For tests predictability
    >>> cluster(samples=2, dimension=3)
    array([[ 0.30471708, -1.03998411,  0.7504512 ],
           [ 0.94056472, -1.95103519, -1.30217951]])
    >>> cluster(samples=4, dimension=2, location=5)
    array([[5.1278404 , 4.68375741],
           [4.98319884, 4.14695607],
           [5.87939797, 5.77779194],
           [5.0660307 , 6.12724121]])
    '''

    means = np.full(dimension, location)
    covariance_matrix = np.identity(dimension)

    return rng.multivariate_normal(means, covariance_matrix, samples)
