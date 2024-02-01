#!/usr/bin/env python3

from collections.abc import Iterable
import os

import numpy as np


class ClusterGenerator(object):
    '''General-purpose data clusters generator.'''

    def __init__(self):
        self.reset(seed=int(os.getenv('PYTHONHASHSEED', '42')))

    def reset(self, seed: int = 42, legacy: bool = False) -> None:
        '''Resets the state of the pseudo-random number generator instance.

        Parameters
        ----------
        seed : int
            The value passed to initialize the generator (default: 42).
        legacy : bool
            Whether the legacy generator shall be used; e.g. useful for tests,
            as it promises to produce the same values always for a given seed,
            no matter what is set in a current NumPy version (default: False).
        '''
        if legacy:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.default_rng(seed)

    def gaussian(self,
                 samples: int = 10,
                 dimension: int = 1,
                 location: float = 0.0,
                 scale: float = 1.0) -> np.ndarray:
        '''Generates a random data cluster involving Gaussian distribution.

        The cluster consists of data vectors with features generated from
        the Gaussian distribution, i.e. every vector element value lies around
        the [location] and the probability density function has a bell shape
        around that point (about 68.3% of values should be within the range
        of [location ± scale] and about 95.4% within [location ± 2 * scale]).

        Parameters
        ----------
        samples : int
            Number of samples in the generated cluster (default: 10).
        dimension : int
            Number of elements in each data vector (default: 1).
        location : float
            The mean value of normal distribution (default: 0.0).
        scale : float
            The standard deviation of normal distribution (default: 1.0).

        Returns
        -------
        cluster : np.ndarray
            Generated array of data vectors.

        Examples
        --------
        >>> generator = ClusterGenerator()
        >>> generator.reset(42, legacy=True)  # For tests predictability
        >>> generator.gaussian(samples=2, dimension=3)
        array([[ 0.49671415, -0.1382643 ,  0.64768854],
               [ 1.52302986, -0.23415337, -0.23413696]])
        >>> generator.gaussian(samples=4, dimension=2, location=3.0, scale=5.0)
        array([[10.89606408,  6.83717365],
               [ 0.65262807,  5.71280022],
               [ 0.68291154,  0.67135123],
               [ 4.20981136, -6.56640122]])
        '''

        shape = (samples, dimension)

        return self.rng.normal(location, scale, size=shape)

    def mvn(self,
            samples: int = 10,
            dimension: int = 1,
            location: float = 0.0,
            scale: float | Iterable[float] = 1.0,
            n_features: float = 1.0,
            n_correlated: float = 0.0,
            covariance: float = 0.5) -> np.ndarray:
        '''Generates a data cluster involving Multivariate Normal distribution.

        Similar to the Gaussian distribution, however it additionally allows to
        specify the part of features that are moved to location and correlated,
        as well as the correlation strength (i.e. covariance values in matrix).

        Parameters
        ----------
        samples : int
            Number of samples in the generated cluster (default: 10).
        dimension : int
            Number of elements in each data vector (default: 1).
        location : float
            The mean value of normal distribution (default: 0.0).
        scale : float or Iterable[float]
            The standard deviation of normal distribution (default: 1.0).
        n_features : float
            Part of features that are moved to location (default: 1.0).
            E.g. for dimension=10, location=5.0, n_features=0.75:
                [5. 5. 5. 5. 5. 5. 5. 0. 0. 0.]
        n_correlated : float
            Part of features that are correlated (default: 0.0).
        covariance : float
            The covariance value between the correlated values within
            the covariance matrix (default: 0.5).
            E.g. for dimension=10, n_correlated=0.5, covariance=0.25:
                [[1.   0.25 0.25 0.25 0.25 0.   0.   0.   0.   0.  ]
                 [0.25 1.   0.25 0.25 0.25 0.   0.   0.   0.   0.  ]
                 [0.25 0.25 1.   0.25 0.25 0.   0.   0.   0.   0.  ]
                 [0.25 0.25 0.25 1.   0.25 0.   0.   0.   0.   0.  ]
                 [0.25 0.25 0.25 0.25 1.   0.   0.   0.   0.   0.  ]
                 [0.   0.   0.   0.   0.   1.   0.   0.   0.   0.  ]
                 [0.   0.   0.   0.   0.   0.   1.   0.   0.   0.  ]
                 [0.   0.   0.   0.   0.   0.   0.   1.   0.   0.  ]
                 [0.   0.   0.   0.   0.   0.   0.   0.   1.   0.  ]
                 [0.   0.   0.   0.   0.   0.   0.   0.   0.   1.  ]]

        Returns
        -------
        cluster : np.ndarray
            Generated array of data vectors.

        Examples
        --------
        >>> generator = ClusterGenerator()
        >>> generator.reset(42, legacy=True)  # For tests predictability
        >>> generator.mvn(samples=2, dimension=3)
        array([[ 0.49671415, -0.1382643 ,  0.64768854],
               [ 1.52302986, -0.23415337, -0.23413696]])
        >>> generator.mvn(samples=4, dimension=2, location=3.0, scale=5.0)
        array([[ 6.53122721,  4.71603622],
               [ 1.95022336,  4.21320114],
               [ 1.96376654,  1.95859661],
               [ 3.54104409, -1.27822469]])
        >>> generator.mvn(samples=3, dimension=4, location=5.0, n_features=0.5)
        array([[ 3.27508217,  4.43771247, -1.01283112,  0.31424733],
               [ 4.09197592,  3.5876963 ,  1.46564877, -0.2257763 ],
               [ 5.0675282 ,  3.57525181, -0.54438272,  0.11092259]])
        >>> generator.mvn(samples=3, dimension=4, scale=[1, 10, 5, 20])
        array([[-0.29169375,  1.18806145, -1.34306894, -5.14739976],
               [-1.05771093,  5.85741792, -0.03018071, -2.69091377],
               [-1.95967012, -3.8606466 ,  0.4670332 ,  3.67853268]])
        '''

        means = np.zeros(shape=(dimension,))
        means[:int(n_features * dimension)] = location

        cov = np.zeros(shape=(dimension, dimension))
        index = int(n_correlated * dimension)
        cov[:index, :index] = covariance
        np.fill_diagonal(cov, scale)

        return self.rng.multivariate_normal(means, cov, size=samples)

    def triangular(self,
                   samples: int = 10,
                   dimension: int = 1,
                   left: float = -1.0,
                   mode: float = 0.0,
                   right: float = 1.0) -> np.ndarray:
        '''Generates a random data cluster involving triangular distribution.

        The cluster consists of data vectors with features generated from
        the triangular distribution, i.e. every vector element value comes
        from the range [left, right] with the probability increasing linearly
        towards the [mode] value.

        Parameters
        ----------
        samples : int
            Number of samples in the generated cluster (default: 10).
        dimension : int
            Number of elements in each data vector (default: 1).
        left : float
            Lower limit for the output values (default: -1.0).
        mode : float
            The peak value of the distribution (default: 0.0).
        right : float
            Upper limit for the output values (default: 1.0).

        Returns
        -------
        cluster : np.ndarray
            Generated array of data vectors.

        Examples
        --------
        >>> generator = ClusterGenerator()
        >>> generator.reset(42, legacy=True)  # For tests predictability
        >>> generator.triangular(samples=2, dimension=3)
        array([[-0.13450578,  0.68603919,  0.26787152],
               [ 0.1040742 , -0.44139703, -0.44144021]])
        >>> generator.triangular(4, 2, left=2.0, mode=3.0, right=5.0)
        array([[2.41743363, 4.10392906],
               [3.45296738, 3.67653314],
               [2.24850248, 4.57509897],
               [3.99733148, 2.79813366]])
        '''

        left, mode, right = sorted((left, mode, right))
        shape = (samples, dimension)

        return self.rng.triangular(left, mode, right, size=shape)

    def uniform(self,
                samples: int = 10,
                dimension: int = 1,
                low: float = -1.0,
                high: float = 1.0) -> np.ndarray:
        '''Generates a random data cluster involving uniform distribution.

        The cluster consists of data vectors with features generated from
        the uniform distribution, i.e. every vector element value comes with
        an equal probability from the range [low; high).

        Parameters
        ----------
        samples : int
            Number of samples in the generated cluster (default: 10).
        dimension : int
            Number of elements in each data vector (default: 1).
        low : float
            Lower boundary for the output interval (default: -1.0).
        high : float
            Upper boundary for the output interval (default: 1.0).

        Returns
        -------
        cluster : np.ndarray
            Generated array of data vectors.

        Examples
        --------
        >>> generator = ClusterGenerator()
        >>> generator.reset(42, legacy=True)  # For tests predictability
        >>> generator.uniform(samples=2, dimension=3)
        array([[-0.25091976,  0.90142861,  0.46398788],
               [ 0.19731697, -0.68796272, -0.68801096]])
        >>> generator.uniform(samples=4, dimension=2, low=3.0, high=5.0)
        array([[3.11616722, 4.73235229],
               [4.20223002, 4.41614516],
               [3.04116899, 4.9398197 ],
               [4.66488528, 3.42467822]])
        '''

        low, high = sorted((low, high))
        shape = (samples, dimension)

        return self.rng.uniform(low, high, size=shape)
