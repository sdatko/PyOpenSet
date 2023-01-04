#!/usr/bin/env python3

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
