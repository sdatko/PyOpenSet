#!/usr/bin/env python3

from unittest import TestCase

import numpy as np

from openset.data import ClusterGenerator


class TestClusterGenerator(TestCase):
    def test_reset(self):
        generator = ClusterGenerator()

        generator.reset(42)
        values1 = generator.gaussian()

        generator.reset(42)
        values2 = generator.gaussian()

        np.testing.assert_almost_equal(values1, values2)

    def test_reset_to_legacy(self):
        generator = ClusterGenerator()

        # > "This generator is considered frozen and will have
        # >  no further improvements. It is guaranteed to produce
        # >  the same values as the final point release of NumPy v1.16."
        # Source: https://numpy.org/doc/stable/reference/random/legacy.html
        generator.reset(seed=42, legacy=True)
        value = generator.gaussian(samples=1, dimension=1)

        np.testing.assert_almost_equal(value, 0.49671415)

    def test_gaussian(self):
        generator = ClusterGenerator()
        generator.reset(seed=42, legacy=True)  # Compatibility guarantee

        actual = generator.gaussian(samples=5, dimension=2,
                                    location=3.0, scale=2.0)
        expected = np.array([
            [3.99342831, 2.72347140],
            [4.29537708, 6.04605971],
            [2.53169325, 2.53172609],
            [6.15842563, 4.53486946],
            [2.06105123, 4.08512009],
        ])

        np.testing.assert_almost_equal(actual, expected)

    def test_mvn(self):
        generator = ClusterGenerator()
        generator.reset(seed=42, legacy=True)  # Compatibility guarantee

        actual = generator.mvn(samples=5, dimension=4, location=3.0, scale=2.0,
                               n_features=0.5, n_correlated=0.5, covariance=1.)
        expected = np.array([
            [+1.31470715, +3.46859663, +0.91596991, -0.19553525],
            [+2.74411984, +3.82943645, +2.23334418, -0.33111966],
            [+3.90430701, +3.24566568, -0.65537159, +0.76729577],
            [+3.10125527, +2.30606062, -2.43940219, -2.70578687],
            [+5.23910924, +3.24181020, -1.28413996, +0.44441284],
        ])

        np.testing.assert_almost_equal(actual, expected)

    def test_traingular(self):
        generator = ClusterGenerator()
        generator.reset(seed=42, legacy=True)  # Compatibility guarantee

        actual = generator.triangular(samples=5, dimension=2,
                                      left=2.0, right=5.0, mode=3.0)
        expected = np.array([
            [3.06279601, 4.45620393],
            [3.73191627, 3.44821100],
            [2.68414613, 2.68409324],
            [2.41743363, 4.10392906],
            [3.45296738, 3.67653314],
        ])

        np.testing.assert_almost_equal(actual, expected)

    def test_uniform(self):
        generator = ClusterGenerator()
        generator.reset(seed=42, legacy=True)  # Compatibility guarantee

        actual = generator.uniform(samples=5, dimension=2,
                                   low=2.0, high=5.0)
        expected = np.array([
            [3.12362036, 4.85214292],
            [4.19598183, 3.79597545],
            [2.46805592, 2.46798356],
            [2.17425084, 4.59852844],
            [3.80334504, 4.12421773],
        ])

        np.testing.assert_almost_equal(actual, expected)
