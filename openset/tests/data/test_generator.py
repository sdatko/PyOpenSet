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
