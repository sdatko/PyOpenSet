#!/usr/bin/env python3

from unittest import TestCase

import numpy as np

from openset.models import Euclidean


class TestEuclidean(TestCase):
    def test_fit(self):
        X = np.array([[0, 0], [0, 4], [2, 0], [2, 4]])

        model = Euclidean()
        model.fit(X)

        actual = model.means[None]
        expected = np.array([1, 2])
        np.testing.assert_almost_equal(actual, expected)

    def test_score(self):
        X = np.array([[0, 0], [0, 4], [2, 0], [2, 4]])

        model = Euclidean()
        model.fit(X)

        actual = model.score(np.array([
            [0, 0],
            [2, 2],
            [1, 3],
            [3, 1],
            [3, 3],
        ]))
        expected = np.array([
            np.sqrt(5),
            1,
            1,
            np.sqrt(5),
            np.sqrt(5),
        ])
        np.testing.assert_almost_equal(actual, expected)
