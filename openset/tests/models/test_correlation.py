#!/usr/bin/env python3

from unittest import TestCase

import numpy as np

from openset.models import Correlation


class TestCorrelation(TestCase):
    def test_fit(self):
        X = np.array([[0, 0], [0, 4], [2, 0], [2, 4]])

        model = Correlation()
        model.fit(X)

        actual = model.means[None]
        expected = np.array([1, 2])
        np.testing.assert_almost_equal(actual, expected)

    def test_score(self):
        X = np.array([[0, 0], [0, 4], [2, 0], [2, 4]])

        model = Correlation()
        model.fit(X)

        actual = model.score(np.array([
            [2, 4],
            [4, 2],
            [1, 3],
            [3, 1],
            [3, 6],
        ]))
        expected = np.array([
            0,
            2,
            0,
            2,
            0,
        ])
        np.testing.assert_almost_equal(actual, expected)
