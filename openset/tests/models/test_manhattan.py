#!/usr/bin/env python3

from unittest import TestCase

import numpy as np

from openset.models import Manhattan


class TestManhattan(TestCase):
    def test_fit(self):
        X = np.array([[0, 0], [0, 4], [2, 0], [2, 4]])

        model = Manhattan()
        model.fit(X)

        actual = model.means[None]
        expected = np.array([1, 2])
        np.testing.assert_almost_equal(actual, expected)

    def test_score(self):
        X = np.array([[0, 0], [0, 4], [2, 0], [2, 4]])

        model = Manhattan()
        model.fit(X)

        actual = model.score(np.array([
            [0, 0],
            [2, 2],
            [1, 3],
            [3, 1],
            [3, 3],
        ]))
        expected = np.array([
            3,
            1,
            1,
            3,
            3,
        ])
        np.testing.assert_almost_equal(actual, expected)
