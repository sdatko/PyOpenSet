#!/usr/bin/env python3

from unittest import TestCase

import numpy as np

from openset.models import MinMaxWindow


class TestMinMaxWindow(TestCase):
    def test_fit(self):
        X = np.array([[0, 0], [0, 4], [2, 0], [2, 4]])

        model = MinMaxWindow()
        model.fit(X)

        actual = model.mins[None]
        expected = np.array([0, 0])
        np.testing.assert_almost_equal(actual, expected)

        actual = model.maxes[None]
        expected = np.array([2, 4])
        np.testing.assert_almost_equal(actual, expected)

    def test_score(self):
        X = np.array([[0, 0], [0, 4], [2, 0], [2, 4]])

        model = MinMaxWindow()
        model.fit(X)

        actual = model.score(np.array([
            [0, 0],
            [2, 2],
            [1, 3],
            [3, 1],
            [3, 3],
            [1, 5],
            [3, 5],
        ]))
        expected = np.array([
            1,
            1,
            1,
            0.5,
            0.5,
            0.5,
            0,
        ])
        np.testing.assert_almost_equal(actual, expected)
