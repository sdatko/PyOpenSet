#!/usr/bin/env python3

from unittest import TestCase

import numpy as np

from openset.models import AngleBasedOutlierFactor


class TestAngleBasedOutlierFactor(TestCase):
    def test_fit(self):
        X = np.array([[0, 0], [0, 4], [2, 0], [2, 4]])

        model = AngleBasedOutlierFactor()
        model.fit(X)

        label = None
        self.assertTrue(label in model.data)

    def test_score(self):
        X = np.array([[0, 0], [0, 4], [2, 0], [2, 4]])

        model = AngleBasedOutlierFactor()
        model.fit(X)

        actual = model.score(np.array([
            [0, 0],
            [2, 2],
            [1, 3],
            [3, 1],
            [3, 3],
        ]))
        expected = np.array([
            0.0005556,
            0.0190972,
            0.0102222,
            0.0081481,
            0.0081481
        ])
        np.testing.assert_almost_equal(actual, expected)
