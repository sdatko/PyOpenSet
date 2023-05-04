#!/usr/bin/env python3

from unittest import TestCase

import numpy as np

from openset.models import IntegratedRankWeightedDepth


class TestIntegratedRankWeightedDepth(TestCase):
    def test_repr(self):
        model = IntegratedRankWeightedDepth(1234)
        self.assertEqual(str(model), 'IntegratedRankWeightedDepth(1234)')

    def test_fit(self):
        X = np.array([[0, 0], [0, 4], [2, 0], [2, 4]])

        model = IntegratedRankWeightedDepth(4)
        model.fit(X)

        actual = model.U[None]
        expected = np.array([[0.2811805, 0.6236807, -0.8317571, 0.3747833],
                             [-0.9596549, 0.7816792, -0.5551397, -0.9271125]])
        np.testing.assert_almost_equal(actual, expected)

        actual = model.M[None]
        expected = np.array([[0.0, 0.0, 0.0, 0.0],
                             [-3.8386196, 3.1267168, -2.2205588, -3.7084498],
                             [0.562361, 1.2473614, -1.6635143, 0.7495665],
                             [-3.2762586, 4.3740782, -3.884073, -2.9588833]])
        np.testing.assert_almost_equal(actual, expected)

    def test_score(self):
        X = np.array([[0, 0], [0, 4], [2, 0], [2, 4]])

        model = IntegratedRankWeightedDepth(4)
        model.fit(X)

        actual = model.score(np.array([
            [0, 0],
            [2, 2],
            [1, 3],
            [3, 1],
            [3, 3],
            [6, 0],
        ]))
        expected = np.array([
            -0.1875,
            -0.4375,
            -0.4375,
            -0.375,
            -0.3125,
            -0.0625,
        ])
        np.testing.assert_almost_equal(actual, expected)
