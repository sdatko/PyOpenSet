#!/usr/bin/env python3

from unittest import TestCase

import numpy as np

from openset.models import KNearestNeighbors


class TestKNearestNeighbors(TestCase):
    def test_fit(self):
        X = np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4],
                      [1, 0], [1, 1], [1, 2], [1, 3], [1, 4],
                      [2, 0], [2, 1], [2, 2], [2, 3], [2, 4]])

        model = KNearestNeighbors()
        model.fit(X)

        label = None
        self.assertTrue(label in model.classifiers)

        actual = model.classifiers[label].p
        expected = 2
        self.assertEqual(actual, expected)

        actual = model.classifiers[label].n_neighbors
        expected = 5
        self.assertEqual(actual, expected)

        actual = model.classifiers[label].mode
        expected = 'distance'
        self.assertEqual(actual, expected)

    def test_score(self):
        X = np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4],
                      [1, 0], [1, 1], [1, 2], [1, 3], [1, 4],
                      [2, 0], [2, 1], [2, 2], [2, 3], [2, 4]])

        model = KNearestNeighbors()
        model.fit(X)

        actual = model.score(np.array([
            [0, 0],
            [2, 2],
            [1, 3],
            [3, 1],
            [3, 3],
        ]))
        expected = np.array([
            1.0828427,
            0.8828427,
            0.8,
            1.612899,
            1.612899,
        ])
        np.testing.assert_almost_equal(actual, expected)