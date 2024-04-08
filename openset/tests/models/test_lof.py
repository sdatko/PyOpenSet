#!/usr/bin/env python3

from unittest import TestCase

import numpy as np

from openset.models import LocalOutlierFactor


class TestLocalOutlierFactor(TestCase):
    def test_repr(self):
        model = LocalOutlierFactor(15)
        self.assertEqual(str(model), 'LocalOutlierFactor(15)')

    def test_fit(self):
        X = np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4],
                      [1, 0], [1, 1], [1, 2], [1, 3], [1, 4],
                      [2, 0], [2, 1], [2, 2], [2, 3], [2, 4]])

        model = LocalOutlierFactor()
        model.fit(X)

        label = None
        self.assertTrue(label in model.classifiers)

        actual = model.classifiers[label].p
        expected = 2
        self.assertEqual(actual, expected)

        actual = model.classifiers[label].n_neighbors
        expected = 10
        self.assertEqual(actual, expected)

        actual = model.classifiers[label].novelty
        expected = True
        self.assertEqual(actual, expected)

    def test_score(self):
        X = np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4],
                      [1, 0], [1, 1], [1, 2], [1, 3], [1, 4],
                      [2, 0], [2, 1], [2, 2], [2, 3], [2, 4]])

        model = LocalOutlierFactor()
        model.fit(X)

        actual = model.score(np.array([
            [0, 0],
            [2, 2],
            [1, 3],
            [3, 1],
            [3, 3],
        ]))
        expected = np.array([
            1.0513142,
            0.9881312,
            0.9881312,
            1.0472042,
            1.0577083,
        ])
        np.testing.assert_almost_equal(actual, expected)

    def test_small_dataset(self):
        X = np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]])

        model = LocalOutlierFactor()
        model.fit(X)

        actual = model.score(np.array([
            [0, 0],
            [2, 2],
        ]))
        expected = np.array([
            0.9258242,
            0.9258242,
        ])
        np.testing.assert_almost_equal(actual, expected)
