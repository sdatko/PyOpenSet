#!/usr/bin/env python3

from unittest import TestCase

import numpy as np

from openset.models import Cosine


class TestCosine(TestCase):
    def test_fit(self):
        X = np.array([[0, 0], [0, 4], [2, 0], [2, 4]])

        model = Cosine()
        model.fit(X)

        actual = model.means[None]
        expected = np.array([1, 2])
        np.testing.assert_almost_equal(actual, expected)

    def test_score(self):
        X = np.array([[0, 0], [0, 4], [2, 0], [2, 4]])

        model = Cosine()
        model.fit(X)

        actual = model.score(np.array([
            [1, 1],
            [2, 2],
            [1, 3],
            [3, 1],
            [3, 3],
        ]))
        expected = np.array([  # 1 - (u dot v) / sqrt(|u|² * |v|²)
            1 - np.dot([1, 1], [1, 2]) / np.sqrt((1 + 1) * (1 + 4)),
            1 - np.dot([2, 2], [1, 2]) / np.sqrt((4 + 4) * (1 + 4)),
            1 - np.dot([1, 3], [1, 2]) / np.sqrt((1 + 9) * (1 + 4)),
            1 - np.dot([3, 1], [1, 2]) / np.sqrt((9 + 1) * (1 + 4)),
            1 - np.dot([3, 3], [1, 2]) / np.sqrt((9 + 9) * (1 + 4)),
        ])
        np.testing.assert_almost_equal(actual, expected)
