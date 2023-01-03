#!/usr/bin/env python3

from unittest import TestCase

import numpy as np

from openset.models import Mahalanobis
from openset.models import MahalanobisSC


class TestMahalanobis(TestCase):
    def test_fit(self):
        X = np.array([[0, 0], [0, 4], [2, 0], [2, 4]])

        model = Mahalanobis()
        model.fit(X)

        actual = model.means[None]
        expected = np.array([1, 2])
        np.testing.assert_almost_equal(actual, expected)

        actual = model.icovs[None]
        expected = np.array([[3/4, 0], [0, 3/16]])
        np.testing.assert_almost_equal(actual, expected)

        # corner case
        X = np.array([[0], [0], [2], [2]])

        model = Mahalanobis()
        model.fit(X)

        actual = model.means[None]
        expected = np.array([1])
        np.testing.assert_almost_equal(actual, expected)

        actual = model.icovs[None]
        expected = np.array([[0.75]])
        np.testing.assert_almost_equal(actual, expected)

    def test_score(self):
        X = np.array([[0, 0], [0, 4], [2, 0], [2, 4]])

        model = Mahalanobis()
        model.fit(X)

        actual = model.score(np.array([
            [0, 0],
            [2, 2],
            [1, 3],
            [3, 1],
            [3, 3],
        ]))
        expected = np.array([
            1.2247449,
            0.8660254,
            0.4330127,
            1.7853571,
            1.7853571,
        ])
        np.testing.assert_almost_equal(actual, expected)


class TestMahalanobisSC(TestCase):
    def test_fit(self):
        X = np.array([[0, 0], [0, 4], [2, 0], [2, 4]])

        model = MahalanobisSC()
        model.fit(X)

        actual = model.means[None]
        expected = np.array([1, 2])
        np.testing.assert_almost_equal(actual, expected)

        actual = model.icov
        expected = np.array([[3/4, 0], [0, 3/16]])
        np.testing.assert_almost_equal(actual, expected)

        # corner case
        X = np.array([[0], [0], [2], [2]])

        model = MahalanobisSC()
        model.fit(X)

        actual = model.means[None]
        expected = np.array([1])
        np.testing.assert_almost_equal(actual, expected)

        actual = model.icov
        expected = np.array([[0.75]])
        np.testing.assert_almost_equal(actual, expected)

    def test_score(self):
        X = np.array([[0, 0], [0, 4], [2, 0], [2, 4]])

        model = MahalanobisSC()
        model.fit(X)

        actual = model.score(np.array([
            [0, 0],
            [2, 2],
            [1, 3],
            [3, 1],
            [3, 3],
        ]))
        expected = np.array([
            1.2247449,
            0.8660254,
            0.4330127,
            1.7853571,
            1.7853571,
        ])
        np.testing.assert_almost_equal(actual, expected)
