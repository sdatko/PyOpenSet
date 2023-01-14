#!/usr/bin/env python3

from unittest import TestCase

import numpy as np

from openset.utils.stats import deciles
from openset.utils.stats import percentiles
from openset.utils.stats import quartiles


class TestStats(TestCase):
    def test_deciles(self):
        data = np.array(range(10001))

        actual = deciles(data)
        expected = np.array([0, 1000, 2000, 3000, 4000, 5000,
                             6000, 7000, 8000, 9000, 10000])
        np.testing.assert_almost_equal(actual, expected)

    def test_percentiles(self):
        data = np.array(range(10001))

        actual = percentiles(data)
        expected = np.array(list(range(0, 10001, 100)))
        np.testing.assert_almost_equal(actual, expected)

    def test_quartiles(self):
        data = np.array(range(10001))

        actual = quartiles(data)
        expected = np.array([0, 2500, 5000, 7500, 10000])
        np.testing.assert_almost_equal(actual, expected)
