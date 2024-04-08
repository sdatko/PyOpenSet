#!/usr/bin/env python3

from unittest import TestCase
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
from pony import orm

from openset.experiments import Correlations
from openset.models import Euclidean


class TestCorrelations(TestCase):
    @patch('openset.experiments.correlations.Correlations.db')
    def test_setup_db(self, mock_db):
        Correlations.setup_db()

        mock_db.bind.assert_called_once()
        mock_db.generate_mapping.assert_called_once()

    @patch('openset.experiments.correlations.Correlations.db')
    def test_setup_db_exceptions(self, mock_db):
        str_e = 'Database object was already bound to SQLite provider'
        exception = orm.core.BindingError(str_e)

        mock_db.bind.side_effect = Mock(side_effect=exception)
        Correlations.setup_db()  # no exception expected here

        str_e = 'This exception should be raised'
        exception = orm.core.BindingError(str_e)

        mock_db.bind.side_effect = Mock(side_effect=exception)
        with self.assertRaises(orm.core.BindingError):
            Correlations.setup_db()

        exception = Exception('This exception should be raised')
        mock_db.bind.side_effect = Mock(side_effect=exception)

        with self.assertRaises(Exception):
            Correlations.setup_db()

    @patch('openset.experiments.correlations.Correlations.setup_db')
    def test_init_uncached(self, mock_setup_db):
        experiment = Correlations()

        actual = experiment._cached
        expected = False
        self.assertEqual(actual, expected)

        mock_setup_db.assert_not_called()

    @patch('openset.experiments.correlations.Correlations.setup_db')
    def test_init_cached(self, mock_setup_db):
        experiment = Correlations(cached=True)

        actual = experiment._cached
        expected = True
        self.assertEqual(actual, expected)

        mock_setup_db.assert_called_once()

    @patch('openset.experiments.correlations.Correlations.db_file', ':memory:')
    @patch.object(Correlations, '_get')
    def test_cache(self, mock_get):
        expected = ([1], [2], [3], 4.0, 5.0)  # dummy values
        mock_get.return_value = expected

        experiment = Correlations(cached=True)
        mock_get.assert_not_called()

        actual = experiment.get(
            distance=8,
            model=Euclidean(),
            seed=42,
            n_correlated=0.5,
            covariance=0.25,
            outliers_correlated=False,
        )

        self.assertEqual(actual, expected)
        mock_get.assert_called_once()

        mock_get.return_value = None

        actual = experiment.get(
            distance=8,
            model=Euclidean(),
            seed=42,
            n_correlated=0.5,
            covariance=0.25,
            outliers_correlated=False,
        )

        self.assertEqual(actual, expected)
        mock_get.assert_called_once()  # second time it comes from the cache

    def test_get(self):
        experiment = Correlations()

        result = experiment.get(
            distance=8,
            model=Euclidean(),
            seed=42,
            n_correlated=0.5,
            covariance=0.25,
            outliers_correlated=False,
        )

        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], np.ndarray)
        self.assertIsInstance(result[2], np.ndarray)
        self.assertIsInstance(result[3], float)
        self.assertIsInstance(result[4], float)

        actual = len(result)
        expected = 5
        self.assertEqual(actual, expected)

        actual = len(result[0])
        expected = 101
        self.assertEqual(actual, expected)

        actual = len(result[1])
        expected = 101
        self.assertEqual(actual, expected)

        actual = len(result[2])
        expected = 101
        self.assertEqual(actual, expected)

        result = experiment.get(
            distance=8,
            model=Euclidean(),
            seed=42,
            n_correlated=0.5,
            covariance=0.25,
            outliers_correlated=True,
        )

        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], np.ndarray)
        self.assertIsInstance(result[2], np.ndarray)
        self.assertIsInstance(result[3], float)
        self.assertIsInstance(result[4], float)

        actual = len(result)
        expected = 5
        self.assertEqual(actual, expected)

        actual = len(result[0])
        expected = 101
        self.assertEqual(actual, expected)

        actual = len(result[1])
        expected = 101
        self.assertEqual(actual, expected)

        actual = len(result[2])
        expected = 101
        self.assertEqual(actual, expected)
