#!/usr/bin/env python3

from unittest import TestCase
from unittest.mock import Mock
from unittest.mock import patch

from pony import orm

from openset.experiments import MVNEstimation


class TestMVNEstimation(TestCase):
    @patch('openset.experiments.properties.MVNEstimation.db')
    def test_setup_db(self, mock_db):
        MVNEstimation.setup_db()

        mock_db.bind.assert_called_once()
        mock_db.generate_mapping.assert_called_once()

    @patch('openset.experiments.properties.MVNEstimation.db')
    def test_setup_db_exceptions(self, mock_db):
        str_e = 'Database object was already bound to SQLite provider'
        exception = orm.core.BindingError(str_e)

        mock_db.bind.side_effect = Mock(side_effect=exception)
        MVNEstimation.setup_db()  # no exception expected here

        str_e = 'This exception should be raised'
        exception = orm.core.BindingError(str_e)

        mock_db.bind.side_effect = Mock(side_effect=exception)
        with self.assertRaises(orm.core.BindingError):
            MVNEstimation.setup_db()

        exception = Exception('This exception should be raised')
        mock_db.bind.side_effect = Mock(side_effect=exception)

        with self.assertRaises(Exception):
            MVNEstimation.setup_db()

    @patch('openset.experiments.properties.MVNEstimation.setup_db')
    def test_init_uncached(self, mock_setup_db):
        experiment = MVNEstimation()

        actual = experiment._cached
        expected = False
        self.assertEqual(actual, expected)

        mock_setup_db.assert_not_called()

    @patch('openset.experiments.properties.MVNEstimation.setup_db')
    def test_init_cached(self, mock_setup_db):
        experiment = MVNEstimation(cached=True)

        actual = experiment._cached
        expected = True
        self.assertEqual(actual, expected)

        mock_setup_db.assert_called_once()

    @patch('openset.experiments.properties.MVNEstimation.db_file', ':memory:')
    @patch.object(MVNEstimation, '_get')
    def test_cache(self, mock_get):
        expected = (1.0, 2.0, 3.0, 4.0, 5.0)  # dummy values
        mock_get.return_value = expected

        experiment = MVNEstimation(cached=True)
        mock_get.assert_not_called()

        actual = experiment.get(
            dimension=10,
            samples=1000,
            n_correlated=0.5,
            covariance=0.25,
            seed=42
        )

        self.assertEqual(actual, expected)
        mock_get.assert_called_once()

        mock_get.return_value = None

        actual = experiment.get(
            dimension=10,
            samples=1000,
            n_correlated=0.5,
            covariance=0.25,
            seed=42
        )

        self.assertEqual(actual, expected)
        mock_get.assert_called_once()  # second time it comes from the cache

    def test_get(self):
        experiment = MVNEstimation()

        result = experiment.get(
            dimension=10,
            samples=1000,
            n_correlated=0.5,
            covariance=0.25,
            seed=42
        )

        self.assertIsInstance(result[0], float)
        self.assertIsInstance(result[1], float)
        self.assertIsInstance(result[2], float)
        self.assertIsInstance(result[3], float)
        self.assertIsInstance(result[4], float)

        actual = len(result)
        expected = 5
        self.assertEqual(actual, expected)
