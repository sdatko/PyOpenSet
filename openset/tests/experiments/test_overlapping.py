#!/usr/bin/env python3

from unittest import TestCase
from unittest.mock import Mock
from unittest.mock import patch

from pony import orm

from openset.experiments import BoundingBoxes


class TestBoundingBoxes(TestCase):
    @patch('openset.experiments.overlapping.BoundingBoxes.db')
    def test_setup_db(self, mock_db):
        BoundingBoxes.setup_db()

        mock_db.bind.assert_called_once()
        mock_db.generate_mapping.assert_called_once()

    @patch('openset.experiments.overlapping.BoundingBoxes.db')
    def test_setup_db_exceptions(self, mock_db):
        str_e = 'Database object was already bound to SQLite provider'
        exception = orm.core.BindingError(str_e)

        mock_db.bind.side_effect = Mock(side_effect=exception)
        BoundingBoxes.setup_db()  # no exception expected here

        str_e = 'This exception should be raised'
        exception = orm.core.BindingError(str_e)

        mock_db.bind.side_effect = Mock(side_effect=exception)
        with self.assertRaises(orm.core.BindingError):
            BoundingBoxes.setup_db()

        exception = Exception('This exception should be raised')
        mock_db.bind.side_effect = Mock(side_effect=exception)

        with self.assertRaises(Exception):
            BoundingBoxes.setup_db()

    @patch('openset.experiments.overlapping.BoundingBoxes.setup_db')
    def test_init_uncached(self, mock_setup_db):
        experiment = BoundingBoxes()

        actual = experiment._cached
        expected = False
        self.assertEqual(actual, expected)

        mock_setup_db.assert_not_called()

    @patch('openset.experiments.overlapping.BoundingBoxes.setup_db')
    def test_init_cached(self, mock_setup_db):
        experiment = BoundingBoxes(cached=True)

        actual = experiment._cached
        expected = True
        self.assertEqual(actual, expected)

        mock_setup_db.assert_called_once()

    @patch('openset.experiments.overlapping.db_filename', ':memory:')
    @patch.object(BoundingBoxes, '_get')
    def test_cache(self, mock_get):
        expected = (1.0, 2.0, 3.0, 4.0)  # dummy values
        mock_get.return_value = expected

        experiment = BoundingBoxes(cached=True)
        mock_get.assert_not_called()

        actual = experiment.get(
            dimension=10,
            distribution='gaussian',
            samples=1000,
            seed=42
        )

        self.assertEqual(actual, expected)
        mock_get.assert_called_once()

        mock_get.return_value = None

        actual = experiment.get(
            dimension=10,
            distribution='gaussian',
            samples=1000,
            seed=42
        )

        self.assertEqual(actual, expected)
        mock_get.assert_called_once()  # second time it comes from the cache

    def test_get(self):
        experiment = BoundingBoxes()

        for distribution in ('correlated-25-25', 'correlated-25-50',
                             'correlated-25-75', 'correlated-50-25',
                             'correlated-50-50', 'correlated-50-75',
                             'correlated-75-25', 'correlated-75-50',
                             'correlated-75-75', 'gaussian',
                             'triangular', 'uniform'):
            result = experiment.get(
                dimension=10,
                distribution=distribution,
                samples=1000,
                seed=42
            )

            self.assertIsInstance(result[0], float)
            self.assertIsInstance(result[1], float)
            self.assertIsInstance(result[2], float)
            self.assertIsInstance(result[3], float)

            actual = len(result)
            expected = 4
            self.assertEqual(actual, expected)

    def test_get_disjoint_sets(self):
        experiment = BoundingBoxes()

        result = experiment.get(
            dimension=10,
            distribution='uniform',
            samples=3,
            seed=42
        )

        self.assertEqual(result[0], float(-1e999))
        self.assertEqual(result[1], 0)
        self.assertEqual(result[2], 0)
