#!/usr/bin/env python3

from unittest import TestCase
from unittest.mock import MagicMock
from unittest.mock import patch

from openset.utils import Runner


class TestRunner(TestCase):
    def test_initializer_argument(self):
        runner = Runner(6)

        actual = runner.nproc
        expected = 6
        self.assertEqual(actual, expected)

    @patch('openset.utils.runner.cpu_count')
    def test_initializer_default(self, mock_cpu_count):
        mock_cpu_count.return_value = 4

        runner = Runner()

        actual = runner.nproc
        expected = 4
        self.assertEqual(actual, expected)

    @patch('openset.utils.runner.tqdm')
    @patch('openset.utils.runner.Pool')
    def test_run_simple(self, mock_pool, mock_tqdm):
        mock_pool_instance = mock_pool.return_value.__enter__.return_value
        mock_pool_instance.imap_unordered.return_value = None

        runner = Runner()

        function = MagicMock()
        arguments = (1, 2, 3, 4)
        runner.run(function, arguments)

        mock_pool_instance.imap_unordered.assert_called_once_with(
            func=function,
            iterable=arguments
        )
        mock_tqdm.assert_called_once_with(None, total=4)

    @patch('openset.utils.runner.tqdm')
    @patch('openset.utils.runner.Pool')
    def test_run_with_iterator(self, mock_pool, mock_tqdm):
        mock_pool_instance = mock_pool.return_value.__enter__.return_value
        mock_pool_instance.imap_unordered.return_value = None

        runner = Runner()

        function = MagicMock()
        arguments = iter([1, 2, 3, 4])
        runner.run(function, arguments)

        mock_pool_instance.imap_unordered.assert_called_once_with(
            func=function,
            iterable=arguments
        )
        mock_tqdm.assert_called_once_with(None, total=None)

    @patch('openset.utils.runner.tqdm')
    @patch('openset.utils.runner.Pool')
    def test_run_unpack(self, mock_pool, mock_tqdm):
        mock_pool_instance = mock_pool.return_value.__enter__.return_value
        mock_pool_instance.imap_unordered.return_value = [1, 2, 3]
        mock_tqdm.return_value = [1, 2, 3]

        runner = Runner()

        function = MagicMock()
        arguments = ([1, 2], [3, 4], [5, 6])
        runner.run(function, arguments, unpack=True)

        mock_pool_instance.imap_unordered.assert_called_once_with(
            func=runner._starmap,
            iterable=arguments
        )
        mock_tqdm.assert_called_once_with([1, 2, 3], total=3)

        actual = runner.func
        expected = function
        self.assertEqual(actual, expected)

    def test_starmap(self):
        runner = Runner(4)
        runner.func = MagicMock()

        args = (1, 2, 3, 4, 5)
        runner._starmap(args)

        runner.func.assert_called_with(1, 2, 3, 4, 5)
