#!/usr/bin/env python3

from unittest import TestCase
from unittest.mock import patch

from pony import orm

from openset.utils import MemCache
from openset.utils import SQLCache


def _noop():  # pragma: no cover
    '''Dummy function for counting the calls.'''
    pass


class TestMemCache(TestCase):
    @patch('openset.tests.utils.test_cache._noop')
    def test_cache(self, mock_noop):
        cache = MemCache()

        @cache
        def fibonacci(n: int) -> int:
            '''Helper function to be decorated in tests.'''
            _noop()
            if n < 2:
                return n
            return fibonacci(n - 1) + fibonacci(n - 2)

        fibonacci(10)
        fibonacci(10)
        fibonacci(10)

        self.assertEqual(mock_noop.call_count, 11)
        self.assertEqual(len(cache.CACHE), 11)

    @patch('openset.tests.utils.test_cache._noop')
    def test_clear(self, mock_noop):
        cache = MemCache()

        @cache
        def fibonacci(n: int) -> int:
            '''Helper function to be decorated in tests.'''
            _noop()
            if n < 2:
                return n
            return fibonacci(n - 1) + fibonacci(n - 2)

        fibonacci(10)
        fibonacci(10)
        fibonacci(10)

        self.assertEqual(mock_noop.call_count, 11)
        self.assertEqual(len(cache.CACHE), 11)

        cache.clear()
        self.assertEqual(len(cache.CACHE), 0)

        fibonacci(10)
        fibonacci(10)
        fibonacci(10)

        self.assertEqual(mock_noop.call_count, 22)
        self.assertEqual(len(cache.CACHE), 11)


class TestSQLCache(TestCase):
    @patch('openset.tests.utils.test_cache._noop')
    def test_cache(self, mock_noop):
        cache = SQLCache(':memory:')

        @cache
        def fibonacci(n: int) -> int:
            '''Helper function to be decorated in tests.'''
            _noop()
            if n < 2:
                return n
            return fibonacci(n - 1) + fibonacci(n - 2)

        fibonacci(10)
        fibonacci(10)
        fibonacci(10)

        self.assertEqual(mock_noop.call_count, 11)
        with orm.db_session():
            self.assertEqual(cache.Cache.select().count(), 11)

    @patch('openset.tests.utils.test_cache._noop')
    def test_clear(self, mock_noop):
        cache = SQLCache(':memory:')

        @cache
        def fibonacci(n: int) -> int:
            '''Helper function to be decorated in tests.'''
            _noop()
            if n < 2:
                return n
            return fibonacci(n - 1) + fibonacci(n - 2)

        fibonacci(10)
        fibonacci(10)
        fibonacci(10)

        self.assertEqual(mock_noop.call_count, 11)
        with orm.db_session():
            self.assertEqual(cache.Cache.select().count(), 11)

        cache.clear()
        with orm.db_session():
            self.assertEqual(cache.Cache.select().count(), 0)

        fibonacci(10)
        fibonacci(10)
        fibonacci(10)

        self.assertEqual(mock_noop.call_count, 22)
        with orm.db_session():
            self.assertEqual(cache.Cache.select().count(), 11)

    @patch('openset.utils.cache.os.path.join')
    def test_default_filename(self, mock_os_path_join):
        mock_os_path_join.return_value = ':memory:'

        SQLCache()

        self.assertEqual(mock_os_path_join.call_count, 1)
        self.assertEqual(mock_os_path_join.call_args.args[-1], 'cache.sqlite')
