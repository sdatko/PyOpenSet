#!/usr/bin/env python3

from functools import wraps
import hashlib
import os
import pickle

from pony import orm


class MemCache(object):
    '''General-purpose in-memory cache.'''

    def __init__(self):
        self.CACHE = {}

    def __call__(self, function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            inputs = hashlib.sha256(pickle.dumps(
                (function.__qualname__, args, kwargs)
            )).hexdigest()

            if inputs in self.CACHE:
                return self.CACHE[inputs]

            else:
                result = function(*args, **kwargs)
                self.CACHE[inputs] = result
                return result

        return wrapper

    def clear(self):
        self.CACHE.clear()


class SQLCache(object):
    '''General-purpose persistent cache.'''

    def __init__(self, filename=None):
        if not filename:
            filename = os.path.join(os.getcwd(), 'cache.sqlite')

        self.db = orm.Database()
        self.db.bind(provider='sqlite', filename=filename, create_db=True)

        class Cache(self.db.Entity):
            inputs = orm.Required(str, index=True, unique=True)
            result = orm.Required(bytes)

        self.db.generate_mapping(create_tables=True)
        self.Cache = Cache

    def __call__(self, function):
        @wraps(function)
        @orm.db_session()
        def wrapper(*args, **kwargs):
            inputs = hashlib.sha256(pickle.dumps(
                (function.__qualname__, args, kwargs)
            )).hexdigest()

            if db_entry := self.Cache.get(inputs=inputs):
                return pickle.loads(db_entry.result)

            else:
                result = function(*args, **kwargs)
                self.Cache(inputs=inputs, result=pickle.dumps(result))
                return result

        return wrapper

    def clear(self):
        self.db.drop_all_tables(with_all_data=True)
        self.db.create_tables()
