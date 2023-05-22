#!/usr/bin/env python3

import os
from time import time

import numpy as np
from pony import orm

from openset.data.generator import ClusterGenerator
from openset.experiments.base import BaseExperiment


db_filename = os.path.join(os.getcwd(), 'overlapping.sqlite')


class BoundingBoxes(BaseExperiment):
    '''Calculates the common volume of bounding boxes around two data clusters.

    Additionally returns the percentage factors of the common volume
    to the volume of bounding box around the first and second data cluster.
    '''

    db = orm.Database()

    class Cache(db.Entity):
        _table_ = __qualname__

        # input
        dimension = orm.Required(int)
        distribution = orm.Required(str)
        samples = orm.Required(int)
        seed = orm.Required(int)

        # output
        volume = orm.Required(float)
        factor1 = orm.Required(float)
        factor2 = orm.Required(float)
        time = orm.Required(float)

        # index
        orm.composite_index(dimension, distribution, samples, seed)

    @classmethod
    def setup_db(cls):
        try:
            cls.db.bind(provider='sqlite',
                        filename=db_filename,
                        create_db=True)
            cls.db.generate_mapping(create_tables=True)
        except orm.core.BindingError as e:
            expected = 'Database object was already bound to SQLite provider'
            if str(e) != expected:
                raise

    def __init__(self, cached=False):
        self._cached = cached

        if self._cached:
            self.setup_db()

    @orm.db_session()
    def _cache(self, dimension, distribution, samples, seed):
        # Try cache
        result = self.Cache.get(
            dimension=dimension,
            distribution=distribution,
            samples=samples,
            seed=seed,
        )

        if not result:  # in cache
            # Compute result
            volume, factor1, factor2, time = self._get(
                dimension, distribution, samples, seed
            )

            # Save in cache
            result = self.Cache(
                dimension=dimension,
                distribution=distribution,
                samples=samples,
                seed=seed,
                volume=volume,
                factor1=factor1,
                factor2=factor2,
                time=time,
            )

        # Return the outcome
        return result.volume, result.factor1, result.factor2, result.time

    def _get(self, dimension, distribution, samples, seed):
        generator = ClusterGenerator()
        generator.reset(seed=seed)

        time1 = time()

        match distribution:
            case 'correlated2525':
                set1 = generator.mvn(samples, dimension,
                                     n_correlated=0.25, covariance=0.25)
                set2 = generator.mvn(samples, dimension,
                                     n_correlated=0.25, covariance=0.25)

            case 'correlated5050':
                set1 = generator.mvn(samples, dimension,
                                     n_correlated=0.5, covariance=0.5)
                set2 = generator.mvn(samples, dimension,
                                     n_correlated=0.5, covariance=0.5)

            case 'correlated7575':
                set1 = generator.mvn(samples, dimension,
                                     n_correlated=0.75, covariance=0.75)
                set2 = generator.mvn(samples, dimension,
                                     n_correlated=0.75, covariance=0.75)

            case 'gaussian':
                set1 = generator.gaussian(samples, dimension)
                set2 = generator.gaussian(samples, dimension)

            case 'triangular':
                set1 = generator.triangular(samples, dimension)
                set2 = generator.triangular(samples, dimension)

            case 'uniform':
                set1 = generator.uniform(samples, dimension)
                set2 = generator.uniform(samples, dimension)

        mins1 = set1.min(axis=0)
        mins2 = set2.min(axis=0)
        maxes1 = set1.max(axis=0)
        maxes2 = set2.max(axis=0)

        common_mins = np.maximum(mins1, mins2)
        common_maxes = np.minimum(maxes1, maxes2)

        # NOTE: with high dimensions like 1000, we can easily hit here
        #       the overflow (float's infinity) when using numpy.prod(),
        #       hence the log scale is more convenient to use (with sum()).
        #       With log10 we receive the order of magnitude for values,
        #       i.e. -3 means the value is equal to 10^-3, so 0.001.
        volume1 = np.sum(np.log10(np.subtract(maxes1, mins1)))
        volume2 = np.sum(np.log10(np.subtract(maxes2, mins2)))
        common_volume = np.sum(np.log10(np.subtract(common_maxes,
                                                    common_mins)))

        factor1 = 100 * 10**(common_volume - volume1)
        factor2 = 100 * 10**(common_volume - volume2)

        time2 = time()

        return common_volume, factor1, factor2, (time2 - time1)
