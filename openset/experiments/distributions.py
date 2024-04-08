#!/usr/bin/env python3

import os
from time import time

import numpy as np
from pony import orm

from openset.data.generator import ClusterGenerator
from openset.experiments.base import BaseExperiment
from openset.utils.stats import percentiles


TESTING_SET_SIZE = 1000


class Generated(BaseExperiment):
    '''Analyse performance of outlier detection models on generated data.

    Generates three data clusters of given distribution type and dimension:
    one training set and two testing sets (one containing only inliers and
    second containing only outliers). Then it calculates the distance values
    according to a given model class and returns the percentiles for each
    data cluster.

    The number of training samples and generator seed are additional parameters
    useful for analysing the stability of model. The fitting and scoring times
    are also recorded for comprehensive study.
    '''

    db = orm.Database()
    db_file = os.path.join(os.getcwd(), 'distributions.sqlite')

    class Cache(db.Entity):
        _table_ = __qualname__

        # input
        dimension = orm.Required(int)
        distance = orm.Required(int)
        distribution = orm.Required(str)
        model = orm.Required(str)
        samples = orm.Required(int)
        seed = orm.Required(int)

        # output
        train = orm.Required(orm.FloatArray)
        known = orm.Required(orm.FloatArray)
        unknown = orm.Required(orm.FloatArray)
        time_fit = orm.Required(float)
        time_score = orm.Required(float)

        # index
        orm.composite_index(dimension, distance,
                            distribution, model,
                            samples, seed)

    @classmethod
    def setup_db(cls):
        try:
            cls.db.bind(provider='sqlite',
                        filename=cls.db_file,
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
    def _cache(self, dimension, distance, distribution, model, samples, seed):
        # Try cache
        result = self.Cache.get(
            dimension=dimension,
            distance=distance,
            distribution=distribution,
            model=str(model),
            samples=samples,
            seed=seed,
        )

        if not result:  # in cache
            # Compute result
            train, known, unknown, time_fit, time_score = self._get(
                dimension, distance, distribution, model, samples, seed
            )

            # Save in cache
            result = self.Cache(
                dimension=dimension,
                distance=distance,
                distribution=distribution,
                model=str(model),
                samples=samples,
                seed=seed,
                train=train,
                known=known,
                unknown=unknown,
                time_fit=time_fit,
                time_score=time_score,
            )

        # Return the outcome
        return (
            result.train, result.known, result.unknown,
            result.time_fit, result.time_score,
        )

    def _get(self, dimension, distance, distribution, model, samples, seed):
        generator = ClusterGenerator()
        generator.reset(seed=seed)

        distance /= np.sqrt(dimension)

        match distribution:
            case 'gaussian':
                training = generator.gaussian(samples, dimension)
                typicals = generator.gaussian(TESTING_SET_SIZE, dimension)
                outliers = generator.gaussian(TESTING_SET_SIZE, dimension,
                                              location=distance)

            case 'triangular':
                training = generator.triangular(samples, dimension)
                typicals = generator.triangular(TESTING_SET_SIZE, dimension)
                outliers = generator.triangular(TESTING_SET_SIZE, dimension,
                                                left=(distance - 1),
                                                mode=distance,
                                                right=(distance + 1))

            case 'uniform':
                training = generator.uniform(samples, dimension)
                typicals = generator.uniform(TESTING_SET_SIZE, dimension)
                outliers = generator.uniform(TESTING_SET_SIZE, dimension,
                                             low=(distance - 1),
                                             high=(distance + 1))

        time1 = time()

        model.fit(training)

        time2 = time()

        train = percentiles(model.score(training))

        time3 = time()

        known = percentiles(model.score(typicals))
        unknown = percentiles(model.score(outliers))

        time_fit = time2 - time1
        time_score = time3 - time2

        return train, known, unknown, time_fit, time_score
