#!/usr/bin/env python3

from decimal import Decimal
import os
from time import time

import numpy as np
from pony import orm

from openset.data.generator import ClusterGenerator
from openset.experiments.base import BaseExperiment
from openset.utils.stats import percentiles


DIMENSION = 1000
TRAINING_SET_SIZE = 2000
TESTING_SET_SIZE = 1000
DEFAULT_VARIANCE = 1.0


class Variances(BaseExperiment):
    '''Analyse performance of outlier detection models considering variances.

    Generates three data clusters (one training set and two testing sets) using
    the MVN distributions of fixed dimensions, numbers of samples and distance
    to outliers. Then it calculates the distance values according to a given
    model class and returns the percentiles for each data cluster. The fitting
    and scoring times are also recorded.

    The additional parameters include the generator seed, a number of features
    that have different variance from default and the strength of the variance.
    '''

    db = orm.Database()
    db_file = os.path.join(os.getcwd(), 'variances.sqlite')

    class Cache(db.Entity):
        _table_ = __qualname__

        # input
        distance = orm.Required(int)
        model = orm.Required(str)
        seed = orm.Required(int)
        n_varied = orm.Required(Decimal)
        variance = orm.Required(Decimal)
        outliers_varied = orm.Required(bool)

        # output
        train = orm.Required(orm.FloatArray)
        known = orm.Required(orm.FloatArray)
        unknown = orm.Required(orm.FloatArray)
        time_fit = orm.Required(float)
        time_score = orm.Required(float)

        # index
        orm.composite_index(distance, model, seed,
                            n_varied, variance, outliers_varied)

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
    def _cache(self, distance, model, seed,
               n_varied, variance, outliers_varied):
        # Try cache
        result = self.Cache.get(
            distance=distance,
            model=str(model),
            seed=seed,
            n_varied=n_varied,
            variance=variance,
            outliers_varied=outliers_varied,
        )

        if not result:  # in cache
            # Compute result
            train, known, unknown, time_fit, time_score = self._get(
                distance, model, seed,
                n_varied, variance, outliers_varied,
            )

            # Save in cache
            result = self.Cache(
                distance=distance,
                model=str(model),
                seed=seed,
                n_varied=n_varied,
                variance=variance,
                outliers_varied=outliers_varied,
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

    def _get(self, distance, model, seed, n_varied, variance, outliers_varied):
        generator = ClusterGenerator()
        generator.reset(seed=seed)

        variances = np.full(DIMENSION, DEFAULT_VARIANCE)
        variances[:int(n_varied * DIMENSION)] = variance

        training = generator.mvn(TRAINING_SET_SIZE, DIMENSION,
                                 scale=variances)
        typicals = generator.mvn(TESTING_SET_SIZE, DIMENSION,
                                 scale=variances)

        distance /= np.sqrt(DIMENSION)

        if outliers_varied:
            outliers = generator.mvn(TESTING_SET_SIZE, DIMENSION,
                                     location=distance,
                                     scale=variances)
        else:
            outliers = generator.mvn(TESTING_SET_SIZE, DIMENSION,
                                     location=distance)

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
