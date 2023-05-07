#!/usr/bin/env python3

import os
from time import time

import numpy as np
from pony import orm

from openset.data.generator import ClusterGenerator
from openset.experiments.base import BaseExperiment
from openset.utils.stats import percentiles


db_filename = os.path.join(os.getcwd(), 'correlations.sqlite')

DIMENSION = 1000
TRAINING_SET_SIZE = 2000
TESTING_SET_SIZE = 1000


class Correlations(BaseExperiment):
    '''Analyse performance of outlier detection models considering correlation.

    Generates three data clusters (one training set and two testing sets) using
    the MVN distributions of fixed dimensions, numbers of samples and distance
    to outliers. Then it calculates the distance values according to a given
    model class and returns the percentiles for each data cluster. The fitting
    and scoring times are also recorded.

    The additional parameters include the generator seed, a number of features
    that are shifted by the distance, a number of features that are correlated
    and the correlation strength (covariance value).
    '''

    db = orm.Database()

    class Cache(db.Entity):
        _table_ = __qualname__

        # input
        distance = orm.Required(int)
        model = orm.Required(str)
        seed = orm.Required(int)
        n_features = orm.Required(float)
        n_correlated = orm.Required(float)
        covariance = orm.Required(float)
        outliers_correlated = orm.Required(bool)

        # output
        train = orm.Required(orm.FloatArray)
        known = orm.Required(orm.FloatArray)
        unknown = orm.Required(orm.FloatArray)
        time_fit = orm.Required(float)
        time_score = orm.Required(float)

        # index
        orm.composite_index(distance, model, seed, outliers_correlated)

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
    def _cache(self, distance, model, seed,
               n_features, n_correlated, covariance, outliers_correlated):
        # Try cache
        result = self.Cache.get(
            model=str(model),
            seed=seed,
            n_features=n_features,
            n_correlated=n_correlated,
            covariance=covariance,
            outliers_correlated=outliers_correlated,
        )

        if not result:  # in cache
            # Compute result
            train, known, unknown, time_fit, time_score = self._get(
                distance, model, seed,
                n_features, n_correlated, covariance, outliers_correlated,
            )

            # Save in cache
            result = self.Cache(
                distance=distance,
                model=str(model),
                seed=seed,
                n_features=n_features,
                n_correlated=n_correlated,
                covariance=covariance,
                outliers_correlated=outliers_correlated,
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

    def _get(self, distance, model, seed,
             n_features, n_correlated, covariance, outliers_correlated):
        generator = ClusterGenerator()
        generator.reset(seed=seed)

        if n_features == 0.0:
            distance = 0.0
        else:
            distance /= np.sqrt(int(n_features * DIMENSION))

        training = generator.mvn(TRAINING_SET_SIZE, DIMENSION,
                                 n_correlated=n_correlated,
                                 covariance=covariance)
        typicals = generator.mvn(TESTING_SET_SIZE, DIMENSION,
                                 n_correlated=n_correlated,
                                 covariance=covariance)

        if outliers_correlated:
            outliers = generator.mvn(TESTING_SET_SIZE, DIMENSION,
                                     location=distance,
                                     n_features=n_features,
                                     n_correlated=n_correlated,
                                     covariance=covariance)
        else:
            outliers = generator.mvn(TESTING_SET_SIZE, DIMENSION,
                                     location=distance,
                                     n_features=n_features)

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
