#!/usr/bin/env python3

import os
from time import time

import numpy as np
from pony import orm

from openset.data.generator import ClusterGenerator
from openset.experiments.base import BaseExperiment


db_filename = os.path.join(os.getcwd(), 'properties.sqlite')


class MVNEstimation(BaseExperiment):
    '''Generate a data cluster and attempt to estimate distribution properties.

    It returns the mean squared errors (MSE) of empirically calculated mean
    and covariance matrix elements. Additionally the error for estimation of
    just variances (diagonal components) and covariances elements (all but
    diagonal components) is reported.
    '''

    db = orm.Database()

    class Cache(db.Entity):
        _table_ = __qualname__

        # input
        dimension = orm.Required(int)
        samples = orm.Required(int)
        n_correlated = orm.Required(float)
        covariance = orm.Required(float)
        seed = orm.Required(int)

        # output
        mse_means = orm.Required(float)
        mse_cov = orm.Required(float)
        mse_vars = orm.Required(float)
        mse_covs = orm.Required(float)
        time = orm.Required(float)

        # index
        orm.composite_index(dimension, samples, seed)

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
    def _cache(self, dimension, samples, n_correlated, covariance, seed):
        # Try cache
        result = self.Cache.get(
            dimension=dimension,
            samples=samples,
            n_correlated=n_correlated,
            covariance=covariance,
            seed=seed,
        )

        if not result:  # in cache
            # Compute result
            mse_means, mse_cov, mse_vars, mse_covs, time = self._get(
                dimension, samples, n_correlated, covariance, seed
            )

            # Save in cache
            result = self.Cache(
                dimension=dimension,
                samples=samples,
                n_correlated=n_correlated,
                covariance=covariance,
                seed=seed,
                mse_means=mse_means,
                mse_cov=mse_cov,
                mse_vars=mse_vars,
                mse_covs=mse_covs,
                time=time,
            )

        # Return the outcome
        return (
            result.mse_means, result.mse_cov,
            result.mse_vars, result.mse_covs,
            result.time,
        )

    def _get(self, dimension, samples, n_correlated, covariance, seed):
        generator = ClusterGenerator()
        generator.reset(seed=seed)

        time1 = time()

        data = generator.mvn(samples, dimension,
                             n_correlated=n_correlated,
                             covariance=covariance)

        means = data.mean(axis=0)
        cov = np.cov(data.T)

        expected_means = np.zeros(shape=(dimension,))
        expected_cov = np.identity(dimension)
        for row in range(int(n_correlated * dimension)):
            for col in range(int(n_correlated * dimension)):
                if row == col:
                    continue
                expected_cov[row, col] = covariance

        delta_means = means - expected_means
        delta_cov = cov - expected_cov

        mse_means = np.square(delta_means).mean()
        mse_cov = np.square(delta_cov).mean()
        mse_vars = np.square(delta_cov * np.identity(dimension)).mean()
        mse_covs = np.square(delta_cov * (1 - np.identity(dimension))).mean()

        time2 = time()

        return mse_means, mse_cov, mse_vars, mse_covs, (time2 - time1)
