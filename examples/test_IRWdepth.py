#!/usr/bin/env python3

from openset.models.IRWdepth import IRWdepth

import numpy as np
from matplotlib import pyplot as plt
from numpy.random import default_rng


def main():
    rng = default_rng()

    d=100
    nproj=500

    # train/test ID data
    n = 100
    mi = np.full(d, 0)
    cov = np.identity(d)

    X_train = rng.multivariate_normal(mi, cov, n)
    X_test = rng.multivariate_normal(mi, cov, n)

    # OOD
    mi2 = np.full(d, 11)
    X_ood = rng.multivariate_normal(mi2, cov, n)


    model = IRWdepth()
    model.fit(X_train, nproj)

    train_scores = [model.score(row) for row in X_train]
    test_scores = [model.score(row) for row in X_test]
    ood_scores = [model.score(row) for row in X_ood]


    if 1:
        print(np.mean(train_scores))
        print(np.mean(test_scores))
        print(np.mean(ood_scores))


    if 1:
        # boxplot
        fig, ax1 = plt.subplots(figsize=(6, 4), constrained_layout=True)

        labels = ['train', 'test', 'ood']
        ax1.boxplot([train_scores, test_scores, ood_scores], labels=labels)
        ax1.grid()
        #plt.show()
        plt.gcf().savefig("./test.png")



if __name__ == '__main__':
    main()

