#!/usr/bin/env python3

from matplotlib import pyplot as plt
import numpy as np

from openset.models.irwd import IntegratedRankWeightedDepth


def main():
    d = 100
    n = 100
    nproj = 500
    rng = np.random.default_rng(42)

    # train and test ID data
    mi = np.full(d, 0)
    cov = np.identity(d)
    X_train = rng.multivariate_normal(mi, cov, n)
    X_test = rng.multivariate_normal(mi, cov, n)

    # test OOD data
    mi2 = np.full(d, 11)
    X_ood = rng.multivariate_normal(mi2, cov, n)

    model = IntegratedRankWeightedDepth(nproj)
    model.fit(X_train)
    train_scores = model.score(X_train)
    test_scores = model.score(X_test)
    ood_scores = model.score(X_ood)

    # Summary
    print(np.mean(train_scores))
    print(np.mean(test_scores))
    print(np.mean(ood_scores))

    # Boxplot
    fig, ax1 = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax1.boxplot([train_scores, test_scores, ood_scores],
                labels=['train', 'test', 'ood'])
    ax1.grid()
    plt.gcf().savefig('./test.png')
    # plt.show()


if __name__ == '__main__':
    main()
