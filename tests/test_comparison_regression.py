from time import perf_counter

import numpy as np
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from morfist import MixedRandomForest, MixedRandomForestLegacy, cross_validation


# Configuration
# Number of tress of the random forest
n_trees = 11
# Cross-validation folds
n_folds = 10
# Original data
x_regression, y_regression = load_boston(return_X_y=True)


def setup_regression_scikit():
    t_start = perf_counter()
    # Fit scikit regression tree
    reg_scikit = RandomForestRegressor(n_estimators=n_trees)

    # Calculate scikit scores using cross-validation
    scores_scikit = cross_val_score(
        reg_scikit,
        x_regression,
        y_regression,
        cv=n_folds,
        scoring='neg_mean_squared_error'
    )

    t_stop = perf_counter()
    time = t_stop - t_start
    print('scikit-learn (rmse):', np.sqrt(-scores_scikit.mean()))
    return time


def setup_regression_morfist():
    t_start = perf_counter()

    reg_morfist = MixedRandomForest(
        n_estimators=n_trees,
        min_samples_leaf=1,
        choose_split='mean'
    )

    # Calculate morfist scores using cross-validation
    scores_morfist = cross_validation(
        reg_morfist,
        x_regression,
        y_regression,
        folds=n_folds
    )
    t_stop = perf_counter()
    time = t_stop - t_start
    print('morfist (rmse):', scores_morfist.mean())
    return time


def setup_regression_morfist_legacy():
    t_start = perf_counter()

    reg_morfist = MixedRandomForestLegacy(
        n_estimators=n_trees,
        min_samples_leaf=1,
        choose_split='mean'
    )

    # Calculate morfist scores using cross-validation
    scores_morfist = cross_validation(
        reg_morfist,
        x_regression,
        y_regression,
        folds=n_folds
    )
    t_stop = perf_counter()
    time = t_stop - t_start
    print('morfist legacy (rmse):', scores_morfist.mean())
    return time


def test_regression():
    time_sci = setup_regression_scikit()
    print("time_scikit", time_sci)
    time_morfist = setup_regression_morfist()
    print("time morfist", time_morfist)
    time_morfist_legacy = setup_regression_morfist_legacy()
    print("time morfist legacy", time_morfist_legacy)

    assert time_sci < 0.5
    assert time_morfist < 8
    assert time_morfist_legacy < 45

    assert time_morfist < time_morfist_legacy