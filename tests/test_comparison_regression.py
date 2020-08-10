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


# Test morfist against scikit-learn for a classification task
def test_classification_scikit():
    t_start = perf_counter()
    # Fit scikit classification tree
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
    print("Elapsed time scikit cls:", time)
    print('\tscikit-learn (rmse):', np.sqrt(-scores_scikit.mean()))


def test_classification_morfist():
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
    print("Elapsed time morfist cls:", time)
    print('\tmorfist (rmse):', scores_morfist.mean())


def test_classification_morfist_legacy():
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
    print("Elapsed time morfist legacy cls:", time)
    print('\tmorfist (rmse):', scores_morfist.mean())
