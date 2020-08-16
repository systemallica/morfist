from time import perf_counter

import numpy as np
from sklearn.datasets import load_boston, load_breast_cancer

from morfist import MixedRandomForest, MixedRandomForestLegacy, cross_validation

# Configuration
# Number of tress of the random forest
n_trees = 11
# Cross-validation folds
n_folds = 10
# Original data
x_regression, y_regression = load_boston(return_X_y=True)
x_classification, y_classification = load_breast_cancer(return_X_y=True)


# Test morfist with a given joint task
def test_mix_1():
    # Create data with data transposed and stacked vertically, X is regression, Y is classification
    x_mix_1, y_mix_1 = x_regression, np.vstack([y_regression, y_regression < y_regression.mean()]).T
    t_start = perf_counter()

    mix_rf = MixedRandomForest(
        n_estimators=n_trees,
        min_samples_leaf=1,
        classification_targets=[1],
        choose_split='mean'
    )

    mix_scores = cross_validation(
        mix_rf,
        x_mix_1,
        y_mix_1,
        folds=n_folds,
        classification_targets=[1]
    )
    t_stop = perf_counter()
    time = t_stop - t_start
    print("Elapsed time morfist mix:", time)
    print('Mixed output: ')
    print('\ttask 1 (original) (rmse):', mix_scores[0])
    print('\ttask 2 (additional) (accuracy)', mix_scores[1])

    t_start = perf_counter()

    mix_rf_l = MixedRandomForestLegacy(
        n_estimators=n_trees,
        min_samples_leaf=1,
        class_targets=[1],
        choose_split='mean'
    )

    mix_scores = cross_validation(
        mix_rf_l,
        x_mix_1,
        y_mix_1,
        folds=n_folds,
        classification_targets=[1]
    )
    t_stop = perf_counter()
    time = t_stop - t_start
    print("Elapsed time morfist legacy mix:", time)
    print('Mixed output: ')
    print('\ttask 1 (original) (rmse):', mix_scores[0])
    print('\ttask 2 (additional) (accuracy)', mix_scores[1])


# Test morfist with a given joint task (2)
def test_mix_2():
    # Create data with data transposed and stacked vertically, X is regression, Y is classification
    x_mix_2, y_mix_2 = x_classification, np.vstack([y_classification, y_classification]).T
    t2_start = perf_counter()

    mix_rf = MixedRandomForest(
        n_estimators=n_trees,
        min_samples_leaf=1,
        classification_targets=np.array([0]),
        choose_split='mean'
    )

    assert mix_rf.classification_targets is not []

    mix_scores = cross_validation(
        mix_rf,
        x_mix_2,
        y_mix_2,
        folds=n_folds,
        classification_targets=[0]
    )
    t2_stop = perf_counter()
    time = t2_stop - t2_start
    print("Elapsed time morfist mix 2:", time)
    print('Mixed output: ')
    print('\ttask 1 (original) (accuracy):', mix_scores[0])
    print('\ttask 2 (additional) (rmse):', mix_scores[1])

    t_start = perf_counter()

    mix_rf_l = MixedRandomForestLegacy(
        n_estimators=n_trees,
        min_samples_leaf=1,
        class_targets=[1],
        choose_split='mean'
    )

    mix_scores = cross_validation(
        mix_rf_l,
        x_mix_2,
        y_mix_2,
        folds=n_folds,
        classification_targets=[1]
    )
    t_stop = perf_counter()
    time = t_stop - t_start
    print("Elapsed time morfist legacy mix 2:", time)
    print('Mixed output: ')
    print('\ttask 1 (original) (accuracy):', mix_scores[0])
    print('\ttask 2 (additional) (rmse):', mix_scores[1])
