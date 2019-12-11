from morfist import MixedRandomForest, cross_validation
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_boston, load_breast_cancer
import numpy as np

# Configuration
# Number of tress of the random forest
n_trees = 11
# Cross-validation folds
n_folds = 10

# Original data
# With categorical variables
x_regression, y_regression = load_boston(return_X_y=True)
# With numerical variables
x_classification, y_classification = load_breast_cancer(return_X_y=True)


# Test morfist against scikit-learn for a classification task
def test_class():
    # Fit morfist classification tree
    cls_morfist = MixedRandomForest(
        n_estimators=n_trees,
        min_samples_leaf=1,
        class_targets=[0]
    )

    # Fit scikit classification tree
    cls_scikit = RandomForestClassifier(n_estimators=n_trees)

    # Calculate morfist scores using cross-validation
    morfist_scores = cross_validation(
        cls_morfist,
        x_classification,
        y_classification,
        class_targets=[0],
        folds=n_folds
    )

    # Calculate scikit scores using cross-validation
    scikit_scores = cross_val_score(
        cls_scikit,
        x_classification,
        y_classification
    )

    print('Classification: ')
    print('\tmorfist (accuracy): {}'.format(morfist_scores.mean()))
    print('\tscikit-learn (accuracy): {}'.format(scikit_scores.mean()))


# Test morfist against scikit-learn for a regression task
def test_reg():
    # Fit morfist regression tree
    reg_morfist = MixedRandomForest(
        n_estimators=n_trees,
        min_samples_leaf=5
    )

    # Fit scikit regression tree
    reg_scikit = RandomForestRegressor(n_estimators=n_trees)

    # Calculate morfist scores using cross-validation
    scores_morfist = cross_validation(
        reg_morfist,
        x_regression,
        y_regression,
        folds=n_folds
    )

    # Calculate scikit scores using cross-validation
    scores_scikit = cross_val_score(
        reg_scikit,
        x_regression,
        y_regression,
        scoring='neg_mean_squared_error'
    )

    print('Regression: ')
    print('\tmorfist (rmse): {}'.format(scores_morfist.mean()))
    print('\tscikit-learn (rmse): {}'.format(np.sqrt(-scores_scikit.mean())))


# Test morfist with a given joint task
def test_mix_1():
    # Create data with data transposed and stacked vertically, X is regression, Y is classification
    x_mix_1, y_mix_1 = x_regression, np.vstack([y_regression, y_regression < y_regression.mean()]).T

    mix_rf = MixedRandomForest(
        n_estimators=n_trees,
        min_samples_leaf=5,
        class_targets=[1]
    )

    mix_scores = cross_validation(
        mix_rf,
        x_mix_1,
        y_mix_1,
        folds=n_folds,
        class_targets=[1]
    )
    print('Mixed output: ')
    print('\ttask 1 (original) (rmse): {}'.format(mix_scores[0]))
    print('\ttask 2 (additional) (accuracy): {}'.format(mix_scores[1]))


# Test morfist with a given joint task (2)
def test_mix_2():
    # Create data with data transposed and stacked vertically, X is regression, Y is classification
    x_mix_2, y_mix_2 = x_classification, np.vstack([y_classification, y_classification]).T

    mix_rf = MixedRandomForest(
        n_estimators=n_trees,
        min_samples_leaf=1,
        class_targets=[0]
    )

    mix_scores = cross_validation(
        mix_rf,
        x_mix_2,
        y_mix_2,
        folds=n_folds,
        class_targets=[0]
    )
    print('Mixed output: ')
    print('\ttask 1 (original) (accuracy): {}'.format(mix_scores[0]))
    print('\ttask 2 (additional) (rmse): {}'.format(mix_scores[1]))


if __name__ == '__main__':
    test_class()
    print('')
    test_reg()
    print('')
    test_mix_1()
    print('')
    test_mix_2()
