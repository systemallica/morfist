from time import perf_counter

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from morfist import MixedRandomForest, MixedRandomForestLegacy, cross_validation

# Configuration
# Number of tress of the random forest
n_trees = 11
# Cross-validation folds
n_folds = 10
# Original data
x_classification, y_classification = load_breast_cancer(return_X_y=True)


def setup_classification_scikit():
    t_start = perf_counter()
    # Fit scikit classification tree
    cls_scikit = RandomForestClassifier(n_estimators=n_trees)

    # Calculate scikit scores using cross-validation
    scores_scikit = cross_val_score(
        cls_scikit, x_classification, y_classification, cv=n_folds
    )

    t_stop = perf_counter()
    time = t_stop - t_start
    print("scikit-learn (accuracy):", scores_scikit.mean())
    return time


def setup_classification_morfist():
    t_start = perf_counter()

    cls_morfist = MixedRandomForest(
        n_estimators=n_trees,
        min_samples_leaf=1,
        classification_targets=[0],
        choose_split="mean",
    )

    # Calculate morfist scores using cross-validation
    morfist_scores = cross_validation(
        cls_morfist,
        x_classification,
        y_classification,
        classification_targets=[0],
        folds=n_folds,
    )
    t_stop = perf_counter()
    time = t_stop - t_start
    print("morfist (accuracy):", morfist_scores.mean())
    return time


def setup_classification_morfist_legacy():
    t_start = perf_counter()

    cls_morfist = MixedRandomForestLegacy(
        n_estimators=n_trees,
        min_samples_leaf=1,
        class_targets=[0],
        choose_split="mean",
    )

    # Calculate morfist scores using cross-validation
    morfist_scores = cross_validation(
        cls_morfist,
        x_classification,
        y_classification,
        classification_targets=[0],
        folds=n_folds,
    )
    t_stop = perf_counter()
    time = t_stop - t_start
    print("morfist legacy (accuracy):", morfist_scores.mean())
    return time


def test_classification():
    time_sci = setup_classification_scikit()
    print("time_scikit", time_sci)
    time_morfist = setup_classification_morfist()
    print("time morfist", time_morfist)
    time_morfist_legacy = setup_classification_morfist_legacy()
    print("time morfist legacy", time_morfist_legacy)

    assert time_sci < 0.5
    assert time_morfist < 12
    assert time_morfist_legacy < 32

    assert time_morfist < time_morfist_legacy
