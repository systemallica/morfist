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


# Test morfist against scikit-learn for a classification task
def test_classification_scikit():
    t_start = perf_counter()
    # Fit scikit classification tree
    cls_scikit = RandomForestClassifier(n_estimators=n_trees)

    # Calculate scikit scores using cross-validation
    scores_scikit = cross_val_score(
        cls_scikit,
        x_classification,
        y_classification,
        cv=n_folds
    )

    t_stop = perf_counter()
    time = t_stop - t_start
    print("Elapsed time scikit cls:", time)
    print('\tscikit-learn (accuracy):', scores_scikit.mean())


def test_classification_morfist():
    t_start = perf_counter()

    cls_morfist = MixedRandomForest(
        n_estimators=n_trees,
        min_samples_leaf=1,
        classification_targets=[0],
        choose_split='mean',
    )

    # Calculate morfist scores using cross-validation
    morfist_scores = cross_validation(
        cls_morfist,
        x_classification,
        y_classification,
        classification_targets=[0],
        folds=n_folds
    )
    t_stop = perf_counter()
    time = t_stop - t_start
    print("Elapsed time morfist cls:", time)
    print('\tmorfist (accuracy):', morfist_scores.mean())


def test_classification_morfist_legacy():
    t_start = perf_counter()

    cls_morfist = MixedRandomForestLegacy(
        n_estimators=n_trees,
        min_samples_leaf=1,
        class_targets=[0],
        choose_split='mean',
    )

    # Calculate morfist scores using cross-validation
    morfist_scores = cross_validation(
        cls_morfist,
        x_classification,
        y_classification,
        classification_targets=[0],
        folds=n_folds
    )
    t_stop = perf_counter()
    time = t_stop - t_start
    print("Elapsed time morfist legacy cls:", time)
    print('\tmorfist (accuracy):', morfist_scores.mean())
