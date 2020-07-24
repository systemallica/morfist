import copy

import numpy as np


def accuracy(y, y_hat):
    # Calculate classification accuracy of model
    return (y.astype(int) == y_hat.astype(int)).sum() / y.size


def rmse(y, y_hat):
    # Calculate root squared mean error of model
    return np.sqrt(((y - y_hat) ** 2).mean())


def cross_validation(model,
                     x,
                     y,
                     folds=10,
                     classification_targets=None,
                     classification_eval=accuracy,
                     reg_eval=rmse,
                     verbose=False):
    """Perform cross validation on a model

    :param model:  model to be validated
    :param x: X values of the data set
    :param y: Y values of the data set
    :param folds: number of folds
    :param classification_targets: features that are part of the classification task
    :param classification_eval: function to evaluate model classification accuracy
    :param reg_eval: function to evaluate model regression accuracy
    :param verbose: used for debug purposes
    :return: scores[]:
                 0: classification accuracy
                 1: regression RMSE
    """
    classification_targets = classification_targets if classification_targets else []

    idx = np.random.permutation(x.shape[0])
    fold_size = int(idx.size / folds)

    if y.ndim == 1:
        y = y.reshape((y.size, 1))

    y_hat = np.zeros((idx.size, y.shape[1]))

    # Perform the cross-validation
    # Train and fit the model for different subsets of the data
    for i in range(folds):
        if verbose:
            print('Running fold {} of {} ...'.format(i + 1, folds))

        fold_start = i * fold_size
        fold_stop = min((i + 1) * fold_size, idx.size)

        mask = np.ones(idx.size, dtype=bool)
        mask[fold_start:fold_stop] = 0

        train_idx = idx[mask]
        test_idx = idx[(1 - mask).astype(bool)]

        m = copy.copy(model)
        m.fit(x[train_idx, :], y[train_idx, :])
        y_hat[test_idx, :] = m.predict(x[test_idx, :])

    scores = np.zeros(y.shape[1])
    # Calculate the classification and regression accuracy of the model
    for i in range(y.shape[1]):
        if i in classification_targets:
            scores[i] = classification_eval(y[:, i], y_hat[:, i])
        else:
            scores[i] = reg_eval(y[:, i], y_hat[:, i])

    return scores
