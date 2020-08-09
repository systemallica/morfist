import numpy as np
from numba import njit

from morfist.algo.histogram import numba_histogram


@njit
def impurity_classification(y_classification):
    # Calculate the impurity value for the classification task

    # Cast to integer
    y_class = y_classification.astype(np.int8)

    # Calculate frequencies
    frequency = np.bincount(y_class) / y_class.size

    result = 0
    for i in range(frequency.size):
        if frequency[i]:
            result += frequency[i] * np.log2(frequency[i])

    return 0 - result


@njit
def impurity_regression(y, y_regression):
    # Calculate the impurity value for the regression task

    if np.unique(y_regression).size < 2:
        return 0

    n_bins = 100
    bin_width = (y.max() - y.min()) / n_bins

    frequency, _ = numba_histogram(y_regression, n_bins)
    frequency_float = frequency.astype(np.float64)
    frequency_float = (frequency_float / len(y)) / bin_width

    probability = (frequency_float + 1) / (frequency_float.sum() + n_bins)

    return 0 - bin_width * (probability * np.log2(probability)).sum()


@njit
def get_gain(imp_n_left, imp_n_right, imp_n, imp_root, n_left, n_right, n_parent):
    impurity_left = imp_n_left / imp_root
    impurity_right = imp_n_right / imp_root
    impurity_parent = imp_n / imp_root

    gain_left = (n_left / n_parent) * (impurity_parent - impurity_left)
    gain_right = (n_right / n_parent) * (impurity_parent - impurity_right)
    return gain_left + gain_right


def get_max_features(max_features, n_features):
    # Maximum number of features to try for the best split
    n_max_features = n_features
    if max_features == 'sqrt':
        n_max_features = int(np.sqrt(n_features))
    elif max_features == 'log2':
        n_max_features = int(np.log2(n_features))
    elif isinstance(max_features, float):
        n_max_features = int(max_features * n_features)
    elif max_features is None:
        n_max_features = n_features

    return n_max_features


class MixedSplitter:
    def __init__(self,
                 x,
                 y,
                 max_features='sqrt',
                 min_samples_leaf=5,
                 choose_split='mean',
                 classification_targets=None):
        """Class in charge of finding the best split at every given moment

        :param x: training data
        :param y: target data
        :param max_features: the number of features to consider when looking for the best split
        :param min_samples_leaf: minimum amount of samples in each leaf
        :param choose_split:  method used to find the best split
        :param classification_targets: features that are part of the classification task
        """
        self.n_train = x.shape[0]
        self.n_features = x.shape[1]
        self.n_targets = y.shape[1]
        self.classification_targets = classification_targets if classification_targets else []
        self.max_features = get_max_features(max_features, self.n_features)
        self.min_samples_leaf = min_samples_leaf
        self.root_impurity = self.__impurity_node(y)
        self.choose_split = choose_split

    def split(self, x, y):
        # If there are not enough features in the leaf, stop splitting
        if x.shape[0] <= self.min_samples_leaf:
            return None, None, np.inf

        # Best feature
        best_feature = None
        # Best value
        best_value = None
        # Best impurity
        best_impurity = -np.inf

        # Random selection of the features to try for the best split
        try_features = np.random.choice(
            np.arange(self.n_features),
            self.max_features,
            replace=False
        )

        # Try each of the selected features and find which of them gives the best split(higher impurity)
        for feature in try_features:
            # Get the unique possible values for this particular feature
            values = np.unique(x[:, feature])

            # Split value selection(random value subsampling): Boström (2011)
            #   Two random feature values are selected, and a split is attempted at their mean
            if values.size < 2:
                continue
            values = (values[:-1] + values[1:]) / 2
            value = np.random.choice(values)

            # Try to split with this specific combination of feature and value
            left_idx = x[:, feature] <= value
            right_idx = x[:, feature] > value

            impurity = self.__impurity_split(y, y[left_idx, :], y[right_idx, :])
            # If it's better than the previous saved one, save the values
            if impurity > best_impurity:
                best_feature, best_value, best_impurity = feature, value, impurity

        return best_feature, best_value, best_impurity

    # Calculate the impurity of a split
    def __impurity_split(self, y_parent, y_left, y_right):
        n_left = y_left.shape[0]
        n_right = y_right.shape[0]
        if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
            return np.inf
        else:
            n_parent = y_parent.shape[0]

            gain = get_gain(self.__impurity_node(y_left),
                            self.__impurity_node(y_right),
                            self.__impurity_node(y_parent),
                            self.root_impurity,
                            n_left,
                            n_right,
                            n_parent)

            if self.choose_split == 'mean':
                return gain.mean()
            elif self.choose_split == 'random':
                return np.random.choice(gain)
            else:
                return gain.max()

    def __impurity_node(self, y):
        # Calculate the impurity of a node
        delta = 0.0001
        impurity = np.zeros(self.n_targets)
        # Calculate the impurity value for each of the targets(classification or regression)
        for i in range(self.n_targets):
            if i in self.classification_targets:
                impurity[i] = impurity_classification(y[:, i]) + delta
            else:
                impurity[i] = impurity_regression(y, y[:, i]) + delta
        return impurity
