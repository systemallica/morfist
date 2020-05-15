import numpy as np
import scipy.stats
import copy
from numba import njit


# Calculate the impurity value for the classification task
@njit
def impurity_classification(y_classification):
    # Cast to integer
    y_class = y_classification.astype(np.int16)

    # Calculate frequencies
    frequency = np.bincount(y_class) / y_class.size

    result = 0
    for i in range(frequency.size):
        if frequency[i]:
            result += frequency[i] * np.log2(frequency[i])

    return 0 - result


# Calculate the impurity value for the regression task
@njit
def impurity_regression(y, y_regression):
    if np.unique(y_regression).size < 2:
        return 0

    n_bins = 100
    bin_width = (y.max() - y.min()) / n_bins

    frequency = np.histogram(y_regression, n_bins)[0]
    frequency_float = frequency.astype(np.float64)
    frequency_float = (frequency_float / len(y)) / bin_width

    probability = (frequency_float + 1) / (frequency_float.sum() + n_bins)

    return 0 - bin_width * (probability * np.log2(probability)).sum()


@njit
def unique(x, feature):
    return np.unique(x[:, feature])


@njit
def get_gain(imp_n_left, imp_n_right, imp_n, imp_root, n_left, n_right, n_parent):
    impurity_left = imp_n_left / imp_root
    impurity_right = imp_n_right / imp_root
    impurity_parent = imp_n / imp_root

    gain_left = (n_left / n_parent) * (impurity_parent - impurity_left)
    gain_right = (n_right / n_parent) * (impurity_parent - impurity_right)
    return gain_left + gain_right


# Class in charge of finding the best split at every given moment
# Parameters:
#   x: training data
#   y: target data
#   max_features: the number of features to consider when looking for the best split
#   min_samples_leaf: minimum amount of samples in each leaf
#   choose_split: method used to find the best split
#   classification_targets: features that are part of the classification task
class MixedSplitter:
    def __init__(self,
                 x,
                 y,
                 max_features='sqrt',
                 min_samples_leaf=5,
                 choose_split='mean',
                 classification_targets=None):
        self.n_train = x.shape[0]
        self.n_features = x.shape[1]
        self.n_targets = y.shape[1]
        self.classification_targets = classification_targets if classification_targets else []
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.root_impurity = self.__impurity_node(y)
        self.choose_split = choose_split

    def split(self, x, y):
        # Maximum number of features to try for the best split
        if self.max_features == 'sqrt':
            self.max_features = int(np.sqrt(self.n_features))
        elif self.max_features == 'log2':
            self.max_features = int(np.log2(self.n_features))
        elif isinstance(self.max_features, float):
            self.max_features = int(self.max_features * self.n_features)
        elif self.max_features is None:
            self.max_features = self.n_features

        return self.__find_best_split(x, y)

    def __find_best_split(self, x, y):
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
            values = unique(x, feature)

            # We ensure that there are at least 2 different values
            if values.size < 2:
                continue

            # Random value sub-sampling
            # Reduces the size by one element
            # This is to avoid using the first value in case it is 0 for regression
            # [0] -> ([0] + [1]) / 2
            values = (values[:-1] + values[1:]) / 2

            # Choose a random amount of values, with a min of 2
            values = np.random.choice(values, min(2, values.size))

            # Try to split with this specific combination of feature and values
            # Here lies the computational burden, as we try every possible split
            # TODO incrementally compute impurity
            for value in values:

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
            else:
                return gain.max()

    # Calculate the impurity of a node
    def __impurity_node(self, y):
        delta = 0.0001
        impurity = np.zeros(self.n_targets)
        # Calculate the impurity value for each of the targets(classification or regression)
        for i in range(self.n_targets):
            if i in self.classification_targets:
                impurity[i] = impurity_classification(y[:, i]) + delta
            else:
                impurity[i] = impurity_regression(y, y[:, i]) + delta
        return impurity


# Build a Random Tree
# Parameters:
#   max_features: the number of features to consider when looking for the best split
#   min_samples_leaf: minimum amount of samples in each leaf
#   choose_split: method used to find the best split
#   classification_targets: features that are part of the classification task
class MixedRandomTree:
    def __init__(self,
                 max_features='sqrt',
                 min_samples_leaf=5,
                 choose_split='mean',
                 classification_targets=None):
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.classification_targets = classification_targets if classification_targets else []
        self.choose_split = choose_split
        self.n_targets = 0
        self.features = []
        self.values = []
        self.leaf_values = []
        self.left_children = []
        self.right_children = []
        self.n = []

    def fit(self, x, y):
        if y.ndim == 1:
            y = y.reshape((y.size, 1))

        self.n_targets = y.shape[1]

        splitter = MixedSplitter(x,
                                 y,
                                 self.max_features,
                                 self.min_samples_leaf,
                                 self.choose_split,
                                 self.classification_targets)

        split_features = []
        split_values = []
        leaf_values = []
        left_children = []
        right_children = []
        n_i = []

        split_queue = [(x, y)]
        i = 0
        # Build the tree until all values are covered
        while len(split_queue) > 0:
            next_x, next_y = split_queue.pop(0)

            leaf_values.append(self._make_leaf(next_y))
            n_i.append(next_y.shape[0])

            feature, value, impurity = splitter.split(next_x, next_y)

            split_features.append(feature)
            split_values.append(value)
            if feature:
                left_children.append(i + len(split_queue) + 1)
                right_children.append(i + len(split_queue) + 2)

                l_idx = next_x[:, feature] <= value
                r_idx = next_x[:, feature] > value

                split_queue.append((next_x[l_idx, :], next_y[l_idx, :]))
                split_queue.append((next_x[r_idx, :], next_y[r_idx, :]))
            else:
                left_children.append(None)
                right_children.append(None)

            i += 1

        self.features = np.array(split_features)
        self.values = np.array(split_values)
        self.leaf_values = np.array(leaf_values)
        self.left_children = np.array(left_children)
        self.right_children = np.array(right_children)
        self.n = np.array(n_i)

    def _make_leaf(self, y):
        y_ = np.zeros(self.n_targets)
        for i in range(self.n_targets):
            if i in self.classification_targets:
                y_[i] = np.argmax(np.bincount(y[:, i].astype(int)))
            else:
                y_[i] = y[:, i].mean()
        return y_

    def predict(self, x):
        n_test = x.shape[0]
        prediction = np.zeros((n_test, self.n_targets))

        def traverse(x_traverse, test_idx, node_idx):
            if test_idx.size < 1:
                return

            if not self.features[node_idx]:
                prediction[test_idx, :] = self.leaf_values[node_idx]
            else:
                left_idx = x_traverse[:, self.features[node_idx]] <= self.values[node_idx]
                right_idx = x_traverse[:, self.features[node_idx]] > self.values[node_idx]

                traverse(x_traverse[left_idx, :], test_idx[left_idx], self.left_children[node_idx])
                traverse(x_traverse[right_idx, :], test_idx[right_idx], self.right_children[node_idx])

        traverse(x, np.arange(n_test), 0)
        return prediction

    def print(self):
        def print_level(level, i):
            if self.features[i]:
                print('\t' * level + '[{} <= {}]:'.format(self.features[i], self.values[i]))
                print_level(level + 1, self.left_children[i])
                print_level(level + 1, self.right_children[i])
            else:
                print('\t' * level + str(self.leaf_values[i]) + ' ({})'.format(self.n[i]))

        print_level(0, 0)


# Build the Random Forest model
# Parameters:
#   n_estimators: number of trees in the forest
#   max_features: the number of features to consider when looking for the best split
#   min_samples_leaf: minimum amount of samples in each leaf
#   choose_split: method used to find the best split
#   classification_targets: features that are part of the classification task
class MixedRandomForest:
    def __str__(self):
        params = "("
        i = 0
        for key in self.__dict__:
            if i == 0:
                params += str(key) + "=" + str(self.__dict__[key]) + ", \n"
            elif i == len(self.__dict__) - 1:
                params += "\t\t\t" + str(key) + "=" + str(self.__dict__[key]) + ")"
            else:
                params += "\t\t\t" + str(key) + "=" + str(self.__dict__[key]) + ", \n"
            i += 1
        return self.__class__.__name__ + params

    def __init__(self,
                 n_estimators=10,
                 max_features='sqrt',
                 min_samples_leaf=5,
                 choose_split='mean',
                 classification_targets=None):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.classification_targets = classification_targets if classification_targets else []
        self.choose_split = choose_split
        self.n_targets = 0
        self.classification_labels = {}
        self.estimators = []

    # Fit the model
    def fit(self, x, y):
        self.estimators = []

        if y.ndim == 1:
            y = y.reshape((y.size, 1))
        self.n_targets = y.shape[1]

        # Get the classification labels
        # It takes the unique labels of the specified classification variables
        for i in filter(lambda j: j in self.classification_targets, range(self.n_targets)):
            self.classification_labels[i] = np.unique(y[:, i])

        n_train = x.shape[0]
        # Train the random trees that are part of the forest
        for i in range(self.n_estimators):
            m = MixedRandomTree(self.max_features,
                                self.min_samples_leaf,
                                self.choose_split,
                                self.classification_targets)

            # It is a random forest so the trees are built with random subsets of the data
            sample_idx = np.random.choice(np.arange(n_train),
                                          n_train,
                                          replace=True)

            m.fit(x[sample_idx, :], y[sample_idx, :])
            self.estimators.append(m)

    # Predict the class/value of an instance
    def predict(self, x):
        n_test = x.shape[0]
        pred = np.zeros((n_test, self.n_targets, self.n_estimators))
        for i, m in enumerate(self.estimators):
            pred[:, :, i] = m.predict(x)

        pred_avg = np.zeros((n_test, self.n_targets))
        for i in range(self.n_targets):
            # Predict categorical value
            if i in self.classification_targets:
                pred_avg[:, i], _ = scipy.stats.mode(pred[:, i, :].T)
            # Predict numerical value
            else:
                pred_avg[:, i] = pred[:, i, :].mean(axis=1)

        return pred_avg

    # Predict the probability of an instance
    def predict_proba(self, x):
        n_test = x.shape[0]
        pred = np.zeros((n_test, self.n_targets, self.n_estimators))
        for i, m in enumerate(self.estimators):
            pred[:, :, i] = m.predict(x)

        pred_avg = np.zeros((n_test, self.n_targets), dtype=object)
        for i in range(self.n_targets):
            if i in self.classification_targets:
                for j in range(n_test):
                    freq = np.bincount(pred[j, i, :].T.astype(int),
                                       minlength=self.classification_labels[i].size)
                    pred_avg[j, i] = freq / self.n_estimators
            else:
                pred_avg[:, i] = pred[:, i, :].mean(axis=1)

        return pred_avg


# Calculate classification accuracy of model
def accuracy(y, y_hat):
    return (y.astype(int) == y_hat.astype(int)).sum() / y.size


# Calculate root squared mean error of model
def rmse(y, y_hat):
    return np.sqrt(((y - y_hat) ** 2).mean())


# Perform cross validation on a model
# Parameters:
#     model: model to be validated
#     x: X values of the data set
#     y: Y values of the data set
#     classification_targets: features that are part of the classification task
#     classification_eval: function to evaluate model classification accuracy
#     reg_eval: function to evaluate model regression accuracy
#     verbose: used for debug purposes
# Returns:
#     scores[]:
#         0: classification accuracy
#         1: regression RMSE
def cross_validation(model,
                     x,
                     y,
                     folds=10,
                     classification_targets=None,
                     classification_eval=accuracy,
                     reg_eval=rmse,
                     verbose=False):
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
