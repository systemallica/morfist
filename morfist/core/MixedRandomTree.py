import numpy as np

from morfist.core.MixedSplitter import MixedSplitter


class MixedRandomTree:
    def __init__(self,
                 max_features='sqrt',
                 min_samples_leaf=5,
                 choose_split='mean',
                 classification_targets=None):
        """Build a Random Tree

        :param max_features: the number of features to consider when looking for the best split
        :param min_samples_leaf: minimum amount of samples in each leaf
        :param choose_split: method used to find the best split
        :param classification_targets: features that are part of the classification task
        """
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
