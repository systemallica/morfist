import numpy as np
import scipy.stats

from morfist.core.MixedRandomTree import MixedRandomTree


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
        """Build a printable Random Forest model

        :param n_estimators: number of trees in the forest
        :param max_features: the number of features to consider when looking for the best split
        :param min_samples_leaf: minimum amount of samples in each leaf
        :param choose_split: method to use to find the best split
        :param classification_targets: features that are part of the classification task
        """
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
