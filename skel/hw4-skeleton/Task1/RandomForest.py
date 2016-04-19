from __future__ import division
import csv
import math
import numpy as np  # http://www.numpy.org
from collections import Counter



### TODO: REMOVE

from scipy.stats import entropy

######

from CSE6242HW4Tester import generateSubmissionFile


"""
Here, X is assumed to be a matrix with n rows and d columns
where n is the number of samples
and d is the number of features of each sample

Also, y is assumed to be a vector of n labels
"""

# Enter your name here
myname = "May-Matthew"

class RandomForest(object):
    class __DecisionTree(object):
        def __init__(self, max_features, max_depth):
            """
            Creates a new decision tree.

            Args:
                max_features: Function. Expression returning the number of
                    features to test for each split. The function is passed the
                    total number of features in the data.
                max_depth: Integer. Maximum depth that the tree can be grown.

            """

            self.max_features = max_features
            self.max_depth = max_depth

        def learn(self, X, y):
            # TODO: train decision tree and store it in self.tree
            self.X = X
            self.y = y

            num_samples = self.X.shape[0]
            num_features = self.X.shape[1]

            # Compute m, the subset of features to use in growing the tree.
            m = self.max_features(num_features)
            indices = np.random.choice(num_features, size=m, replace=False)

            self.tree = self.create_tree(X, y, indices)

        def classify(self, test_instance):
            # TODO: return predicted label for a single instance using self.tree

            node = self.tree

            while isinstance(node, TreeNode):
                if test_instance[node.fi] > node.thresh:
                    node = node.b_2
                else:
                    node = node.b_1

            return node

        def create_tree(self, X, y, indices, d = 0):
            """ Creates a decision tree. """

            #print("creating tree; depth is %d, len(y) is %d" % (d, len(y)))
            # Conditions for stopping.
            if Utils.entropy(y) == 0 or d == self.max_depth or len(y) < 100:
                return self.most_common_val(y)

            # Find the best split.
            fi, thresh = self.best_split(X, y, indices)

            # Split.
            X_1, y_1, X_2, y_2 = self.split(X, y, fi, thresh)

            # If we're empty on either side, return the most common value.
            if y_1.shape[0] == 0 or y_2.shape[0] == 0:
                return self.most_common_val(y)

            # Split into branches.
            new_depth = d+1
            b_1 = self.create_tree(X_1, y_1, indices, new_depth)
            b_2 = self.create_tree(X_2, y_2, indices, new_depth)

            return TreeNode(fi, thresh, b_1, b_2)

        def best_split(self, X, y, indices):
            """ Finds the best split. """

            # Initialize.
            gain, fi, thresh = 0, 0, 0

            # Loop through, looking for the feature that maximizes our gain.
            for i in indices:
                vals = X[:, i]
                vals = sorted(set(vals))

                for v in xrange(len(vals) - 1):
                    new_thresh = (vals[v] + vals[v+1]) / 2

                    X_1, y_1, X_2, y_2 = self.split(X, y, i, new_thresh)

                    new_gain = Utils.gain(y, y_1, y_2)

                    if new_gain > gain:
                        gain, fi, thresh = new_gain, i, new_thresh

            return fi, thresh

        def split(self, X, y, fi, thresh):
            """
            Splits both features and labels into two groups each, based on a
            defined threshold.

            """

            less_than = np.where(X[:,fi] <= thresh)
            greater_than = np.where(X[:,fi] > thresh)

            # Features and labels less than or equal to the threshold.
            X_1 = X[less_than]
            y_1 = y[less_than]

            # Features and labels greater than the threshold.
            X_2 = X[greater_than]
            y_2 = y[greater_than]

            return X_1, y_1, X_2, y_2

        def most_common_val(self, y):
            """ Returns the most common value in an array. """

            return np.bincount(y).argmax()

    decision_trees = []

    def __init__(self, num_trees = 100, max_features = math.sqrt,
                 max_depth = 10, bootstrap = 0.9):
        """
        Creates a new random forest.

        Args:
            num_trees: Integer. Number of trees to grow in constructing the
                forest.
            max_features: Expression returning the number of
                features to test for each split. The function is passed the
                total number of features in the data.
            max_depth: Integer. Maximum depth that each tree will be grown.
            bootstrap: Float. Percentage of data to select as a subset for
                growing each tree.

        """
        # TODO: do initialization here, you can change the function signature according to your need

        self.num_trees = num_trees
        self.bootstrap = bootstrap

        # Build the forest.
        self.decision_trees = [self.__DecisionTree(max_features, max_depth)] * num_trees

    # You MUST NOT change this signature
    def fit(self, X, y):
        # TODO: train `num_trees` decision trees

        subset = int(self.bootstrap * len(y))

        for tree in self.decision_trees:
            # Shuffle the data.
            X, y = Utils.shuffle(X, y)

            # Retrieve the subset of the data to grow the tree on.
            subset_X, subset_y = X[:subset], y[:subset]

            # Train the tree.
            tree.learn(subset_X, subset_y)

    # You MUST NOT change this signature
    def predict(self, X):
        y = np.array([], dtype = int)

        for instance in X:
            votes = np.array([decision_tree.classify(instance)
                              for decision_tree in self.decision_trees])

            counts = np.bincount(votes)

            y = np.append(y, np.argmax(counts))

        return y

class TreeNode(object):
    def __init__(self, fi, thresh, b_1, b_2):
        self.fi = fi
        self.thresh = thresh
        self.b_1 = b_1
        self.b_2 = b_2

class Utils(object):
    @staticmethod
    def shuffle(list_a, list_b):
        """
        Shuffles two lists, maintaining index relationships between them. The
        two lists should be the same length.

        Returns the shuffled lists.

        """

        assert len(list_a) == len(list_b)
        perm = np.random.permutation(len(list_a))
        return list_a[perm], list_b[perm]

    @staticmethod
    def entropy(Y):
        """
        Computes information entropy of the labels.

        """

        print('computing entropy')
        dist, n_labs = Counter(Y), len(Y)
        return -np.sum([(count_y/n_labs) * math.log(count_y/n_labs, 2) for y, count_y in dist.items()])

    @staticmethod
    def gain(parent, child1, child2):
        """
        Computes expected information gain from a split into two child groups.

        The expected information gain is the change in information entropy from
        a prior state to a new state that takes some information as given:
        https://en.wikipedia.org/wiki/Information_gain_in_decision_trees

        """

        def weighted_entropy(child, n_labs):
            return len(child) / n_labs * Utils.entropy(child)

        n_labs = len(parent)
        return Utils.entropy(parent) - \
                (weighted_entropy(child1, n_labs) + weighted_entropy(child2, n_labs))

def main():
    X = []
    y = []

    # Load data set
    with open("hw4-data.csv") as f:
        next(f, None)

        for line in csv.reader(f, delimiter = ","):
            X.append(line[:-1])
            y.append(line[-1])

    X = np.array(X, dtype = float)
    y = np.array(y, dtype = int)

    # Split training/test sets
    # You need to modify the following code for cross validation
    X, y = Utils.shuffle(X, y)

    K = 10
    trees = 10

    lbound = 0
    bound_size = X.shape[0] / K
    rbound = 0 + bound_size

    accuracies = []

    for i in xrange(10):
        X_train = np.concatenate((X[:lbound,:], X[rbound:,:]))
        y_train = np.concatenate((y[:lbound], y[rbound:]))
        X_test = X[lbound:rbound,:]
        y_test = y[lbound:rbound]

        print(lbound, rbound)

        # self, num_trees = 100, max_features = math.sqrt,
        #             max_depth = 10, bootstrap = 0.9

        # Initialize according to your implementation
        randomForest = RandomForest(num_trees = 10, max_features = math.sqrt,
                                    max_depth = 10, bootstrap = 0.9)

        randomForest.fit(X_train, y_train)

        y_predicted = randomForest.predict(X_test)

        results = [prediction == truth for prediction, truth in zip(y_predicted, y_test)]

        # Accuracy
        accuracy = float(results.count(True)) / float(len(results))
        accuracies.append(accuracy)

        # generateSubmissionFile(myname, randomForest)

        print "accuracy: %.4f" % accuracy

        lbound += bound_size
        rbound += bound_size

    print("Final accuracy: %.4f" % np.average(accuracies))

main()