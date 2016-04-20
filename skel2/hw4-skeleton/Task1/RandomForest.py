import csv
import numpy as np  # http://www.numpy.org
import math

from CSE6242HW4Tester import generateSubmissionFile


"""
Here, X is assumed to be a matrix with n rows and d columns
where n is the number of samples
and d is the number of features of each sample

Also, y is assumed to be a vector of n labels
"""

# Enter your name here
myname = "LastName-FirstName"

class RandomForest(object):
    class __DecisionTree(object):
        tree = {}

        def __init__(self, m = math.sqrt, max_depth = 10, min_for_split = 2):
            self.m = m
            self.max_depth = max_depth
            self.min_for_split = min_for_split

        def learn(self, X, y):
            self.X = X
            self.y = y

            # Compute the number of sub-features to split on.
            num_features = X.shape[1]
            num_sub_features = int(self.m(num_features))

            # Find the indices of the sub-features.
            indices = np.random.choice(num_features, size=num_sub_features,
                                       replace=False)

            self.build_tree(X, y, indices)


        def classify(self, test_instance):
            # TODO: return predicted label for a single instance using self.tree
            return 0

        def build_tree(self, X, y, indices, depth = 0):
            # If any of our stopping conditions are met,
            if self.gini(y) == 0.0 or \
               self.max_depth == depth or \
               len(y) < self.min_for_split:
                # Return the most common value.
                return self.mode(y)

        def best_split(self, X, y, indices):
            """
            Finds the best split for the given feature indices. Returns the
            feature index and threshold.

            """

            # Initialize.
            gain, fi, thresh = 0, 0, 0

            # For each feature index,
            for i in indices:
                # Sort the values for the feature.
                vals = sorted(set(X[:, i]))

                # For every row,
                for v in xrange(len(vals) - 1):
                    # Compute a threshold.
                    new_thresh = (vals[v] + vals[v+1]) / 2

                    # Perform a split base on the threshold.
                    X_1, y_1, X_2, y_2 = self.split(X, y, i, new_thresh)

                    # Compute the new gain value.
                    new_gain = self.gain(y, y_1, y_2)

                    # If the new gain exceeds the current gain, update the
                    # gain, feature index, and threshold.
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

        def mode(self, y):
            """
            Returns the most common value in a set of labels y.

            """

            return np.bincount(y).argmax()

        def gini_gain(self, y, left, right):
            """
            Computes the Gini gain of splitting a set of labels into two sets,
                left and right.

            """

            def weighted_gini(part, num_labs):
                return (len(part) / num_labs) * self.gini(part)

            num_labs = float(len(y))
            weighted_sum = weighted_gini(left, num_labs) + \
                           weighted_gini(right, num_labs)

            return self.gini(y) - weighted_sum

        def gini(self, y):
            """
            Computes the Gini index for a set of labels y.

            """

            num_y = float(len(y))
            counts = np.bincount(y)
            return 1 - np.sum([(count / num_y)**2 for count in counts])

    decision_trees = []

    def __init__(self, num_trees, m = math.sqrt):
        # TODO: do initialization here, you can change the function signature according to your need
        self.num_trees = num_trees
        self.decision_trees = [self.__DecisionTree(m)] * num_trees

    # You MUST NOT change this signature
    def fit(self, X, y):
        for tree in self.decision_trees:
            tree.learn(X, y)

    # You MUST NOT change this signature
    def predict(self, X):
        y = np.array([], dtype = int)

        for instance in X:
            votes = np.array([decision_tree.classify(instance)
                              for decision_tree in self.decision_trees])

            counts = np.bincount(votes)

            y = np.append(y, np.argmax(counts))

        return y


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

    lbound = 0
    bound_size = X.shape[0] / K
    rbound = lbound + bound_size

    accuracies = []

    for i in xrange(K):
        # Prepare a training set.
        X_train = np.concatenate((X[:lbound,:], X[rbound:,:]))
        y_train = np.concatenate((y[:lbound], y[rbound:]))

        # Prepare a test set.
        X_test = X[lbound:rbound,:]
        y_test = y[lbound:rbound]

        print(lbound, rbound)

        # Initialize according to your implementation
        randomForest = RandomForest(10)

        # Fit the classifier.
        randomForest.fit(X_train, y_train)

        # Predict results of the test set.
        y_predicted = randomForest.predict(X_test)

        # Determine our successes, and failures.
        results = [prediction == truth for prediction, truth in zip(y_predicted, y_test)]

        # Compute accuracy.
        accuracy = float(results.count(True)) / float(len(results))
        accuracies.append(accuracy)

        # generateSubmissionFile(myname, randomForest)

        print "Accuracy: %.4f" % accuracy

        lbound += bound_size
        rbound += bound_size

    print("Final accuracy: %.4f" % np.average(accuracies))

main()