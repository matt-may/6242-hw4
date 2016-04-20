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

        def __init__(self, m):
            self.m = m

        def learn(self, X, y):
            self.X = X
            self.y = y

            num_samples = X.shape[0]
            num_features = X.shape[1]
            num_sub_features = int(self.m(num_features))

            print("num sub features: %d" % num_sub_features)

            indices = np.random.choice(num_features, size=num_sub_features,
                                       replace=False)

            

        def classify(self, test_instance):
            # TODO: return predicted label for a single instance using self.tree
            return 0

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
        print(len(X_train), len(y_train), len(X_test), len(y_test))

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