import csv
import numpy as np  # http://www.numpy.org

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

        def learn(self, X, y):
            # TODO: train decision tree and store it in self.tree
            pass

        def classify(self, test_instance):
            # TODO: return predicted label for a single instance using self.tree
            return 0

    decision_trees = []

    def __init__(self, num_trees):
        # TODO: do initialization here, you can change the function signature according to your need
        self.num_trees = num_trees
        self.decision_trees = [self.__DecisionTree()] * num_trees

    # You MUST NOT change this signature
    def fit(self, X, y):
        # TODO: train `num_trees` decision trees
        pass

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

        # Accuracy
        accuracy = float(results.count(True)) / float(len(results))
        accuracies.append(accuracy)

        # generateSubmissionFile(myname, randomForest)

        print "accuracy: %.4f" % accuracy

        lbound += bound_size
        rbound += bound_size

    print("Final accuracy: %.4f" % np.average(accuracies))

main()