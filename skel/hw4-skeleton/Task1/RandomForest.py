from __future__ import division
import csv
import numpy as np  # http://www.numpy.org
from collections import Counter

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
        tree = {}

        def learn(self, X, y):
            # TODO: train decision tree and store it in self.tree
            print(y)
            print Utils.entropy(np.array([1,0,0,0,0,1,1]))
            print Utils.shuffle(np.array([1,2]), np.array([3,4]))
            exit()
            # self.tree = self.build_tree(X, y)

        def classify(self, test_instance):
            # TODO: return predicted label for a single instance using self.tree
            return 0

        def build_tree(self, X, y):
            pass

    decision_trees = []

    def __init__(self, num_trees):
        # TODO: do initialization here, you can change the function signature according to your need
        self.num_trees = num_trees
        self.decision_trees = [self.__DecisionTree()] * num_trees

    # You MUST NOT change this signature
    def fit(self, X, y):
        # TODO: train `num_trees` decision trees
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

    @staticmethod
    def entropy(Y):
        """
        Computes information entropy of the labels.

        """

        dist, n_labs = Counter(Y), len(Y)
        return -np.sum([(count_y/n_labs) * np.log2(count_y/n_labs) for y, count_y in dist.items()])

    @staticmethod
    def gain(parent, child1, child2):
        """
        Computes expected information gain from a split into two child groups.

        The expected information gain is the change in information entropy from
        a prior state to a new state that takes some information as given:
        https://en.wikipedia.org/wiki/Information_gain_in_decision_trees

        """

        def weighted_entropy(child, n_labs):
            return len(child) / n_labs * entropy(child)

        n_labs = len(parent)
        return entropy(parent) - \
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
    K = 10
    X_train = np.array([x for i, x in enumerate(X) if i % K != 9], dtype = float)
    y_train = np.array([z for i, z in enumerate(y) if i % K != 9], dtype = int)
    X_test  = np.array([x for i, x in enumerate(X) if i % K == 9], dtype = float)
    y_test  = np.array([z for i, z in enumerate(y) if i % K == 9], dtype = int)

    randomForest = RandomForest(999)  # Initialize according to your implementation

    randomForest.fit(X_train, y_train)

    y_predicted = randomForest.predict(X_test)

    results = [prediction == truth for prediction, truth in zip(y_predicted, y_test)]

    # # Accuracy
    # accuracy = float(results.count(True)) / float(len(results))
    # print "accuracy: %.4f" % accuracy
    #
    # generateSubmissionFile(myname, randomForest)


main()
