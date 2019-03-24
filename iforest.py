# Follows algo from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf
import math
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from concurrent.futures import ProcessPoolExecutor


class IsolationTreeEnsemble:

    def __init__(self, sample_size, n_trees=10):
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.height_limit = math.ceil(math.log2(self.sample_size))
        self.N = None
        self.trees = [IsolationTree(height_limit=self.height_limit)
                      for i in range(self.n_trees)]

    def fit(self, X: np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.N = X.shape[0]

        for itree in self.trees:
            X_dash = X[np.random.choice(X.shape[0], size=self.sample_size,
                                        replace=False)].copy()
            itree.fit(X_dash, improved=improved)

        return self

    def path_length(self, X: np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        return np.array([np.mean([itree.path_length(x) for itree in self.trees])
                         for x in X])

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        return np.power(2, -(self.path_length(X) / c(self.N)))

    def predict_from_anomaly_scores(self, scores: np.ndarray,
                                    threshold: float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        return (scores >= threshold).astype(np.int)

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."""
        scores = anomaly_score(X)
        return predict_from_anomaly_scores(scores, threshold)


def H(i):
    return math.log(i) + 0.5772156649


def c(n):
    if n > 2:
        return (2.0 * H(n - 1.0)) - ((2.0 * (n - 1.0)) / n)
    if n == 2:
        return 1
    else:
        return 0


class IsolationTree:

    def __init__(self, height_limit=None, level=0):
        self.left = None
        self.right = None
        self.is_ex = True
        self.split_value = None
        self.split_axis = None
        self.level = level
        self.height_limit = height_limit
        self.size = None
        self.n_nodes = 0
        self.split_ratio_threshold = 0.25

    def fit(self, X: np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.

        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """
        self.size = X.shape[0]
        if X.shape[0] < 2:
            return self
        if self.level >= self.height_limit:
            return self

        self.split_axis = np.random.choice(X.shape[1])
        v_min = X[:, self.split_axis].min()
        v_max = X[:, self.split_axis].max()
        if v_min == v_max:
            return self

        if not improved:
            self.split_point = np.random.uniform(low=v_min,
                                                 high=v_max)
            d_left = X[X[:, self.split_axis] < self.split_point, :]
            d_right = X[X[:, self.split_axis] >= self.split_point, :]

        if improved:
            i = 0
            while True:
                self.split_point = np.random.uniform(low=v_min, high=v_max)
                d_left = X[X[:, self.split_axis] < self.split_point, :]
                d_right = X[X[:, self.split_axis] >= self.split_point, :]
                if X.shape[0] < 10:
                    break
                if (d_left.shape[0] > 0) and (d_right.shape[0] > 0):
                    if float(d_left.shape[0]) / d_right.shape[0] > (1 / self.split_ratio_threshold):
                        break
                    if float(d_left.shape[0]) / d_right.shape[0] < self.split_ratio_threshold:
                        break
                if i > 20:
                    break
                i += 1

        self.left = IsolationTree(level=self.level + 1,
                                  height_limit=self.height_limit
                                  ).fit(d_left, improved=improved)
        self.right = IsolationTree(level=self.level + 1,
                                   height_limit=self.height_limit
                                   ).fit(d_right, improved=improved)
        self.is_ex = False
        self.n_nodes = self.left.n_nodes + self.right.n_nodes + 2
        return self

    def path_length(self, x: np.ndarray):
        if self.is_ex:
            return self.level + c(self.size)
        if x[self.split_axis] < self.split_point:
            return self.left.path_length(x)
        else:
            return self.right.path_length(x)


def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """
    threshold = 1.0
    while True:
        y_pred = (scores >= threshold).astype(np.int)
        confusion = confusion_matrix(y, y_pred)
        TN, FP, FN, TP = confusion.flat
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        # print(f'threshold : {threshold}, TPR: {TPR}, FPR: {FPR}')
        if TPR >= desired_TPR:
            return threshold, FPR
        else:
            threshold -= 0.001
