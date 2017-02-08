import numpy as np
from scipy.sparse import vstack
from numpy.linalg import norm
import operator

from metrics import CategoricalMetric

class NaiveBayes():

    def __init__(self, posterior=False):
        self.name = "N"
        self.priors = None
        self.w = None
        self.class_names = None
        self.fitted = False

        self.metrics = CategoricalMetric()
        self.most_recent_y_hat = None
        self.predict = self.predict_post if posterior else self.predict_map
        self.argop = np.argmax

    def is_fitted(self):
        return self.fitted

    def unfit(self):
        self.priors = None
        self.w = None
        self.class_names = None

        self.fitted = False

    def update_metrics(self, y_hat, y_test):
        self.metrics.update(y_hat, y_test)

    def print_metrics(self):
        self.metrics.print()

    def get_metrics(self):
        return self.metrics.get_metrics()

    def fit(self, x, y):
        _, alpha = x.shape
        n = len(y)

        classes = {}
        counts = {}
        for i in range(n):
            y_i = y[i][0]
            if y_i in classes.keys():
                classes[y_i] += 1
                counts[y_i] += x[i]
            else:
                classes[y_i] = 1
                counts[y_i] = x[i]

        self.class_names = np.array(sorted(classes.keys()))
        self.priors = np.array([classes[y_i] + 1 for y_i in self.class_names])/(n + len(self.class_names))
        
        counts = vstack([counts[y_i] for y_i in self.class_names]).todense()

        total_counts = np.sum(counts, axis=1).reshape(-1,1)
        count_proportions = (counts + 1)/(total_counts + alpha)
        self.w = count_proportions
        self.fitted = True

    def calc_sum(self, x):
        if not self.fitted:
            return "Run NaiveBayes.fit() before predicting."

        x = x.todense().A

        return np.log(self.priors) + np.dot(x, np.log(self.w.T))

    def predict_map(self, x):
        indices = self.argop(self.calc_sum(x), axis=1).flat

        self.most_recent_y_hat = self.class_names[indices].reshape(-1,1)
        return self.most_recent_y_hat

    # def predict_post(self, x):

    #     predictors = self.calc_sum(x)
    #     posterior = predictors/norm(predictors, axis=1, ord=1).reshape(-1,1)

    #     k = len(self.priors)
    #     post_sample = lambda p: np.random.choice(k, size=1, p=p)

    #     indices = np.apply_along_axis(post_sample, 1, posterior).flat

    #     self.most_recent_y_hat = self.class_names[indices].reshape(-1,1)
    #     return self.most_recent_y_hat


class WCNB(NaiveBayes):
    """
    Doesn't really work with count matrices - just tf-idf matrices
    """
    def __init__(self, posterior=False):
        super().__init__(posterior)
        self.predict = self.predict_post if posterior else self.predict_map
        self.argop = np.argmin
        self.name="C"

    def fit(self, x, y):
        _, alpha = x.shape
        n = len(y)
        x = x.todense().A

        # x is already a tfidf matrix
        # "Poor Assumptions" ss 4.3
        x = x/norm(x, ord=2, axis=1).reshape(-1,1)

        # "Poor Assumptions" ss 3.1
        self.class_names = set(np.unique(y))
        classes = {c:0 for c in self.class_names}
        counts = {c:0 for c in self.class_names}
        for i in range(n):
            y_i = y[i][0]
            classes[y_i] += 1
            counts[y_i] += x[i]

        self.class_names = np.array(sorted(classes.keys()))
        # self.priors = np.array([classes[y_i] + 1 for y_i in self.class_names])/(n + len(self.class_names))
        
        outclass = lambda c: sum(counts[cl] for cl in set(np.unique(y)) - {c})

        counts = np.vstack(list(map(outclass, self.class_names)))
        total_counts = np.sum(counts, axis=1).reshape(-1,1)

        # "Poor Assumptions" ss 3.2
        w = np.log((counts + 1)/(total_counts + alpha))
        self.w = w/norm(w, axis=1, ord=1).reshape(-1,1)
        self.fitted = True

    def calc_sum(self, x):
        if not self.fitted:
            return "Run NaiveBayes.fit() before predicting."

        x = x.todense().A

        return np.dot(x, self.w.T)