import numpy as np
from scipy.sparse import vstack

from metrics import CategoricalMetric

class NaiveBayes():

    def __init__(self):
        self.priors = None
        self.fitted = False
        self.metrics = CategoricalMetric()

    def is_fitted(self):
        return self.fitted

    def unfit(self):
        self.priors = None
        self.w = None
        self.fitted = False
        self.class_names = None

    def update_metrics(self, y_hat, y_test):
        self.metrics.update(y_hat, y_test)

    def print_metrics(self):
        self.metrics.print()

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
        self.priors = (np.array([classes[y_i] for y_i in self.class_names])/n)
        
        counts = vstack([counts[y_i] for y_i in self.class_names]).todense()

        total_counts = np.sum(counts, axis=1)
        self.w = (counts + 1)/(total_counts + alpha)
        self.fitted = True

    def predict(self, x):
        if not self.fitted:
            return "Run NaiveBayes.fit() before predicting."

        x = x.todense().A
        indices = np.argmax(self.priors + np.dot(x, self.w.T), axis=1).A1
        return self.class_names[indices].reshape(-1,1)


