import numpy as np
from scipy.sparse import vstack

class NaiveBayes():

    def __init__(self):
        self.priors = None

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

    def predict(self, x):
        try:
            self.priors
        except e:
            return "Run NaiveBayes.fit() before predicting."

        x = x.todense().A
        indices = np.argmax(self.priors + np.dot(x, self.w.T), axis=1).A1
        return self.class_names[indices].reshape(-1,1)


