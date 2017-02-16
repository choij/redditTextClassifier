import numpy as np
import operator
from scipy import sparse
from sklearn.preprocessing import normalize

from metrics import CategoricalMetric
from bootstrap import BootstrapModel

class NaiveBayes(BootstrapModel):

    def __init__(self, preproc=None, posterior=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "N"
        self.priors = None
        self.w = None
        self.class_names = None

        self.predict = self.predict_post if posterior else self.predict_map
        self.argop = np.argmax
        self.preproc = preproc

    def unfit(self):
        self.priors = None
        self.w = None
        self.class_names = None

        self.fitted = False

    def fit(self, x, y):
        if self.preproc:
            self.preproc.fit(x)
            x = self.preproc.transform(x)
        _, alpha = x.shape
        n = len(y)

        self.class_names = np.sort(np.unique(y))
        cset = set(self.class_names)
        class_indices = {c: np.where(y == c)[0] for c in cset}
        class_sum = lambda c: x[class_indices(c),:].sum(axis=0).flatten()
        class_freq = [len(class_indices[c]) + 1 for c in self.class_names]

        self.priors = np.array(class_freq)/(n + len(cset))
        counts = np.vstack(class_sum(c) for c in self.class_names)
        total_counts = counts.sum(axis=1).reshape(-1,1)

        w = sparse.csr_matrix((counts + 1)/(total_counts + alpha))
        self.w = w.transpose()
        self.fitted = True

    def calc_sum(self, x):
        if not self.fitted:
            return "Run NaiveBayes.fit() before predicting."

        weights = sparse.csr_matrix(np.log(self.w))
        return np.log(self.priors) + np.dot(x, weights).todense()

    def predict_map(self, x):
        if self.preproc:
            x = self.preproc.transform(x)
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
    def __init__(self, preproc=None, posterior=False, *args, **kwargs):
        super().__init__(preproc=preproc, posterior=posterior, *args, **kwargs)
        self.predict = self.predict_post if posterior else self.predict_map
        self.argop = np.argmin
        self.name="C"

    def fit(self, x, y):
        if self.preproc:
            self.preproc.fit(x)
            x = self.preproc.transform(x)
        _, alpha = x.shape

        # x is already a tfidf matrix
        # "Poor Assumptions" ss 3.1
        self.class_names = np.sort(np.unique(y))
        cset = set(self.class_names)
        class_indices = {c: np.where(y == c)[0] for c in cset}
        outclass = lambda c: np.hstack(class_indices[cl] for cl in cset - {c})
        out_sum = lambda c: x[outclass(c),:].sum(axis=0).flatten()

        counts = np.vstack(out_sum(c) for c in self.class_names)
        total_counts = counts.sum(axis=1).reshape(-1,1)

        # "Poor Assumptions" ss 3.2
        w = sparse.csr_matrix(np.log((counts + 1)/(total_counts + alpha)))
        self.w = normalize(w, norm='l1', axis=1).transpose()
        self.fitted = True

    def calc_sum(self, x):
        if not self.fitted:
            return "Run NaiveBayes.fit() before predicting."

        return np.dot(x, self.w).todense()
