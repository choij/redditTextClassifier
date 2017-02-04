import numpy as np
from scipy.sparse import vstack

def fit_nb(x, y, cols=None):
    if cols is None: cols = np.arange(x.shape[1])
    x = x[:,cols]

    alpha = len(cols)
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

    class_names = np.array(sorted(classes.keys()))
    b = (np.array([classes[y_i] for y_i in class_names])/n)
    
    counts = vstack([counts[y_i] for y_i in class_names]).todense()

    total_counts = np.sum(counts, axis=1)
    w = (counts + 1)/(total_counts + alpha).A

    def predict(x):
        if len(cols) < x.shape[1]:
            x = x[:, cols]
        x = x.todense().A

        indices = np.argmax(b + np.dot(x, w.T), axis=1).A1
        return class_names[indices].reshape(-1,1)

    return predict

