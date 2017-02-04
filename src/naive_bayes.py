import numpy as np

def fit_nb(x, y, cols=None):
    if cols is None: cols = np.arange(len(x[0]))
    x = x[:,cols]

    alpha = len(cols)
    n = len(y)

    classes = {}
    counts = {}
    for i in range(n):
        if y[i] in classes.keys():
            classes[y[i]] += 1
            counts[y[i]] += x[i]
        else:
            classes[y[i]] = 1
            counts[y[i]] = x[i]

    b = np.array([classes[y_i] for y_i in sorted(classes.keys())])/n
    
    counts = np.array([counts[y_i] for y_i in sorted(classes.keys())])
    total_counts = np.sum(counts, axis=1)[np.newaxis].T
    w = (counts + 1)/(total_counts + alpha)

    def predict(x):
        if len(cols) < len(x[0]):
            x = x[:, cols]
        return np.argmax(b + np.dot(x, w.T))[newaxis].T

# fit_cols_nb = lambda cols: partial(fit_nb, cols=cols)


