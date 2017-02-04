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

    class_names = sorted(classes.keys())
    b = np.array([classes[y_i] for y_i in class_names])/n
    
    counts = np.array([counts[y_i] for y_i in class_names])
    total_counts = np.sum(counts, axis=1)[np.newaxis].T
    w = (counts + 1)/(total_counts + alpha)


    def predict(x):
        if len(cols) < len(x[0]):
            x = x[:, cols]
        indices = np.argmax(b + np.dot(x, w.T))
        predictions = np.array([class_names[i] for i in indices])[newaxis].T

# fit_cols_nb = lambda cols: partial(fit_nb, cols=cols)
