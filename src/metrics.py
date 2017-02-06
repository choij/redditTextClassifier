import numpy as np
from tabulate import tabulate
from sklearn.metrics import precision_recall_fscore_support as prf
from sklearn.metrics import accuracy_score

from tools import update_moving_average

class CategoricalMetric:
    def __init__(self):
        self.prf = 0
        self.acc = 0
        self.n = None

    def update(self, y_hat, y_test):
        y_hat, y_test = y_hat.flatten(), y_test.flatten()

        new_prf = np.array(prf(y_test, y_hat, average='weighted'))[:-1]
        new_acc = accuracy_score(y_test, y_hat)

        self.prf = update_moving_average(self.prf, new_prf, self.n)
        self.acc = update_moving_average(self.acc, new_acc, self.n)

        self.n = 1 if self.n is None else self.n + 1

    def print(self):
        headers = ["Accuracy", "Precision", "Recall", "F1-Measure"]
        data = [self.acc] + self.prf.tolist()
        table = [headers, data]
        print(tabulate(table, headers="firstrow"))
