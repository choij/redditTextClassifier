from FeatureEngineering import FeatureEngineering, WordVectorizer
from bootstrap import BootstrapModel, Bootstrap
import numpy as np
import math
import operator as op
from tools import timit
from collections import defaultdict


class KNN(BootstrapModel):

    def __init__(self):
        super().__init__()

    @timit
    def calcKNN(self, test=None, k=10):
        d_list = defaultdict(lambda: 0)
        for i in range(self.x_train.shape[0]):
            label = self.y_train[i].item() #label e.g. news, politics
            d = 1/np.linalg.norm(self.x_train[i] - test)
            d_list[label] += d

        d_list = sorted(d_list.items(), key=lambda k_v: k_v[1], reverse=True)
        return d_list[0][0]

    # @timit
    def calcKNN_E(self, test=None, k=10):
        s = np.sum(np.abs(self.x_train - test) ** 2, axis=-1) ** (1. / 2)
        s = 1/s
        label = self.y_train.flatten().tolist()
        d_list = dict(zip(label, s))
        d_list = sorted(d_list.items(), key=lambda k_v: k_v[1], reverse=True)
        return d_list[0][0]

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y
        self.class_names = np.sort(np.unique(y))
        self.fitted = True

    def predict(self, x_test):
        # predictions = []
        # for row in x_test:
        #     p = self.calcKNN_E(row)
        #     # p = self.calcKNN(row)
        #     predictions.append(p)

        predictions = [self.calcKNN_E(row) for row in x_test]
        return np.array(predictions).reshape(-1,1)

if __name__ == "__main__":
    fe = FeatureEngineering()
    wv = WordVectorizer()
    x_ser = fe.read_clean_x_train_features().head(10000)
    y_mat = fe.read_y_train_features()
    # x_mat = fe.calc_count_matrix(x_ser)
    # x_mat = fe.calc_tfid_matrix(x_ser)
    x_mat = wv.transform(x_ser)
    # x_train = fe.merge_matrix(x_mat.todense(), y_mat)[:200, :]
    # x_test = fe.merge_matrix(x_mat.todense(), y_mat)[10:20, :]
    # knn = KNN()
    # knn.calcKNN(x_y_train_mat)


    loss = lambda y_hat, y: np.vectorize(int)(y_hat == y)
    bootstrap = Bootstrap(x_mat, y_mat, [KNN()], loss, num_samples=10)
    bootstrap.run()
    bootstrap.print_summary()