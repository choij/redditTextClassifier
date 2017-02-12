from FeatureEngineering import FeatureEngineering, WordVectorizer
from bootstrap import BootstrapModel, Bootstrap
from metrics import CategoricalMetric
import numpy as np
import pandas as pd
import os
from tools import find_project_dir
import math
import operator as op
from tools import timit
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import decomposition
from sklearn import preprocessing



class KNN(BootstrapModel):

    def __init__(self):
        super().__init__()

    # @timit
    def calcKNN(self, test=None, k=10):
        d_list = defaultdict(lambda: 0)
        for i in range(self.x_train.shape[0]):
            label = self.y_train[i].item() #label e.g. news, politics
            d = 1/np.linalg.norm(self.x_train[i] - test)
            d_list[label] += d

        d_list = sorted(d_list.items(), key=lambda k_v: k_v[1], reverse=True)
        return d_list[0][0]

    # @timit
    def calcKNN_E(self, test=None, k=50):
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
    x_ser = fe.read_clean_x_train_features().head(5000)
    y_mat = fe.read_y_train_features()
    # x_mat = fe.calc_count_matrix(x_ser)
    # x_mat = fe.calc_tfid_matrix(x_ser)
    x_mat = wv.transform(x_ser)

    X_train, X_test, y_train, y_test = train_test_split(x_mat, y_mat, test_size=0.2,
                                                                         random_state=1)

    pca = decomposition.PCA(n_components=50)
    pca.fit(X_train)
    X_train = pca.transform(X_train)

    pca = decomposition.PCA(n_components=50)
    pca.fit(X_test)
    X_test = pca.transform(X_test)

    X_train = preprocessing.normalize(X_train, norm='l2')
    X_test = preprocessing.normalize(X_test, norm='l2')

    knn = KNN()
    knn.fit(X_train, y_train)
    y_hat = knn.predict(X_test)

    # Actual predictions on test data
    # x_test = fe.read_clean_x_test_features()
    # fullpath = lambda path: os.path.join(find_project_dir(), path)
    #
    # y_hat = knn.predict(wv.transform(x_test))
    # pd.DataFrame(y_hat,).to_csv(fullpath("models/knn_output.csv"), header=['category'], index_label='id')

    cm = CategoricalMetric()
    cm.update(y_hat, y_test)
    cm.print()


    # loss = lambda y_hat, y: np.vectorize(int)(y_hat == y)
    # bootstrap = Bootstrap(x_mat, y_mat, [KNN()], loss, num_samples=10)
    # bootstrap.run()
    # bootstrap.print_summary()