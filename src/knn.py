import operator
from collections import Counter
from collections import defaultdict

import numpy as np
from FeatureEngineering import FeatureEngineering, WordVectorizer
from bootstrap import BootstrapModel
from matplotlib import pyplot as plt
from metrics import CategoricalMetric
from sklearn import preprocessing, decomposition
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import pandas as pd
from tools import find_project_dir
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from naive_bayes import NaiveBayes, WCNB

class KNN(BootstrapModel):

    def __init__(self):
        super().__init__()

    def calcKNN_E(self, test=None, k=30):
        """
        Calculates the K nearest neighbours and chooses the neighbour
        with the closest distance.
        :param test: test set
        :param k: number of neighbours
        :return: string specifying category
        """
        s = np.sum(np.abs(self.x_train - test) ** 2, axis=-1) ** (1. / 2)
        s = 1/np.power(s, 2)
        # s = 1/s
        label = self.y_train.flatten().tolist()
        vals = [(k, v) for k, v in zip(label, s)]
        vals.sort(key=operator.itemgetter(1), reverse=True)
        nearest_n = [vals[n][0] for n in range(k)]
        nn_count = dict(Counter(nearest_n))
        return max(nn_count.items(), key=operator.itemgetter(1))[0]

    def fit(self, x, y):
        """
        Sets properties for use in program
        :param x: training set x
        :param y: training set y
        """
        self.x_train = x
        self.y_train = y
        self.class_names = np.sort(np.unique(y))
        self.fitted = True

    def predict(self, x_test):
        """
        Calculates predictions for each element in test set
        :param x_test:
        :return:
        """
        predictions = [self.calcKNN_E(row) for row in x_test]
        return np.array(predictions).reshape(-1,1)

    def plot(self):
        """
        Plots Accuracy, Precision, and F1 Measure
        :return:
        """

        x = np.arange(0, 11000, 1000)
        Accuracy = np.array([0.0, 0.545, 0.645, 0.651667, 0.62625, 0.661, 0.665, 0.665, 0.668125, 0.679444, 0.7015])
        Precision = np.array(
            [0.0, 0.678586, 0.670159, 0.680672, 0.634279, 0.675912, 0.680216, 0.67933, 0.687349, 0.701095, 0.718402])
        # Recall = np.array([0.0, 0.545, 0.645, 0.651667, 0.62625, 0.661, 0.665, 0.665, 0.668125, 0.679444, 0.7015])
        F1_Measure = np.array(
            [0.0, 0.544855, 0.640022, 0.649099, 0.619129, 0.661816, 0.662813, 0.664641, 0.666558, 0.676791, 0.699615])

        with plt.style.context('fivethirtyeight'):
            plt.plot(x, Accuracy, label="Accuracy")
            plt.plot(x, Precision, label="Precision")
            plt.plot(x, F1_Measure, label="F1_Measure")

        plt.xlabel('Samples')
        plt.ylabel('Measure')
        plt.title('KNN Measurements vs. Number of Samples')
        plt.legend(loc="lower right")
        plt.show()

    def plot_roc_curve(self, categories, y_score, y_test):
        """
        Plots ROC score
        :param categories: number of categories
        :param y_score:
        :param y_test:
        :return:
        """
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(categories):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.figure()
        lw = 2
        plt.plot(fpr[2], tpr[2], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('KNN 5000 Samples')
        plt.legend(loc="lower right")
        plt.show()


if __name__ == "__main__":
    fullpath = lambda path: os.path.join(find_project_dir(), path)


    fe = FeatureEngineering()
    wv = WordVectorizer()
    # x_ser = fe.read_clean_x_train_features().head(10)
    x_ser = fe.read_x_train_features().head(5000)
    y_mat = fe.read_y_train_features()
    # x_mat = fe.calc_count_matrix(x_ser)
    # x_mat = fe.calc_tfid_matrix(x_ser)
    x_mat = wv.transform(x_ser)

    X_train, X_test, y_train, y_test = train_test_split(x_mat, y_mat, test_size=0.2,
                                                                         random_state=1)

    # pca = decomposition.PCA(n_components=50)
    # pca.fit(X_train)
    # X_train = pca.transform(X_train)
    #
    # pca = decomposition.PCA(n_components=50)
    # pca.fit(X_test)
    # X_test = pca.transform(X_test)

    X_train = preprocessing.normalize(X_train, norm='l2')
    X_test = preprocessing.normalize(X_test, norm='l2')
    #

    X_train = preprocessing.normalize(X_train, norm='l2')
    X_test = preprocessing.normalize(X_test, norm='l2')

    fe = FeatureEngineering()
    x_ser_clean = fe.read_clean_x_train_features().head(5000)
    y_mat = fe.read_y_train_features()

    X_train, X_test, y_train, y_test = train_test_split(x_ser_clean, y_mat, test_size=0.2,
                                                        random_state=1)

    preproc = TfidfVectorizer(analyzer='word', ngram_range=(1, 5), min_df=0.00001, max_df=0.5, norm='l2')
    model = WCNB(preproc=None)
    x_mat = preproc.fit_transform(x_ser_clean)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)


    # knn = KNN()
    # knn.fit(X_train, y_train)
    # y_hat = knn.predict(X_test)
    #
    # p_test = pd.DataFrame(y_test)
    # p_test = pd.get_dummies(p_test).as_matrix()
    #
    # y_score = pd.DataFrame(y_hat)
    # y_score = pd.get_dummies(y_score).as_matrix()
    #
    # classes = y_score.shape[1]
    #
    # knn.plot_roc_curve(classes, y_score, p_test)

















    knn = KNN()
    # knn.fit(X_train, y_train)
    # y_hat = knn.predict(X_test)
    y_hat = pd.read_csv(fullpath("data/niko_yhat.csv"), nrows=5000)
    y_hat = y_hat['category']
    y_score = pd.get_dummies(y_hat).as_matrix()

    # p_test = pd.DataFrame(y_test)


    # p_test = pd.get_dummies(y_hat).as_matrix()


    # y_score = pd.DataFrame(y_hat)
    p_test = pd.read_csv(fullpath("data/train_output.csv"), nrows=5000)
    p_test = p_test['category']
    p_test = pd.get_dummies(p_test).as_matrix()
    #
    classes = y_score.shape[1]

    knn.plot_roc_curve(classes, p_test, y_score)

    # loss = lambda y_hat, y: np.vectorize(int)(y_hat == y)
    # bootstrap = Bootstrap(X_train, y_train, [KNN()], loss, num_samples=5)
    # bootstrap.run()
    # bootstrap.print_summary()

    # Actual predictions on test data
    # x_test = fe.read_clean_x_test_features()
    # fullpath = lambda path: os.path.join(find_project_dir(), path)

    # y_hat = knn.predict(wv.transform(x_test))
    # pd.DataFrame(y_hat,).to_csv(fullpath("models/knn_output.csv"), header=['category'], index_label='id')

    # cm = CategoricalMetric()
    # cm.update(y_hat, y_test)
    # cm.print()
