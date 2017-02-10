from FeatureEngineering import FeatureEngineering
from bootstrap import BootstrapModel, Bootstrap
import numpy as np
import math
import operator as op
from tools import timit
from collections import defaultdict
from sklearn import cross_validation

class KNN(BootstrapModel):

    def __init__(self):
        super().__init__()

    @timit
    def calcKNN(self, test=None, k=10):
        d_list = defaultdict(lambda: 0)
        for i in range(self.x_train.shape[0]):
            label = self.y_train[i].item() #label e.g. news, politics
            d = 1/np.linalg.norm(self.x_train[i].todense() - test.todense())
            d_list[label] += d

        d_list = sorted(d_list.items(), key=lambda k_v: k_v[1], reverse=True)
        return d_list[0][0]

    # def calcEuDist(self, x1, x2):
    #     d = 0
    #     x1 = x1[:-1]
    #     x2 = x2[:-1]
    #     for i in range(len(x1)):
    #         d += pow(x1[i] - x2[i],2)
    #     ed = math.sqrt(d)
    #     return ed

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y
        self.class_names = np.unique(y)
        self.fitted = True

    def predict(self, x_test):
        predictions = []
        for row in x_test:
            p = self.calcKNN(row)
            predictions.append(p)
        return np.array(predictions).reshape(-1,1)

    def predict_values(self, train_set, test_set, y_test):
        predictions = []
        for row in test_set:
            p = self.calcKNN(train_set, row)[0]
            predictions.append(p)

        test = y_test[:,-1].tolist()
        correct = 0
        wrong = 0

        for predicted, actual in zip(predictions, test):
            if predicted == actual:
                correct += 1
                print("Correct")
            else:
                print("Actual value: " + actual + "\nPredicted value: " + predicted)
                wrong += 1

        print('Percentage predicted correctly: ' + str(correct/len(predictions)))

if __name__ == "__main__":
    fe = FeatureEngineering()
    x_ser = fe.read_x_train_features()
    y_mat = fe.read_y_train_features()
    x_mat = fe.calc_count_matrix(x_ser)
    # x_train = fe.merge_matrix(x_mat.todense(), y_mat)[:200, :]
    # x_test = fe.merge_matrix(x_mat.todense(), y_mat)[10:20, :]
    y_test = y_mat[10:20, :]
    # knn = KNN()
    # knn.calcKNN(x_y_train_mat)
    loss = lambda y_hat, y: np.vectorize(int)(y_hat == y)
    bootstrap = Bootstrap(x_mat, y_mat, loss, [KNN()], num_samples=10)
    bootstrap.run()
    bootstrap.print_summary()

    # scores = knn.predict_values(x_train, x_test, y_test)

    data1 = [2, 2, 2, 'a']
    data2 = [4, 4, 4, 'b']