from FeatureEngineering import FeatureEngineering
import numpy as np
import math
import operator as op
from sklearn import cross_validation

class KNN:

    def calcKNN(self, train, test=None, k=10):
        # test = train[0]
        d_list = []
        for row in train:
            t = row.tolist()[0]
            d = self.calcEuDist(t,test.tolist()[0])
            d_list.append((t,d))

        d_list.sort(key=op.itemgetter(1))
        # print([(d[0][-1], d[-1]) for d in d_list])
        n = [d_list[i][0] for i in range(k)]

        # Sum euclidian distances instead of the below method.
        popular = {}
        for i in n:
            neighbour = i[-1]
            if neighbour in popular:
                popular[neighbour] += 1
            else:
                popular[neighbour] = 1

        value = max(popular.items(), key=op.itemgetter(1))
        return value

    def calcKernal(self, distance):

    def calcEuDist(self, x1, x2):
        d = 0
        x1 = x1[:-1]
        x2 = x2[:-1]
        for i in range(len(x1)):
            d += pow(x1[i] - x2[i],2)
        ed = math.sqrt(d)
        return ed

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
    x_train = fe.merge_matrix(x_mat.todense(), y_mat)[:200, :]
    x_test = fe.merge_matrix(x_mat.todense(), y_mat)[10:20, :]
    y_test = y_mat[10:20, :]
    knn = KNN()
    # knn.calcKNN(x_y_train_mat)

    scores = knn.predict_values(x_train, x_test, y_test)

    data1 = [2, 2, 2, 'a']
    data2 = [4, 4, 4, 'b']