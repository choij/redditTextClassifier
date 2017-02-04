import numpy as np

from naive_bayes import fit_nb
from FeatureEngineering import FeatureEngineering

def main():
    fe = FeatureEngineering()
    x_df = fe.read_x_train_features()
    y_mat = fe.read_y_train_features()
    x_mat = fe.calc_count_matrix(x_df)
    x_y_train_mat = fe.merge_matrix(x_mat.todense(),y_mat)

    fitted = fit_nb(x_mat, y_mat)
    y_hat = fitted(x_mat)
    acc = float(sum(y_hat == y_mat)/len(y_hat))
    print("Accuracy on training set: {}".format(acc))

if __name__ == '__main__':
	main()