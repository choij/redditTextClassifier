import numpy as np
import pandas as pd
import os

from naive_bayes import NaiveBayes
from FeatureEngineering import FeatureEngineering
from tools import find_project_dir
from bootstrap import Bootstrap


def main():
    fullpath = lambda path: os.path.join(find_project_dir(), path)

    fe = FeatureEngineering()
    x_ser = fe.read_x_train_features()
    x_ser_clean = fe.read_clean_x_train_features()
    y_mat = fe.read_y_train_features()

    k = 500
    x_ser = x_ser.head(k)
    x_ser_clean = x_ser_clean.head(k)
    y_mat = y_mat[:k,:]

    x_mat = fe.calc_count_matrix(x_ser)
    x_mat_clean = fe.calc_count_matrix(x_ser_clean)
    # x_y_train_mat = fe.merge_matrix(x_mat.todense(),y_mat)

    loss = lambda y_hat, y: np.vectorize(int)(y_hat==y)
    models = [NaiveBayes()]
    bootstrap = Bootstrap(x_mat, y_mat, loss, models, num_samples=20)
    bootstrap.run()

    bootstrap.print_summary()

    models = [NaiveBayes()]
    bootstrap = Bootstrap(x_mat_clean, y_mat, loss, models, num_samples=20)
    bootstrap.run()

    bootstrap.print_summary()

    # pd.DataFrame(y_hat).to_csv(fullpath("nb_predictions.csv"), header=False, index=False)

if __name__ == '__main__':
	main()