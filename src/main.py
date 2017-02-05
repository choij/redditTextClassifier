import numpy as np
import pandas as pd
import os

from naive_bayes import NaiveBayes
from FeatureEngineering import FeatureEngineering
from tools import find_project_dir
from bootstrap import bootstrap


def main():
    fullpath = lambda path: os.path.join(find_project_dir(), path)

    fe = FeatureEngineering()
    x_df = fe.read_x_train_features()
    y_mat = fe.read_y_train_features()

    x_df = x_df.head(5000)
    y_mat = y_mat[:5000,:]

    x_mat = fe.calc_count_matrix(x_df)
    # x_y_train_mat = fe.merge_matrix(x_mat.todense(),y_mat)


    """
    TODO: make bootstrap a class
    """
    loss = lambda y_hat, y: np.vectorize(int)(y_hat==y)
    models = [NaiveBayes]
    acc = bootstrap(x_mat, y_mat, loss, models, num_samples=200)

    acc = float(acc[0])
    print("Accuracy on training set: {}".format(acc))

    # pd.DataFrame(y_hat).to_csv(fullpath("nb_predictions.csv"), header=False, index=False)

if __name__ == '__main__':
	main()