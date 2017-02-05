import numpy as np
import pandas as pd
import os

from naive_bayes import NaiveBayes
from FeatureEngineering import FeatureEngineering
from tools import find_project_dir


def main():
    fullpath = lambda path: os.path.join(find_project_dir(), path)

    fe = FeatureEngineering()
    x_df = fe.read_x_train_features()
    y_mat = fe.read_y_train_features()

    x_df = x_df.head(100)
    y_mat = y_mat[:100,:]

    x_mat = fe.calc_count_matrix(x_df)
    # x_y_train_mat = fe.merge_matrix(x_mat.todense(),y_mat)


    nb = NaiveBayes()
    nb.fit(x_mat, y_mat)
    y_hat = nb.predict(x_mat)
    acc = float(sum(y_hat == y_mat)/len(y_hat))
    print("Accuracy on training set: {}".format(acc))
    pd.DataFrame(y_hat).to_csv(fullpath("nb_predictions.csv"), header=False, index=False)

if __name__ == '__main__':
	main()