import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureEngineering:
    def read_x_train_features(self):
        df = pd.read_csv("data/train_input.csv")
        df = df.apply(lambda x: self.replace_unwanted_str(x['conversation']), axis=1)
        df = df.head(10)

        return df

    def replace_unwanted_str(self, row):

        mapping = [('<speaker_1>', ''), ('<number>', ''), ('</s>', '3')]
        for k, v in mapping:
            row = row.replace(k, v)

        return row

    def calc_count_matrix(self, df):
        count_vector = CountVectorizer()
        count_matrix = count_vector.fit_transform(df.tolist())

        return count_matrix

    def calc_tfid_matrix(self, df):
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
        tfidf_matrix = tf.fit_transform(df.tolist())
        feature_names = tf.get_feature_names()

        return tfidf_matrix

    def read_y_train_features(self):
        df = pd.read_csv("data/train_output.csv")
        df = df.drop('id', axis=1)
        df = df.head(10)
        y_train = df.as_matrix()
        return y_train

    def merge_matrix(self, mat1, mat2):
        return np.concatenate((mat1,mat2), axis=1)

if __name__ == '__main__':
    fe = FeatureEngineering()
    x_df = fe.read_x_train_features()
    y_mat = fe.read_y_train_features()
    x_mat = fe.calc_count_matrix(x_df)
    x_y_train_mat = fe.merge_matrix(x_mat.todense(),y_mat)

    from naive_bayes import fit_nb

    print(y_mat)
    print(fit_nb(x_mat, y_mat))

