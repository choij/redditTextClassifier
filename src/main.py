import numpy as np
import pandas as pd
import os
import itertools

from naive_bayes import NaiveBayes, WCNB
from sklearn.feature_extraction.text import TfidfVectorizer
from FeatureEngineering import FeatureEngineering
from tools import find_project_dir
from bootstrap import Bootstrap


def sweep(loss, csv=True, cln=["Clean "], ngrams=[2,4], min_df=[0], max_df=[0.7]):

    fe = FeatureEngineering()
    x_ser = fe.read_x_train_features()
    x_ser_clean = fe.read_clean_x_train_features()
    y_mat = fe.read_y_train_features()

    k = 5000
    x_ser = x_ser.head(k)
    x_ser_clean = x_ser_clean.head(k)
    y_mat = y_mat[:k,:]

    def make_ser(cln, ngrams, min_df, max_df):
        ser = x_ser_clean if cln else x_ser
        return ser

    def get_maker(csv):
        desc_print = "{}TF-IDF Data; max_ngrams:{}, min_df: {}, max_df: {}"
        desc_csv = "{}TF-IDF Data, {}, {}, {}"
        desc = desc_csv if csv else desc_print

        make_tup = lambda x: (make_ser(*x), desc.format(*x), *x)
        return make_tup

    csv = True
    if csv: print("data, max_ngrams, min_df, max_df, model, predictor, accuracy, precision, recall, f1, boot_acc")
    
    params = itertools.product(cln, ngrams, min_df, max_df)

    for x, dat, cln, ngrams, min_df, max_df in map(get_maker(csv), params):

        tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, ngrams), min_df=min_df, max_df=max_df)
        x_mat = tfidf.fit_transform(x_ser_clean)

        if not csv: print("\n{}:".format(dat))

        models = [WCNB(preproc=None)]
        bootstrap = Bootstrap(x_mat, y_mat, models, num_samples=20)
        bootstrap.run()

        def prepend(x):
            typ = 'M'
            return [dat, x.name, typ]
        if csv:
            bootstrap.comma_separated_metrics(prepend=prepend)
        else:
            bootstrap.print_summary()

def main():

    fullpath = lambda path: os.path.join(find_project_dir(), path)
    loss = lambda y_hat, y: np.vectorize(int)(y_hat==y)

    if True: 
        sweep(loss)
    else:
        fe = FeatureEngineering()
        # x_ser = fe.read_x_train_features()
        x_ser_clean = fe.read_clean_x_train_features()
        y_mat = fe.read_y_train_features()

        # k = 5000
        # k_tfidf = 500
        # nsamp = 10
        # x_ser = x_ser.head(k)
        # x_ser_clean = x_ser_clean.head(k)
        # y_mat = y_mat[:k,:]
        # y_mat_tfidf = y_mat[:k_tfidf,:]

        # x_mat = fe.calc_count_matrix(x_ser)
        # x_mat_clean = fe.calc_count_matrix(x_ser_clean)
        # x_tfidf_clean = fe.calc_tfid_matrix(x_ser_clean, max_ngrams=3, min_df=0.0001)
        # count = CountVectorizer()
        # tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0.0001) 
        # x_mat_clean = fe.calc_count_matrix(x_ser_clean)

        print("done preproc")

        """
        Run bootstrap.
        """
        # bootstrap = Bootstrap(x_ser_clean.head(500), y_mat_tfidf, [WCNB(preproc=tfidf)], num_samples=nsamp)
        # bootstrap.run()
        # bootstrap.print_summary()
        # bootstrap.models[0].save(fullpath('models/wcnb3'))

        # bootstrap = Bootstrap(x_ser_clean, y_mat, [NaiveBayes()], num_samples=nsamp)
        # bootstrap.run()
        # bootstrap.print_summary()
        # bootstrap.models[0].save(fullpath('models/nb'))

        """
        Run fe.cv.transform() or fe.tf.transform() to get features
        after learning a model. Right now you have to run fe.calc_XXX_matrix
        on the data that was used to train the model first, then
        fe.XX.transform(x), where x is a Pandas series.
        """
        model = WCNB(preproc=TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0.00001, max_df=0.7) )
        model.fit(x_ser_clean, y_mat)
        model.save(fullpath('models/wcnb3'))
        model = WCNB.load(fullpath('models/wcnb3'))
        x_test = fe.read_clean_x_test_features()
        y_hat = model.predict(x_test)
        pd.DataFrame(y_hat).to_csv(fullpath("models/wcnb3_output.csv"), header=['category'], index_label='id')   

        # model = NaiveBayes(preproc=count)
        # model.fit(x_mat_clean, y_mat)
        # model.save(fullpath('models/nb'))
        # model = WCNB.load(fullpath('models/nb'))
        # x_test = fe.read_clean_x_test_features()
        # y_hat = model.predict(x_test)
        # pd.DataFrame(y_hat,).to_csv(fullpath("models/nb_output.csv"), header=['category'], index_label='id')


if __name__ == '__main__':
	main()