import numpy as np
import pandas as pd
import os
import itertools
import time
import sys

from naive_bayes import NaiveBayes, WCNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from FeatureEngineering import FeatureEngineering
from tools import find_project_dir, parmap
from bootstrap import Bootstrap


def sweep(loss, csv=True, cln=["Clean "], ngrams=[1, 4, 5, 6, 7], min_df=[0.00001], max_df=[0.5, 0.6, 0.7], K=[]):

    fullpath = lambda path: os.path.join(find_project_dir(), path)
    nsamp = [1000, 500, 200, 200, 200, 100, 100, 100, 50, 30, 20, 10, 10]
    fe = FeatureEngineering()
    x_ser = fe.read_x_train_features()
    x_ser_clean = fe.read_clean_x_train_features()
    y_mat = fe.read_y_train_features()
    x_ser_test = fe.read_clean_x_test_features()

    def get_maker(csv):
        desc_print = "{}TF-IDF Data; min_ngrams:{}, max_ngrams:{}, min_df: {}, max_df: {}"
        desc_csv = "{}TF-IDF Data, {}, {}, {}, {}, {}"
        desc = desc_csv if csv else desc_print

        def make_tup(x):
            x = list(x)
            min_ngrams, max_ngrams = x[1]
            x[1] = min_ngrams
            x.insert(2, max_ngrams)
            return (x_ser_clean, desc.format(*x), *x)

        return make_tup

    if csv: print("data, min_ngrams, max_ngrams, min_df, max_df, model, predictor, k, accuracy, precision, recall, f1, boot_acc")
    
    # ngrams = itertools.combinations(ngrams, 2)
    ngrams = [(1, i) for i in ngrams]
    params = itertools.product(cln, ngrams, min_df, max_df, range(len(K)-1, -1, -1))

    tups = list(map(get_maker(csv), params))
    n = len(tups)
    start = time.time()
    i = 1
    for tup in tups:
        x, dat, cln, min_ngrams, max_ngrams, min_df, max_df, j = tup

        k = K[j]
        n_samp = nsamp[j]
        x_ser = x_ser.head(k)
        x_ser_clean = x_ser_clean.head(k)
        y_mat = y_mat[:k,:]
        print("Starting {}...".format(dat), file=sys.stderr, flush=True, end='')

        tfidf = TfidfVectorizer(analyzer='word', ngram_range=(min_ngrams, max_ngrams), min_df=min_df, max_df=max_df, norm='l2')
        # count = CountVectorizer(analyzer='word', ngram_range=(min_ngrams, max_ngrams), min_df=min_df, max_df=max_df)
        try:
            x_mat = tfidf.fit_transform(x_ser_clean)
            # count.fit(x_ser_clean)
            # x_mat_train = count.transform(x_ser_clean)
            # x_mat_test = tfidf.transform(x_ser_test)
        except ValueError as e:
            continue

        if not csv: print("\n{}:".format(dat))

        models = [WCNB(preproc=None)]
        bootstrap = Bootstrap(x_mat, y_mat, models, num_samples=n_samp)
        bootstrap.run()

        def prepend(x):
            typ = 'M'
            return [dat, x.name, typ]

        if csv:
            bootstrap.comma_separated_metrics(prepend=prepend)
        else:
            bootstrap.print_summary()

        finish =time.time()
        print("Done tup {}/{} in {}".format(i, n, finish-start), file=sys.stderr, flush=True)
        i += 1
        start = finish


def main():

    fullpath = lambda path: os.path.join(find_project_dir(), path)
    loss = lambda y_hat, y: np.vectorize(int)(y_hat==y)

    if True: 
        sweep(loss, csv=True, K=[50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400, 165000], ngrams=[5], max_df=[0.5])
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

        print("done preproc A")

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
        preproc=TfidfVectorizer(analyzer='word', ngram_range=(1, 5), min_df=0.00001, max_df=0.5, norm='l2')
        count = CountVectorizer(analyzer='word', ngram_range=(1, 5), min_df=0.00001, max_df=0.5)
        k = "1.5.-5.5_count"
        x_mat = preproc.fit_transform(x_ser_clean)
        count.fit(x_ser_clean)
        print("done preproc B")
        model = WCNB()
        model.fit(x_mat, y_mat)
        model.save(fullpath('models/wcnb{}'.format(k)))
        model = WCNB.load(fullpath('models/wcnb{}'.format(k)))
        x_test = count.transform(fe.read_clean_x_test_features())
        y_hat = model.predict(x_test)
        pd.DataFrame(y_hat).to_csv(fullpath("models/wcnb{}_output.csv".format(k)), header=['category'], index_label='id')   

        # params = [(5,.5),(5,.6),(5,.7),(5,.4),(6,.5),(6,.7),(7,.3),(7,.6),(8,.7),(8,.8)]

        # for max_ngrams, max_df in params:
        #     preproc=TfidfVectorizer(analyzer='word', ngram_range=(1, max_ngrams), min_df=0.00001, max_df=max_df, norm='l2')
        #     k = "1.{}.-5.{}".format(max_ngrams, max_df)
        #     x_mat = preproc.fit_transform(x_ser_clean)
        #     print("done preproc B")
        #     model = WCNB()
        #     model.fit(x_mat, y_mat)
        #     model.save(fullpath('models/top10/wcnb{}'.format(k)))
        #     model = WCNB.load(fullpath('models/top10/wcnb{}'.format(k)))
        #     x_test = preproc.transform(fe.read_clean_x_test_features())
        #     y_hat = model.predict(x_test)
        #     pd.DataFrame(y_hat).to_csv(fullpath("models/top10/wcnb{}_output.csv".format(k)), header=['category'], index_label='id')   


        # model = NaiveBayes(preproc=count)
        # model.fit(x_mat_clean, y_mat)
        # model.save(fullpath('models/nb'))
        # model = WCNB.load(fullpath('models/nb'))
        # x_test = fe.read_clean_x_test_features()
        # y_hat = model.predict(x_test)
        # pd.DataFrame(y_hat,).to_csv(fullpath("models/nb_output.csv"), header=['category'], index_label='id')


if __name__ == '__main__':
	main()