import numpy as np
import pandas as pd
import os
import itertools

from naive_bayes import NaiveBayes, WCNB
from FeatureEngineering import FeatureEngineering
from tools import find_project_dir
from bootstrap import Bootstrap

def sweep(loss, csv=True, cln=("", "Clean "), ngrams=(1, 2, 3, 4, 5), mdf=(0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.5)):

    def make_mat(cln, ngrams, mdf):
        ser = x_ser_clean if cln else x_ser
        mat = fe.calc_tfid_matrix(ser, max_ngrams=ngrams, min_df=mdf)
        return mat

    def get_maker(csv):
        desc = "{}TF-IDF Data; max_ngrams:{}, min_df: {}"
        desc_csv = "{}TF-IDF Data, {}, {}"
        make_tup = lambda x: (make_mat(*x), desc.format(*x))
        make_tup_csv = lambda x: (make_mat(*x), desc_csv.format(*x))
        return make_tup_csv if csv else make_tup

    fe = FeatureEngineering()
    x_ser = fe.read_x_train_features()
    x_ser_clean = fe.read_clean_x_train_features()
    y_mat = fe.read_y_train_features()

    k = 300
    x_ser = x_ser.head(k)
    x_ser_clean = x_ser_clean.head(k)
    y_mat = y_mat[:k,:]

    x_mat = fe.calc_count_matrix(x_ser)
    x_mat_clean = fe.calc_count_matrix(x_ser_clean)

    csv = True
    if csv: print("data, max_ngrams, min_df, model, predictor, accuracy, precision, recall, f1, boot_acc")
    
    params = itertools.product(cln, ngrams, mdf)
    count_tups = [(x_mat, "Original Data,,"), (x_mat_clean, "Clean Data,,")]

    for x, dat in itertools.chain.from_iterable([count_tups, map(get_maker(csv), params)]):

        if not csv: print("\n{}:".format(dat))

        models = [WCNB()]
        bootstrap = Bootstrap(x, y_mat, loss, models, num_samples=20)
        bootstrap.run()

        def prepend(x):
            typ = 'M' #'P' if x.posterior else 'M'
            return [dat, x.name, typ]
        if csv:
            bootstrap.comma_separated_metrics(prepend=prepend)
        else:
            bootstrap.print_summary()

def main():

    fullpath = lambda path: os.path.join(find_project_dir(), path)
    loss = lambda y_hat, y: np.vectorize(int)(y_hat==y)


    if False: 
        sweep(loss)
    else:
        fe = FeatureEngineering()
        x_ser = fe.read_x_train_features()
        x_ser_clean = fe.read_clean_x_train_features()
        y_mat = fe.read_y_train_features()

        k = 5000
        k_tfidf = 50000
        nsamp = 1000
        x_ser = x_ser.head(k)
        x_ser_clean = x_ser_clean.head(k)
        y_mat = y_mat[:k,:]
        y_mat_tfidf = y_mat[:k_tfidf,:]

        # x_mat = fe.calc_count_matrix(x_ser)
        x_mat_clean = fe.calc_count_matrix(x_ser_clean)
        x_tfidf_clean = fe.calc_tfid_matrix(x_ser_clean, max_ngrams=3, min_df=0.0001)
        # x_mat_clean = fe.calc_count_matrix(x_ser_clean)

        print("done preproc")

        # bootstrap = Bootstrap(x_tfidf_clean, y_mat_tfidf, loss, [WCNB()], num_samples=nsamp)
        # bootstrap.run()
        # bootstrap.print_summary()
        # bootstrap.models[0].save(fullpath('models/wcnb2'))

        # bootstrap = Bootstrap(x_mat_clean, y_mat, loss, [NaiveBayes()], num_samples=nsamp)
        # bootstrap.run()
        # bootstrap.print_summary()
        # bootstrap.models[0].save(fullpath('models/nb'))

        # """
        # Note: features are different
        # """
        # model = WCNB()
        # model.fit(x_tfidf_clean, y_mat)
        # model.save(fullpath('models/wcnb'))
        # model = WCNB.load(fullpath('models/wcnb'))
        # x_test = fe.tf.transform(fe.read_clean_x_test_features())
        # y_hat = model.predict(x_test)
        # pd.DataFrame(y_hat).to_csv(fullpath("models/wcnb_output.csv"))        
        model = NaiveBayes()
        model.fit(x_mat_clean, y_mat)
        model.save(fullpath('models/nb'))
        model = WCNB.load(fullpath('models/nb'))
        x_test = fe.cv.transform(fe.read_clean_x_test_features())
        y_hat = model.predict(x_test)
        pd.DataFrame(y_hat,).to_csv(fullpath("models/nb_output.csv"), header=['category'], index_label='id')


if __name__ == '__main__':
	main()