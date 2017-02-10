import pandas as pd
import numpy as np
import os
import re
import string
import spacy

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords, wordnet
from nltk.stem.snowball import EnglishStemmer
from nltk import wordpunct_tokenize, WordNetLemmatizer, sent_tokenize, pos_tag
from itertools import chain
from collections import defaultdict
from scipy import sparse

from tools import find_project_dir

class FeatureEngineering:
    def __init__(self):
        self.dir = lambda path: os.path.join(find_project_dir(), path)

        self.stopwords = set(stopwords.words('english'))
        self.punct = set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = EnglishStemmer()

        self.tf = None

    def lemmatize(self, t):
        """
        Converts Treebank tags into WordNet tags and calls the lemmatizer.
        """
        token, tag = t
        tag = {
            'V': wordnet.VERB,
            'R': wordnet.ADV,
            'J': wordnet.ADJ,
            'S': wordnet.ADJ_SAT
        }.get(tag[0], wordnet.NOUN)

        return self.lemmatizer.lemmatize(token, tag)

    def clean_row(self, row):
        """
        Helper function for 'clean'. Inspired by http://bbengfort.github.io/tutorials/2016/05/19/text-classification-nltk-sckit-learn.html
        """
        validate = lambda x: not (all(char in self.punct for char in x[0]) or x[0] in self.stopwords) 

        sentences = sent_tokenize(row[0])
        tokens = map(wordpunct_tokenize, sentences)
        tagged_tokens = chain.from_iterable(map(pos_tag, tokens))
        valid_tokens = filter(validate, tagged_tokens)
        lemmatized_tokens = map(self.lemmatize, valid_tokens)
        stemmed = map(self.stemmer.stem, lemmatized_tokens)

        return " ".join(stemmed)

    def clean(self, df, p=False):
        """
        Runs the following procedure on each row:
            1. Tokenizes the sentences
            2. Tokenizes the words in each sentence
            3. Tags each word with a POS (part of speech) tag
            4. Removes stop-words and punctuation
            5. Lemmatizes each word
            6. Stems each word
            7. Merges the list of rows into space-separated string
        """

        df = pd.DataFrame(df)
        df = pd.DataFrame(df.apply(self.clean_row, axis=1))

        return df

    def read_clean_x_train_features(self):
        fp = self.dir("data/cleaned_train_input.csv")
        return pd.read_csv(fp, header=None)[0]

    def read_clean_x_test_features(self):
        fp = self.dir("data/cleaned_test_input.csv")
        return pd.read_csv(fp, header=None)[0]

    def read_x_train_features(self):
        df = pd.read_csv(self.dir("data/train_input.csv"))
        ser = df.apply(lambda x: self.replace_unwanted_str(x['conversation']), axis=1)

        return ser

    def read_x_test_features(self):
        df = pd.read_csv(self.dir("data/test_input.csv"))
        ser = df.apply(lambda x: self.replace_unwanted_str(x['conversation']), axis=1)

        return ser

    def replace_unwanted_str(self, row):
        """
        Removes HTML tags.
        """
        return re.sub('<[^<]+?>', '', row)

    def calc_count_matrix(self, ser):
        self.cv = CountVectorizer()
        count_matrix = self.cv.fit_transform(ser.tolist())

        return count_matrix

    def calc_tfid_matrix(self, ser, max_ngrams=3, min_df=0):
        self.tf = TfidfVectorizer(analyzer='word', ngram_range=(1, max_ngrams), min_df=min_df)
        tfidf_matrix = self.tf.fit_transform(ser.tolist())
        feature_names = self.tf.get_feature_names()

        return tfidf_matrix

    def read_y_train_features(self):
        df = pd.read_csv(self.dir("data/train_output.csv"))
        df = df.drop('id', axis=1)
        y_train = df.as_matrix()
        return y_train

    def merge_matrix(self, mat1, mat2):
        return np.concatenate((mat1,mat2), axis=1)

class WordVectorizer:
    def __init__(self):
        self.nlp = spacy.load('en')

    def fit(self, x):
        pass

    def transform(self, x):
        return np.vstack([nlp(x_i).vector for x_i in x])


if __name__ == '__main__':
    fe = FeatureEngineering()
    x_ser = fe.read_clean_x_train_features()
    y_mat = fe.read_y_train_features()
    x_mat = fe.calc_count_matrix(x_ser)
    x_tfidf = fe.calc_tfid_matrix(x_ser)
    # x_y_train_mat = fe.merge_matrix(x_mat.todense(),y_mat)

    y_mat = y_mat[:500,:]
