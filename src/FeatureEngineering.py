import textmining as tm
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureEngineering:
    def read_features(self):
        df = pd.read_csv("data/train_input.csv")
        df = df.apply(lambda x: self.replace_unwanted_str(x['conversation']), axis=1)
        df = df.head(10)

        self.count_occur_of_words(df)

    def replace_unwanted_str(self, row):

        mapping = [('<speaker_1>', ''), ('<number>', ''), ('</s>', '3')]
        for k, v in mapping:
            row = row.replace(k, v)

        return row

    def count_occur_of_words(self, df):
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
        tfidf_matrix = tf.fit_transform(df.tolist())
        feature_names = tf.get_feature_names()


if __name__ == '__main__':
    fe = FeatureEngineering()
    fe.read_features()
