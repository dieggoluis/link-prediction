import numpy as np
import nltk

from library import read_files

stpwds = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.PorterStemmer()

# the columns of node_info are
# (1) paper unique ID (integer)
# (2) publication year (integer)
# (3) paper title (string)
# (4) authors (strings separated by ,)
# (5) name of journal (optional) (string)
# (6) abstract (string) - lowercased, free of punctuation except intra-word dashes

class FeatureExtractor(object):

    def __init__(self):
        file_names = ["node_information.csv"]
        self.node_info = read_files(file_names)[0]

    def fit(self, X, y):
        pass

    def transform(self, X):
        return None
