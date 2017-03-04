# coding=utf-8
import numpy as np
import nltk
import pandas as pd
import gensim
from library import read_files, clean_text_simple

#data_path = '../data/'
#model = gensim.models.word2vec.Word2Vec.load(data_path + 'custom_w2v_model.txt')
#model.intersect_word2vec_format(data_path + 'GoogleNews-vectors-negative300.bin.gz', binary=True)

file_names = ['node_information.csv', 'cleaned_titles.csv', 'cleaned_abstracts.csv']
node_info = pd.read_csv(file_names[0], index_col=0, header=None)
cleaned_titles = pd.read_csv(file_names[1], index_col=0, header=None)
cleaned_abstracts = pd.read_csv(file_names[2], index_col=0, header=None)

### TMP
# - Overlap dos textos dos abstracts ok
# - Overlap de keywords dos abstracts
# - Número de pares de sinônimos entre as keywords
# - Overlap entre títulos e abstracts opostos ok
# - authors overlap ok
### TMP

# the columns of node_info are
# (1) paper unique ID (integer)
# (2) publication year (integer)
# (3) paper title (string)
# (4) authors (strings separated by ,)
# (5) name of journal (optional) (string)
# (6) abstract (string) - lowercased, free of punctuation except intra-word dashes

class FeatureExtractor(object):

    def __init__(self):
        pass

    def __intersect_array(self, a, b):
        array_a = []
        for row in a:
            if row[0]!=row[0] :
                array_a.append([])
            else:
                array_a.append([s.strip() for s in row[0].split(',')])
        # doc2
        array_b = []
        for row in b:
            if row[0]!=row[0] :
                array_b.append([])
            else:
                array_b.append([s.strip() for s in row[0].split(',')])

        # check same size
        assert len(array_a)==len(array_b)

        intersec = [len(set(x).intersection(set(y))) for x, y in zip(array_a, array_b)]
        return intersec

    def fit(self, X, y):
        pass

    def transform(self, X):

        print 'transform...'

        idx_docs1 = X[:,0]
        idx_docs2 = X[:,1]
        features = []

        # difference publication year
        year_docs1 = np.array(node_info.loc[idx_docs1,1:1].astype(int))
        year_docs2 = np.array(node_info.loc[idx_docs2,1:1].astype(int))
        features.append(year_docs1 - year_docs2)
        # -----
        
        # titles overlap
        titles_docs1 = np.array(cleaned_titles.loc[idx_docs1])
        titles_docs2 = np.array(cleaned_titles.loc[idx_docs2])
        intersec = self.__intersect_array(titles_docs1, titles_docs2)
        intersec = np.reshape(np.array(intersec), (-1,1))
        features.append(intersec)
        # -----

        # abstracts overlap
        abstract_doc1 = np.array(cleaned_abstracts.loc[idx_docs1])
        abstract_doc2 = np.array(cleaned_abstracts.loc[idx_docs2])
        intersec = self.__intersect_array(abstract_doc1, abstract_doc2)
        intersec = np.reshape(np.array(intersec), (-1,1))
        features.append(intersec)
        # -----

        # titles and abstract overlap
        intersec1 = np.array(self.__intersect_array(titles_docs1, abstract_doc2))
        intersec2 = np.array(self.__intersect_array(abstract_doc1, titles_docs2))
        intersec = np.reshape(intersec1 + intersec2, (-1,1))
        features.append(intersec)
        # -----

        # authors overlap
        authors_doc1 = np.array(node_info.loc[idx_docs1,3:3])
        authors_doc2 = np.array(node_info.loc[idx_docs2,3:3])
        intersec = self.__intersect_array(authors_doc1, authors_doc2)
        intersec = np.reshape(np.array(intersec), (-1,1))
        features.append(intersec)
        # -----

        features_array = np.concatenate(tuple(features), axis=1)

        return features_array
