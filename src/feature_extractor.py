# coding=utf-8
import numpy as np
import nltk
import pandas as pd
import gensim
import igraph
from library import read_files, clean_text_simple
from sklearn.feature_extraction.text import TfidfVectorizer


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
# - mesmo journal ok
# - tf-idf vectors cosine similarity
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

    def __tfidf_cossine_similarity(self, a, b):
        documents = np.concatenate((a,b), axis = 0)
        newdoc = []
        for item in documents:
            if type(item[0]) == str:
                newdoc.append(item[0].replace(","," "))
            else:
                newdoc.append("")
        tfidf = TfidfVectorizer().fit_transform(newdoc)

        #print "fitted"
        dot_products = []
        n = len(a)
        #print "n = ", n
        for i in range(n):
            if i%10000==0:
                print i
            dot_products.append((tfidf[i]*tfidf[i+n].T).A[0] )
        #print "gerou"
        return np.array(dot_products)            


    def fit(self, X, y):
        ## build graph of the citation network
        ids1 = X[:,0]
        ids2 = X[:,1]
        vertices = list(set(ids1.astype(str)).union(ids2.astype(str)))
        edges = [tuple([str(row[0]), str(row[1])]) for row, link in zip(X,y) if link == 1]

        self.network_graph = igraph.Graph() 
        self.network_graph.add_vertices(vertices)
        self.network_graph.add_edges(edges)

        vs = zip(vertices, range(len(vertices)))
        self.hash_vs = {a:b for a,b in vs}
        print 'calculating shortest paths...'
        self.dmatrix = np.array(self.network_graph.shortest_paths())

    def transform(self, X):

        print 'Transform...'

        idx_docs1 = X[:,0]
        idx_docs2 = X[:,1]
        features = []
        
        # cosine similarity of titles text
        print 'titles cosine similarity...'
        titles_docs1 = np.array(cleaned_titles.loc[idx_docs1])
        titles_docs2 = np.array(cleaned_titles.loc[idx_docs2])
        features.append(self.__tfidf_cossine_similarity(titles_docs1, titles_docs2))            
        
        # cosine similarity of abstract text
        print 'abstract cosine similarity...'
        abstract_doc1 = np.array(cleaned_abstracts.loc[idx_docs1])
        abstract_doc2 = np.array(cleaned_abstracts.loc[idx_docs2])
        features.append(self.__tfidf_cossine_similarity(abstract_doc1, abstract_doc2))

        # difference publication year
        print 'difference publication year...'
        year_docs1 = np.array(node_info.loc[idx_docs1,1:1].astype(int))
        year_docs2 = np.array(node_info.loc[idx_docs2,1:1].astype(int))
        features.append(year_docs1 - year_docs2)

        # titles overlap
        print 'titles overlap...'
        titles_docs1 = np.array(cleaned_titles.loc[idx_docs1])
        titles_docs2 = np.array(cleaned_titles.loc[idx_docs2])
        intersec = self.__intersect_array(titles_docs1, titles_docs2)
        intersec = np.reshape(np.array(intersec), (-1,1))
        features.append(intersec)

        # abstracts overlap
        print 'abstracts overlap...'
        abstract_doc1 = np.array(cleaned_abstracts.loc[idx_docs1])
        abstract_doc2 = np.array(cleaned_abstracts.loc[idx_docs2])
        intersec = self.__intersect_array(abstract_doc1, abstract_doc2)
        intersec = np.reshape(np.array(intersec), (-1,1))
        features.append(intersec)

        # titles and abstract overlap
        print 'title and abstract overlap...'
        intersec1 = np.array(self.__intersect_array(titles_docs1, abstract_doc2))
        intersec2 = np.array(self.__intersect_array(abstract_doc1, titles_docs2))
        intersec = np.reshape(intersec1 + intersec2, (-1,1))
        features.append(intersec)

        # authors overlap
        print 'authors overlap...'
        authors_doc1 = np.array(node_info.loc[idx_docs1,3:3])
        authors_doc2 = np.array(node_info.loc[idx_docs2,3:3])
        intersec = self.__intersect_array(authors_doc1, authors_doc2)
        intersec = np.reshape(np.array(intersec), (-1,1))
        features.append(intersec)

        # same journal
        print 'same journal...'
        journal_doc1 = np.array(node_info.loc[idx_docs1, 4:4])
        journal_doc2 = np.array(node_info.loc[idx_docs2, 4:4])
        empties = np.array(["" for i in range(len(journal_doc1))])
        unknown_journal = np.array((journal_doc1=="") | (journal_doc2=="")).astype(int)
        is_same_journal = np.array((journal_doc1!="") & (journal_doc2!="") | (journal_doc1==journal_doc2)).astype(int)
        is_other_journal = np.array((journal_doc1!="") & (journal_doc2!="") & (journal_doc1!=journal_doc2)).astype(int)
        features.append(unknown_journal)
        features.append(is_same_journal)
        features.append(is_other_journal)

        # degrees
        print 'vertex degrees...'
        degrees_vs = self.network_graph.indegree()
        distances = [self.dmatrix[self.hash_vs[str(idx1)], self.hash_vs[str(idx2)]] for idx1, idx2 in zip(idx_docs1, idx_docs2)]
        degrees_idx1 = [degrees_vs[self.hash_vs[str(idx1)]] for idx1 in idx_docs1]
        degrees_idx2 = [degrees_vs[self.hash_vs[str(idx2)]] for idx2 in idx_docs2]

        degrees_idx1 = np.reshape(np.array(degrees_idx1), (-1, 1))
        degrees_idx2 = np.reshape(np.array(degrees_idx2), (-1, 1))
        features.append(degrees_idx1)
        features.append(degrees_idx2)

        # shortest path 
        print 'shortest path...'
        shortest_dist_3 = [1 if d <= 3 else 0 for d in distances]
        shortest_dist_5 = [1 if d <= 5 and d > 3 else 0 for d in distances]
        shortest_dist_10 = [1 if d <= 10 and d > 5 else 0 for d in distances]
        shortest_dist_inf = [1 if d > 10 else 0 for d in distances]

        shortest_dist_3 = np.reshape(np.array(shortest_dist_3), (-1,1))
        shortest_dist_5 = np.reshape(np.array(shortest_dist_5), (-1,1))
        shortest_dist_10 = np.reshape(np.array(shortest_dist_10), (-1,1))
        shortest_dist_inf = np.reshape(np.array(shortest_dist_inf), (-1,1))

        features.append(shortest_dist_3)
        features.append(shortest_dist_5)
        features.append(shortest_dist_10)
        features.append(shortest_dist_inf)

        features_array = np.concatenate(tuple(features), axis=1)

        return features_array
