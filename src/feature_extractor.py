# coding=utf-8
import numpy as np
import nltk
import pandas as pd
import gensim
import igraph
from library import read_files, clean_text_simple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import scale
import networkx as nx
import itertools
from networkx.algorithms.connectivity import local_edge_connectivity
from networkx.algorithms.connectivity import(
            build_auxiliary_edge_connectivity)
from networkx.algorithms.flow import build_residual_network
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

    def __intersect_array(self, a, b, union=False):
        array_a = []
        for row in a:
            if len(row)==0 or row[0]!=row[0] :
                array_a.append([])
            else:
                array_a.append([s.strip() for s in row[0].split(',')])
        # doc2
        array_b = []
        for row in b:
            if len(row)==0 or row[0]!=row[0] :
                array_b.append([])
            else:
                array_b.append([s.strip() for s in row[0].split(',')])

        # check same size
        assert len(array_a)==len(array_b)
        intersec = [len(set(x).intersection(set(y))) for x, y in zip(array_a, array_b)]
        if(not union):
            return intersec
        else:
            uni = [len(x)+len(y)-inter for x,y,inter in zip(array_a, array_b, intersec)]
            return intersec, uni

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
        # normalizing
        # dot_products = scale(dot_products)
        #print "gerou"
        return np.array(dot_products)            


    def fit(self, X, y):
        ## build graph of the citation network
        ids1 = X[:,0]
        ids2 = X[:,1]
        vertices = list(set(ids1.astype(str)).union(ids2.astype(str)))
        edges = [tuple([str(row[0]), str(row[1])]) for row, link in zip(X,y) if link == 1]

        self.graph = igraph.Graph() 
        self.graph.add_vertices(vertices)
        self.graph.add_edges(edges)

        self.di_network_graph = nx.DiGraph() 
        self.di_network_graph.add_nodes_from(vertices)
        self.di_network_graph.add_edges_from(edges)

        self.un_network_graph = nx.Graph() 
        self.un_network_graph.add_nodes_from(vertices)
        self.un_network_graph.add_edges_from(edges)

        vs = zip(vertices, range(len(vertices)))
        self.hash_vs = {a:b for a,b in vs}
        #print 'calculating shortest paths...'
        #self.dmatrix = np.array(self.graph.shortest_paths())

        #uncomment
        # WARNING: cutoff should not be set in our final submission, which is equivalent to set it to infinity
        print 'calculating betweenness centrality'
        self.di_b_centrality = self.graph.betweenness(directed=True)
        self.un_b_centrality = self.graph.betweenness(directed=False)

        print 'calculating local edge connectivity'
        H = build_auxiliary_edge_connectivity(self.di_network_graph)
        R = build_residual_network(H, 'capacity')
        self.di_connectivity = dict.fromkeys(self.di_network_graph, dict())
        for u, v in itertools.combinations(self.di_network_graph, 2):
            k = local_edge_connectivity(self.di_network_graph, u, v, auxiliary=H, residual=R)
            self.di_connectivity[u][v] = k
        H = build_auxiliary_edge_connectivity(self.un_network_graph)
        R = build_residual_network(H, 'capacity')
        self.un_connectivity = dict.fromkeys(self.un_network_graph, dict())
        for u, v in itertools.combinations(self.un_network_graph, 2):
            k = local_edge_connectivity(self.un_network_graph, u, v, auxiliary=H, residual=R)
            self.un_connectivity[u][v] = k


    def transform(self, X):

        print 'Transform...'

        idx_docs1 = X[:,0]
        idx_docs2 = X[:,1]
        features = []

        # difference in number of inlinks
        print "difference in number of inlinks + number of times 'to' is cited"
        degrees_vs = self.graph.indegree()
        degrees_idx1 = [degrees_vs[self.hash_vs[str(idx1)]] for idx1 in idx_docs1]
        degrees_idx2 = [degrees_vs[self.hash_vs[str(idx2)]] for idx2 in idx_docs2]
        diff_degrees = [a-b for a,b in zip(degrees_idx1,degrees_idx2)]
        # degrees_idx2 = scale(degrees_idx2)
        # diff_degrees = scale(diff_degrees)
        degrees_idx2 = np.reshape(np.array(degrees_idx2), (-1,1))
        diff_degrees = np.reshape(np.array(diff_degrees), (-1,1))
        #degrees_idx1 = np.reshape(np.array(degrees_idx1), (-1, 1))
        #degrees_idx2 = np.reshape(np.array(degrees_idx2), (-1, 1))
        #print degrees_idx1
        #features.append(degrees_idx1)

        features.append(degrees_idx2)
        features.append(diff_degrees)

        # whether papers were classified in the same cluster or not
        # TO IMPLEMENT YET


        # uncomment
        
        print 'difference in betweeness centrality'
        
        di_b_centrality_idx1 = np.array([self.di_b_centrality[self.hash_vs[str(nodeid)]] for nodeid in idx_docs1])
        di_b_centrality_idx2 = np.array([self.di_b_centrality[self.hash_vs[str(nodeid)]] for nodeid in idx_docs2])
        diff_di_b_centrality = di_b_centrality_idx2 - di_b_centrality_idx1
        # diff_di_b_centrality = scale(di_b_centrality)
        diff_di_b_centrality = np.reshape(np.array(diff_di_b_centrality), (-1, 1))
        
        un_b_centrality_idx1 = np.array([self.un_b_centrality[self.hash_vs[str(nodeid)]] for nodeid in idx_docs1])
        un_b_centrality_idx2 = np.array([self.un_b_centrality[self.hash_vs[str(nodeid)]] for nodeid in idx_docs2])
        diff_un_b_centrality = un_b_centrality_idx2 - un_b_centrality_idx1
        # diff_un_b_centrality = scale(diff_un_b_centrality)
        diff_un_b_centrality = np.reshape(np.array(diff_un_b_centrality), (-1, 1))

        features.append(diff_di_b_centrality)
        features.append(diff_un_b_centrality)

        # edge connectivity
        di_connectivities = [self.di_connectivity[self.hash_vs[str(idx1)], self.di_connectivity[str(idx2)]] for idx1, idx2 in zip(idx_docs1, idx_docs2)]
        un_connectivities = [self.un_connectivity[self.hash_vs[str(idx1)], self.un_connectivity[str(idx2)]] for idx1, idx2 in zip(idx_docs1, idx_docs2)]
        di_connectivities = np.reshape(np.array(di_connectivities), (-1,1))
        un_connectivities = np.reshape(np.array(un_connectivities), (-1,1))
        features.append(di_connectivities)
        features.append(un_connectivities)
        


        # shortest path 
        # uncomment
        '''        
        print 'shortest path...'
        distances = [self.dmatrix[self.hash_vs[str(idx1)], self.hash_vs[str(idx2)]] for idx1, idx2 in zip(idx_docs1, idx_docs2)]

        shortest_dist_3 = [1 if d <= 3 else 0 for d in distances]
        shortest_dist_5 = [1 if d <= 5 and d > 3 else 0 for d in distances]
        shortest_dist_10 = [1 if d <= 10 and d > 5 else 0 for d in distances]
        shortest_dist_inf = [1 if d > 10 else 0 for d in distances]

        shortest_dist_3 = np.reshape(np.array(shortest_dist_3), (-1,1))
        shortest_dist_5 = np.reshape(np.array(shortest_dist_5), (-1,1))
        shortest_dist_10 = np.reshape(np.array(shortest_dist_10), (-1,1))
        shortest_dist_inf = np.reshape(np.array(shortest_dist_inf), (-1,1))
        
        distances = scale(distances)
        distances = np.reshape(np.array(distances), (-1,1))
        features.append(distances)
        features.append(shortest_dist_3)
        features.append(shortest_dist_5)
        features.append(shortest_dist_10)
        features.append(shortest_dist_inf)'''

        print "we study neighbors"
        set_vs = set(self.graph.vs['name'])
        is_in_graph1 = [(str(v) in set_vs) for v in idx_docs1]
        is_in_graph2 = [(str(v) in set_vs) for v in idx_docs2]
        neighbors1 = []
        for node,is_in in zip(idx_docs1, is_in_graph1):
            if is_in:
                neighbors1.append(",".join([str(v) for v in self.graph.adjacent(str(node), mode='ALL')])) # both in and out neighbors
            else:
                neighbors1.append("")
        neighbors2 = []
        for node,is_in in zip(idx_docs2, is_in_graph2):
            if is_in:
                neighbors2.append(",".join([str(v) for v in self.graph.adjacent(str(node), mode='ALL')])) # both in and out neighbors
            else:
                neighbors2.append("")
        #neighbors1 = [",".join([str(v) for v in self.graph.adjacent(int(node))]) if is_in else "" for node,is_in in zip(idx_docs1,is_in_graph1)]        
        #neighbors2 = [",".join([str(v) for v in self.graph.adjacent(int(node))]) if is_in else "" for node,is_in in zip(idx_docs2,is_in_graph2)]        
        intersec, uni = self.__intersect_array(neighbors1,neighbors2,union=True)
        jaccard_coefficient = [float(a)/float(b) if b!=0 else 0 for a,b in zip(intersec,uni)]
        #intersec = scale(intersec)
        #jaccard_coefficient = scale(jaccard_coefficient)
        intersec = np.reshape(np.array(intersec), (-1,1))
        jaccard_coefficient = np.reshape(np.array(jaccard_coefficient), (-1,1))
        features.append(intersec)
        features.append(jaccard_coefficient)

        # difference in number of inlinks


        # cosine similarity of titles text
        # uncomment
        
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
        #features.append(year_docs1 - year_docs2)
        diff_year = year_docs1 - year_docs2
        # diff_year = scale(diff_year)
        features.append(diff_year)
        # titles overlap
        print 'titles overlap...'
        titles_docs1 = np.array(cleaned_titles.loc[idx_docs1])
        titles_docs2 = np.array(cleaned_titles.loc[idx_docs2])
        intersec = self.__intersect_array(titles_docs1, titles_docs2)
        # intersec = scale(intersec)
        intersec = np.reshape(np.array(intersec), (-1,1))
        features.append(intersec)

        # abstracts overlap
        print 'abstracts overlap...'
        abstract_doc1 = np.array(cleaned_abstracts.loc[idx_docs1])
        abstract_doc2 = np.array(cleaned_abstracts.loc[idx_docs2])
        intersec = self.__intersect_array(abstract_doc1, abstract_doc2)
        # intersec = scale(intersec)
        intersec = np.reshape(np.array(intersec), (-1,1))
        features.append(intersec)

        # titles and abstract overlap
        print 'title and abstract overlap...'
        intersec1 = np.array(self.__intersect_array(titles_docs1, abstract_doc2))
        intersec2 = np.array(self.__intersect_array(abstract_doc1, titles_docs2))
        intersec = np.reshape(intersec1 + intersec2, (-1,1))
        # intersec = scale(intersec)
        features.append(intersec)

        # authors overlap
        print 'authors overlap and self-citations'
        authors_doc1 = np.array(node_info.loc[idx_docs1,3:3])
        authors_doc2 = np.array(node_info.loc[idx_docs2,3:3])
        intersec = self.__intersect_array(authors_doc1, authors_doc2)
        self_citation = [1 if x>0 else 0 for x in intersec]
        # intersec = scale(intersec)
        intersec = np.reshape(np.array(intersec), (-1,1))
        self_citation = np.reshape(np.array(self_citation), (-1,1))
        features.append(intersec)
        features.append(self_citation)

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

        features_array = np.concatenate(tuple(features), axis=1)

        return features_array
