import csv
import string
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
# requires nltk 3.2.1
from nltk import pos_tag
import re

def preprocessing():
    file_names = ['node_information.csv', 'cleaned_titles.csv', 'cleaned_abstracts.csv']
    node_info = pd.read_csv(file_names[0], index_col=0, header=None)

    # titles
    print "cleaning titles..."
    titles = np.array(node_info.loc[:,2:2])
    titles_array = [','.join(clean_text_simple(t[0])) for t in titles]
    df_titles = pd.DataFrame(data=titles_array, index=node_info.index)
    print "writing ", file_names[1]
    df_titles.to_csv(file_names[1], header=False)

    # abstracts
    print "cleaning abstracts..."
    abstracts = np.array(node_info.loc[:,5:5])
    abstract_array = [','.join(clean_text_simple(a[0])) for a in abstracts]
    df_abstracts = pd.DataFrame(data=abstract_array, index=node_info.index)
    print "writing ", file_names[2]
    df_abstracts.to_csv(file_names[2], header=False)

def read_files(file_names):
    """ read data and return a list of files """
    assert type(file_names) == list
    
    file_list = []
    for file_name in file_names:
        print "Reading ", file_name
        with open(file_name, "r") as f:
            reader = csv.reader(f)
            file_list.append(list(reader))

    return file_list


def clean_text_simple(text, remove_stopwords=True, pos_filtering=True, stemming=True):

    punct = string.punctuation.replace('-', '')
    words_problem = ['aed']
    
    # remove formatting
    text =  re.sub('\s+', ' ', text)
    # convert to lower case
    text = text.lower()
    # remove punctuation (preserving intra-word dashes)
    text = ''.join(l for l in text if l not in punct)
    # strip extra white space
    text = re.sub(' +',' ',text)
    # strip leading and trailing white space
    text = text.strip()
    # tokenize (split based on whitespace)
    tokens = text.split(' ')
    if pos_filtering == True:
        # apply POS-tagging
        tagged_tokens = pos_tag(tokens)
        # retain only nouns and adjectives
        tokens_keep = []
        for i in range(len(tagged_tokens)):
            item = tagged_tokens[i]
            if (
            item[1] == 'NN' or
            item[1] == 'NNS' or
            item[1] == 'NNP' or
            item[1] == 'NNPS' or
            item[1] == 'JJ' or
            item[1] == 'JJS' or
            item[1] == 'JJR'
            ):
                tokens_keep.append(item[0])
        tokens = tokens_keep
    if remove_stopwords:
        stpwds = stopwords.words('english')
        # remove stopwords
        tokens = [token for token in tokens if token not in stpwds]
    if stemming:
        stemmer = nltk.stem.PorterStemmer()
        # apply Porter's stemmer
        tokens_stemmed = list()
        for token in tokens:
            #print token
            if token not in words_problem:
                tokens_stemmed.append(stemmer.stem(token))
        #tokens = list(set(tokens_stemmed))

    return(tokens)

def find_keywords(text):
    return text
