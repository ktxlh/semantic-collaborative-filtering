# To do: 把dict改成list + word_id 複雜度會小很多
import pickle
import logging
import nltk
import string
import numpy as np
import pandas as pd
from CSimilarity import Paths
from local_stopwords import words
from nltk.corpus import stopwords

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARN)
stop_words = set(stopwords.words('english') + list(string.punctuation) + words.course)


def tokenize(text):

    ''' Helper: Tokenize a sentence '''
    
    tokens = []
    for word in nltk.word_tokenize(text):
        word = word.lower()
        if word not in stop_words and not word.isnumeric():
            tokens.append(word)
    return tokens

def calculate_bow(corpus):
    """
    Calculate bag of words representations of corpus
    Parameters
    ----------
    corpus: list
        Documents represented as a list of string

    Returns
    ----------
    corpus_bow: list
        List of tuple, each tuple contains raw text and vectorized text
    vocab: list
    """
    # YOUR CODE HERE
    corpus_bow, vocab = [], []
    vocab = set(word for doc in corpus for word in tokenize(doc) )
    vocab = list(vocab)
    word_tf = dict()

    word2id = dict(zip(vocab, range( len(vocab) ) ))

    for doc in corpus:
        bow = np.zeros(len(vocab), dtype=int)
        for word in tokenize(doc):
            bow[word2id[word]] += 1
            if word in word_tf:
                word_tf[word] += 1
            else:
                word_tf[word] = 1
        corpus_bow.append((doc, bow))

    pickle.dump(word_tf, open('dataset/word_tf.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    return corpus_bow, vocab


def calculate_idf(corpus, vocab):

    def inversedocfreq(matrix, term):
        try:
            # YOUR CODE HERE
            N = len(matrix)
            n_t = 0
            for doc in matrix:
                bow = doc[1]
                if bow[term] > 0:
                    n_t += 1
            idf = N/n_t
            return idf
        except ZeroDivisionError:
            return 0
    
    word_idf = dict()
    for word_id, word in enumerate(vocab):
        word_idf[word] = inversedocfreq(corpus, word_id)   
    idfs = sorted(list(word_idf.items()), key=lambda tup: tup[1])

    return idfs
    #pickle.dump(word_idf, open('dataset/word_idf.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)


def word_idf():

    ''' One-off stop words generation '''
    
    cinfo = pd.read_csv(Paths().cinfo_path, dialect="excel")
    all_sents = list()
    for dummy, row in cinfo.iterrows():
        all_sents.append(row['Long Title'])
        all_sents.append(row['DESCRIPTION'])

    corpus_bow, vocab = calculate_bow(all_sents)
    idfs = calculate_idf(corpus_bow, vocab)

    df = pd.DataFrame()
    df['words'] = [word for (word, idf) in idfs]
    df['idf'] = [idf for (word, idf) in idfs]
    #df.to_csv("word_idf.csv", index=False)


def word_freq():

    ''' One-off stop words generation '''
    
    df = pd.read_csv("course_vecs.csv")
    corpus = [row['Long Title'] + " " + row['DESCRIPTION'] for _, row in df.iterrows()]    
    corpus_bow, vocab = calculate_bow(corpus)
    freqs = [sum(corpus_bow[j][1][i] for j in range(len(corpus_bow))) for i in range(len(vocab))]
    word2freq = sorted(list(zip(vocab, freqs)), key=lambda tup: tup[1], reverse=True)
    
    df = pd.DataFrame()
    df['words'] = [word for (word, tf) in word2freq]
    df['tf'] = [tf for (word, tf) in word2freq]
    #df.to_csv("word_tf.csv", index=False)
    

if __name__ == "__main__":
    #word_idf()
    #word_freq()
    