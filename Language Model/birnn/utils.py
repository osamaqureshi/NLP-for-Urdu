import numpy as np
import os, sys, csv, random, time
import gensim, re, string, nltk
from collections import Counter
from pickle import dump, load
import tensorflow as tf
from tensorflow import keras
from gensim.models import Word2Vec

##------------------------------ Supporting Functions ------------------------------##

def tokenize(corpus, oov_token='"انک"'):
    tokenizer = keras.preprocessing.text.Tokenizer(filters='', split=' ', oov_token=oov_token)
    tokenizer.fit_on_texts(corpus)
    tensor = tokenizer.texts_to_sequences(corpus)
    tensor = keras.preprocessing.sequence.pad_sequences(tensor,  padding='post')
    return tensor, tokenizer
  
def create_sequences(corpus, max_len=256):
    sequences = []
    for line in corpus:
        if len(line) > max_len: 
            sequences += [line[i:min(i+max_len, len(line))] for i in range(0, len(line)-max_len//2, max_len//2)]
        else:
            sequences += [line]
    return sequences

def reduce_corpus(corpus, min_len=10):
    corpus = [line for line in corpus if len(line)>=min_len]
    return corpus

def reduce_vocab(corpus, unk_token, min_count=5):
    counter = Counter([token for line in corpus for token in line])
    corpus = [[token if counter[token]>=min_count else unk_token for token in row] for row in corpus]
    return corpus

def load_embeddings(embedding_path: str, tokenizer, vocab_size: int, embedding_dim: int, unk_token: str, 
                    start_token: str, end_token: str):
    model = Word2Vec.load(embedding_path)  
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    w2v = dict([(word, v) for word,v in zip(model.wv.vocab, model.wv.vectors)])

    for index, word in tokenizer.index_word.items():
        if word in [unk_token]: 
            unk_index = index
        elif word in [start_token, end_token]: 
            embedding_matrix[index] = np.random.rand(embedding_dim)*0.1 # assign random vector of embeddings for start and end token
        else:
            embedding_matrix[index] = w2v[word]
        embedding_matrix[unk_index] = np.mean(embedding_matrix, axis=0) # assign average vector of embeddings for unk token
    return embedding_matrix