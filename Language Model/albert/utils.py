import numpy as np
import os, sys, csv, random, time
import gensim, re, string, nltk
from collections import Counter
from pickle import dump, load
import tensorflow as tf
from tensorflow import keras

##------------------------------ Supporting Functions ------------------------------##

def tokenize_subword(corpus, reserved_tokens:list):
  corpus = [' '.join(line) for line in corpus]
  tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(corpus, target_vocab_size=2**13, reserved_tokens=reserved_tokens)
  tensor = keras.preprocessing.sequence.pad_sequences([tokenizer.encode(line) for line in corpus],  padding='post')
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