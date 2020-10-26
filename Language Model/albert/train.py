import numpy as np
import pandas as pd
import os, sys, csv, random, time
import gensim, re, string, nltk
from collections import Counter
from pickle import dump, load
import tensorflow as tf
from tensorflow import keras
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from utils import tokenize_subword
from utils import create_sequences
from utils import reduce_corpus
from utils import reduce_vocab
from model import ALBERT
from model import loss_function
from model import CustomSchedule

@tf.function
def train_step(batch):
    loss = 0
    with tf.GradientTape() as tape:
        pred = albert(batch, training=True)
        loss += loss_function(batch, pred)

    batch_loss = (loss / int(batch.shape[0]))
    variables = albert.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    train_loss(batch_loss)
    return batch_loss

@tf.function
def test_step(batch):
    loss = 0

    with tf.GradientTape() as tape:
        pred = albert(batch, training=True)
        loss += loss_function(batch, pred)
    batch_loss = (loss / int(batch.shape[0]))
    test_loss(batch_loss)
    return batch_loss

##------------------------------ Define constants and hyperparameters ------------------------------##

MASK_TOKEN = '"م"'
START_TOKEN = '"س"'
END_TOKEN = '"ش"'
EPOCHS = 20
MIN_SEQ_LEN = 20 # min/max length of input sequences
MAX_SEQ_LEN = 64
MIN_COUNT = 10 # minimum count for vocab words
TEST_SIZE = 0.15
RANDOM_STATE = 42
BATCH_SIZE = 64
D_MODEL = 512
EMBEDDING_DIM = 128
DFF = 512
NUM_LAYERS = 4
NUM_HEADS = 8
DROPOUT_RATE = 0.1
BUFFER_SIZE = 10000
CORPUS_PATH = 'data/corpus.txt'
EMBEDDING_PATH = 'data/word2vec256.bin'
CHECKPOINT_PATH = 'Checkpoints'
LOG_DIR = 'Logs'

##------------------------------ Preprocess ------------------------------##

corpus = open(CORPUS_PATH).read().split('\n')   # read corpus
corpus = [[token for token in line.split(' ') if token != ''] for line in corpus] # tokenize corpus into words
print('Pre-processing')
print('vocab size: ', len(set([token for line in corpus for token in line])))
print('num. of lines: ', len(corpus))
print(' '.join(corpus[0]))
corpus = reduce_corpus(corpus, min_len=MIN_SEQ_LEN) # reduce corpus size - remove lines with length less than MIN_SEQ_LEN
corpus = [[START_TOKEN]+line+[END_TOKEN] for line in corpus] # add start and end tokens
corpus = reduce_vocab(corpus, UNK_TOKEN, min_count=MIN_COUNT) # reduce vocab size - remove token with count less than MIN_COUNT
vocab = list(set([token for line in corpus for token in line])) # extract vocabulary of corpus
corpus = create_sequences(corpus, max_len=MAX_SEQ_LEN) # create sequences of max length MAX_SEQ_LEN
print('\nPost-processing')
print('vocab size: ', len(vocab))
print('num. of lines: ', len(corpus))
print(' '.join(corpus[0]))

tensor, lang = tokenize_subword(corpus, reserved_tokens=[MASK_TOKEN,START_TOKEN,END_TOKEN])    # tokenize corpus and prepare padded Tensor sequences 
train, test = train_test_split(tensor, test_size=TEST_SIZE, random_state=RANDOM_STATE)  # split dataset into train and test 
train = tf.data.Dataset.from_tensor_slices(train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
test = tf.data.Dataset.from_tensor_slices(test).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
del tensor, corpus

vocab_size = lang.vocab_size+1
mask_token = lang.encode(MASK_TOKEN)[0]

##------------------------------ Setup training  ------------------------------##

albert = ALBERT(NUM_LAYERS, D_MODEL, EMBEDDING_DIM, NUM_HEADS, DFF, vocab_size, mask_token, DROPOUT_RATE)

learning_rate = CustomSchedule(D_MODEL)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

train_summary_writer = tf.summary.create_file_writer(LOG_DIR+'/train')
test_summary_writer = tf.summary.create_file_writer(LOG_DIR+'/test')

ckpt = tf.train.Checkpoint(albert=albert, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=2)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored')
  
##------------------------------ Training  ------------------------------##

for epoch in range(EPOCHS):
    start = time.time()
    total_loss = 0

    for i, batch in enumerate(train):
        batch_loss = train_step(batch)
        total_loss += batch_loss
        if i % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, i, train_loss.result()))
            
    with train_summary_writer.as_default():
        tf.summary.scalar('train loss', train_loss.result(), step=epoch)
        tf.summary.scalar('train accuracy', train_accuracy.result(), step=epoch)
    
    total_loss = 0
    for i, batch in enumerate(test):
        total_loss += test_step(batch)
        if i % 100 == 0:
            print('Epoch {} Batch {} Test Loss {:.4f}'.format(epoch + 1, i, train_loss.result()))

    with test_summary_writer.as_default():
        tf.summary.scalar('test loss', test_loss.result(), step=epoch)
        tf.summary.scalar('test accuracy', test_accuracy.result(), step=epoch)
    
    print ('Saving checkpoint for epoch {}'.format(epoch+1))
    ckpt_manager.save()
    print('Epoch {} Train Loss {:.4f}'.format(epoch + 1, train_loss.result()))
    print('Epoch {} Test Loss {:.4f}'.format(epoch + 1, test_loss.result()))
    print('Time taken for epoch: {} sec\n'.format(time.time() - start))