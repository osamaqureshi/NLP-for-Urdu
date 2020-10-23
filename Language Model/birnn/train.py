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

from utils import tokenize
from utils import create_sequences
from utils import reduce_corpus
from utils import reduce_vocab
from utils import load_embeddings
from model import BiRNN
from model import loss_function
from model import mask_sequences

@tf.function
def train_step(batch, loss_object):
    loss = 0

    with tf.GradientTape() as tape:
        for t in range(1, batch.shape[0]):
            inp, tar = mask_sequences(batch, t=t)
            pred = birnn(inp, predict=True)
            loss += loss_function(tar, pred, loss_object)

    batch_loss = (loss / int(batch.shape[0]))
    variables = birnn.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    train_loss(batch_loss)
    return batch_loss

##------------------------------ Define constants and hyperparameters ------------------------------##


UNK_TOKEN = '"انک"'
START_TOKEN = '"س"'
END_TOKEN = '"ش"'
EPOCHS = 10
MIN_SEQ_LEN = 10 # min/max length of input sequences
MAX_SEQ_LEN = 128
MIN_COUNT = 10 # minimum count for vocab words
TEST_SIZE = 0.15
RANDOM_STATE = 42
BATCH_SIZE = 32
UNITS = 256
EMBEDDING_DIM = 256
PROJECTION_UNITS = 256
BUFFER_SIZE = 10000
CORPUS_PATH = 'data/corpus.txt'
EMBEDDING_PATH = 'data/word2vec256.bin'
CHECKPOINT_PATH = 'checkpoints'
LOG_DIR = 'logs'

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

tensor, lang = tokenize(corpus, oov_token=UNK_TOKEN)    # tokenize corpus and prepare padded Tensor sequences 
train, test = train_test_split(tensor, test_size=TEST_SIZE, random_state=RANDOM_STATE)  # split dataset into train and test 
del tensor, corpus

steps_per_epoch = len(train)//BATCH_SIZE
vocab_size = max(lang.word_index.values())+1
print('\ntrain shape: ', train.shape, '\tbatches: ', steps_per_epoch)

train = tf.data.Dataset.from_tensor_slices(train).shuffle(BUFFER_SIZE)
train = train.batch(BATCH_SIZE, drop_remainder=True)

# Load embeddings matrix
embedding_matrix = load_embeddings(
    embedding_path=EMBEDDING_PATH, 
    tokenizer=lang,
    vocab_size=vocab_size, 
    embedding_dim=EMBEDDING_DIM, 
    unk_token=UNK_TOKEN, 
    start_token=START_TOKEN, 
    end_token=END_TOKEN)

##------------------------------ Setup training  ------------------------------##

birnn = BiRNN(UNITS, PROJECTION_UNITS, MAX_SEQ_LEN, vocab_size, EMBEDDING_DIM, embedding_matrix)
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)

ckpt = tf.train.Checkpoint(birnn=birnn)
ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=2)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored')
  
##------------------------------ Training  ------------------------------##

print('\ntraining')
for epoch in range(EPOCHS):
    start = time.time()
    total_loss = 0

    for i, batch in enumerate(train.take(steps_per_epoch)):
        print('batch: ', i)
        batch_loss = train_step(batch, loss_object)
        total_loss += batch_loss 
        if i % 10 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, i, batch_loss.numpy()))
    
        if i % 100 == 0:
            print('Epoch {} Batch {} Saving weights'.format(epoch + 1, i))
            ckpt_save_path = ckpt_manager.save()
            
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
    
            
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,ckpt_save_path))
    print('Epoch {} Loss {:.4f}'.format(epoch + 1, train_loss.result()))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
