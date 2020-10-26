import numpy as np
import tensorflow as tf

class Bidirectional(tf.keras.Model):
    def __init__(self, units: int, 
                 projection_units: int):
        super(Bidirectional, self).__init__()
        self.units = units
        self.projection_units = projection_units
        self.Layers = [tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.units, 
                                                                         return_sequences=True, 
                                                                         return_state=True, 
                                                                         recurrent_initializer='glorot_uniform', 
                                                                         name='birnn')),
                       tf.keras.layers.Dense(self.projection_units, name='projection')]

    def call(self, inp):
        out, _, _ = self.Layers[0](inp)
        out = self.Layers[1](out)
        return out

class BiRNN(tf.keras.Model):
    def __init__(self, units: int,projection_units: int,max_seq_length: int,
                 vocab_size: int,embedding_dim: int,embedding_matrix = None):
        super(BiRNN, self).__init__()
        self.units = units
        self.projection_units=projection_units
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embeddings = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim,
                                                    weights = [embedding_matrix], 
                                                    trainable=False, name='embeddings')
        self.Layers = [Bidirectional(units=self.units, projection_units=self.projection_units),
                       tf.keras.layers.Add(),
                       Bidirectional(units=self.units, projection_units=self.projection_units),
                       tf.keras.layers.Dense(self.vocab_size, activation='softmax', name='softmax')]

    def call(self, inp, predict=False):
        inp = self.embeddings(inp)
        out1 = self.Layers[0](inp)
        out2 = self.Layers[1]([inp, out1])
        out3 = self.Layers[2](out2)
        if predict is False:
            return out3
        else:
            out4 = self.Layers[3](out3)
            return out4

def loss_function(real, pred, loss_object):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

def mask_sequences(seq, t):
    mask = np.zeros(seq.shape)
    mask[:,:t] = 1
    inp = tf.math.multiply(seq, mask)
    mask[:,:t+1] = 1
    tar = tf.math.multiply(seq, mask)

    return inp, tar