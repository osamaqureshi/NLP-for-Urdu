import numpy as np
import tensorflow as tf

def point_wise_feed_forward_network(embedding_dim, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(embedding_dim)  # (batch_size, seq_len, d_model)
  ])

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, embedding_dim):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    assert d_model % self.num_heads == 0
    self.depth = d_model // self.num_heads
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)
    self.dense = tf.keras.layers.Dense(embedding_dim)

  def split_heads(self, x, batch_size):
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q):
    batch_size = tf.shape(q)[0]
    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v)
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
    concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
    return output, attention_weights

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, embedding_dim, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()
    self.mha = MultiHeadAttention(d_model, num_heads, embedding_dim)
    self.ffn = point_wise_feed_forward_network(embedding_dim, dff)
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, x, training):
    print()
    attn_output, _ = self.mha(x, x, x)  # (batch_size, seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, seq_len, d_model)
    ffn_output = self.ffn(out1)  # (batch_size, seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, seq_len, d_model)
    return out2

class ALBERT(tf.keras.Model):
  def __init__(self, num_layers: int, d_model: int, embedding_dim:int, num_heads: int, dff: int, vocab_size: int, mask_token: int, rate):
    super(ALBERT, self).__init__()
    self.num_layers = num_layers
    self.d_model = d_model
    self.embedding_dim = embedding_dim
    self.num_heads = num_heads
    self.dff = dff
    self.vocab_size = vocab_size
    self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)
    self.pos_encoding = positional_encoding(self.vocab_size, self.embedding_dim)
    self.encoder = EncoderLayer(d_model, embedding_dim, num_heads, dff, rate)
    self.dropout = tf.keras.layers.Dropout(rate)
    self.dense = tf.keras.layers.Dense(self.vocab_size)

  def call(self, inp, training=False): 
    seq_len = tf.shape(inp)[1]
    inp = mask_sequence(inp, mask_token)
    seq = self.embedding(inp)  # (batch_size, seq_len, d_model)
    seq *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    seq += self.pos_encoding[:, :seq_len, :]
    seq = self.dropout(seq, training=training)
    for i in range(self.num_layers):
      seq = self.encoder(seq, training)
    
    if training:
      seq = self.dense(seq)  # (batch_size, seq_len, vocab_size)

    return seq  # (batch_size, seq_len, d_model) 

def scaled_dot_product_attention(q, k, v):
  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
  return output, attention_weights
  
def get_angles(pos, embedding_dim):
  pos = np.arange(pos)[:, np.newaxis]
  angles = 1/np.power(10000, 2*np.arange(embedding_dim)/embedding_dim)[np.newaxis, :]
  return pos * angles

def positional_encoding(vocab_size, embedding_dim):
  angles = get_angles(vocab_size, embedding_dim)
  angles[:,::2] = np.sin(angles[:,::2])
  angles[:,1::2] = np.cos(angles[:,::2])
  pos_encoding = angles[np.newaxis, ...]
    
  return tf.cast(pos_encoding, dtype=tf.float32)

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)
  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  return tf.reduce_mean(loss_)

def mask_sequence(seq, mask_token):
  mask = np.random.choice([True, False], size=seq.shape, p=[.85, .15])
  seq = tf.where(mask, seq, tf.constant(mask_token, dtype=seq.dtype, shape=seq.shape))
  return seq

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)
    self.warmup_steps = warmup_steps
    
  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)