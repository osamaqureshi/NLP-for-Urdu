Pre-trained Bi-Directional RNN (BiRNN) Language Model (LM), can be used to extract contextual embeddings for Urdu text sequences. 

### Model Architecture 
The BiRNN LM architecture is inspired by the work of [ELMo](https://arxiv.org/pdf/1802.05365.pdf), [TagLM](https://arxiv.org/pdf/1705.00108.pdf) and other RNN based Language Models introduced by recent research. Instead of character convolutions, input sequences are tokenized into words, and their embeddings are initialized with pre-trained Word2Vec embeddings with 256 dimensions. The model used L=2 BiGRU layers with 256 units, 256 dimension projections, and a residual connection from the first to second layer. The model also has a final dense layer with softmax, used during training to optimize the network on Language Modeling task, but this layer is not used when extracting contextual embeddings. (Total parameters = 1.8M)

### Training BiRNN

The vocabulary is limited to tokens with a minimum count of 10, resulting in a vocabulary size of around 13000. Input sequences dimensionality is 128, and each sequence is appended with a start and end token. The model is trained with SGD, using Adam optimizer, with a batch size of 32.
