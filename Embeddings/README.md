Word Embeddings have experienced a tremendous improvements over the last decade, so much so that Pre-trianed Word Embeddings have become an ubiquitous component of NLP systems. These Pre-trained embeddings trained on a large corpus of text, using one of a number of different algorithms, such as [Word2Vec](https://arxiv.org/pdf/1301.3781.pdf), [GloVe](https://nlp.stanford.edu/pubs/glove.pdf), [FastText](https://arxiv.org/pdf/1607.04606.pdf) etc. 

For most languages, such Pre-trained embeddings are readily available, but as with other low-resource languages, Urdu has very limited resources for researchers and practioners looking to download and use Pre-trained embeddings. This repository makes available Pre-trained Word2Vec and FastText word embeddings for various dimensions, in the form of trained Gensim models, and json files. 

Word2Vec noteboook also provides a brief guide to building Word2Vec with SkipGram embeddings.

### Word2Vec

| Dimension | Gensim | Json |
| --- | --- | --- |
| 50 | [model](https://drive.google.com/file/d/1J8UMvIGCXoj5Je5EcFTDNx-e_KOTiAQb/view?usp=sharing) | [json](https://drive.google.com/file/d/1-ByfTSY3WIuE_q40KI2Q2YFKKe6zJGCT/view?usp=sharing) |
| 64 | [model](https://drive.google.com/file/d/1V-Afwe_oF7YNknAGbKOHFUfycZetMlpw/view?usp=sharing) | [json](https://drive.google.com/file/d/1-1tX6eh687DD5Rzgr-RT5JE8g9G-scDf/view?usp=sharing) |
| 100 | [model](https://drive.google.com/file/d/1GmC2vEbe776enURLCh9KiM3tODDEeyPj/view?usp=sharing) | [json](https://drive.google.com/file/d/1DqwzBhrp75CTAHbiXrvBcyoyZdcX-zUs/view?usp=sharing) |
| 256 | [model](https://drive.google.com/file/d/1IddLk7oCQYaabSGH46QAVYFFeITJOwWi/view?usp=sharing) | [json](https://drive.google.com/file/d/1-0kWn-yrqusruQEGNRDKU6hBsWZrEMd-/view?usp=sharing) |
| 300 | [model](https://drive.google.com/file/d/1-7NDl0BJ__6rE8spdXZJLizpYCTgsgct/view?usp=sharing) | [json](https://drive.google.com/file/d/1-7imJacHeZVGD-eenZI2Ks29fW2vkgwv/view?usp=sharing) |

```python
from gensim.models import Word2Vec
EMBEDDING_PATH = '../word2vec256.bin'
model =. Word2Vec.load(EMBEDDING_PATH)  
```

### FastText

| Dimension | Gensim | Json |
| --- | --- | --- |
| 50 | [model](https://drive.google.com/file/d/1-4W7oxwDJpShOi4z_cS5uMgjL6V_dRpE/view?usp=sharing) | [json](https://drive.google.com/file/d/1-W9VgwPmdTp1DpUc4aC8UOGUrugUks3F/view?usp=sharing) |
| 100 | [model](https://drive.google.com/file/d/1-5TJvHZI8jN3z775ts-tYaaYcNoB1IwL/view?usp=sharing) | [json](https://drive.google.com/file/d/1-RmucdUs9ktGbJC_y-nsVUuYQ6rMpg1o/view?usp=sharing) |
| 256 | [model](https://drive.google.com/file/d/1-EPLBaKUlYSHKnGJ1hrCxO-7db56DdnZ/view?usp=sharing) | [json](https://drive.google.com/file/d/1-LkPYV0hG-j6x26bmXB9bD1aARmmZGN6/view?usp=sharing) |
| 300 | [model](https://drive.google.com/file/d/1-BuWN8C0baXHiPGp2M7T8Qo0o_i_cOsM/view?usp=sharing) | [json](https://drive.google.com/file/d/1-XrQewx2wOmliahyY4OzNkHtKC21xzUS/view?usp=sharing) |

```python
from gensim.models import FastText
EMBEDDING_PATH = '../fasttext256.bin'
model =. FastText.load(EMBEDDING_PATH)  
```
