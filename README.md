# NLP-for-Urdu
A resource for NLP research with Urdu.

## Dataset
Urdu is very low resource language for NLP, as there are very limited resources of significant, high quality text data. I have attempted to collect a diverse, high quality corpus of text for training embeddings and language models. This includes compiling a new corpus of Urdu short stories, extracting data from Wikipedia Urdu (https://dumps.wikimedia.org/urwiki), and collecting other available sources of Urdu text prepared and provided by other researchers, such as Makhzan (https://github.com/zeerakahmed/makhzan). The entire corpus was cleaned and processed to ensure data quality. I intend on continuing to expanding this corpus, and plan on extracting text from the Urdu portion of CommonCrawl next.

##Emebddings
Word Embeddings have become an ubiquitous component of NLP systems. This repostory releases pre-trained Word2Vec and FastText embeddings for multiple vector dimensions. These embeddings are provided in the form of trained Gensim models, and json files.

In addition pre-trained embeddings, the repository also provides a notebook with a Skip-Gram Word2Vec with Negative sampling implementation for those interested in learning how to implement Word2Vec themselves.

## Language Models
Langauge models have become an essential component of NLU research. However, training state-of-the-art language models requires a large degree of subject matter expertise, along with significant computational resources. Becuase of this, Powerful Pre-trained language models, such as for BERT and GPT, have become an incredible resources for researchers and practioners working on a broad range of NLU problems. There are several resources for downloading pre-trained language models for English and a number of other langauages (such as https://huggingface.co/transformers), but there is no current resource that includes pre-trained language models for Urdu. This repository aims to develop and train various models architectures, including ELMo, BERT, GPT etc, and make them available as an open resource.
