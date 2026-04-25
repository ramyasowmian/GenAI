
# ===============================================================================
# Word2Vec(..., sg=0)  # CBOW #by default
# Word2Vec(..., sg=1)  # Skip-gram
# ===============================================================================

# # custom_model =gensim.models.Word2Vec(
#     window=10,
#     min_count=5,
#     vector_size=150
# )
## these all the value are the hyperparameter


import os
import re
import numpy as np
import pandas as pd
import nltk
import warnings
import gensim

from nltk import sent_tokenize
from gensim.utils import simple_preprocess

warnings.filterwarnings('ignore')
nltk.download('all')

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "..", "Resource", "Reviews.csv")

df = pd.read_csv(csv_path, nrows=150)

raw_sentences = df["Text"].dropna().astype(str).tolist()
tokenize_sentences = [sent_tokenize(sentence) for sentence in raw_sentences]
# Flatten the list of sentences just for understanding of flattern with tokenize_sentences
sentences = [[sentence] for sentence in raw_sentences]

all_words = []
for sentence in raw_sentences:
    all_words.append(simple_preprocess(sentence))

# ----------- Word2Vec Model Training (CBOW) #sg=0 - by default -------------
custom_model_CBOW = gensim.models.Word2Vec(
    sentences=raw_sentences,
    vector_size=150,
    window=10,
    min_count=5,  
)
custom_model_CBOW.build_vocab(all_words, update=True) 
curpus_count_CBOW = custom_model_CBOW.corpus_count
custom_model_CBOW.train(raw_sentences, total_examples=custom_model_CBOW.corpus_count, epochs=10)

wv = custom_model_CBOW.wv["good"]
most_similar = custom_model_CBOW.wv.most_similar("good")
similarity = custom_model_CBOW.wv.similarity("good", "bad")

print("Word Vector for 'good':", wv)
print("most_similar vector for 'good':", most_similar)
print("similarity Vector for 'good':", similarity)


# ----------- Word2Vec Model Training (Skip-gra) # sg=1-------------
custom_model_SP = gensim.models.Word2Vec(
    sentences=raw_sentences,
    vector_size=150,
    window=10,
    min_count=5,  
    sg=1
)
custom_model_SP.build_vocab(all_words, update=True) 
curpus_count_SP = custom_model_SP.corpus_count
custom_model_SP.train(raw_sentences, total_examples=custom_model_SP.corpus_count, epochs=10)

wv = custom_model_SP.wv["good"]
most_similar = custom_model_SP.wv.most_similar("good")
similarity = custom_model_SP.wv.similarity("good", "bad")

print("Word Vector for 'good':", wv)
print("most_similar vector for 'good':", most_similar)
print("similarity Vector for 'good':", similarity)
