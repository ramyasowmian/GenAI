# ===================================================================================================================
# One-Hot Encoding Implementation for NLP

# One-hot encoding represents each word in the vocabulary as a binary vector
# where only one element is 1 (hot) and all others are 0 (cold).
# ===================================================================================================================

import numpy as np
import pandas as pd
import re

from prompt_toolkit import document

import warnings 
warnings.filterwarnings('ignore')

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Download the necessary NLTK resources (run once)
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger")
nltk.download('averaged_perceptron_tagger_eng')  


# Sample text data for demonstration
sentences = [
    "I love machine learning",
    "Natural language processing is fascinating",
    "Deep learning models are powerful",
    "Text encoding is important for NLP",
    "One hot encoding creates sparse vectors"
]

print("sentences:")
for i, sentence in enumerate(sentences, 1):
    print(f"{i}. {sentence}")
print()

# Step 2: Initialize stop words and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()    

# ---------------------------- Text Clean up ----------------------
# Initialize stop words and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer() 

# ----------------- WordNet POS tag for lemmatization -------------
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun if no specific POS tag is found  
    
def clean_tokens(text):
    # Lowercase the text
    text = text.lower()

    # Remove punctuation and numbers using regex
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)      # Remove numbers

    # ----------------------------- (Token) -------------------------
    # Tokenize the text (Converts sentences into list of words)
    tokens = word_tokenize(text) 
    
    # Get POS tags for the tokens
    pos_tags = pos_tag(tokens)  

    # ------------------- Stopwords, Lemmatization -------------------
    # Filter out stop words and non-alphabetic tokens and apply lemmatization based on POS tags              
    filtered_tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos))
        for word, pos in zip(tokens, [tag for _, tag in pos_tags])
        if word.isalpha() and word not in stop_words
    ]  
    return filtered_tokens  

# Flatten all words
all_words = []
for sentence in sentences:
    all_words.extend(sentence.lower().split())


# -------------------- Tokens --------------------------------
clean_tokens = [clean_tokens(sentence) for sentence in sentences]  # Apply the cleaning function to each review
clean_text = [word for tokens in clean_tokens for word in tokens]   # Join tokens back to text for vectorization

print("tokens =", clean_tokens)  
print("clean_text =", clean_text) 

# ------------------- One-hot encoding using scikit-learn -------------------------
def sklearn_one_hot_encoding(all_words):
    
    # Get unique words
    vocabulary = sorted(list(set(all_words)))
   
    # Create label encoder
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(vocabulary)
    print("integer_encoded =", integer_encoded)

    # Reshape for OneHotEncoder
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    print("integer_encoded =", integer_encoded)

    # Create OneHotEncoder
    onehot_encoder = OneHotEncoder(sparse_output=False)
    onehot_encoded_vector = onehot_encoder.fit_transform(integer_encoded)

    return onehot_encoded_vector, vocabulary, integer_encoded

# ------------------------- VECTOR with Preprocessing -------------------------
onehot_encoded_vector, vocabulary, integer_encoded = sklearn_one_hot_encoding(clean_text)
print()
print("vocabulary(Preprocessing) = ", vocabulary)
print()
print("vector(Preprocessing) = ", onehot_encoded_vector)
print()

# ------------------------- VECTOR without Preprocessing -------------------------
onehot_encoded_vector, vocabulary, integer_encoded = sklearn_one_hot_encoding(all_words)

print()
print("vocabulary = ", vocabulary)
print()
print("vector = ", onehot_encoded_vector)
print()


print("=== ONE-HOT ENCODING CHARACTERISTICS ===")
print("✓ Each word gets a unique binary vector")
print("✓ Vector length equals vocabulary size")
print("✓ Only one element is 1, all others are 0")
print("✓ No semantic relationships captured")
print("✓ Creates very sparse representations")
print("✓ Memory intensive for large vocabularies")
print("✓ Simple and interpretable")
print("✓ Good for categorical variables")
print()

print("=== USE CASES ===")
print("• Input to neural networks")
print("• Categorical feature encoding")
print("• Baseline for text classification")
print("• Multi-class classification problems")
print("• When vocabulary size is manageable")