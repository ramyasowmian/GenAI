
# ===================================================================================================================
# BOW Implementation for NLP
# Clean up -> (remove Stop words, punctuation, numbers, convert to lowercase, apply lemmatization based on POS tags)
# Text, (clean up), Tokenization, Token, BOW Vectorizer, Vector(Matrix), Vocabulary(Feature Names)
# ===================================================================================================================

import pandas as pd
import re
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from sklearn.feature_extraction.text import CountVectorizer  # BOW Vectorizer

# Download the necessary NLTK resources (run once)
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger")
nltk.download('averaged_perceptron_tagger_eng')     

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
resource_dir = os.path.join(script_dir, "..", "Resource")
csv_path = os.path.join(resource_dir, "Reviews.csv") 

# ------------------- Text -------------------------
# Step 1: Load the first 10 reviews from the CSV file
df = pd.read_csv(csv_path, nrows=10)   
text = df["Text"].str.lower()

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

# -------------------- Tokens --------------------------------
clean_tokens = text.apply(clean_tokens)  # Apply the cleaning function to each review
clean_text = clean_tokens.apply(lambda x: ' '.join(x))  # Join tokens back to text for vectorization

print("tokens =", clean_tokens)  
print("clean_text =", clean_text)   

#  ------------------- BOW VECTORIZER --------------------------
vectorizer = CountVectorizer()

# ------------------------- VECTOR -------------------------
# Numerical representation(Matrix) of the TEXT using BOW
vector = vectorizer.fit_transform(clean_text)

# -------------------- VOCABULARY(FEATURE NAMES) -------------------
# Vocabulary is the set of unique words (features) extracted from the text after preprocessing.
vocabulary = vectorizer.get_feature_names_out() 

bow_df = pd.DataFrame(vector.toarray(), columns=vocabulary)

print("bow_matrix:", vector)
print("Vocabulary (Feature Names):", vocabulary)
print("BOW Matrix (DataFrame):\n", bow_df)  

