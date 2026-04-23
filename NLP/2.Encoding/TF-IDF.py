
# ===================================================================================================================
# TF-IDF Implementation for NLP
# TF-IDF with Custom Lemmatization Tokenizer
# Clean up -> (remove Stop words, punctuation, numbers, convert to lowercase, apply lemmatization based on POS tags)
# Text, (clean up), Tokenization, Token, TF-IDF Vectorizer, Vector(TF-IDF Matrix), Vocabulary(Feature Names)
# ===================================================================================================================

# Create TF-IDF vectorizer with stopwords removal and lowercase conversion
# Punctuation and numbers are automatically ignored by the TfidfVectorizer when stop_words='english' is used, 
# but we will also apply custom cleaning in the tokenizer to ensure that all unwanted characters are removed before tokenization.

import re
import os
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from sklearn.feature_extraction.text import TfidfVectorizer # TF-IDF Vectorizer

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
# Load the first 10 reviews from the CSV file
df = pd.read_csv(csv_path, nrows=10)
text = df["Text"].str.lower()

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
    
def clean_Token(text):
    # Lowercase the text
    text = text.lower()

    # Remove punctuation and numbers using regex
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)      # Remove numbers

    # Tokenize the text (Converts sentences into list of words)
    tokens = word_tokenize(text) 
    
    # Get POS tags for the tokens
    pos_tags = pos_tag(tokens)  

    # ------------------- Stopwords, Lemmatization -------------------
    # Filter out stop words and non-alphabetic tokens and apply lemmatization based on POS tags              
    tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos))
        for word, pos in zip(tokens, [tag for _, tag in pos_tags])
        if word.isalpha() and word not in stop_words
    ]  
    return tokens  

# ------------------- TF-IDF VECTORIZER -------------------
# Create TF-IDF vectorizer with the custom tokenizer
vectorizer = TfidfVectorizer(stop_words='english', 
                             lowercase=True,
                             tokenizer=clean_Token)

# ------------------------- VECTOR -------------------------
# Numerical representation(Matrix) of the TEXT
vector = vectorizer.fit_transform(text)  

# -------------------- VOCABULARY(FEATURE NAMES) -------------------
# Vocabulary is the set of unique words (features) extracted from the text after preprocessing.
vocabulary = vectorizer.get_feature_names_out()

# -------------------- OUTPUT -------------------
df = pd.DataFrame(vector.toarray(), columns=vocabulary)

print("token:", text.apply(clean_Token))
print("vector:", vector)
print("Vocabulary", vocabulary)
print("TF-IDF Matrix Shape:", vector.shape)
print("TF-IDF Matrix Shape:", df.shape)