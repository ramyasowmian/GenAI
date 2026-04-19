
# =============================================================
# TF-IDF with Custom Lemmatization Tokenizer
# Clean up -> (remove Stop words, punctuation, numbers, convert to lowercase, apply lemmatization based on POS tags)
# Text, (clean up), Tokenization, Token, TF-IDF Vectorizer, Vector(TF-IDF Matrix), Vocabulary(Feature Names)
# =============================================================

import re
import os
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer

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
# Load the first 100 reviews from the CSV file
df = pd.read_csv(csv_path, nrows=10)
lowerText = df["Text"].str.lower()

# -------------------- Tokenizer --------------------
# This is the Custom Lemmatization Tokenizer
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
    
def custom_lemmatizer_tokenizer(text):
    # Remove punctuation and numbers using regex
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)      # Remove numbers

    # Tokenize the text  
    tokens = word_tokenize(text) 
    
    # Get POS tags for the tokens
    pos_tags = pos_tag(tokens)  

    # ------------------- Stopwords, Lemmatization -------------------
    # Filter out stop words and non-alphabetic tokens and apply lemmatization based on POS tags              
    filtered_tokens = [
        lemmatizer.lemmatize(word.lower(), get_wordnet_pos(pos))
        for word, pos in zip(tokens, [tag for _, tag in pos_tags])
        if word.isalpha() and word.lower() not in stop_words
    ]  
    return filtered_tokens  

# ------------------- TF-IDF VECTORIZER -------------------
# Create TF-IDF vectorizer with the custom tokenizer
vectorizer = TfidfVectorizer(stop_words='english', 
                             tokenizer=custom_lemmatizer_tokenizer)

# ------------------------- VECTOR -------------------------
# TF-IDF numerical representation of the TEXT
tfidf_matrix = vectorizer.fit_transform(lowerText)  

# -------------------- VOCABULARY(FEATURE NAMES) -------------------
vocabulary = vectorizer.get_feature_names_out()

# -------------------- OUTPUT -------------------
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vocabulary)


print("Vocabulary (Feature Names):", vocabulary)
print("TF-IDF Matrix Shape:", tfidf_matrix.shape)
print("TF-IDF Matrix Shape:", tfidf_df)



