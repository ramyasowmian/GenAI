# TF-IDFStopwordLemmatization.py
# This code demonstrates how to create a TF-IDF matrix from a collection of text reviews 
# while performing punctuation and numbers removal, stopwords removal, and lemmatization using NLTK.
#
# Note: 
#   The Punctuation and numbers are automatically ignored in TF-IDF vectorization 
#   when stop_words='english' is used, as it focuses on extracting meaningful tokens.
#
#   The TF-IDF vectorizer does not perform lemmatization, you must provide a custom tokenizer
#   Custom Lemmatizer Tokenizer(NLTK) + TF-IDF Vectorizer(SciKit Learn)

import re
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

# Load the first 100 reviews from the CSV file
df = pd.read_csv("ZReviews.csv", nrows=100)              
lowerText = df["Text"].str.lower()  

# Initialize stop words and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()    

# Define a function to get the WordNet POS tag for lemmatization
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

    # Filter out stop words and non-alphabetic tokens and apply lemmatization based on POS tags              
    filtered_tokens = [
        lemmatizer.lemmatize(word.lower(), get_wordnet_pos(pos))
        for word, pos in zip(tokens, [tag for _, tag in pos_tags])
        if word.isalpha() and word.lower() not in stop_words
    ]  
    return filtered_tokens

# Create TF-IDF 
vectorizer = TfidfVectorizer(stop_words='english', 
                            lowercase=True,
                            tokenizer=custom_lemmatizer_tokenizer)

# Fit and transform the text data to create the TF-IDF matrix
matrix = vectorizer.fit_transform(lowerText)     

# Get the feature names (tokens)
feature_names = vectorizer.get_feature_names_out()     

# Convert the feature names to a list
all_token = feature_names.tolist()       

print("all_token =", all_token)   
