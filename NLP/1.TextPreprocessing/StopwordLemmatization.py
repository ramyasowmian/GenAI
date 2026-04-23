# NLTKStopwordLemmatization.py
# This code performs tokenization, stop words removal, and lemmatization using NLTK.

import os
import pandas as pd 
import nltk
from nltk.tokenize import word_tokenize  
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Download the necessary NLTK resources (run once)
nltk.download("punkt")                                      
nltk.download("stopwords")                                  
nltk.download("wordnet")   
nltk.download("omw-1.4")                                 
nltk.download("averaged_perceptron_tagger")  
nltk.download('averaged_perceptron_tagger_eng')              

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "Reviews.csv")

# Step 1: Load the first 100 reviews from the CSV file
df = pd.read_csv(csv_path, nrows=100)              

# Step 2: Tokenize the reviews into words
lowerText = df["Text"].str.lower()    

# Step 3: Initialize stop words and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()  

# Step 4: Define a function to get the WordNet POS tag for lemmatization
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
    

# Step 5: NLTK Tokenization, Stop Words Removal, and Lemmatization
# isalpha() is used to filter out non-alphabetic tokens (e.g., punctuation, numbers)
# Stopping words are common words that do not carry significant meaning in text analysis (e.g., "the", "is", "and").
# Removing them helps to focus on the more meaningful tokens in the text.  
# Lemmatization is the process of reducing words to their base or root form (lemma) (e.g., "running" becomes "run").  
def remove_stop_words(text):
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
 
# Step 5: Apply the function to each review and flatten the list
all_token = lowerText.apply(remove_stop_words).explode().tolist()  

print("all_token =", all_token)

