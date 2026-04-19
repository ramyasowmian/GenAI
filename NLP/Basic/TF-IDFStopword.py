# TF-IDFStopword.py
# This code performs tokenization and stop words removal using TF-IDF vectorizer from SciKit Learn

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the first 100 reviews from the CSV file
df = pd.read_csv("ZReviews.csv", nrows=100)              
lowerText = df["Text"].str.lower()            

# Create TF-IDF vectorizer with stopwords removal and lowercase conversion
# Punctuation and numbers are automatically ignored by the TfidfVectorizer when stop_words='english' is used, 
# as it focuses on extracting meaningful tokens.
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)     

# Fit and transform the text data to create the TF-IDF matrix
matrix = vectorizer.fit_transform(lowerText)     

# Get the feature names (tokens)
feature_names = vectorizer.get_feature_names_out()     

# Convert the feature names to a list
all_token = feature_names.tolist()       

print("all_token =", all_token)   
