# NLTKStopword.py
# This code performs tokenization and stop words removal using NLTK
 
import os
import pandas as pd 
import nltk
from nltk.tokenize import word_tokenize  
from nltk.corpus import stopwords   

# Download the necessary NLTK resources (run once)
nltk.download("punkt")                                      
nltk.download("stopwords")   

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
resource_dir = os.path.join(script_dir, "..", "Resource")
csv_path = os.path.join(resource_dir, "Reviews.csv")

# Step 1: Load the first 100 reviews from the CSV file
df = pd.read_csv(csv_path, nrows=100)              

# Step 2: lowercase the reviews
lowerText = df["Text"].str.lower()    

# Step 3: Initialize stop words
stop_words = set(stopwords.words("english"))

# Step 4: NLTK Tokenization, Stop Words Removal
# isalpha() is used to filter out non-alphabetic tokens (e.g., punctuation, numbers)
# Stopping words are common words that do not carry significant meaning in text analysis (e.g., "the", "is", "and").
# Removing them helps to focus on the more meaningful tokens in the text.    
def remove_stop_words(text):
    # Tokenize the text  
    tokens = word_tokenize(text)   

    # Filter out stop words and non-alphabetic tokens                  
    filtered_tokens = [
        word.lower()
        for word in tokens 
            if word.isalpha() and word.lower() not in stop_words
    ]  
    return filtered_tokens 

# Step 5: Apply the function to each review and flatten the list
all_token = lowerText.apply(remove_stop_words).explode().tolist()  

print("all_token =", all_token)

