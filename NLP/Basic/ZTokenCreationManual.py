import pandas as ps 

df = ps.read_csv("Reviews.csv")                        # Read the CSV file containing reviews into a DataFrame
lowerText = df["Text"].head(10).str.lower()            # First 100 reviews in lowercase

sentencesToken = lowerText.str.split()                 # Tokenize the sentences into words
all_token = []                                         # Extend the all_token list with the tokens from each review
for review in sentencesToken:
    all_token.extend(review)

print("all_token", all_token)
