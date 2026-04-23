# ===================================================================================================================
# One-Hot Encoding Implementation for NLP

# One-hot encoding represents each word in the vocabulary as a binary vector
# where only one element is 1 (hot) and all others are 0 (cold).

# This file demonstrates multiple approaches to one-hot encoding:
# 1. Manual implementation
# 2. Using scikit-learn
# 3. Document-level encoding
# ===================================================================================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Sample text data for demonstration
sample_sentences = [
    "I love machine learning",
    "Natural language processing is fascinating",
    "Deep learning models are powerful",
    "Text encoding is important for NLP",
    "One hot encoding creates sparse vectors"
]

print("Sample Sentences:")
for i, sentence in enumerate(sample_sentences, 1):
    print(f"{i}. {sentence}")
print()


def sklearn_one_hot_encoding(sentences):
    """One-hot encoding using scikit-learn"""
    print("=== SCIKIT-LEARN ONE-HOT ENCODING ===")

    # Flatten all words
    all_words = []
    for sentence in sentences:
        all_words.extend(sentence.lower().split())

    # Get unique words
    unique_words = sorted(list(set(all_words)))
    print(f"Unique words: {unique_words}")
    print()

    # Create label encoder
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(unique_words)

    # Reshape for OneHotEncoder
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    # Create OneHotEncoder
    onehot_encoder = OneHotEncoder(sparse_output=False)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    print("One-hot encoded vectors for each word:")
    for i, word in enumerate(unique_words):
        print(f"{word}: {onehot_encoded[i].astype(int)}")

    return onehot_encoded, unique_words, label_encoder

# Apply sklearn encoding
sklearn_encoded, unique_words, label_encoder = sklearn_one_hot_encoding(sample_sentences)
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