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

# ============================================================================
# METHOD 1: MANUAL ONE-HOT ENCODING
# ============================================================================

# ------- Create vocabulary from list of sentences -------------
def create_vocabulary(sentences):
    vocabulary = set()
    for sentence in sentences:
        words = sentence.lower().split()
        vocabulary.update(words)
    return sorted(list(vocabulary))

# -------------Manual implementation of one-hot encoding --------------
def manual_one_hot_encoding(sentences):
    # Create vocabulary
    vocabulary = create_vocabulary(sentences)
    vocab_size = len(vocabulary)
    word_to_index = {word: idx for idx, word in enumerate(vocabulary)}

    print("=== MANUAL ONE-HOT ENCODING ===")
    print(f"Vocabulary Size: {vocab_size}")
    print(f"Vocabulary: {vocabulary}")
    print()
    print(f"word_to_index: {word_to_index}")
    print()

    # Encode each sentence
    encoded_sentences = []

    for sentence in sentences:
        words = sentence.lower().split()

        # Create one-hot matrix for this sentence
        sentence_matrix = np.zeros((len(words), vocab_size))

        print()
        print("Encoding sentence:", sentences)
        print("Encoding words:", words)
        print("Encoding sentence_matrix:", sentence_matrix)
        print()

        for i, word in enumerate(words):
            if word in word_to_index:
                sentence_matrix[i, word_to_index[word]] = 1

        encoded_sentences.append(sentence_matrix)

    return encoded_sentences, vocabulary

# Apply manual encoding
manual_encoded, vocab = manual_one_hot_encoding(sample_sentences)

print("Manual Encoding Results:")
print(f"First sentence: '{sample_sentences[0]}'")
print(f"Words: {sample_sentences[0].lower().split()}")
print(f"One-hot matrix shape: {manual_encoded[0].shape}")
print("One-hot matrix (rows=words, columns=vocabulary):")
print(manual_encoded[0])
print()

# ============================================================================
# METHOD 2: USING SCIKIT-LEARN
# ============================================================================

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

# ============================================================================
# METHOD 3: DOCUMENT-LEVEL ONE-HOT ENCODING (Binary BoW)
# ============================================================================

def document_level_one_hot(sentences):
    """Document-level one-hot encoding (shows word presence/absence)"""
    print("=== DOCUMENT-LEVEL ONE-HOT ENCODING ===")

    vocabulary = create_vocabulary(sentences)
    vocab_size = len(vocabulary)
    word_to_index = {word: idx for idx, word in enumerate(vocabulary)}

    # Create binary vectors for each document
    document_vectors = []

    for sentence in sentences:
        words = set(sentence.lower().split())  # Use set to remove duplicates
        doc_vector = np.zeros(vocab_size)

        for word in words:
            if word in word_to_index:
                doc_vector[word_to_index[word]] = 1

        document_vectors.append(doc_vector)

    document_vectors = np.array(document_vectors)

    print(f"Document vectors shape: {document_vectors.shape}")
    print("Each row represents a document, each column represents a word (1=present, 0=absent)")
    print()

    # Create DataFrame for better visualization
    df = pd.DataFrame(document_vectors, columns=vocabulary,
                     index=[f"Doc {i+1}" for i in range(len(sentences))])

    print("Document-Term Matrix:")
    print(df)
    print()

    return document_vectors, vocabulary

# Apply document-level encoding
doc_encoded, vocab = document_level_one_hot(sample_sentences)

# ============================================================================
# METHOD 4: WORD-LEVEL ONE-HOT ENCODING FUNCTION
# ============================================================================

def encode_single_word(word, vocabulary):
    """Encode a single word to one-hot vector"""
    if word not in vocabulary:
        return None

    word_to_index = {w: idx for idx, w in enumerate(vocabulary)}
    one_hot = np.zeros(len(vocabulary))
    one_hot[word_to_index[word]] = 1

    return one_hot

print("=== WORD-LEVEL ONE-HOT ENCODING ===")
test_words = ["machine", "learning", "artificial", "intelligence"]

for word in test_words:
    vector = encode_single_word(word, vocab)
    if vector is not None:
        print(f"'{word}': {vector.astype(int)}")
    else:
        print(f"'{word}': Not in vocabulary")

print()

# ============================================================================
# SUMMARY AND CHARACTERISTICS
# ============================================================================

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