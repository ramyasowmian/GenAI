# ===================================================================================================================
# Word2Vec Implementation for NLP
# This file implements a simple skip-gram Word2Vec model from scratch using numpy.
# It loads text data, cleans it, generates training pairs, trains embeddings, and
# demonstrates nearest-neighbor word similarity.
# ===================================================================================================================

import os
import re
import numpy as np
import pandas as pd

# ---------------------- Text preprocessing ----------------------

# -------- Lowercase, remove punctuation and numbers, and tokenize text -------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    return tokens

# ------ Build vocabulary and mapping dictionaries from tokenized sentences ------
def build_vocabulary(sentences):
    tokens = [token for sentence in sentences for token in sentence]
    vocabulary = sorted(set(tokens))
    word_to_index = {word: idx for idx, word in enumerate(vocabulary)}
    index_to_word = {idx: word for word, idx in word_to_index.items()}
    return vocabulary, word_to_index, index_to_word

# ------ Generate skip-gram training pairs from tokenized sentences ------
def generate_skipgram_pairs(sentences, word_to_index, window_size=2):
    pairs = []
    for sentence in sentences:
        indexed_sentence = [word_to_index[word] for word in sentence if word in word_to_index]
        for index, target_word in enumerate(indexed_sentence):
            start = max(0, index - window_size)
            end = min(len(indexed_sentence), index + window_size + 1)
            for context_index in range(start, end):
                if context_index != index:
                    context_word = indexed_sentence[context_index]
                    pairs.append((target_word, context_word))
    return pairs

# ------ Compute softmax values for each set of scores in x ------
def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

# ------- Train a skip-gram Word2Vec model using full softmax ------
def train_skipgram(pairs, vocab_size, embedding_dim=50, learning_rate=0.03, epochs=1000):

    W1 = np.random.randn(vocab_size, embedding_dim) * 0.01
    W2 = np.random.randn(embedding_dim, vocab_size) * 0.01

    for epoch in range(1, epochs + 1):
        loss = 0.0
        for target_idx, context_idx in pairs:
            hidden = W1[target_idx]
            output = np.dot(W2.T, hidden)
            predicted = softmax(output)
            loss -= np.log(predicted[context_idx] + 1e-9)

            error = predicted
            error[context_idx] -= 1.0

            dW2 = np.outer(hidden, error)
            dW1 = np.dot(W2, error)

            W2 -= learning_rate * dW2
            W1[target_idx] -= learning_rate * dW1

        if epoch % 200 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs}  Loss: {loss:.3f}")
    return W1

# ------ Compute cosine similarity between two vectors ------
def cosine_similarity(vec_a, vec_b):
    if np.linalg.norm(vec_a) == 0 or np.linalg.norm(vec_b) == 0:
        return 0.0
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

# ------ Return the top-N most similar words to the given word -----
def get_similar_words(word, embeddings, word_to_index, index_to_word, top_n=5):
    if word not in word_to_index:
        return []
    word_idx = word_to_index[word]
    word_vec = embeddings[word_idx]
    similarities = []
    for idx, candidate_vec in enumerate(embeddings):
        if idx == word_idx:
            continue
        sim = cosine_similarity(word_vec, candidate_vec)
        similarities.append((index_to_word[idx], sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "..", "Resource", "Reviews.csv")

    df = pd.read_csv(csv_path, nrows=1)
    raw_sentences = df["Text"].dropna().astype(str).tolist()
    tokenized_sentences = [clean_text(sentence) for sentence in raw_sentences if len(sentence.strip()) > 0]

    vocabulary, word_to_index, index_to_word = build_vocabulary(tokenized_sentences)
    pairs = generate_skipgram_pairs(tokenized_sentences, word_to_index, window_size=2)

    print("raw_sentences", raw_sentences)
    print("")
    print("tokenized_sentences", tokenized_sentences)
    print("")
    print("word_to_index", word_to_index)
    print("")
    print("index_to_word", index_to_word)
    print("")
    print("Vocabulary :", vocabulary)
    print()
    print("Vocabulary size:", len(vocabulary))
    print()
    print("Training pairs:", len(pairs))
    print()

    embeddings = train_skipgram(pairs, vocab_size=len(vocabulary), embedding_dim=50, learning_rate=0.05, epochs=1000)

    print("embeddings = \(embeddings)")

    examples = ["good", "bad", "movie", "love", "product"]
    for example in examples:
        if example in word_to_index:
            similar = get_similar_words(example, embeddings, word_to_index, index_to_word, top_n=5)
            print(f"\nTop similar words to '{example}':")
            for neighbor, score in similar:
                print(f"  {neighbor}: {score:.3f}")
        else:
            print(f"\nWord '{example}' is not in vocabulary.")
