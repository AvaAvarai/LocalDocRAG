import pandas as pd
from gensim.models import Word2Vec
import numpy as np

# Load the dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    sentences = df['sentence'].apply(str.split).tolist()  # Split sentences into words
    return sentences, df['sentence']

# Train a Word2Vec model with negative sampling
def train_word2vec(sentences, vector_size=50, window=5, min_count=1, workers=4, epochs=10, lr=0.01, negative=20):
    model = Word2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=workers, 
                     alpha=lr, sg=1, negative=negative)  # sg=1 for skip-gram, use negative sampling
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=epochs)
    return model

# Calculate sentence embeddings
def get_sentence_embeddings(model, sentences):
    sentence_embeddings = []
    for sentence in sentences:
        words = [word for word in sentence if word in model.wv]
        if words:
            word_vectors = np.array([model.wv[word] for word in words])
            sentence_vector = np.mean(word_vectors, axis=0)
            sentence_embeddings.append(sentence_vector)
        else:
            # If no words in the sentence are in the model's vocabulary, use a zero vector
            sentence_embeddings.append(np.zeros(model.vector_size))
    return np.array(sentence_embeddings)

# Save the sentence embeddings to a CSV file
def save_embeddings(embeddings, sentences, output_file):
    df = pd.DataFrame(embeddings, index=sentences)
    df.to_csv(output_file)

if __name__ == "__main__":
    sentences, original_sentences = load_data('cleaned_texts.csv')
    word2vec_model = train_word2vec(sentences)
    sentence_embeddings = get_sentence_embeddings(word2vec_model, sentences)
    save_embeddings(sentence_embeddings, original_sentences, 'sentence_embeddings_word2vec.csv')
