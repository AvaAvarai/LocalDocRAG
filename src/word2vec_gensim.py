import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from tqdm import tqdm
import concurrent.futures
import numpy as np
import json
import os

def sentence_vector(sentence, model):
    words = simple_preprocess(sentence)
    if not words:
        return np.zeros(model.vector_size).tolist()  # Return a zero vector if no words are found
    return np.mean([model.wv[word] for word in words if word in model.wv], axis=0).tolist()

def process_row(row, model):
    return sentence_vector(row['sentence'], model)

def main():
    # Load data
    df = pd.read_csv('cleaned_texts.csv')

    # Prepare sentences for Word2Vec training
    sentences = [simple_preprocess(sentence) for sentence in df['sentence']]

    # Train Word2Vec model
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=os.cpu_count())

    # Serialize the model for use in the worker processes
    model.save("word2vec.model")
    model = Word2Vec.load("word2vec.model")

    # Setting up parallel processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Map process_row function to each row in the DataFrame
        # Pass the model as a constant argument
        futures = [executor.submit(process_row, row, model) for row in df.to_dict('records')]
        results = list(tqdm(concurrent.futures.as_completed(futures), total=len(futures)))
        results = [future.result() for future in results]

    # Assign results back to the DataFrame
    df['embedding'] = [json.dumps(embedding) for embedding in results]

    # Save embeddings to a new CSV file
    df.to_csv('embeddings_word2vec.csv', index=False)

if __name__ == '__main__':
    main()
