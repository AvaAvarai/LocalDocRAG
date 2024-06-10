import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from tqdm import tqdm
import numpy as np
import concurrent.futures

def sentence_vector(sentence, model):
    words = simple_preprocess(sentence)
    if not words:
        return np.zeros(model.vector_size)  # Return a zero vector if no words are found
    return sum(model.wv[word] for word in words if word in model.wv) / len([word for word in words if word in model.wv])

def process_row(row, model):
    return sentence_vector(row['sentence'], model)

def main():
    # Load data
    df = pd.read_csv('cleaned_texts.csv')

    # Prepare sentences for Word2Vec training
    sentences = [simple_preprocess(sentence) for sentence in df['sentence']]

    # Train Word2Vec model
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    # Setting up parallel processing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Map process_row function to each row in the DataFrame
        # Use tqdm to show the progress bar
        # Pass model as a part of the function to avoid pickling the whole model
        results = list(tqdm(executor.map(process_row, df.to_dict('records'), [model]*len(df)), total=len(df)))

    # Assign results back to the DataFrame
    df['embedding'] = results

    # Save embeddings to a new CSV file
    df.to_csv('embeddings_word2vec.csv', index=False)

if __name__ == '__main__':
    main()
