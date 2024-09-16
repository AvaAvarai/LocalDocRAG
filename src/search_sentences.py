import csv
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import warnings
import torch

# Suppress warnings
warnings.filterwarnings('ignore')

def load_embeddings(csv_file):
    sentences = []
    sources = []
    embeddings = []

    # Read the CSV file
    with open(csv_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sentence = row['Sentence']
            source = row['Source']
            embedding_str = row['Embedding']
            # Convert the JSON string back to a NumPy array
            embedding = np.array(json.loads(embedding_str))

            sentences.append(sentence)
            sources.append(source)
            embeddings.append(embedding)

    # Convert lists to NumPy arrays for efficient computation
    embeddings = np.vstack(embeddings)
    return sentences, sources, embeddings

def main():
    # Load the embeddings from the CSV file
    csv_file = 'extracted_sentences.csv'  # Ensure this matches your output file
    print("Loading embeddings from CSV...")
    sentences, sources, embeddings = load_embeddings(csv_file)
    print(f"Loaded {len(sentences)} sentences.")

    # Initialize the embedding model
    model_name = 'all-MiniLM-L6-v2'  # Ensure this matches the model used previously

    # Correct device assignment
    device_embedder = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_summarizer = 0 if torch.cuda.is_available() else -1

    embedder = SentenceTransformer(model_name, device=device_embedder)

    # Initialize the summarization model
    summarizer = pipeline('summarization', model='t5-small', device=device_summarizer)

    while True:
        # Accept user query
        print("\nEnter your query (or type 'exit' to quit): ", end='', flush=True)
        query = input()
        if query.lower() == 'exit':
            break

        # Accept similarity threshold
        print("Enter the similarity threshold (e.g., 0.7): ", end='', flush=True)
        threshold_input = input()
        try:
            threshold = float(threshold_input)
            if not (0.0 < threshold <= 1.0):
                print("Please enter a number between 0 and 1 for the threshold.")
                continue
        except ValueError:
            print("Invalid input for threshold. Please enter a number between 0 and 1.")
            continue

        # Generate embedding for the query
        print("Generating query embedding...")
        query_embedding = embedder.encode([query])[0]

        # Compute cosine similarities
        print("Computing similarities...")
        similarities = cosine_similarity([query_embedding], embeddings)[0]

        # Select sentences above the threshold
        relevant_indices = np.where(similarities >= threshold)[0]

        if len(relevant_indices) == 0:
            print("No sentences found with similarity above the threshold.")
            continue

        # Retrieve relevant sentences and their sources
        relevant_sentences = [sentences[idx] for idx in relevant_indices]
        relevant_sources = [sources[idx] for idx in relevant_indices]
        relevant_similarities = [similarities[idx] for idx in relevant_indices]

        # Combine the relevant sentences into a single text
        combined_text = ' '.join(relevant_sentences)

        # Synthesize the final answer
        print("\nSynthesizing answer...")
        max_length = min(512, len(combined_text.split()))
        summary = summarizer(
            combined_text,
            max_length=max_length,
            min_length=40,
            do_sample=False,
            truncation=True,
        )[0]['summary_text']

        # Display the synthesized answer
        print("\nFinal Answer:")
        print(summary)

        # Display citations
        print("\nCitations:")
        for i, idx in enumerate(relevant_indices):
            print(f"[{i+1}] {sentences[idx]} (Source: {sources[idx]}, Similarity: {similarities[idx]:.4f})")

if __name__ == "__main__":
    main()
