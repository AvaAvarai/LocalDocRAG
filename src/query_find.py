import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from scipy.spatial.distance import chebyshev, minkowski, mahalanobis, hamming, jaccard
from concurrent.futures import ProcessPoolExecutor
import os
import time
from sentence_transformers import SentenceTransformer

# Function to normalize embeddings
def normalize_embeddings(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norms

# Function to binarize embeddings for Hamming and Jaccard distances
def binarize_embeddings(X):
    return (X > np.median(X, axis=0)).astype(int)

# Function to compute max and min differences
def compute_max_min_entry(args):
    i, j, X = args
    diffs = np.abs(X[i] - X[j])
    max_diff = np.max(diffs)
    min_diff = np.min(diffs)
    return i, j, max_diff, min_diff

def compute_max_min_difference_matrices(X, batch_size=1000):
    n = X.shape[0]
    max_diff_matrix = np.zeros((n, n))
    min_diff_matrix = np.zeros((n, n))
    
    with ProcessPoolExecutor() as executor:
        for batch_start in range(0, n, batch_size):
            start_time = time.time()
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Computing max and min differences batch: {batch_start} to {min(batch_start + batch_size, n)}")
            tasks = [(i, j, X) for i in range(batch_start, min(batch_start + batch_size, n)) for j in range(i, n)]
            results = executor.map(compute_max_min_entry, tasks)
            
            for i, j, max_diff, min_diff in results:
                max_diff_matrix[i, j] = max_diff_matrix[j, i] = max_diff
                min_diff_matrix[i, j] = min_diff_matrix[j, i] = min_diff
            
            end_time = time.time()
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Batch {batch_start} to {min(batch_start + batch_size, n)} completed in {end_time - start_time:.2f} seconds")
    
    return max_diff_matrix, min_diff_matrix

# Function to compute other distance matrices
def compute_other_distances(X, metric, **kwargs):
    n = X.shape[0]
    distance_matrix = np.zeros((n, n))
    if metric == 'chebyshev':
        for i in range(n):
            for j in range(i, n):
                distance = chebyshev(X[i], X[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
    elif metric == 'minkowski':
        p = kwargs.get('p', 3)
        for i in range(n):
            for j in range(i, n):
                distance = minkowski(X[i], X[j], p)
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
    elif metric == 'mahalanobis':
        VI = kwargs.get('VI', np.linalg.inv(np.cov(X.T)))
        for i in range(n):
            for j in range(i, n):
                distance = mahalanobis(X[i], X[j], VI)
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
    elif metric == 'manhattan':
        distance_matrix = manhattan_distances(X)
    elif metric == 'hamming' or metric == 'jaccard':
        X_bin = binarize_embeddings(X)
        for i in range(n):
            for j in range(i, n):
                if metric == 'hamming':
                    distance = hamming(X_bin[i], X_bin[j])
                else:
                    distance = jaccard(X_bin[i], X_bin[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
    return distance_matrix

# Function to compute the angular distance matrix
def compute_angular_distance_matrix(cosine_sim_matrix):
    return np.arccos(np.clip(cosine_sim_matrix, -1.0, 1.0))

# Function to generate and save heatmap with highlighted sentences
def generate_heatmap(matrix, title, output_dir, indices_to_highlight=None):
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, cmap='viridis', cbar=True)
    plt.title(f'{title} Heatmap of Embeddings')
    plt.xlabel('Sentences')
    plt.ylabel('Sentences')
    if indices_to_highlight:
        for idx in indices_to_highlight:
            plt.scatter([idx[1]], [idx[0]], color='red')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.savefig(os.path.join(output_dir, f'{title}_heatmap.png'))
    plt.close()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Heatmap for {title} saved")

# Function to print the top 10 most similar sentences with their similarity scores
def print_top_similar_sentences(matrix, metadata, metric_name, output_dir):
    n = matrix.shape[0]
    similarities = []
    for i in range(n):
        for j in range(i + 1, n):
            similarities.append((matrix[i, j], metadata['sentence'][i], metadata['sentence'][j]))

    similarities.sort(key=lambda x: x[0], reverse=(metric_name != 'euclidean' and metric_name != 'manhattan'))

    output_path = os.path.join(output_dir, f'{metric_name}_top_10_similar_sentences.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        for score, sent1, sent2 in similarities[:10]:
            f.write(f"Score: {score:.4f}\nSentence 1: {sent1}\nSentence 2: {sent2}\n\n")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Top 10 similar sentences for {metric_name} saved")

# Main function
def main():
    # Load the CSV file into a DataFrame
    csv_file = r'embeddings_SBERT_382.62s.csv'
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading data from {csv_file}")
    df = pd.read_csv(csv_file)

    # Extract embeddings and metadata
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Extracting embeddings and metadata")
    embedding_columns = [str(i) for i in range(384)]
    embeddings = df[embedding_columns].values
    metadata = df[['sentence', 'document']]

    # Normalize embeddings
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Normalizing embeddings")
    embeddings = normalize_embeddings(embeddings)

    # Create output directories
    original_heatmaps_dir = 'original_heatmaps'
    marked_heatmaps_dir = 'marked_heatmaps'
    similar_query_sentences_dir = 'similar_query_sentences'
    similar_paired_sentences_dir = 'similar_paired_sentences'
    
    os.makedirs(original_heatmaps_dir, exist_ok=True)
    os.makedirs(marked_heatmaps_dir, exist_ok=True)
    os.makedirs(similar_query_sentences_dir, exist_ok=True)
    os.makedirs(similar_paired_sentences_dir, exist_ok=True)

    # Get query from user and generate its embedding
    query = input("Enter your query: ")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = normalize_embeddings(model.encode([query]))

    # Define metrics
    metrics = [
        ('cosine', cosine_similarity),
        ('euclidean', euclidean_distances),
        ('manhattan', manhattan_distances),
        ('chebyshev', lambda X: compute_other_distances(X, 'chebyshev')),
        ('minkowski', lambda X: compute_other_distances(X, 'minkowski', p=3)),
        ('mahalanobis', lambda X: compute_other_distances(X, 'mahalanobis', VI=np.linalg.inv(np.cov(X.T)))),
        ('hamming', lambda X: compute_other_distances(X, 'hamming')),
        ('jaccard', lambda X: compute_other_distances(X, 'jaccard')),
        ('max_diff', lambda X: compute_max_min_difference_matrices(X)[0]),
        ('min_diff', lambda X: compute_max_min_difference_matrices(X)[1]),
        ('angular', lambda X: compute_angular_distance_matrix(cosine_similarity(X)))
    ]

    for metric_name, metric_func in metrics:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Computing {metric_name} distance/similarity")
        matrix = metric_func(embeddings)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {metric_name.capitalize()} matrix computed")

        # Save original heatmap
        generate_heatmap(matrix, metric_name.capitalize(), original_heatmaps_dir)
        
        # Print top 10 similar sentences
        print_top_similar_sentences(matrix, metadata, metric_name, similar_paired_sentences_dir)

        # Compute distances/similarities to the query
        query_distances = metric_func(np.vstack((query_embedding, embeddings)))[0, 1:]
        top_10_indices = query_distances.argsort()[:10]
        indices_to_highlight = [(idx, idx) for idx in top_10_indices]

        # Save heatmap with highlighted top 10 similar sentences
        generate_heatmap(matrix, f'{metric_name.capitalize()}_Query', marked_heatmaps_dir, indices_to_highlight)

        # Save top 10 similar sentences to the query
        with open(os.path.join(similar_query_sentences_dir, f'{metric_name}_query_similar_sentences.txt'), 'w', encoding='utf-8') as f:
            for idx in top_10_indices:
                f.write(f"Score: {query_distances[idx]:.4f}\nSentence: {metadata['sentence'][idx]}\n\n")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Top 10 similar sentences to query for {metric_name} saved")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] All tasks completed successfully.")

if __name__ == '__main__':
    main()
