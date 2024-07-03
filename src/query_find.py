import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from scipy.spatial.distance import directed_hausdorff, chebyshev, minkowski, mahalanobis, hamming, jaccard
import os
import time
from sentence_transformers import SentenceTransformer

# Function to normalize embeddings
def normalize_embeddings(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norms

# Function to compute a single entry in the Hausdorff distance matrix
def compute_hausdorff_entry(i, j, X):
    return max(
        directed_hausdorff(X[i].reshape(1, -1), X[j].reshape(1, -1))[0], 
        directed_hausdorff(X[j].reshape(1, -1), X[i].reshape(1, -1))[0]
    )

# Function to compute the Hausdorff distance matrix
def hausdorff_distance_matrix(X):
    n = X.shape[0]
    hausdorff_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            value = compute_hausdorff_entry(i, j, X)
            hausdorff_matrix[i, j] = value
            hausdorff_matrix[j, i] = value
    return hausdorff_matrix

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
    elif metric == 'hamming':
        for i in range(n):
            for j in range(i, n):
                distance = hamming(X[i], X[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
    elif metric == 'jaccard':
        for i in range(n):
            for j in range(i, n):
                distance = jaccard(X[i], X[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
    return distance_matrix

# Function to compute a single entry for max and min differences
def compute_max_min_entry(i, j, X):
    diffs = np.abs(X[i] - X[j])
    max_diff = np.max(diffs)
    min_diff = np.min(diffs)
    return max_diff, min_diff

# Function to compute the max and min difference matrices
def compute_max_min_difference_matrices(X):
    n = X.shape[0]
    max_diff_matrix = np.zeros((n, n))
    min_diff_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            max_diff, min_diff = compute_max_min_entry(i, j, X)
            max_diff_matrix[i, j] = max_diff_matrix[j, i] = max_diff
            min_diff_matrix[i, j] = min_diff_matrix[j, i] = min_diff
    return max_diff_matrix, min_diff_matrix

# Function to compute the angular distance matrix
def compute_angular_distance_matrix(cosine_sim_matrix):
    return np.arccos(np.clip(cosine_sim_matrix, -1.0, 1.0))

# Function to generate and save heatmap
def generate_heatmap(matrix, title, output_dir):
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, cmap='viridis', cbar=True)
    plt.title(f'{title} Heatmap of Embeddings')
    plt.xlabel('Sentences')
    plt.ylabel('Sentences')
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

    with open(os.path.join(output_dir, f"top_10_similar_sentences_{metric_name}.txt"), 'w', encoding='utf-8') as f:
        f.write(f"Top 10 most similar sentences for {metric_name}:\n")
        for score, sent1, sent2 in similarities[:10]:
            f.write(f"Score: {score:.4f}\nSentence 1: {sent1}\nSentence 2: {sent2}\n\n")

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

    # Create output directories for heatmaps and top 10 similar sentences
    original_heatmaps_dir = 'heatmaps/original_heatmaps'
    marked_heatmaps_dir = 'heatmaps/marked_heatmaps'
    similar_query_sentences_dir = 'similar_query_sentences'
    similar_paired_sentences_dir = 'similar_paired_sentences'
    os.makedirs(original_heatmaps_dir, exist_ok=True)
    os.makedirs(marked_heatmaps_dir, exist_ok=True)
    os.makedirs(similar_query_sentences_dir, exist_ok=True)
    os.makedirs(similar_paired_sentences_dir, exist_ok=True)

    # Prompt for the query
    query = input("Enter your query: ")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])
    query_embedding = normalize_embeddings(query_embedding)

    # Add the query embedding to the embeddings matrix
    embeddings_with_query = np.vstack([embeddings, query_embedding])
    metadata_with_query = pd.concat([metadata, pd.DataFrame([{'sentence': query, 'document': 'query'}])], ignore_index=True)

    # Function to handle distance calculations and heatmap generation for a given metric
    def handle_metric(metric_name, compute_func, **kwargs):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Computing {metric_name} distance/similarity")
        if metric_name in ['max_diff', 'min_diff']:
            matrix = compute_func(embeddings_with_query)
            if metric_name == 'max_diff':
                matrix = matrix[0]  # Select the max_diff matrix
            else:
                matrix = matrix[1]  # Select the min_diff matrix
        else:
            matrix = compute_func(embeddings_with_query, **kwargs)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {metric_name.capitalize()} matrix computed")
        generate_heatmap(matrix[:-1, :-1], metric_name.capitalize(), original_heatmaps_dir)
        print_top_similar_sentences(matrix[:-1, :-1], metadata, metric_name, similar_paired_sentences_dir)
        generate_heatmap(matrix, f"{metric_name.capitalize()}_Query", marked_heatmaps_dir)
        print_top_similar_sentences(matrix[-1:], metadata_with_query, f"{metric_name}_query", similar_query_sentences_dir)

    # Compute and save metrics
    handle_metric('cosine', cosine_similarity)
    handle_metric('euclidean', euclidean_distances)
    handle_metric('manhattan', manhattan_distances)
    handle_metric('chebyshev', compute_other_distances, metric='chebyshev')
    handle_metric('minkowski', compute_other_distances, metric='minkowski', p=3)
    handle_metric('mahalanobis', compute_other_distances, metric='mahalanobis', VI=np.linalg.inv(np.cov(embeddings.T)))
    handle_metric('hamming', compute_other_distances, metric='hamming')
    handle_metric('jaccard', compute_other_distances, metric='jaccard')
    handle_metric('max_diff', compute_max_min_difference_matrices)
    handle_metric('min_diff', compute_max_min_difference_matrices)
    handle_metric('angular', compute_angular_distance_matrix)
    handle_metric('hausdorff', hausdorff_distance_matrix)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] All tasks completed successfully.")

if __name__ == '__main__':
    main()
