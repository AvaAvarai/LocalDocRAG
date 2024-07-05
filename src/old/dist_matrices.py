import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from scipy.spatial.distance import directed_hausdorff, chebyshev, minkowski, mahalanobis, hamming, jaccard
from concurrent.futures import ProcessPoolExecutor
import os
import time

# Function to normalize embeddings
def normalize_embeddings(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norms

# Function to binarize embeddings for Hamming and Jaccard distances
def binarize_embeddings(X):
    return (X > np.median(X, axis=0)).astype(int)

# Function to compute a single entry in the Hausdorff distance matrix
def compute_hausdorff_entry(args):
    i, j, X = args
    return i, j, max(
        directed_hausdorff(X[i].reshape(1, -1), X[j].reshape(1, -1))[0], 
        directed_hausdorff(X[j].reshape(1, -1), X[i].reshape(1, -1))[0]
    )

# Function to compute the Hausdorff distance matrix in parallel
def hausdorff_distance_matrix(X, batch_size=1000):
    n = X.shape[0]
    hausdorff_matrix = np.zeros((n, n))
    
    with ProcessPoolExecutor() as executor:
        for batch_start in range(0, n, batch_size):
            start_time = time.time()
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Computing Hausdorff distance batch: {batch_start} to {min(batch_start + batch_size, n)}")
            tasks = [(i, j, X) for i in range(batch_start, min(batch_start + batch_size, n)) for j in range(i, n)]
            results = executor.map(compute_hausdorff_entry, tasks)
            
            for i, j, value in results:
                hausdorff_matrix[i, j] = value
                hausdorff_matrix[j, i] = value
            
            end_time = time.time()
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Batch {batch_start} to {min(batch_start + batch_size, n)} completed in {end_time - start_time:.2f} seconds")
    
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

# Function to compute a single entry for max and min differences
def compute_max_min_entry(args):
    i, j, X = args
    diffs = np.abs(X[i] - X[j])
    max_diff = np.max(diffs)
    min_diff = np.min(diffs)
    return i, j, max_diff, min_diff

# Function to compute the max and min difference matrices in parallel
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

    similarities.sort(key=lambda x: x[0], reverse=(metric_name != 'euclidean' and metric_name != 'manhattan' and metric_name != 'chebyshev'))

    file_path = os.path.join(output_dir, f'top_10_similar_sentences_{metric_name}.txt')
    with open(file_path, 'w', encoding='utf-8') as f:
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

    # Create output directory for heatmaps
    output_dir = 'heatmaps'
    os.makedirs(output_dir, exist_ok=True)

    # Create output directory for top 10 similar sentences
    similar_output_dir = os.path.join(output_dir, 'top_similar_sentences')
    os.makedirs(similar_output_dir, exist_ok=True)

    # Compute and save the cosine similarity matrix
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Computing cosine similarity matrix")
    cosine_sim_matrix = cosine_similarity(embeddings)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Cosine similarity matrix computed")
    generate_heatmap(cosine_sim_matrix, 'Cosine Similarity', output_dir)
    print_top_similar_sentences(cosine_sim_matrix, metadata, 'cosine_similarity', similar_output_dir)

    # Compute and save the angular distance matrix
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Computing angular distance matrix")
    angular_dist_matrix = compute_angular_distance_matrix(cosine_sim_matrix)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Angular distance matrix computed")
    generate_heatmap(angular_dist_matrix, 'Angular Distance', output_dir)
    print_top_similar_sentences(angular_dist_matrix, metadata, 'angular_distance', similar_output_dir)

    # Delete matrices and dataframes to free up memory
    del cosine_sim_matrix, angular_dist_matrix

    # Compute and save the Euclidean distance matrix
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Computing Euclidean distance matrix")
    euclidean_distances_matrix = euclidean_distances(embeddings)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Euclidean distance matrix computed")
    generate_heatmap(euclidean_distances_matrix, 'Euclidean Distance', output_dir)
    print_top_similar_sentences(euclidean_distances_matrix, metadata, 'euclidean_distance', similar_output_dir)

    # Compute and save the Manhattan distance matrix
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Computing Manhattan distance matrix")
    manhattan_distances_matrix = manhattan_distances(embeddings)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Manhattan distance matrix computed")
    generate_heatmap(manhattan_distances_matrix, 'Manhattan Distance', output_dir)
    print_top_similar_sentences(manhattan_distances_matrix, metadata, 'manhattan_distance', similar_output_dir)

    # Compute and save the Chebyshev distance matrix
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Computing Chebyshev distance matrix")
    chebyshev_distances_matrix = compute_other_distances(embeddings, 'chebyshev')
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Chebyshev distance matrix computed")
    generate_heatmap(chebyshev_distances_matrix, 'Chebyshev Distance', output_dir)
    print_top_similar_sentences(chebyshev_distances_matrix, metadata, 'chebyshev_distance', similar_output_dir)

    # Compute and save the Minkowski distance matrix
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Computing Minkowski distance matrix")
    minkowski_distances_matrix = compute_other_distances(embeddings, 'minkowski', p=3)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Minkowski distance matrix computed")
    generate_heatmap(minkowski_distances_matrix, 'Minkowski Distance', output_dir)
    print_top_similar_sentences(minkowski_distances_matrix, metadata, 'minkowski_distance', similar_output_dir)

    # Compute and save the Mahalanobis distance matrix
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Computing Mahalanobis distance matrix")
    mahalanobis_distances_matrix = compute_other_distances(embeddings, 'mahalanobis', VI=np.linalg.inv(np.cov(embeddings.T)))
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Mahalanobis distance matrix computed")
    generate_heatmap(mahalanobis_distances_matrix, 'Mahalanobis Distance', output_dir)
    print_top_similar_sentences(mahalanobis_distances_matrix, metadata, 'mahalanobis_distance', similar_output_dir)

    # Compute and save the Hamming distance matrix
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Computing Hamming distance matrix")
    hamming_distances_matrix = compute_other_distances(embeddings, 'hamming')
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Hamming distance matrix computed")
    generate_heatmap(hamming_distances_matrix, 'Hamming Distance', output_dir)
    print_top_similar_sentences(hamming_distances_matrix, metadata, 'hamming_distance', similar_output_dir)

    # Compute and save the Jaccard distance matrix
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Computing Jaccard distance matrix")
    jaccard_distances_matrix = compute_other_distances(embeddings, 'jaccard')
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Jaccard distance matrix computed")
    generate_heatmap(jaccard_distances_matrix, 'Jaccard Distance', output_dir)
    print_top_similar_sentences(jaccard_distances_matrix, metadata, 'jaccard_distance', similar_output_dir)

    # Delete matrices and dataframes to free up memory
    del euclidean_distances_matrix, manhattan_distances_matrix, chebyshev_distances_matrix, minkowski_distances_matrix, mahalanobis_distances_matrix, hamming_distances_matrix, jaccard_distances_matrix

    # Compute and save the Hausdorff distance matrix
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Computing Hausdorff distance matrix")
    hausdorff_dist_matrix = hausdorff_distance_matrix(embeddings)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Hausdorff distance matrix computed")
    generate_heatmap(hausdorff_dist_matrix, 'Hausdorff Distance', output_dir)
    print_top_similar_sentences(hausdorff_dist_matrix, metadata, 'hausdorff_distance', similar_output_dir)

    # Delete matrices and dataframes to free up memory
    del hausdorff_dist_matrix

    # Compute and save the max and min difference matrices
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Computing max and min difference matrices")
    max_diff_matrix, min_diff_matrix = compute_max_min_difference_matrices(embeddings)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Max and min difference matrices computed")
    generate_heatmap(max_diff_matrix, 'Max Differences', output_dir)
    generate_heatmap(min_diff_matrix, 'Min Differences', output_dir)

    # Delete matrices and dataframes to free up memory
    del max_diff_matrix, min_diff_matrix

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] All heatmaps and top similar sentences generated and saved successfully.")

if __name__ == '__main__':
    main()
