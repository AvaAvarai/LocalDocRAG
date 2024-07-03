import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import directed_hausdorff
from concurrent.futures import ProcessPoolExecutor
import os
import time

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
                print(f"Max diff between {i} and {j}: {max_diff}, Min diff: {min_diff}")
            
            end_time = time.time()
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Batch {batch_start} to {min(batch_start + batch_size, n)} completed in {end_time - start_time:.2f} seconds")
    
    return max_diff_matrix, min_diff_matrix

# Function to compute the angular distance matrix
def compute_angular_distance_matrix(cosine_sim_matrix):
    return np.arccos(np.clip(cosine_sim_matrix, -1.0, 1.0))

# Function to generate and save heatmap
def generate_heatmap(matrix, title, output_dir, downscale_factor=10):
    # Downscale the matrix
    matrix = matrix[::downscale_factor, ::downscale_factor]

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

# Main function
def main():
    # Load the CSV file into a DataFrame
    csv_file = 'embeddings_all-MiniLM-L6-v2_139.30s.csv'
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading data from {csv_file}")
    df = pd.read_csv(csv_file)

    # Extract embeddings and metadata
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Extracting embeddings and metadata")
    embedding_columns = [str(i) for i in range(384)]
    embeddings = df[embedding_columns].values
    metadata = df[['sentence', 'document']]

    # Create output directory for heatmaps
    output_dir = 'heatmaps'
    os.makedirs(output_dir, exist_ok=True)

    # Compute and save the cosine similarity matrix
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Computing cosine similarity matrix")
    cosine_sim_matrix = cosine_similarity(embeddings)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Cosine similarity matrix computed")
    cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=metadata['sentence'], columns=metadata['sentence'])
    generate_heatmap(cosine_sim_df, 'Cosine Similarity', output_dir)

    # Compute and save the angular distance matrix
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Computing angular distance matrix")
    angular_dist_matrix = compute_angular_distance_matrix(cosine_sim_matrix)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Angular distance matrix computed")
    angular_dist_df = pd.DataFrame(angular_dist_matrix, index=metadata['sentence'], columns=metadata['sentence'])
    generate_heatmap(angular_dist_df, 'Angular Distance', output_dir)

    # Delete matrices and dataframes to free up memory
    del cosine_sim_matrix, cosine_sim_df, angular_dist_matrix, angular_dist_df

    # Compute and save the Euclidean distance matrix
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Computing Euclidean distance matrix")
    euclidean_distances_matrix = euclidean_distances(embeddings)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Euclidean distance matrix computed")
    euclidean_distances_df = pd.DataFrame(euclidean_distances_matrix, index=metadata['sentence'], columns=metadata['sentence'])
    generate_heatmap(euclidean_distances_df, 'Euclidean Distance', output_dir)

    # Delete matrices and dataframes to free up memory
    del euclidean_distances_matrix, euclidean_distances_df

    # Compute and save the Hausdorff distance matrix
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Computing Hausdorff distance matrix")
    hausdorff_dist_matrix = hausdorff_distance_matrix(embeddings)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Hausdorff distance matrix computed")
    hausdorff_distance_df = pd.DataFrame(hausdorff_dist_matrix, index=metadata['sentence'], columns=metadata['sentence'])
    generate_heatmap(hausdorff_distance_df, 'Hausdorff Distance', output_dir)

    # Delete matrices and dataframes to free up memory
    del hausdorff_dist_matrix, hausdorff_distance_df

    # Compute and save the max and min difference matrices
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Computing max and min difference matrices")
    max_diff_matrix, min_diff_matrix = compute_max_min_difference_matrices(embeddings)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Max and min difference matrices computed")
    max_diff_df = pd.DataFrame(max_diff_matrix, index=metadata['sentence'], columns=metadata['sentence'])
    min_diff_df = pd.DataFrame(min_diff_matrix, index=metadata['sentence'], columns=metadata['sentence'])
    generate_heatmap(max_diff_df, 'Max Differences', output_dir)
    generate_heatmap(min_diff_df, 'Min Differences', output_dir)

    # Delete matrices and dataframes to free up memory
    del max_diff_matrix, min_diff_matrix, max_diff_df, min_diff_df

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] All heatmaps generated and saved successfully.")

if __name__ == '__main__':
    main()
