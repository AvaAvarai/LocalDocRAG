import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import directed_hausdorff
from concurrent.futures import ThreadPoolExecutor
import os

# Function to compute a single entry in the Hausdorff distance matrix
def compute_hausdorff_entry(i, j, X):
    return max(
        directed_hausdorff(X[i].reshape(1, -1), X[j].reshape(1, -1))[0], 
        directed_hausdorff(X[j].reshape(1, -1), X[i].reshape(1, -1))[0]
    )

# Function to compute the Hausdorff distance matrix in parallel
def hausdorff_distance_matrix(X):
    n = X.shape[0]
    hausdorff_matrix = np.zeros((n, n))
    
    with ThreadPoolExecutor() as executor:
        futures = {}
        for i in range(n):
            for j in range(n):
                if i <= j:
                    futures[(i, j)] = executor.submit(compute_hausdorff_entry, i, j, X)
        
        for (i, j), future in futures.items():
            hausdorff_matrix[i, j] = future.result()
            hausdorff_matrix[j, i] = hausdorff_matrix[i, j]
    
    return hausdorff_matrix

# Load the CSV file into a DataFrame
csv_file = 'embeddings_all-MiniLM-L6-v2_139.30s.csv'
df = pd.read_csv(csv_file)

# Extract embeddings and metadata
embedding_columns = [str(i) for i in range(384)]
embeddings = df[embedding_columns].values
metadata = df[['sentence', 'document']]

# Compute the distance matrices
cosine_sim_matrix = cosine_similarity(embeddings)
euclidean_distances_matrix = euclidean_distances(embeddings)
hausdorff_dist_matrix = hausdorff_distance_matrix(embeddings)

# Create DataFrames for the distance matrices to use with seaborn
cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=metadata['sentence'], columns=metadata['sentence'])
euclidean_distances_df = pd.DataFrame(euclidean_distances_matrix, index=metadata['sentence'], columns=metadata['sentence'])
hausdorff_distance_df = pd.DataFrame(hausdorff_dist_matrix, index=metadata['sentence'], columns=metadata['sentence'])

# Create output directory for heatmaps
output_dir = 'heatmaps'
os.makedirs(output_dir, exist_ok=True)

# Plot the distance matrices as heatmaps and save the images
distance_matrices = {
    'Cosine Similarity': cosine_sim_df,
    'Euclidean Distance': euclidean_distances_df,
    'Hausdorff Distance': hausdorff_distance_df
}

for name, dist_df in distance_matrices.items():
    plt.figure(figsize=(12, 10))
    sns.heatmap(dist_df, cmap='viridis', cbar=True)
    plt.title(f'{name} Heatmap of Embeddings')
    plt.savefig(os.path.join(output_dir, f'{name}_heatmap.png'))
    plt.close()
