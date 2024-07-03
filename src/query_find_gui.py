import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from scipy.spatial.distance import chebyshev, minkowski, mahalanobis, hamming, jaccard, directed_hausdorff
from sentence_transformers import SentenceTransformer

# Function to normalize embeddings
def normalize_embeddings(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norms

# Function to compute the angular distance matrix
def compute_angular_distance_matrix(cosine_sim_matrix):
    return np.arccos(np.clip(cosine_sim_matrix, -1.0, 1.0))

# Function to compute the Hausdorff distance
def compute_hausdorff_entry(args):
    i, j, X = args
    return i, j, max(
        directed_hausdorff(X[i].reshape(1, -1), X[j].reshape(1, -1))[0], 
        directed_hausdorff(X[j].reshape(1, -1), X[i].reshape(1, -1))[0]
    )

# Function to compute the Hausdorff distance matrix
def hausdorff_distance_matrix(X):
    n = X.shape[0]
    hausdorff_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            distance = max(
                directed_hausdorff(X[i].reshape(1, -1), X[j].reshape(1, -1))[0], 
                directed_hausdorff(X[j].reshape(1, -1), X[i].reshape(1, -1))[0]
            )
            hausdorff_matrix[i, j] = distance
            hausdorff_matrix[j, i] = distance
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
def compute_max_min_entry(args):
    i, j, X = args
    diffs = np.abs(X[i] - X[j])
    max_diff = np.max(diffs)
    min_diff = np.min(diffs)
    return i, j, max_diff, min_diff

# Function to compute the max and min difference matrices
def compute_max_min_difference_matrices(X):
    n = X.shape[0]
    max_diff_matrix = np.zeros((n, n))
    min_diff_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            diffs = np.abs(X[i] - X[j])
            max_diff = np.max(diffs)
            min_diff = np.min(diffs)
            max_diff_matrix[i, j] = max_diff_matrix[j, i] = max_diff
            min_diff_matrix[i, j] = min_diff_matrix[j, i] = min_diff
    return max_diff_matrix, min_diff_matrix

# Function to find top K similar sentences
def find_top_k_similar_sentences(matrix, metadata, k=10):
    n = matrix.shape[0]
    similarities = []
    for i in range(n - 1):
        similarities.append((matrix[-1, i], metadata['sentence'][i]))
    similarities.sort(key=lambda x: x[0], reverse=True)
    return similarities[:k]

# Function to generate the distance matrix
def generate_distance_matrix(embeddings, metric, **kwargs):
    if metric == 'cosine':
        return cosine_similarity(embeddings)
    elif metric == 'euclidean':
        return euclidean_distances(embeddings)
    elif metric == 'manhattan':
        return manhattan_distances(embeddings)
    elif metric == 'chebyshev':
        return compute_other_distances(embeddings, 'chebyshev')
    elif metric == 'minkowski':
        return compute_other_distances(embeddings, 'minkowski', p=3)
    elif metric == 'mahalanobis':
        return compute_other_distances(embeddings, 'mahalanobis', VI=np.linalg.inv(np.cov(embeddings.T)))
    elif metric == 'hamming':
        return compute_other_distances(embeddings, 'hamming')
    elif metric == 'jaccard':
        return compute_other_distances(embeddings, 'jaccard')
    elif metric == 'angular':
        cosine_sim_matrix = cosine_similarity(embeddings)
        return compute_angular_distance_matrix(cosine_sim_matrix)
    elif metric == 'hausdorff':
        return hausdorff_distance_matrix(embeddings)
    elif metric == 'max_diff':
        max_diff_matrix, _ = compute_max_min_difference_matrices(embeddings)
        return max_diff_matrix
    elif metric == 'min_diff':
        _, min_diff_matrix = compute_max_min_difference_matrices(embeddings)
        return min_diff_matrix
    else:
        raise ValueError(f"Unknown metric: {metric}")

# Function to visualize top K similar sentences in a 1D space for each metric
def visualize_similar_sentences(metrics, embeddings, query_embedding, metadata, k=10):
    # Normalize embeddings
    embeddings = normalize_embeddings(embeddings)
    query_embedding = normalize_embeddings(query_embedding.reshape(1, -1))

    # Combine the query embedding with the original embeddings
    embeddings_with_query = np.vstack((embeddings, query_embedding))
    metadata_with_query = metadata.append({'sentence': 'QUERY', 'document': 'query'}, ignore_index=True)

    # Create a Tkinter window
    root = tk.Tk()
    root.title("Top K Similar Sentences for Each Metric")

    # Create tabs for each metric
    tab_control = ttk.Notebook(root)
    tabs = {}
    for metric in metrics:
        tab = ttk.Frame(tab_control)
        tab_control.add(tab, text=metric.capitalize())
        tabs[metric] = tab

    tab_control.pack(expand=1, fill='both')

    for metric in metrics:
        distance_matrix = generate_distance_matrix(embeddings_with_query, metric)
        top_k_similar = find_top_k_similar_sentences(distance_matrix, metadata_with_query, k)

        # Plot the similar sentences in 1D space
        tab = tabs[metric]
        canvas = tk.Canvas(tab, width=800, height=400)
        canvas.pack()

        # Draw the query point
        query_x = 400
        query_y = 200
        canvas.create_oval(query_x - 5, query_y - 5, query_x + 5, query_y + 5, fill='blue')

        # Draw the similar points
        for i, (score, sentence) in enumerate(top_k_similar):
            x = query_x + (i + 1) * 30
            y = query_y
            canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill='red')
            canvas.create_text(x, y + 10, text=f"{score:.4f}", anchor='n')

        # Display the sentences in a listbox
        listbox = tk.Listbox(tab)
        listbox.pack(fill='both', expand=1)
        for score, sentence in top_k_similar:
            listbox.insert(tk.END, f"Score: {score:.4f} - Sentence: {sentence}")

    root.mainloop()

# Load the CSV file into a DataFrame
csv_file = 'embeddings_SBERT_382.62s.csv'
print(f"Loading data from {csv_file}")
df = pd.read_csv(csv_file)

# Extract embeddings and metadata
print(f"Extracting embeddings and metadata")
embedding_columns = [str(i) for i in range(384)]
embeddings = df[embedding_columns].values
metadata = df[['sentence', 'document']]

# Normalize embeddings
print(f"Normalizing embeddings")
embeddings = normalize_embeddings(embeddings)

# Load the pre-trained model
model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

# Query input
query = input("Enter your query: ")
query_embedding = model.encode([query])

# Define metrics
metrics = [
    'cosine', 'euclidean', 'manhattan', 'chebyshev', 'minkowski',
    'mahalanobis', 'hamming', 'jaccard', 'angular', 'hausdorff',
    'max_diff', 'min_diff'
]

# Visualize similar sentences
visualize_similar_sentences(metrics, embeddings, query_embedding, metadata, k=10)
