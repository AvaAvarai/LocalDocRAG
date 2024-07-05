import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import chebyshev, minkowski, mahalanobis, hamming, jaccard, directed_hausdorff
from sentence_transformers import SentenceTransformer

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

# Function to find top K similar sentences using NearestNeighbors for cosine similarity
def find_top_k_similar_sentences_cosine(query_embedding, embeddings, metadata, k=10):
    nbrs = NearestNeighbors(n_neighbors=min(k, len(embeddings)), metric='cosine').fit(embeddings)
    distances, indices = nbrs.kneighbors(query_embedding.reshape(1, -1), n_neighbors=min(k, len(embeddings)))
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        similarity_score = 1 - distance  # Convert cosine distance to similarity
        results.append((similarity_score, metadata['sentence'].iloc[idx], metadata['document'].iloc[idx]))
    return results

# Function to find top K similar sentences for other metrics
def find_top_k_similar_sentences(distance_matrix, metadata, k=10):
    n = distance_matrix.shape[0]
    similarities = []
    for i in range(n - 1):
        similarities.append((distance_matrix[-1, i], metadata['sentence'].iloc[i], metadata['document'].iloc[i]))
    similarities.sort(key=lambda x: x[0])
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
    # Combine the query embedding with the original embeddings
    embeddings_with_query = np.vstack((embeddings, query_embedding))
    metadata_with_query = pd.concat([metadata, pd.DataFrame([{'sentence': 'QUERY', 'document': 'query'}])], ignore_index=True)

    # Check for NaNs in embeddings
    if np.isnan(embeddings_with_query).any():
        print("NaN values detected in embeddings.")
        return

    # Create a Tkinter window
    root = tk.Tk()
    root.title("Top K Similar Sentences for Each Metric")

    # Create a frame for query input and controls
    control_frame = ttk.Frame(root)
    control_frame.pack(pady=10)

    # Query input
    query_label = ttk.Label(control_frame, text="Query:")
    query_label.grid(row=0, column=0, padx=5)
    query_entry = ttk.Entry(control_frame, width=50)
    query_entry.grid(row=0, column=1, padx=5)

    # K input
    k_label = ttk.Label(control_frame, text="K:")
    k_label.grid(row=0, column=2, padx=5)
    k_spinbox = ttk.Spinbox(control_frame, from_=1, to=100, width=5)
    k_spinbox.set(10)
    k_spinbox.grid(row=0, column=3, padx=5)

    # Query button
    def query_action():
        query = query_entry.get()
        k = int(k_spinbox.get())
        query_embedding = model.encode([query])
        visualize_sentences(metrics, embeddings, query_embedding, metadata, k)

    query_button = ttk.Button(control_frame, text="Query", command=query_action)
    query_button.grid(row=0, column=4, padx=5)

    # Create tabs for each metric
    tab_control = ttk.Notebook(root)
    tabs = {}
    for metric in metrics:
        tab = ttk.Frame(tab_control)
        tab_control.add(tab, text=metric.capitalize())
        tabs[metric] = tab

    tab_control.pack(expand=1, fill='both')

    def visualize_sentences(metrics, embeddings, query_embedding, metadata, k):
        # Generate a unique color for each document
        unique_docs = metadata['document'].unique()
        color_palette = sns.color_palette("husl", len(unique_docs)).as_hex()
        doc_color_map = {doc: color for doc, color in zip(unique_docs, color_palette)}

        for metric in metrics:
            if metric == 'cosine':
                top_k_similar = find_top_k_similar_sentences_cosine(query_embedding, embeddings, metadata, k)
            else:
                distance_matrix = generate_distance_matrix(np.vstack((embeddings, query_embedding)), metric)
                top_k_similar = find_top_k_similar_sentences(distance_matrix, metadata, k)

            # Create a color gradient for visualization
            colors = sns.color_palette("coolwarm", n_colors=k).as_hex()

            # Plot all sentences
            tab = tabs[metric]
            for widget in tab.winfo_children():
                widget.destroy()
            canvas = tk.Canvas(tab, width=800, height=400)
            canvas.pack()

            # Draw all sentences with a unique color per document
            all_x = np.linspace(50, 750, len(metadata))
            all_y = 200
            for i, (x, sentence, doc) in enumerate(zip(all_x, metadata['sentence'], metadata['document'])):
                color = doc_color_map[doc]
                canvas.create_oval(x - 5, all_y - 5, x + 5, all_y + 5, fill=color)

            # Highlight the similar points
            for i, (score, sentence, doc) in enumerate(top_k_similar):
                index = metadata[metadata['sentence'] == sentence].index[0]
                x = all_x[index]
                canvas.create_oval(x - 5, all_y - 5, x + 5, all_y + 5, fill=colors[i])

            # Display the sentences in a listbox
            listbox = tk.Listbox(tab)
            listbox.pack(fill='both', expand=1)
            for i, (score, sentence, doc) in enumerate(top_k_similar):
                listbox.insert(tk.END, f"Score: {score:.4f} - Sentence: {sentence} - Document: {doc}")

            # Add a legend
            legend_frame = ttk.Frame(tab)
            legend_frame.pack(pady=5)
            for doc, color in doc_color_map.items():
                legend_label = tk.Label(legend_frame, text=doc, bg=color, width=20)
                legend_label.pack(side=tk.LEFT, padx=5)

    root.mainloop()

# Load the CSV file into a DataFrame
csv_file = 'embeddings_all-MiniLM-L6-v2_420.44s.csv'
print(f"Loading data from {csv_file}")
df = pd.read_csv(csv_file)

# Extract embeddings and metadata
print(f"Extracting embeddings and metadata")
embedding_columns = [str(i) for i in range(384)]
embeddings = df[embedding_columns].values
metadata = df[['sentence', 'document']]

# Check for NaNs in embeddings
if np.isnan(embeddings).any():
    print("NaN values detected in embeddings.")
else:
    # Load the pre-trained model
    model_name = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)

    # Define metrics
    metrics = [
        'cosine', 'euclidean', 'manhattan', 'chebyshev', 'minkowski',
        'mahalanobis', 'hamming', 'jaccard', 'angular', 'hausdorff',
        'max_diff', 'min_diff'
    ]

    # Run the main GUI loop
    visualize_similar_sentences(metrics, embeddings, np.zeros_like(embeddings[0]), metadata, k=10)
