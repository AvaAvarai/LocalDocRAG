import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file into a DataFrame
csv_file = 'embeddings_all-MiniLM-L6-v2_139.30s.csv'
df = pd.read_csv(csv_file)

# Extract embeddings and metadata
embedding_columns = [str(i) for i in range(384)]
embeddings = df[embedding_columns].values
metadata = df[['sentence', 'document']]

# Perform PCA
pca = PCA()
pca.fit(embeddings)

# Plot explained variance to determine the number of components to retain
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Number of Principal Components')
plt.grid()
plt.show()

# Choose the number of components that explain a desired amount of variance (e.g., 95%)
desired_variance = 0.95
cumulative_variance = pca.explained_variance_ratio_.cumsum()
num_components = next(i for i, total_var in enumerate(cumulative_variance) if total_var >= desired_variance) + 1

print(f"Number of components to retain {desired_variance*100}% variance: {num_components}")

# Perform PCA with the selected number of components
pca = PCA(n_components=num_components)
reduced_embeddings = pca.fit_transform(embeddings)

# Create a DataFrame for the reduced embeddings
reduced_df = pd.DataFrame(reduced_embeddings, columns=[f"PC{i+1}" for i in range(num_components)])
reduced_df['sentence'] = df['sentence']
reduced_df['document'] = df['document']

# Map documents to unique numeric IDs
unique_documents = reduced_df['document'].unique()
document_id_map = {doc: idx for idx, doc in enumerate(unique_documents)}
reduced_df['document_id'] = reduced_df['document'].map(document_id_map)

# Select the components to visualize (all components after PCA)
pc_columns = [f"PC{i+1}" for i in range(num_components)]

# Create the parallel coordinates plot
fig = px.parallel_coordinates(
    reduced_df,
    dimensions=pc_columns,
    color='document_id',
    color_continuous_scale=px.colors.qualitative.Plotly,
    labels={f"PC{i+1}": f"PC{i+1}" for i in range(num_components)},
    title="Parallel Coordinates Plot of Reduced Sentence Embeddings"
)

# Show the plot
fig.show()
