import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import plotly.express as px

# Load the CSV file into a DataFrame
csv_file = 'embeddings_all-MiniLM-L6-v2_139.30s.csv'
df = pd.read_csv(csv_file)

# Extract embeddings and metadata
embedding_columns = [str(i) for i in range(384)]
embeddings = df[embedding_columns].values
metadata = df[['sentence', 'document']]

# Perform t-SNE
tsne = TSNE(n_components=2, metric='cosine', perplexity=30, n_iter=1000, random_state=42)
tsne_embeddings = tsne.fit_transform(embeddings)

# Create a DataFrame for t-SNE embeddings
tsne_df = pd.DataFrame(tsne_embeddings, columns=['Dim1', 'Dim2'])
tsne_df['sentence'] = df['sentence']
tsne_df['document'] = df['document']

# Plot t-SNE embeddings using plotly for interactivity
fig = px.scatter(
    tsne_df, x='Dim1', y='Dim2', color='document', hover_data=['sentence'],
    title='t-SNE Visualization of Embeddings', 
    color_discrete_sequence=px.colors.qualitative.Set3
)

fig.update_layout(
    title='t-SNE Visualization of Embeddings',
    xaxis_title='Dimension 1',
    yaxis_title='Dimension 2',
    legend_title='Document'
)

fig.show()
