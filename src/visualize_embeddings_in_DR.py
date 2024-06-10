import warnings

# Suppress specific FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

import os

# Set environment variables to suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, SpectralEmbedding
import umap.umap_ as umap
from ast import literal_eval
from sklearn.preprocessing import StandardScaler

# Load the embeddings data
def load_embeddings(file_path):
    df = pd.read_csv(file_path)
    return df

# Prepare data for visualization
def prepare_data_for_visualization(df):
    # Split the embeddings into separate columns
    embeddings = pd.DataFrame(df['embedding'].apply(literal_eval).tolist(), index=df.index)
    embeddings.columns = [f'embedding_{i}' for i in range(embeddings.shape[1])]
    
    # Normalize the embeddings
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)
    
    # Combine with the original dataframe
    df = df.drop(columns=['embedding'])
    embeddings_df = pd.DataFrame(embeddings, columns=[f'embedding_{i}' for i in range(embeddings.shape[1])], index=df.index)
    df = pd.concat([df, embeddings_df], axis=1)
    
    # Convert 'pdf_source' to categorical codes for color mapping
    df['pdf_source_code'] = pd.Categorical(df['pdf_source']).codes
    
    return df, embeddings

# Create 3D plot and save to HTML
def create_3d_plot(df, reduced_embeddings, title, filename):
    reduced_embeddings = pd.DataFrame(reduced_embeddings, columns=['x', 'y', 'z'])
    reduced_embeddings['sentence'] = df['sentence']
    reduced_embeddings['pdf_source'] = df['pdf_source']
    
    fig = px.scatter_3d(reduced_embeddings, 
        x='x', y='y', z='z', 
        color='pdf_source',
        hover_data={'sentence': True},
        labels={'color': 'PDF Source'},
        title=title)

    # Add custom JavaScript for copying sentence to clipboard
    fig_html = fig.to_html(include_plotlyjs='cdn')
    custom_js = """
                <script>
                document.addEventListener('DOMContentLoaded', function() {
                    const plot = document.querySelector('.plotly-graph-div');
                    plot.on('plotly_click', function(data) {
                        const sentence = data.points[0].customdata[0];
                        navigator.clipboard.writeText(sentence).then(function() {
                            console.log('Text copied to clipboard');
                        }).catch(function(err) {
                            console.error('Could not copy text: ', err);
                        });
                    });
                });
                </script>
                """
    fig_html = fig_html.replace('</body>', custom_js + '</body>')

    with open(filename, 'w') as f:
        f.write(fig_html)

# Apply dimensionality reduction and create plots
def apply_dimensionality_reduction_and_plot(prepared_df, embeddings):
    # PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(embeddings)
    create_3d_plot(prepared_df, pca_result, 'PCA 3D Plot', 'pca_3d_plot.html')
    
    # t-SNE with explicitly set parameters
    tsne = TSNE(n_components=3, perplexity=30, n_iter=1000, metric='cosine', init='pca', learning_rate=200.0)
    tsne_result = tsne.fit_transform(embeddings)
    create_3d_plot(prepared_df, tsne_result, 't-SNE 3D Plot', 'tsne_3d_plot.html')
    
    # UMAP: Shows the best results for this purpose of semantic sentence clustering
    umap_model = umap.UMAP(n_components=3, metric='cosine')
    umap_result = umap_model.fit_transform(embeddings)
    create_3d_plot(prepared_df, umap_result, 'UMAP 3D Plot', 'umap_3d_plot.html')

    # Not enough memory for Spectral Embedding
    # spectral = SpectralEmbedding(n_components=3, affinity='nearest_neighbors')
    # spectral_result = spectral.fit_transform(embeddings)
    # create_3d_plot(prepared_df, spectral_result, 'Spectral Embedding 3D Plot', 'spectral_embedding_3d_plot.html')


if __name__ == "__main__":
    # Load the embeddings CSV
    embeddings_df = load_embeddings('embeddings_word2vec.csv')
    
    # Prepare the data for visualization
    prepared_df, embeddings = prepare_data_for_visualization(embeddings_df)
    
    # Apply dimensionality reduction and create plots
    apply_dimensionality_reduction_and_plot(prepared_df, embeddings)
