import os

# Set environment variables to suppress specific TensorFlow messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, SpectralEmbedding
import umap.umap_ as umap
from ast import literal_eval
from concurrent.futures import ThreadPoolExecutor

# Load the embeddings data
def load_embeddings(file_path):
    df = pd.read_csv(file_path)
    return df

# Prepare data for visualization
def prepare_data_for_visualization(df):
    embeddings = pd.DataFrame(df['embedding'].apply(literal_eval).tolist(), index=df.index)
    embeddings.columns = [f'embedding_{i}' for i in range(embeddings.shape[1])]
    df = df.drop(columns=['embedding'])
    df = pd.concat([df, embeddings], axis=1)
    df['pdf_source_code'] = pd.Categorical(df['pdf_source']).codes
    return df, embeddings

# Create 3D plot and save to HTML
def create_3d_plot(df, reduced_embeddings, title, filename):
    reduced_embeddings = pd.DataFrame(reduced_embeddings, columns=['x', 'y', 'z'])
    reduced_embeddings['sentence'] = df['sentence']
    reduced_embeddings['pdf_source'] = df['pdf_source']
    
    fig = px.scatter_3d(reduced_embeddings, x='x', y='y', z='z', color='pdf_source', hover_data={'sentence': True}, labels={'color': 'PDF Source'}, title=title)
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
    def pca_reduction():
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(embeddings)
        create_3d_plot(prepared_df, pca_result, 'PCA 3D Plot', 'pca_3d_plot.html')

    def tsne_reduction():
        tsne = TSNE(n_components=3, perplexity=30, n_iter=1000, metric='cosine', init='pca')
        tsne_result = tsne.fit_transform(embeddings)
        create_3d_plot(prepared_df, tsne_result, 't-SNE 3D Plot', 'tsne_3d_plot.html')

    def umap_reduction():
        umap_model = umap.UMAP(n_components=3, metric='cosine', n_jobs=-1)
        umap_result = umap_model.fit_transform(embeddings)
        create_3d_plot(prepared_df, umap_result, 'UMAP 3D Plot', 'umap_3d_plot.html')

    def spectral_reduction():
        spectral = SpectralEmbedding(n_components=3, affinity='nearest_neighbors', n_jobs=-1)
        spectral_result = spectral.fit_transform(embeddings)
        create_3d_plot(prepared_df, spectral_result, 'Spectral Embedding 3D Plot', 'spectral_embedding_3d_plot.html')

    with ThreadPoolExecutor() as executor:
        executor.submit(pca_reduction)
        executor.submit(tsne_reduction)
        executor.submit(umap_reduction)
        executor.submit(spectral_reduction)

if __name__ == "__main__":
    embeddings_df = load_embeddings('embeddings.csv')
    prepared_df, embeddings = prepare_data_for_visualization(embeddings_df)
    apply_dimensionality_reduction_and_plot(prepared_df, embeddings)
