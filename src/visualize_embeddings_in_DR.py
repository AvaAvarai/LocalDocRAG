import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

# Load the embeddings data
def load_embeddings(file_path):
    df = pd.read_csv(file_path)
    return df

# Prepare data for visualization
def prepare_data_for_visualization(df):
    # Split the embeddings into separate columns
    embeddings = pd.DataFrame(df['embedding'].apply(eval).tolist(), index=df.index)  # Ensure 'embedding' is evaluated to a list
    embeddings.columns = [f'embedding_{i}' for i in range(embeddings.shape[1])]
    
    # Combine with the original dataframe
    df = df.drop(columns=['embedding'])
    df = pd.concat([df, embeddings], axis=1)
    
    # Convert 'pdf_source' to categorical codes for color mapping
    df['pdf_source_code'] = pd.Categorical(df['pdf_source']).codes
    
    return df, embeddings

# Create 3D plot
def create_3d_plot(df, reduced_embeddings, title):
    fig = px.scatter_3d(reduced_embeddings, 
                        x=0, y=1, z=2, 
                        color=df['pdf_source'],
                        labels={'color': 'PDF Source'},
                        title=title)
    fig.show()

if __name__ == "__main__":
    # Load the embeddings CSV
    embeddings_df = load_embeddings('embeddings.csv')
    
    # Prepare the data for visualization
    prepared_df, embeddings = prepare_data_for_visualization(embeddings_df)
    
    # PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(embeddings)
    create_3d_plot(prepared_df, pd.DataFrame(pca_result), 'PCA 3D Plot')
    
    # t-SNE
    tsne = TSNE(n_components=3, perplexity=30, n_iter=300)
    tsne_result = tsne.fit_transform(embeddings)
    create_3d_plot(prepared_df, pd.DataFrame(tsne_result), 't-SNE 3D Plot')
    
    # UMAP
    umap_model = umap.UMAP(n_components=3)
    umap_result = umap_model.fit_transform(embeddings)
    create_3d_plot(prepared_df, pd.DataFrame(umap_result), 'UMAP 3D Plot')
