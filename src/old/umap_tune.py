import os

# Set environment variable
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import umap.umap_ as umap
from ast import literal_eval

# Load embeddings data
def load_embeddings(file_path):
    df = pd.read_csv(file_path)
    return df

# Prepare data for visualization
def prepare_data_for_visualization(df):
    embeddings = pd.DataFrame(df['embedding'].apply(literal_eval).tolist(), index=df.index)
    embeddings.columns = [f'embedding_{i}' for i in range(embeddings.shape[1])]
    df = df.drop(columns=['embedding'])
    df = pd.concat([df, embeddings], axis=1)
    return df, embeddings

# Load and prepare data
embeddings_df = load_embeddings('embeddings.csv')
prepared_df, embeddings = prepare_data_for_visualization(embeddings_df)

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("UMAP Hyperparameter Tuning"),
    
    html.Label("Number of Neighbors"),
    dcc.Slider(id='n_neighbors', min=2, max=50, step=1, value=15,
               marks={i: str(i) for i in range(2, 51, 2)}),
    
    html.Label("Minimum Distance"),
    dcc.Slider(id='min_dist', min=0.0, max=1.0, step=0.01, value=0.1,
               marks={i/10: str(i/10) for i in range(11)}),
    
    html.Label("Metric"),
    dcc.Dropdown(id='metric', options=[
        {'label': 'euclidean', 'value': 'euclidean'},
        {'label': 'manhattan', 'value': 'manhattan'},
        {'label': 'cosine', 'value': 'cosine'},
        {'label': 'correlation', 'value': 'correlation'},
        {'label': 'chebyshev', 'value': 'chebyshev'}
    ], value='euclidean'),
    
    dcc.Graph(id='umap-graph', style={'height': '80vh'})  # Adjust the height here
], style={'width': '90%', 'margin': '0 auto'})  # Center the content and adjust width

@app.callback(
    Output('umap-graph', 'figure'),
    [Input('n_neighbors', 'value'),
     Input('min_dist', 'value'),
     Input('metric', 'value')]
)
def update_umap_plot(n_neighbors, min_dist, metric):
    umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, n_components=3)
    umap_result = umap_model.fit_transform(embeddings)
    
    reduced_embeddings = pd.DataFrame(umap_result, columns=['x', 'y', 'z'])
    reduced_embeddings['sentence'] = prepared_df['sentence']
    reduced_embeddings['pdf_source'] = prepared_df['pdf_source']
    
    fig = px.scatter_3d(reduced_embeddings, 
                        x='x', y='y', z='z', 
                        color='pdf_source',
                        hover_data={'sentence': True},
                        labels={'color': 'PDF Source'},
                        title='UMAP 3D Plot')
    
    return fig

if __name__ == "__main__":
    app.run_server(debug=True)
