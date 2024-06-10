import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

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
    
    return df

# Create a color mapping for pdf_source
def assign_colors_to_sources(df):
    unique_sources = df['pdf_source'].unique()
    color_scale = px.colors.qualitative.Plotly
    color_mapping = {source: color_scale[i % len(color_scale)] for i, source in enumerate(unique_sources)}
    df['color'] = df['pdf_source'].map(color_mapping)
    return df, color_mapping

# Create Parallel Coordinates plot
def create_parallel_coordinates_plot(df, color_mapping):
    dimensions = [dict(label=col, values=df[col]) for col in df.columns if 'embedding_' in col]

    fig = go.Figure(data=go.Parcoords(
        line=dict(color=df['color'].apply(lambda c: int(c.lstrip('#'), 16))),
        dimensions=dimensions
    ))

    # Update the layout to add a legend
    for source, color in color_mapping.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            legendgroup=source,
            showlegend=True,
            name=source
        ))

    fig.show()

if __name__ == "__main__":
    # Load the embeddings CSV
    embeddings_df = load_embeddings('embeddings_word2vec.csv')
    
    # Prepare the data for visualization
    prepared_df = prepare_data_for_visualization(embeddings_df)
    
    # Assign colors to sources
    prepared_df, color_mapping = assign_colors_to_sources(prepared_df)
    
    # Create the Parallel Coordinates plot
    create_parallel_coordinates_plot(prepared_df, color_mapping)
