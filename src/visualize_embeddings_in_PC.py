import pandas as pd
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
    
    # Convert 'pdf_source' to categorical codes for color mapping
    df['pdf_source_code'] = pd.Categorical(df['pdf_source']).codes
    
    return df

# Create Parallel Coordinates plot
def create_parallel_coordinates_plot(df):
    # Create the plot
    fig = px.parallel_coordinates(df,
                                  dimensions=[col for col in df.columns if 'embedding_' in col],
                                  color='pdf_source_code',
                                  labels={col: col for col in df.columns if 'embedding_' in col},
                                  color_continuous_scale=px.colors.diverging.Tealrose,
                                  color_continuous_midpoint=df['pdf_source_code'].max() / 2)
    fig.show()

if __name__ == "__main__":
    # Load the embeddings CSV
    embeddings_df = load_embeddings('embeddings.csv')
    
    # Prepare the data for visualization
    prepared_df = prepare_data_for_visualization(embeddings_df)
    
    # Create the Parallel Coordinates plot
    create_parallel_coordinates_plot(prepared_df)
