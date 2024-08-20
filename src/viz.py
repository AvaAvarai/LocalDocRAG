import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import tkinter as tk
from tkinter import filedialog
import seaborn as sns

# Step 1: Load the CSV file containing embeddings
def load_embeddings(filename):
    df = pd.read_csv(filename)
    return df

# Step 2: Prepare the data
def prepare_data_for_parallel_coordinates(df):
    # Exclude the 'sentence' column from the DataFrame
    numeric_columns = df.drop(columns=['sentence'])
    return numeric_columns

# Step 3: Standard Parallel Coordinates Plot with Distinct Colors
def visualize_and_save_parallel_coordinates(df, output_file='parallel_coordinates.png'):
    plt.figure(figsize=(30, 10))  # Adjust figure size based on the number of dimensions

    # Create a custom color palette based on the number of unique classes
    unique_classes = df['document'].unique()
    palette = sns.color_palette("husl", len(unique_classes))

    # Create the parallel coordinates plot with distinct colors
    parallel_coordinates(df, class_column='document', color=palette, alpha=0.5)
    
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to avoid clipping
    
    plt.savefig(output_file, dpi=300)  # Save as a high-resolution PNG file
    plt.close()
    print(f"Plot saved as {output_file}")

# Step 4: Split the plot into parts if necessary
def split_and_visualize_parallel_coordinates(df, output_file_prefix='parallel_coordinates', dims_per_plot=50):
    num_plots = (len(df.columns) - 1 + dims_per_plot - 1) // dims_per_plot  # Calculate the number of plots required
    for i in range(num_plots):
        start_dim = i * dims_per_plot
        end_dim = min((i + 1) * dims_per_plot, len(df.columns) - 1)
        df_subset = df.iloc[:, start_dim:end_dim].copy()  # Copy the subset to avoid slice warnings
        df_subset['document'] = df['document']  # Ensure the 'document' column is included
        output_file = f'{output_file_prefix}_part_{i+1}.png'
        visualize_and_save_parallel_coordinates(df_subset, output_file)
        print(f"Part {i+1} saved as {output_file}")

# Step 5: File picker to select the CSV file
def select_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    root.destroy()
    return file_path

if __name__ == "__main__":
    # Open file picker to select the CSV file
    filename = select_file()

    if filename:  # Proceed if a file was selected
        # Load embeddings from the CSV file
        df = load_embeddings(filename)
        
        # Prepare the data for visualization
        embedding_df = prepare_data_for_parallel_coordinates(df)
        
        # Visualize and save the embeddings, splitting into multiple plots if necessary
        split_and_visualize_parallel_coordinates(embedding_df)
    else:
        print("No file selected. Exiting...")
