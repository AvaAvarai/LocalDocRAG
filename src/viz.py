import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import tkinter as tk
from tkinter import filedialog

# Step 1: Load the entire CSV file containing embeddings
def load_embeddings(filename):
    df = pd.read_csv(filename)
    return df

# Step 2: Prepare the data by excluding the 'sentence' column
def prepare_data_for_parallel_coordinates(df):
    numeric_columns = df.drop(columns=['sentence'])
    return numeric_columns

# Step 3: Parallel Coordinates Plot with Flat Alpha
def visualize_and_save_parallel_coordinates(df, output_file='parallel_coordinates.png'):
    plt.figure(figsize=(30, 10))  # Adjust figure size based on the number of dimensions

    # Use Matplotlib's color map to create distinct colors
    unique_classes = df['document'].unique()
    colors = plt.colormaps['tab10'](range(len(unique_classes)))

    # Create the parallel coordinates plot with a flat alpha
    parallel_coordinates(df, class_column='document', color=colors, alpha=0.1)
    
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
        # Load the entire dataset
        df = load_embeddings(filename)
        
        # Prepare the data for visualization
        embedding_df = prepare_data_for_parallel_coordinates(df)
        
        # Visualize and save the embeddings, splitting into multiple plots if necessary
        split_and_visualize_parallel_coordinates(embedding_df)
    else:
        print("No file selected. Exiting...")
