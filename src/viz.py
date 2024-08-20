import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Step 1: Load the CSV file containing embeddings
def load_embeddings(filename):
    df = pd.read_csv(filename)
    return df

# Step 2: Prepare the data
def prepare_data_for_parallel_coordinates(df):
    # Exclude the 'sentence' column from the DataFrame
    numeric_columns = df.drop(columns=['sentence'])
    return numeric_columns

# Step 3: Custom Parallel Coordinates Plot with Spaced-Out Axes
def visualize_and_save_parallel_coordinates(df, output_file='parallel_coordinates.png'):
    num_vars = len(df.columns) - 1  # Exclude 'document' column
    fig, axes = plt.subplots(1, num_vars, sharey=False, figsize=(min(num_vars * 1.5, 200), 10))

    # Define the color mapping for each document category
    categories = df['document'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))
    color_map = dict(zip(categories, colors))

    for i, col in enumerate(df.columns[:-1]):  # Exclude the last column 'document'
        for category in categories:
            subset = df[df['document'] == category]
            axes[i].plot([i] * len(subset), subset[col], marker='o', linestyle='-', color=color_map[category], alpha=0.5)
        axes[i].set_title(col)
        axes[i].set_xticks([i])
    
    # Reduce the spacing between axes
    plt.subplots_adjust(wspace=0.2)  # Adjust the width between subplots to reduce size

    # Create unique legend handles
    handles = []
    labels = []
    for category in categories:
        handles.append(plt.Line2D([0], [0], color=color_map[category], lw=2))
        labels.append(category)
    
    # Add a legend outside the plot
    fig.legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(output_file, dpi=300, bbox_inches='tight')  # Save as a high-resolution PNG file
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
