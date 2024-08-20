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

# Step 3: Custom Parallel Coordinates Plot with Reduced Spacing
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

    # Add a legend outside the plot
    fig.legend(handles=[plt.Line2D([0], [0], color=color_map[category], lw=2) for category in categories],
               labels=categories, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(output_file, dpi=300, bbox_inches='tight')  # Save as a high-resolution PNG file
    plt.close()
    print(f"Plot saved as {output_file}")

# Step 4: File picker to select the CSV file
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
        
        # Visualize the embeddings and save as PNG
        visualize_and_save_parallel_coordinates(embedding_df)
    else:
        print("No file selected. Exiting...")
