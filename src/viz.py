import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
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

# Step 3: Visualize the embeddings using Matplotlib and save as PNG
def visualize_and_save_parallel_coordinates(df, output_file='parallel_coordinates.png'):
    plt.figure(figsize=(30, 15))  # Adjust figure size to accommodate more dimensions
    
    # Use 'document' column as class_column for coloring
    parallel_coordinates(df, class_column='document', colormap='viridis', linewidth=0.5)
    
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to avoid clipping
    
    plt.savefig(output_file, dpi=300)  # Save as a high-resolution PNG file
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
