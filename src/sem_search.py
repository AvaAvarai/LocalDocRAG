import os
import re
from pdf2docx import Converter
import docx
import pandas as pd
import time
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Convert PDF to Word Document using pdf2docx
def convert_pdf_to_docx(pdf_file, docx_file):
    cv = Converter(pdf_file)
    cv.convert(docx_file)
    cv.close()

# Convert Word Document to Text
def convert_docx_to_txt(docx_file, txt_file):
    doc = docx.Document(docx_file)
    with open(txt_file, 'w', encoding='utf-8') as f:
        for para in doc.paragraphs:
            f.write(para.text + '\n')

# Function to clean up redundant spacing
def clean_sentence(sentence):
    sentence = sentence.replace('\n', ' ').replace('\t', ' ')
    return re.sub(r'\s+', ' ', sentence).strip()

# Function to extract sentences from a text file
def extract_sentences_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    sentences = [clean_sentence(sentence) for sentence in text.split('. ') if clean_sentence(sentence)]
    return sentences

# Function to find all PDF files in a directory and its subdirectories
def find_all_pdfs(directory):
    pdf_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

# Function to find the most relevant sentences for a query using a specific model
def find_relevant_sentences(query, model, embeddings, top_n=5):
    query_embedding = model.encode([query])[0].reshape(1, -1)
    nbrs = NearestNeighbors(n_neighbors=min(top_n, len(embeddings)), metric='cosine').fit(embeddings)
    distances, indices = nbrs.kneighbors(query_embedding, n_neighbors=min(top_n, len(embeddings)))
    results = []
    for idx in indices[0]:
        results.append({
            'query': query,
            'sentence': df.iloc[idx]['sentence'],
            'document': df.iloc[idx]['document'],
            'distance': distances[0][indices[0].tolist().index(idx)]
        })
    return results

# Load the pre-trained models with a fixed seed
models = {
    'SBERT': SentenceTransformer('all-MiniLM-L6-v2'),
    'SciBERT': SentenceTransformer('allenai/scibert_scivocab_uncased')
}
np.random.seed(42)

# Directory containing the PDF documents
pdf_dir = 'ref'

# Output directory for DOCX and TXT files
output_dir = 'output'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Find all PDF files in the directory and its subdirectories
pdf_files = find_all_pdfs(pdf_dir)

# Convert PDFs to Word and then to Text
txt_files = []
for pdf_file in pdf_files:
    print(f"Converting file: {pdf_file}")
    relative_path = os.path.relpath(pdf_file, pdf_dir)
    docx_file = os.path.join(output_dir, relative_path.replace('.pdf', '.docx'))
    txt_file = os.path.join(output_dir, relative_path.replace('.pdf', '.txt'))

    # Ensure the output subdirectory exists
    os.makedirs(os.path.dirname(docx_file), exist_ok=True)
    os.makedirs(os.path.dirname(txt_file), exist_ok=True)

    convert_pdf_to_docx(pdf_file, docx_file)
    convert_docx_to_txt(docx_file, txt_file)
    txt_files.append(txt_file)

# Load and chunk documents into sentences
docs_sentences = []
doc_references = []

for txt_file in txt_files:
    print(f"Processing file: {txt_file}")
    sentences = extract_sentences_from_txt(txt_file)
    if sentences:
        docs_sentences.extend(sentences)
        doc_references.extend([txt_file] * len(sentences))
    else:
        print(f"No sentences extracted from file: {txt_file}")

print(f"Total number of sentences: {len(docs_sentences)}")

# Check if any sentences were extracted
if len(docs_sentences) == 0:
    raise ValueError("No sentences were extracted from the text files. Please check the content and extraction logic.")

# Create a DataFrame for easy handling
df = pd.DataFrame({'sentence': docs_sentences, 'document': doc_references})

# Generate embeddings for all sentences using all models
embeddings = {}
for model_name, model in models.items():
    print(f"Generating embeddings using {model_name}...")
    start_time = time.time()
    embeddings[model_name] = model.encode(df['sentence'].tolist(), show_progress_bar=True)
    end_time = time.time()
    training_time = end_time - start_time
    if len(embeddings[model_name]) == 0:
        raise ValueError(f"Embeddings for {model_name} are empty. Please check the model and the input data.")
    # Save embeddings to CSV with training time in the filename
    embeddings_df = pd.DataFrame(embeddings[model_name])
    embeddings_df['sentence'] = df['sentence']
    embeddings_df['document'] = df['document']
    filename = f'embeddings_{model_name}_{training_time:.2f}s.csv'
    embeddings_df.to_csv(filename, index=False)
    print(f"Embeddings for {model_name} saved to '{filename}' with training time {training_time:.2f} seconds.")

# List of example queries for testing
queries = [
    "What is the purpose of Visual Knowledge Discovery?",
    "How are visual machine learning models built?",
    "What are the challenges of implementing AI in high-risk scenarios?",
    "What are Parallel Coordinates used for?",
    "What are the difficulties with Visual Knowledge Discovery?"
]

# Process each query and save results to a CSV file
for query in queries:
    all_results = []
    for model_name, model in models.items():
        results = find_relevant_sentences(query, model, embeddings[model_name])
        for result in results:
            result['model'] = model_name
            all_results.append(result)
    results_df = pd.DataFrame(all_results)
    csv_filename = f'query_results_{queries.index(query) + 1}.csv'
    results_df.to_csv(csv_filename, index=False)
    print(f"Results for query '{query}' saved to '{csv_filename}'")
