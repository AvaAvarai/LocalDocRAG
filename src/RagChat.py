import os
import re
import numpy as np
import pandas as pd
import random
import torch
import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
from threading import Thread
from sentence_transformers import SentenceTransformer
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever
from transformers import BartTokenizer, BartForConditionalGeneration
from docx import Document
import fitz  # PyMuPDF
import markdown

# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Function to convert DOCX files to text
def convert_docx_to_text(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

# Function to convert MD files to text
def convert_md_to_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return markdown.markdown(text)

# Function to convert PDF files to text using PyMuPDF
def convert_pdf_to_text(file_path):
    doc = fitz.open(file_path)
    full_text = []
    for page in doc:
        full_text.append(page.get_text())
    return '\n'.join(full_text)

# Convert files to documents
def convert_files(directory):
    docs = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith('.pdf'):
                text = convert_pdf_to_text(file_path)
            elif file.endswith('.docx'):
                text = convert_docx_to_text(file_path)
            elif file.endswith('.md'):
                text = convert_md_to_text(file_path)
            else:
                continue
            doc = {"content": text, "meta": {"name": file}}
            docs.append(doc)
    return docs

# Function to save embeddings to a CSV file
def save_embeddings(embeddings, sentences, references, filename):
    df = pd.DataFrame(embeddings)
    df['sentence'] = sentences
    df['document'] = references
    df.to_csv(filename, index=False)

# Function to load embeddings from a CSV file
def load_embeddings(filename):
    df = pd.read_csv(filename)
    embeddings = df.iloc[:, :-2].values
    sentences = df['sentence'].tolist()
    references = df['document'].tolist()
    return embeddings, sentences, references

# Function to generate and save embeddings
def generate_and_save_embeddings():
    global sentence_embeddings, sentences, references
    # Convert and index documents
    docs = convert_files('ref')
    doc_texts = [doc['content'] for doc in docs]
    doc_names = [doc['meta']['name'] for doc in docs]

    # Split documents into sentences and generate embeddings
    sentences = []
    references = []
    for text, name in zip(doc_texts, doc_names):
        for sentence in re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text):
            if sentence.strip():
                sentences.append(sentence.strip())
                references.append(name)

    sentence_embeddings = sentence_model.encode(sentences, show_progress_bar=True)

    # Save embeddings to a file
    save_file = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if save_file:
        save_embeddings(sentence_embeddings, sentences, references, save_file)
        messagebox.showinfo("Info", "Embeddings saved successfully!")

# Function to load embeddings from a file
def load_existing_embeddings():
    global sentence_embeddings, sentences, references
    load_file = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if load_file:
        sentence_embeddings, sentences, references = load_embeddings(load_file)
        messagebox.showinfo("Info", "Embeddings loaded successfully!")

# Function to find similar passages using dot product
def find_similar_passages(query, top_k=5):
    query_embedding = sentence_model.encode([query])[0]
    scores = np.dot(sentence_embeddings, query_embedding)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [(sentences[i], references[i], scores[i]) for i in top_indices]

# Initialize the BART model for summarization
bart_model_name = 'facebook/bart-large-cnn'
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name)

# Function to generate summary with BART
def generate_summary(text):
    inputs = bart_tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = bart_model.generate(
        inputs['input_ids'],
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
        do_sample=False  # Disable sampling for determinism
    )
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Function to process the query and return the answer with citations
def process_query(query):
    similar_passages = find_similar_passages(query)
    combined_text = ' '.join([f"{text} (Source: {source})" for text, source, score in similar_passages])
    summary = generate_summary(combined_text)
    return summary

# Function to handle the query submission
def submit_query():
    query = query_entry.get()
    chat_history.config(state=tk.NORMAL)
    chat_history.insert(tk.END, f"You: {query}\n", 'user')
    chat_history.insert(tk.END, "Bot: Generating response, please wait...\n", 'bot_loading')
    chat_history.yview(tk.END)
    
    def run_query():
        response = process_query(query)
        chat_history.config(state=tk.NORMAL)
        chat_history.delete('end-2l', 'end-1l')  # Remove the "Generating response" line
        chat_history.insert(tk.END, f"Bot: {response}\n", 'bot')
        chat_history.config(state=tk.DISABLED)
        query_entry.delete(0, tk.END)
    
    Thread(target=run_query).start()

# Function to show the starting menu
def show_start_menu():
    menu = tk.Toplevel(root)
    menu.title("Start Menu")

    tk.Label(menu, text="Choose an option:").pack(pady=10)

    generate_button = tk.Button(menu, text="Generate New Embeddings", command=lambda: [generate_and_save_embeddings(), menu.destroy()])
    generate_button.pack(pady=5)

    load_button = tk.Button(menu, text="Load Existing Embeddings", command=lambda: [load_existing_embeddings(), menu.destroy()])
    load_button.pack(pady=5)

    menu.transient(root)
    menu.grab_set()
    root.wait_window(menu)

# Initialize Sentence Transformer model
model_name = 'multi-qa-mpnet-base-dot-v1'
sentence_model = SentenceTransformer(model_name)

# Set up the GUI
root = tk.Tk()
root.title("Document QA Chatbot")

# Chat history text area
chat_history = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled')
chat_history.tag_config('user', foreground='blue')
chat_history.tag_config('bot', foreground='green')
chat_history.tag_config('bot_loading', foreground='lightgray')
chat_history.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

# Query entry
query_entry = tk.Entry(root, width=80)
query_entry.grid(row=1, column=0, padx=10, pady=10)

# Submit button
submit_button = tk.Button(root, text="Submit", command=submit_query)
submit_button.grid(row=1, column=1, padx=10, pady=10)

# Show start menu
show_start_menu()

# Run the application
root.mainloop()
