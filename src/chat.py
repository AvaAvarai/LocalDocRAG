import os

# Suppress TensorFlow oneDNN optimization messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import tkinter as tk
from tkinter import scrolledtext, Spinbox
import pandas as pd
import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoTokenizer, pipeline
from sentence_transformers import util
import warnings
from huggingface_hub.file_download import logger as hf_logger
import logging

# Load ONNX model
scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", resume_download=True)
providers = [("DmlExecutionProvider", {"device_id": 0})]
session = ort.InferenceSession("scibert.onnx", providers=providers)

# Function to generate embeddings using ONNX model
def generate_embeddings_onnx(texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = scibert_tokenizer(batch_texts, return_tensors='np', padding=True, truncation=True, max_length=128)
        input_ids = inputs['input_ids'].astype(np.int64)
        attention_mask = inputs['attention_mask'].astype(np.int64)
        onnx_inputs = {session.get_inputs()[0].name: input_ids, session.get_inputs()[1].name: attention_mask}
        batch_embeddings = session.run(None, onnx_inputs)[0]
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)

# Load pre-saved embeddings from CSV
def load_embeddings_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    
    # Print the available columns to debug the column names issue
    print("Available columns in the CSV file:", df.columns.tolist())
    
    # Ensure the columns exist and handle missing columns
    if 'sentence' not in df.columns or 'embedding' not in df.columns or 'pdf_source' not in df.columns:
        raise KeyError("The CSV file does not contain the necessary columns: 'sentence', 'embedding', 'pdf_source'")
    
    embeddings = np.array(df['embedding'].apply(eval).tolist()).astype(np.float32)
    sentences = list(zip(df['sentence'], df['sentence'], df['pdf_source']))
    return sentences, embeddings

# Convert cosine similarity to angular distance
def cosine_to_angular_distance(cosine_similarity):
    return (1 - np.arccos(cosine_similarity) / np.pi)

# Find top K similar sentences
def find_top_k_similar(question, embeddings, sentences, k):
    question_embedding = generate_embeddings_onnx([question])[0]
    similarities = util.pytorch_cos_sim(torch.tensor(question_embedding), torch.tensor(embeddings))[0]
    angular_distances = cosine_to_angular_distance(similarities.numpy())
    top_k_indices = angular_distances.argsort()[-k:][::-1]
    top_k_sentences = [sentences[i] for i in top_k_indices]
    top_k_angular_distances = [angular_distances[i] for i in top_k_indices]
    return top_k_sentences, top_k_angular_distances

# Generate response to the question using top K similar sentences
def generate_response(question, top_k_sentences, original_texts):
    qa_pipeline = pipeline("question-answering", model="ixa-ehu/SciBERT-SQuAD-QuAC")
    context = []
    total_tokens = 0
    max_tokens = 512  # Assuming a reasonable limit for the context
    for sentence, _, source in top_k_sentences:
        idx = original_texts[source].index(sentence)
        start_idx = max(0, idx - 1)
        end_idx = min(len(original_texts[source]), idx + 2)
        context_sentences = original_texts[source][start_idx:end_idx]
        context_text = ' '.join(context_sentences)
        context_tokens = len(context_text.split())
        if total_tokens + context_tokens > max_tokens:
            break
        context.extend(context_sentences)
        total_tokens += context_tokens
    combined_context = ' '.join(context)
    answer = qa_pipeline(question=question, context=combined_context)
    return answer['answer']

# GUI Chat Interface
class ChatInterfaceApp:
    def __init__(self, root, sentences, embeddings, original_texts):
        self.root = root
        self.sentences = sentences
        self.embeddings = embeddings
        self.original_texts = original_texts
        self.k = 5  # Default value for top K

        self.root.title("CWU-VKD-LAB Question and Answer System")

        self.text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Helvetica", 12), padx=10, pady=10, state=tk.DISABLED, borderwidth=2, relief="solid")
        self.text_area.grid(row=0, column=0, padx=10, pady=10, columnspan=3, sticky="nsew")

        self.text_area.tag_config("user", foreground="blue")
        self.text_area.tag_config("system", foreground="green")
        self.text_area.tag_config("ai", foreground="purple")

        self.entry = tk.Entry(root, font=("Helvetica", 12))
        self.entry.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        self.send_button = tk.Button(root, text="Send", command=self.query, font=("Helvetica", 12))
        self.send_button.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        self.k_label = tk.Label(root, text="Top K:", font=("Helvetica", 12))
        self.k_label.grid(row=1, column=2, padx=10, pady=10, sticky="e")

        self.k_spinbox = Spinbox(root, from_=1, to=10, width=3, font=("Helvetica", 12), command=self.update_k)
        self.k_spinbox.grid(row=1, column=3, padx=10, pady=10, sticky="w")
        self.k_spinbox.delete(0, "end")
        self.k_spinbox.insert(0, self.k)  # Set default value
        
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=0)
        self.root.grid_columnconfigure(2, weight=0)
        self.root.grid_columnconfigure(3, weight=0)

    def update_k(self):
        self.k = int(self.k_spinbox.get())

    def query(self):
        question = self.entry.get()
        self.update_k()
        if question:
            self.text_area.config(state=tk.NORMAL)
            self.text_area.insert(tk.END, f"You: {question}\n", "user")
            self.entry.delete(0, tk.END)
            
            top_k_sentences, top_k_similarities = find_top_k_similar(question, self.embeddings, self.sentences, self.k)
            self.text_area.insert(tk.END, "\nSYSTEM: Top most similar sentences to the question:\n", "system")
            for i, (original, _, source) in enumerate(top_k_sentences):
                self.text_area.insert(tk.END, f"{i+1}. {original} (Source: {source}) (Similarity: {top_k_similarities[i]:.2f})\n", "system")
            
            response = generate_response(question, top_k_sentences, self.original_texts)
            self.text_area.insert(tk.END, f"\nAnswer: {response}\n", "ai")
            
            self.text_area.insert(tk.END, "\n" + "="*80 + "\n\n")
            self.text_area.config(state=tk.DISABLED)

def main():
    csv_file = 'embeddings.csv'  # Path to the pre-saved embeddings CSV file

    print("Loading embeddings from CSV...")
    sentences, embeddings = load_embeddings_from_csv(csv_file)
    
    original_texts = {source: [] for _, _, source in sentences}
    for original, cleaned, source in sentences:
        original_texts[source].append(original)
    
    print("Launching GUI...")
    root = tk.Tk()
    ChatInterfaceApp(root, sentences, embeddings, original_texts)
    root.mainloop()

if __name__ == "__main__":
    main()
