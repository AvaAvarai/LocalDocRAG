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

# Function to clean up redundant spacing, remove newlines and tabs, and remove punctuation
def clean_sentence(sentence, seen_sentences):
    sentence = sentence.encode('utf-8', 'ignore').decode('utf-8')
    sentence = sentence.replace('\n', ' ').replace('\t', ' ').replace('  ', ' ')
    if sentence not in seen_sentences:
        seen_sentences.add(sentence)
        return sentence
    return None

# Function to remove unwanted references from text
def remove_references(text):
    text = re.sub(r'\b(Fig|Table|Fig\.|Table\.|Figure)\s?\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(Fig|Table|Figure)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(Sect|Sec|Section|Sect\.|Sec\.)\s?\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(Sect|Sec|Section|Sect\.|Sec\.)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(depicted in|shown in|seen in|illustrated in|presented in|described in)\s?\b.*\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(depicted in|shown in|seen in|illustrated in|presented in|described in)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\(\d+\)', '', text)
    text = text.replace('  ', ' ')
    return text

# Remove sentences which are too short after cleaning and filtering ignoring 1 and 2 letter words
def filter_too_short(text, min_length=5):
    words_in_text = text.split(' ')
    copy = words_in_text.copy()
    for word in words_in_text:
        if len(word) <= 2:
            copy.remove(word)
    return len(copy) >= min_length

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

# Function to clean and filter sentences
def clean_and_filter_sentences(text):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    seen_sentences = set()
    cleaned_sentences = []
    for sentence in sentences:
        cleaned_sentence = clean_sentence(sentence, seen_sentences)
        if cleaned_sentence:
            cleaned_sentence = remove_references(cleaned_sentence)
            if cleaned_sentence and filter_too_short(cleaned_sentence):
                cleaned_sentences.append(cleaned_sentence)
    return cleaned_sentences

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

    # Split documents into sentences, clean, and generate embeddings
    sentences = []
    references = []
    for text, name in zip(doc_texts, doc_names):
        cleaned_sentences = clean_and_filter_sentences(text)
        for sentence in cleaned_sentences:
            sentences.append(sentence)
            references.append(name)

    # Check if there are any sentences to encode
    if not sentences:
        messagebox.showwarning("Warning", "No documents or valid sentences found. Please load documents and try again.")
        return

    # Generate sentence embeddings
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

# Function to find similar passages using cosine similarity
def find_similar_passages(query, top_k=5, threshold=0.65):
    query_embedding = sentence_model.encode([query])[0]
    query_embedding_norm = np.linalg.norm(query_embedding)
    sentence_embeddings_norm = np.linalg.norm(sentence_embeddings, axis=1)
    cosine_similarities = np.dot(sentence_embeddings, query_embedding) / (sentence_embeddings_norm * query_embedding_norm)
    
    # Sort indices by similarity score in descending order
    sorted_indices = np.argsort(cosine_similarities)[::-1]
    top_passages = [(sentences[i], references[i], cosine_similarities[i]) for i in sorted_indices if cosine_similarities[i] >= threshold]
    
    # Return top_k results
    return top_passages[:top_k]

# Initialize the BART model for summarization
bart_model_name = 'facebook/bart-large-cnn'
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name)

# Function to generate summary with BART
def generate_summary(text):
    inputs = bart_tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = bart_model.generate(
        inputs['input_ids'],
        max_length=300,  # Adjusted for relative length limits
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
        do_sample=False  # Disable sampling for determinism
    )
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Function to generate sub-answer for each similar sentence
def generate_sub_answer(sentence):
    inputs = bart_tokenizer([sentence], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = bart_model.generate(
        inputs['input_ids'],
        max_length=150,  # Adjusted for relative length limits
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
        do_sample=False  # Disable sampling for determinism
    )
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Function to process the query and return the answer with citations
def process_query(query):
    threshold = similarity_threshold.get() / 100.0
    k_value = k_dial.get()
    similar_passages = find_similar_passages(query, top_k=k_value, threshold=threshold)
    if not similar_passages:
        return "Sorry, the bot cannot answer this question.", "", []

    # Extracting the most relevant parts of the similar sentences
    combined_text = ' '.join([text for text, _, _ in similar_passages])

    # Summarize the combined relevant parts using BART
    summary = generate_summary(combined_text)
    detailed_info = "\n\n".join([f"Similar sentence: {text}\n[Source: {source}]\nSimilarity score: {score:.2f}" for text, source, score in similar_passages])
    sources = ", ".join({source for _, source, _ in similar_passages})

    return summary, detailed_info, sources

# Function to handle the query submission
def submit_query(event=None):
    query = query_entry.get()
    chat_history.config(state=tk.NORMAL)
    chat_history.insert(tk.END, f"You: {query}\n", 'user')
    chat_history.insert(tk.END, "Bot: Generating response, please wait...\n", 'bot_loading')
    query_entry.delete(0, tk.END)
    chat_history.yview(tk.END)
    
    def run_query():
        summary, detailed_info, sources = process_query(query)
        chat_history.config(state=tk.NORMAL)
        chat_history.delete('end-2l', 'end-1l')  # Remove the "Generating response" line
        chat_history.insert(tk.END, f"Bot: {summary}\n", 'bot')
        if sources:
            chat_history.insert(tk.END, f"Sources: {sources}\n", 'bot_sources')
        if show_details.get() == 1 and detailed_info:
            chat_history.insert(tk.END, f"{detailed_info}\n", 'bot_detail')
        chat_history.config(state=tk.DISABLED)
    
    Thread(target=run_query).start()

# Function to show the starting menu
def show_start_menu():
    menu = tk.Toplevel(root)
    menu.title("Start Menu")
    menu.config(bg='#f0f0f0')
    
    tk.Label(menu, text="Choose an option:", bg='#f0f0f0', font=('Arial', 12)).pack(pady=10)

    generate_button = tk.Button(menu, text="Generate New Embeddings", command=lambda: [generate_and_save_embeddings(), menu.destroy()])
    generate_button.pack(pady=5)

    load_button = tk.Button(menu, text="Load Existing Embeddings", command=lambda: [load_existing_embeddings(), menu.destroy()])
    load_button.pack(pady=5)

    menu.update_idletasks()
    width = menu.winfo_width()
    height = menu.winfo_height()
    x = (menu.winfo_screenwidth() // 2) - (width // 2)
    y = (menu.winfo_screenheight() // 2) - (height // 2)
    menu.geometry(f'{width}x{height}+{x}+{y}')
    
    menu.transient(root)
    menu.grab_set()
    root.wait_window(menu)

# Function to center the main window
def center_window(window):
    window.update_idletasks()
    width = window.winfo_width()
    height = window.winfo_height()
    x = (window.winfo_screenwidth() // 2) - (width // 2)
    y = (window.winfo_screenheight() // 2) - (height // 2)
    window.geometry(f'{width}x{height}+{x}+{y}')

# Initialize Sentence Transformer model
model_name = 'multi-qa-mpnet-base-dot-v1'
sentence_model = SentenceTransformer(model_name)

# Set up the GUI
root = tk.Tk()
root.title("Document QA Chatbot")
root.config(bg='#f0f0f0')

# Bind the Escape key to exit the application
root.bind('<Escape>', lambda e: root.quit())

# Configure grid to expand with window size
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=0)

# Chat history frame
chat_frame = tk.Frame(root, bg='#f0f0f0')
chat_frame.grid(row=0, column=0, columnspan=3, padx=10, pady=10, sticky='nsew')

# Chat history text area
chat_history = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, state='disabled', font=('Arial', 10), bg='#ffffff')
chat_history.tag_config('user', foreground='blue', font=('Arial', 10, 'bold'))
chat_history.tag_config('bot', foreground='green', font=('Arial', 10, 'italic'))
chat_history.tag_config('bot_loading', foreground='lightgray')
chat_history.tag_config('bot_sources', foreground='orange')
chat_history.tag_config('bot_detail', foreground='purple')
chat_history.pack(fill='both', expand=True)

# Query frame
query_frame = tk.Frame(root, bg='#f0f0f0')
query_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky='ew')
query_frame.grid_columnconfigure(0, weight=1)

# Query entry
query_entry = tk.Entry(query_frame, font=('Arial', 12))
query_entry.pack(fill='x', side='left', expand=True, padx=(0, 5))
query_entry.bind("<Return>", submit_query)

# Submit button
submit_button = tk.Button(query_frame, text="Submit", command=submit_query, font=('Arial', 12))
submit_button.pack(side='left')

# Control frame
control_frame = tk.Frame(root, bg='#f0f0f0')
control_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky='ew')
control_frame.grid_columnconfigure(0, weight=1)

# Radio button for showing detailed information
show_details = tk.IntVar()
show_details.set(0)
details_radio = tk.Checkbutton(control_frame, text="Show detailed info", variable=show_details, bg='#f0f0f0', font=('Arial', 10))
details_radio.pack(side='left', padx=5)

# Similarity threshold slider
similarity_threshold = tk.Scale(control_frame, from_=0, to=100, orient=tk.HORIZONTAL, label="Similarity Threshold", bg='#f0f0f0', font=('Arial', 10), length=200)
similarity_threshold.set(65)  # Default value
similarity_threshold.pack(side='left', padx=5)

# K dial
k_dial = tk.Scale(control_frame, from_=1, to=50, orient=tk.HORIZONTAL, label="Top K", bg='#f0f0f0', font=('Arial', 10))
k_dial.set(5)  # Default value
k_dial.pack(side='left', padx=5)

# Center the main window
center_window(root)

# Show start menu
show_start_menu()

# Run the application
root.mainloop()
