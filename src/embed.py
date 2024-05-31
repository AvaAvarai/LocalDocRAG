# embed.py - Extracts text from PDF files, cleans and splits text into sentences,
# generates sentence embeddings, saves embeddings to CSV, visualizes embeddings
# with PCA and t-SNE, and launches a GUI for question-answering using the embeddings.

import os

# Suppress TensorFlow oneDNN optimization messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import re
import tkinter as tk
from tkinter import scrolledtext, Spinbox
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import warnings
from huggingface_hub.file_download import logger as hf_logger
import logging
from multiprocessing import Pool, cpu_count
from bs4 import BeautifulSoup
from transformers import pipeline

# Ensure the necessary resources are downloaded once
def initialize_resources():
    nltk.download('punkt')
    nltk.download('stopwords')

def load_pdfs(directory):
    pdf_files = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.pdf')]
    with Pool(cpu_count()) as pool:
        pdf_texts = pool.map(extract_text_from_pdf, pdf_files)
    return list(zip([os.path.basename(filepath) for filepath in pdf_files], pdf_texts))

def extract_text_from_pdf(filepath):
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def clean_text(text):
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove excessive whitespace
    text = re.sub(r'\.{2,}', '.', text)  # Replace sequences of two or more periods with a single period
    return text

def is_semantically_useful(sentence):
    global nlp
    doc = nlp(sentence)
    
    if len(doc) < 5:
        return False
    if len([token for token in doc if token.is_alpha]) / len(doc) < 0.5:
        return False
    if re.match(r'^NUM\b', sentence) or re.match(r'.*\bNUM\b$', sentence) or re.match(r'.*\bpp\b$', sentence):
        return False

    return True

def clean_and_split_text(text, source):
    global nlp
    
    text = clean_text(text)
    chunks = [text[i:i+1000000] for i in range(0, len(text), 1000000)]  # Split text into manageable chunks
    sentences = []

    for chunk in chunks:
        doc = nlp(chunk)
        sentences.extend([sent.text.strip() for sent in doc.sents])

    cleaned_sentences = []
    for sentence in sentences:
        if is_semantically_useful(sentence):
            words = word_tokenize(sentence)
            words = [word for word in words if word.isalpha() and word.lower() not in stopwords.words('english')]
            seen = set()
            cleaned_sentence = ' '.join([x for x in words if not (x in seen or seen.add(x))])
            if len(cleaned_sentence) > 2:
                cleaned_sentences.append((sentence, cleaned_sentence, source))
    
    unique_sentences = list(dict.fromkeys(cleaned_sentences))
    return unique_sentences, sentences

def generate_embeddings(sentences):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        hf_logger.setLevel(logging.ERROR)
        
        model = SentenceTransformer('sentence-transformers/nli-roberta-large')
        embeddings = model.encode([cleaned for _, cleaned, _ in sentences])
    
    return embeddings, model.get_sentence_embedding_dimension()

def save_embeddings_to_csv(sentences, embeddings, output_file):
    df = pd.DataFrame({
        'source': [source for _, _, source in sentences],
        'original_sentence': [original for original, _, _ in sentences],
        'cleaned_sentence': [cleaned for _, cleaned, _ in sentences],
        'embedding': [emb.tolist() for emb in embeddings]
    })
    df.to_csv(output_file, index=False, encoding='utf-8')

def plot_embeddings(embeddings, sources):
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(embeddings)

    tsne = TSNE(n_components=3, perplexity=40, n_iter=300)
    tsne_result = tsne.fit_transform(embeddings)

    unique_sources = list(set(sources))
    num_sources = len(unique_sources)
    colors = plt.get_cmap('viridis')
    source_to_color = {source: colors(i / num_sources) for i, source in enumerate(unique_sources)}

    fig = plt.figure(figsize=(14, 6))

    ax = fig.add_subplot(121, projection='3d')
    for source in unique_sources:
        idx = [i for i, s in enumerate(sources) if s == source]
        ax.scatter(pca_result[idx, 0], pca_result[idx, 1], pca_result[idx, 2], color=source_to_color[source], label=source, alpha=0.25, edgecolors='w', s=30)
    ax.set_title("PCA 3D Plot")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.set_zlabel("PCA3")
    ax.view_init(elev=20, azim=120)

    ax = fig.add_subplot(122, projection='3d')
    for source in unique_sources:
        idx = [i for i, s in enumerate(sources) if s == source]
        ax.scatter(tsne_result[idx, 0], tsne_result[idx, 1], tsne_result[idx, 2], color=source_to_color[source], label=source, alpha=0.25, edgecolors='w', s=30)
    ax.set_title("t-SNE 3D Plot")
    ax.set_xlabel("t-SNE1")
    ax.set_ylabel("t-SNE2")
    ax.set_zlabel("t-SNE3")
    ax.view_init(elev=20, azim=120)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=7)

    plt.show()

def viz_embeddings(embeddings, sources):
    # PCA for 50 dimensions
    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(embeddings)
    embeddings = pca_result
    unique_sources = list(set(sources))
    num_sources = len(unique_sources)
    colors = plt.get_cmap('viridis')
    source_to_color = {source: colors(i / num_sources) for i, source in enumerate(unique_sources)}

    df = pd.DataFrame(embeddings, columns=[f"dim_{i+1}" for i in range(embeddings.shape[1])])
    df['source'] = sources

    plt.figure(figsize=(15, 8))

    for source in unique_sources:
        subset = df[df['source'] == source]
        plt.plot(subset.drop(columns=['source']).T, color=source_to_color[source], alpha=0.2)

    plt.title("Parallel Coordinates Plot of Embeddings")
    plt.xlabel("Dimensions")
    plt.ylabel("Values")
    plt.legend(unique_sources, loc='upper right', bbox_to_anchor=(1.1, 1), title='Sources')
    plt.show()

def load_embeddings_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    embeddings = np.array(df['embedding'].apply(eval).tolist()).astype(np.float32)
    sentences = list(zip(df['original_sentence'], df['cleaned_sentence'], df['source']))
    return sentences, embeddings

# "Universal Sentence Encoder" by Google Research claims that angular distance
# performs better on average with sentence embeddings than cosine similarity.
# Angular distance is defined as 1 - arccos(cosine_similarity) / pi.
# See paper at: https://arxiv.org/abs/1803.11175 for more details.
def cosine_to_angular_distance(cosine_similarity):
    return (1 - np.arccos(cosine_similarity) / np.pi)

def find_top_k_similar(question, embeddings, sentences, k):
    model = SentenceTransformer('sentence-transformers/nli-roberta-large')
    question_embedding = model.encode(question).astype(np.float32)
    similarities = util.pytorch_cos_sim(question_embedding, embeddings)[0]
    angular_distances = cosine_to_angular_distance(similarities.numpy())
    top_k_indices = angular_distances.argsort()[-k:][::-1]
    top_k_sentences = [sentences[i] for i in top_k_indices]
    top_k_angular_distances = [angular_distances[i] for i in top_k_indices]
    return top_k_sentences, top_k_angular_distances

def generate_response(question, top_k_sentences, original_texts):
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-large-squad2")
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
    global nlp
    nlp = spacy.load("en_core_web_sm")  # Load spaCy model once and set the maximum length limit
    nlp.max_length = 1500000  # Increase the limit
    
    initialize_resources()
    
    pdf_directory = 'ref'
    output_file = 'embeddings.csv'

    print("Loading PDFs...")
    pdf_texts = load_pdfs(pdf_directory)
    
    print("Cleaning and splitting text...")
    all_sentences = []
    original_texts = {}
    for filename, text in pdf_texts:
        sentences, original_sentences = clean_and_split_text(text, filename)
        all_sentences.extend(sentences)
        original_texts[filename] = original_sentences

    print("Generating embeddings...")
    embeddings, embedding_dim = generate_embeddings(all_sentences)
    
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Number of sentences: {len(all_sentences)}")
    
    print("Saving embeddings to CSV...")
    save_embeddings_to_csv(all_sentences, embeddings, output_file)

    print("Visualizing embeddings with PCA and t-SNE...")
    plot_embeddings(embeddings, [source for _, _, source in all_sentences])
    
    print("Visualizing embeddings with parallel coordinates using PCA for 50 dimensions...")
    viz_embeddings(embeddings, [source for _, _, source in all_sentences])
    
    print("Loading embeddings from CSV...")
    sentences, embeddings = load_embeddings_from_csv(output_file)
    
    print("Launching GUI...")
    root = tk.Tk()
    ChatInterfaceApp(root, sentences, embeddings, original_texts)
    root.mainloop()


if __name__ == "__main__":
    main()
