

import os
import re
import tkinter as tk
from tkinter import scrolledtext
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from multiprocessing import Pool, cpu_count
from bs4 import BeautifulSoup

# Load the model globally to avoid reloading in each process
MODEL = SentenceTransformer('sentence-transformers/nli-roberta-large')

# Ensure the necessary resources are downloaded once
def initialize_resources():
    nltk.download('punkt')
    nltk.download('stopwords')

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")

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
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_semantically_useful(sentence):
    doc = nlp(sentence)
    if len(doc) < 5:
        return False
    if len([token for token in doc if token.is_alpha]) / len(doc) < 0.5:
        return False
    return not any(ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART"] for ent in doc.ents)

def clean_and_split_text(text, source):
    text = clean_text(text)
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    cleaned_sentences = []
    for sentence in sentences:
        if is_semantically_useful(sentence):
            words = word_tokenize(sentence)
            words = [word for word in words if word.isalpha() and word.lower() not in stopwords.words('english')]
            cleaned_sentence = ' '.join(dict.fromkeys(words))
            if len(cleaned_sentence) > 2:
                cleaned_sentences.append((sentence, cleaned_sentence, source))
    return list(dict.fromkeys(cleaned_sentences))

def generate_embeddings(sentences):
    embeddings = MODEL.encode([cleaned for _, cleaned, _ in sentences])
    return embeddings, MODEL.get_sentence_embedding_dimension()

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
    colors = plt.get_cmap('viridis')
    source_to_color = {source: colors(i / len(unique_sources)) for i, source in enumerate(unique_sources)}
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
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=6)
    plt.show()

def load_embeddings_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    embeddings = np.array(df['embedding'].apply(eval).tolist()).astype(np.float32)
    sentences = list(zip(df['original_sentence'], df['cleaned_sentence'], df['source']))
    return sentences, embeddings

def find_top_k_similar(question, embeddings, sentences, k=5):
    question_embedding = MODEL.encode(question).astype(np.float32)
    similarities = util.pytorch_cos_sim(question_embedding, embeddings)[0]
    top_k_indices = similarities.topk(k).indices
    top_k_sentences = [sentences[i] for i in top_k_indices]
    top_k_similarities = [similarities[i].item() for i in top_k_indices]
    return top_k_sentences, top_k_similarities

class ChatInterfaceApp:
    def __init__(self, root, sentences, embeddings):
        self.root = root
        self.sentences = sentences
        self.embeddings = embeddings
        self.root.title("Embedding Similarity Search")

        self.text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=20, state=tk.DISABLED)
        self.text_area.grid(column=0, row=0, padx=10, pady=10, columnspan=2)

        self.entry = tk.Entry(root, width=80)
        self.entry.grid(column=0, row=1, padx=10, pady=10, sticky="w")

        self.send_button = tk.Button(root, text="Send", command=self.query)
        self.send_button.grid(column=1, row=1, padx=10, pady=10, sticky="e")

    def query(self):
        question = self.entry.get()
        if question:
            self.text_area.config(state=tk.NORMAL)
            self.text_area.insert(tk.END, f"You: {question}\n")
            self.entry.delete(0, tk.END)

            top_k_sentences, top_k_similarities = find_top_k_similar(question, self.embeddings, self.sentences)
            self.text_area.insert(tk.END, "AI: Here are the most similar sentences:\n")
            for i, (sentence, similarity) in enumerate(zip(top_k_sentences, top_k_similarities)):
                self.text_area.insert(tk.END, f"{i+1}. {sentence[0]} (Source: {sentence[2]}, Similarity: {similarity:.4f})\n")
            
            self.text_area.insert(tk.END, "\n" + "="*80 + "\n\n")
            self.text_area.config(state=tk.DISABLED)

def main():
    initialize_resources()
    
    pdf_directory = 'ref'
    output_file = 'embeddings.csv'

    print("Loading PDFs...")
    pdf_texts = load_pdfs(pdf_directory)
    
    print("Cleaning and splitting text...")
    all_sentences = []
    for filename, text in pdf_texts:
        sentences = clean_and_split_text(text, filename)
        all_sentences.extend(sentences)

    print("Generating embeddings...")
    embeddings, embedding_dim = generate_embeddings(all_sentences)
    
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Number of sentences: {len(all_sentences)}")
    
    print("Saving embeddings to CSV...")
    save_embeddings_to_csv(all_sentences, embeddings, output_file)

    print("Visualizing embeddings with PCA and t-SNE...")
    plot_embeddings(embeddings, [source for _, _, source in all_sentences])

    print("Process completed successfully.")
    
    print("Loading embeddings from CSV...")
    sentences, embeddings = load_embeddings_from_csv(output_file)
    
    print("Launching GUI...")
    root = tk.Tk()
    app = ChatInterfaceApp(root, sentences, embeddings)
    root.mainloop()

if __name__ == "__main__":
    main()
