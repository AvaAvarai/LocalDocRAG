import fitz
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to preprocess the extracted text
def preprocess_text(text):
    sentences = text.split('\n')
    cleaned_sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return cleaned_sentences

# Function to load PDFs and preprocess text
def load_and_preprocess_pdfs():
    pdf_paths = filedialog.askopenfilenames(filetypes=[("PDF files", "*.pdf")])
    all_text = ""
    for pdf_path in pdf_paths:
        all_text += extract_text_from_pdf(pdf_path) + "\n"
    processed_text = preprocess_text(all_text)
    return processed_text

# Function to generate embeddings and save to CSV
def generate_and_save_embeddings(processed_text):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(processed_text, convert_to_tensor=True)
    data = {'text': processed_text, 'embedding': [embedding.numpy().tolist() for embedding in embeddings]}
    df = pd.DataFrame(data)
    df.to_csv('embeddings.csv', index=False)
    return df

# Function to find the top 5 most relevant chunks for a query
def find_top_5_relevant_chunks(query, model, df):
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = cosine_similarity([query_embedding], df['embedding'].tolist())[0]
    top_5_indices = np.argsort(similarities)[-5:][::-1]

    seen_texts = set()
    top_5_texts = []
    for idx in top_5_indices:
        text = df.iloc[idx]['text']
        if text not in seen_texts:
            top_5_texts.append(text)
            seen_texts.add(text)
        if len(top_5_texts) == 5:
            break
    return top_5_texts

# GUI for the QA Chatbot
class QABotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("QA Chatbot")
        
        self.load_button = tk.Button(root, text="Load PDFs", command=self.load_pdfs)
        self.load_button.pack(pady=10)
        
        self.query_label = tk.Label(root, text="Enter your question:")
        self.query_label.pack(pady=5)
        
        self.query_entry = tk.Entry(root, width=100)
        self.query_entry.pack(pady=5)
        
        self.ask_button = tk.Button(root, text="Ask", command=self.answer_query)
        self.ask_button.pack(pady=10)
        
        self.answer_text = scrolledtext.ScrolledText(root, width=100, height=20)
        self.answer_text.pack(pady=10)
        
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.df = None

    def load_pdfs(self):
        processed_text = load_and_preprocess_pdfs()
        self.df = generate_and_save_embeddings(processed_text)
        messagebox.showinfo("Info", "PDFs loaded and embeddings generated.")

    def answer_query(self):
        query = self.query_entry.get()
        if self.df is None:
            messagebox.showerror("Error", "No PDFs loaded. Please load PDFs first.")
            return
        top_5_answers = find_top_5_relevant_chunks(query, self.model, self.df)
        self.answer_text.delete(1.0, tk.END)
        for idx, answer in enumerate(top_5_answers, 1):
            self.answer_text.insert(tk.END, f"Answer {idx}:\n{answer}\n\n")

# Run the GUI application
if __name__ == "__main__":
    root = tk.Tk()
    app = QABotApp(root)
    root.mainloop()
