import os
import csv
import fitz
import nltk
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import json
import torch

def download_nltk_data():
    print("Downloading NLTK data files...")
    nltk.download('punkt')
    print("NLTK data files downloaded.")

def is_valid_sentence(sentence):
    sentence = sentence.strip()
    if len(sentence.split()) < 5:  # At least 5 words
        return False
    if not re.search('[a-zA-Z]', sentence):
        return False
    if re.search(r'\S+@\S+|http\S+|www\S+|©|\d{4}', sentence):
        return False
    if re.search(r'Fig\.|Table|Figure|Equation|Section|References|Copyright|Abstract|Introduction', sentence, re.IGNORECASE):
        return False
    if sentence.isupper():
        return False
    return True

def extract_sentences_from_pdf(pdf_path):
    sentences = []
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text:
            # Remove hyphenation at line breaks
            text = re.sub(r'-\s*\n', '', text)
            text = re.sub(r'\n', ' ', text)
            # Clean text
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
            text = re.sub(r'\S+@\S+', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            # Tokenize sentences
            page_sentences = nltk.sent_tokenize(text)
            sentences.extend(page_sentences)
    # Filter sentences
    sentences = [s.strip() for s in sentences if is_valid_sentence(s)]
    return sentences

def process_pdfs():
    pdf_folder = 'pdfs'  # Ensure this folder exists and contains your PDF files
    output_csv = 'extracted_sentences.csv'

    model_name = 'all-MiniLM-L6-v2'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SentenceTransformer(model_name, device=device)

    pdf_files = [filename for filename in os.listdir(pdf_folder) if filename.endswith('.pdf')]

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Sentence', 'Source', 'Embedding']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for filename in tqdm(pdf_files, desc="Processing PDFs", unit="pdf"):
            pdf_path = os.path.join(pdf_folder, filename)

            try:
                sentences = extract_sentences_from_pdf(pdf_path)
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")
                continue

            if sentences:
                embeddings = model.encode(sentences, show_progress_bar=True, batch_size=64)

                for sentence, embedding in zip(sentences, embeddings):
                    embedding_str = json.dumps(embedding.tolist())
                    writer.writerow({
                        'Sentence': sentence,
                        'Source': filename,
                        'Embedding': embedding_str
                    })

    print(f"Data saved to {output_csv}")

if __name__ == "__main__":
    print("Starting PDF processing...")
    download_nltk_data()
    process_pdfs()
    print("PDF processing completed.")
