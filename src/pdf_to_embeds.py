import os
import csv
import pdfplumber
import nltk
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import json

def download_nltk_data():
    print("Downloading NLTK data files...")
    nltk.download('punkt')
    nltk.download('punkt_tab')
    print("NLTK data files downloaded.")

def is_valid_sentence(sentence):
    sentence = sentence.strip()
    if len(sentence) < 10:
        return False
    if not re.search('[a-zA-Z]', sentence):
        return False
    if re.search(r'\S+@\S+|http\S+|www\S+', sentence):
        return False
    if sentence.isupper():
        return False
    return True

def extract_sentences_from_pdf(pdf_path):
    sentences = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in tqdm(pdf.pages, desc=f"Processing {os.path.basename(pdf_path)}", unit="page"):
            text = page.extract_text()
            if text:
                text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
                text = re.sub(r'\S+@\S+', '', text)
                text = text.replace('\n', ' ').strip()
                text = re.sub(r'-\s+', '', text)  # Remove hyphens and spaces following them
                page_sentences = nltk.sent_tokenize(text, language='english')
                sentences.extend(page_sentences)
    sentences = [s.strip() for s in sentences if is_valid_sentence(s)]
    return sentences

def process_pdfs():
    pdf_folder = 'pdfs'
    output_csv = 'extracted_sentences.csv'

    model_name = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)

    pdf_files = [filename for filename in os.listdir(pdf_folder) if filename.endswith('.pdf')]

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Sentence', 'Source', 'Embedding']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for filename in tqdm(pdf_files, desc="Processing PDFs", unit="pdf"):
            pdf_path = os.path.join(pdf_folder, filename)

            sentences = extract_sentences_from_pdf(pdf_path)

            if sentences:
                embeddings = model.encode(sentences, show_progress_bar=True)

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
