import os
import csv
import pdfplumber
import nltk
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import json

# Download NLTK data files
nltk.download('punkt')
nltk.download('punkt_tab')  # Required for NLTK 3.8 and above

def is_valid_sentence(sentence):
    sentence = sentence.strip()
    if len(sentence) < 10:  # Exclude sentences shorter than 10 characters
        return False
    if not re.search('[a-zA-Z]', sentence):  # Exclude sentences without alphabetic characters
        return False
    if re.search(r'\S+@\S+|http\S+|www\S+', sentence):  # Exclude sentences containing emails or URLs
        return False
    if sentence.isupper():  # Exclude sentences in all uppercase letters
        return False
    return True

def extract_sentences_from_pdf(pdf_path):
    sentences = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in tqdm(pdf.pages, desc=f"Processing {os.path.basename(pdf_path)}", unit="page"):
            text = page.extract_text()
            if text:
                # Remove URLs and emails
                text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
                text = re.sub(r'\S+@\S+', '', text)
                # Remove extra whitespaces and newlines
                text = text.replace('\n', ' ').strip()
                # Tokenize sentences using NLTK
                page_sentences = nltk.sent_tokenize(text, language='english')
                sentences.extend(page_sentences)
    # Filter out unwanted sentences
    sentences = [s.strip() for s in sentences if is_valid_sentence(s)]
    return sentences

def process_pdfs():
    pdf_folder = 'pdfs'  # Ensure this folder exists and contains your PDFs
    output_csv = 'extracted_sentences.csv'

    # Initialize the sentence transformer model
    model_name = 'all-MiniLM-L6-v2'  # You can choose other models if you wish
    model = SentenceTransformer(model_name)

    # List all PDF files in the directory
    pdf_files = [filename for filename in os.listdir(pdf_folder) if filename.endswith('.pdf')]

    # Open the CSV file for writing
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Sentence', 'Source', 'Embedding']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Use tqdm to add a progress bar for PDF processing
        for filename in tqdm(pdf_files, desc="Processing PDFs", unit="pdf"):
            pdf_path = os.path.join(pdf_folder, filename)

            # Extract sentences from PDF
            sentences = extract_sentences_from_pdf(pdf_path)

            # Generate embeddings for sentences
            if sentences:
                embeddings = model.encode(sentences, show_progress_bar=True)

                # Write data to CSV
                for sentence, embedding in zip(sentences, embeddings):
                    # Convert embedding to JSON string
                    embedding_str = json.dumps(embedding.tolist())
                    writer.writerow({
                        'Sentence': sentence,
                        'Source': filename,
                        'Embedding': embedding_str
                    })

    print(f"Data saved to {output_csv}")

if __name__ == "__main__":
    process_pdfs()
