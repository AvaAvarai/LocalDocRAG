import os
import csv
import pdfplumber
import nltk
import re
from tqdm import tqdm  # Import tqdm for progress bars

# Download NLTK data files (if not already downloaded)
nltk.download('punkt')
nltk.download('punkt_tab')  # Required for sentence tokenization in NLTK 3.8 and above

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
        # Use tqdm to add a progress bar for page processing
        for page in tqdm(pdf.pages, desc=f"Processing {os.path.basename(pdf_path)}", unit="page"):
            text = page.extract_text()
            if text:
                # Remove URLs and emails
                text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
                text = re.sub(r'\S+@\S+', '', text)
                # Remove extra whitespaces and newlines
                text = text.replace('\n', ' ').strip()
                # Tokenize sentences using NLTK
                page_sentences = nltk.sent_tokenize(text, language='english')  # Specify language
                sentences.extend(page_sentences)
    # Filter out unwanted sentences
    sentences = [s.strip() for s in sentences if is_valid_sentence(s)]
    return sentences

def process_pdfs():
    pdf_folder = 'pdfs'
    output_csv = 'extracted_sentences.csv'

    # List all PDF files in the directory
    pdf_files = [filename for filename in os.listdir(pdf_folder) if filename.endswith('.pdf')]

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Sentence', 'Source'])

        # Use tqdm to add a progress bar for PDF processing
        for filename in tqdm(pdf_files, desc="Processing PDFs", unit="pdf"):
            pdf_path = os.path.join(pdf_folder, filename)

            # Extract sentences from PDF
            sentences = extract_sentences_from_pdf(pdf_path)

            # Write valid sentences to CSV
            for sentence in sentences:
                writer.writerow([sentence, filename])

if __name__ == "__main__":
    process_pdfs()
