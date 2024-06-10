import fitz  # PyMuPDF
import os
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import download
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm

STOPWORDS = set(stopwords.words('english')).union(ENGLISH_STOP_WORDS)

def extract_text_from_pdf(pdf_path):
    try:
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        return text.encode('utf-8', errors='ignore').decode('utf-8')
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""
    
def clean_text(text):
    text = re.sub(r'-\s*\n', '', text)         # Remove hyphens at line breaks and join the words
    text = text.replace('\n', ' ')             # Replace newlines with spaces
    text = text.replace(':', ' ')              # Replace colons with spaces
    text = re.sub(r'\b"\s', '"', text)         # Remove misplaced space after an opening quote
    text = re.sub(r'\s"\b', '"', text)         # Remove misplaced space before a closing quote
    text = re.sub(r'"\s', '"', text)           # Remove space after opening quote
    text = re.sub(r'\s"', '"', text)           # Remove space before closing quote
    text = re.sub(r'“\s', '“', text)           # Handle different quote marks
    text = re.sub(r'\s”', '”', text)           # Handle different quote marks
    text = re.sub(r'\d+\s*\|\s*\d+', '', text) # Remove page numbers (e.g., "1 | 2")
    text = text.lower()                        # Convert text to lowercase
    text = re.sub(r'\s+', ' ', text)           # Replace multiple whitespace characters with a single space
    text = re.sub(r',,', ',', text)            # Replace double commas with a single comma
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-UTF-8 characters
    text = re.sub(r'\.{2,}', '.', text)        # Replace sequences of periods with a single period

    # Separate sentences with spaces after periods and remove extra spaces
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    sentences = [sentence.strip() for sentence in sentences if sentence]

    return sentences

def process_pdf(file_info):
    author, file_path = file_info
    text = extract_text_from_pdf(file_path)
    if not text:
        return []
    cleaned_sentences = clean_text(text)
    pdf_texts = [{'author': author, 'pdf_source': os.path.basename(file_path), 'sentence': sentence} for sentence in cleaned_sentences]
    return pdf_texts

def process_chunk(chunk):
    pdf_texts = []
    for file_info in chunk:
        result = process_pdf(file_info)
        pdf_texts.extend(result)
    return pdf_texts

def load_and_clean_pdfs_parallel(directory):
    pdf_files = []
    for subdir, _, files in os.walk(directory):
        author = os.path.basename(subdir)
        for file in files:
            if file.endswith(".pdf"):
                pdf_files.append((author, os.path.join(subdir, file)))

    # Split the files into chunks
    cpu_count = multiprocessing.cpu_count()
    chunk_size = max(1, len(pdf_files) // cpu_count)
    chunks = [pdf_files[i:i + chunk_size] for i in range(0, len(pdf_files), chunk_size)]

    pdf_texts = []

    with ProcessPoolExecutor(max_workers=cpu_count) as executor:  # Increase number of workers
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing PDF chunks"):
            result = future.result()
            pdf_texts.extend(result)

    return pdf_texts

def save_cleaned_text_to_csv(cleaned_texts, output_file='cleaned_texts.csv'):
    df = pd.DataFrame(cleaned_texts)
    df.to_csv(output_file, index=False, encoding='utf-8')

def print_stats(cleaned_texts):
    num_sentences = len(cleaned_texts)
    print(f"\nTotal number of sentences processed: {num_sentences}")

if __name__ == "__main__":
    # Download stopwords and punkt if not already available
    download('stopwords')
    download('punkt')

    STOPWORDS = set(stopwords.words('english')).union(ENGLISH_STOP_WORDS)
    
    directory = 'ref'
    cleaned_texts = load_and_clean_pdfs_parallel(directory)
    save_cleaned_text_to_csv(cleaned_texts)
    print_stats(cleaned_texts)
