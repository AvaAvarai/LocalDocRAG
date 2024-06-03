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

# Compile regex patterns
patterns = [
    re.compile(r'\b(?:figure|table|definition|appendix|advances|ultra-strong|manuscript submitted|url|available at|alternatively|abstract|institute (for|of)|we.*\backnowledge\b|\bwe.*\backnowledgments\b|contribution|contributions|funding|grant|grants|resources provided by|bits per byte|pp\.)\b'),
    re.compile(r'^\[\d+\]\s|^[a-z]\.?\s|^\[?\d+\]?\.$|^input:\s|.*\|-\s*\(.*\)$|.*\bph\b.*\b(ps|ch)\b|.*\b(nn0|cc|zz)\b.*|.*\b(?:[online]|acm sigkdd int|[a-z]\d+)\b.*|.*\[online\].*'),
    re.compile(r'\b(?:[12]\d{3}|vol\.|no\.|doi:|ed\.|pages|chapter|cambridge university press|journal|conference|proceedings|in press|forthcoming|ISBN\s(?:97[89][-– ])?\d{1,5}[-– ]?\d+[-– ]?\d+[-– ]?[\dX])\b'),
    re.compile(r'https?://\S+|www\.\S+|http\s?:\s?//\S+'),
    re.compile(r'\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b'),
    re.compile(r'\bin fig\b|\bin table\b|\bin diagram\b|\bin screenshot\b|\bprint\((.*)\)\b'),
    re.compile(r'^(\d+\s?[,|\.|\s])+[a-z]*\d*$'),
    re.compile(r'\b(?:linear|relu|train loss|test loss|out figure|joint f|case studies|baseline prediction|naive prediction|mix prediction|combined prediction|input: print\(|input:\s|proof of)\b'),
    re.compile(r'^\d+\s*\[.*\]\s*[a-z\s,]*$'),
    re.compile(r'^[a-z\s]+and\s[a-z\s]+(\.|,)$|^[a-z\s,]+\.$|^[a-z\s,]+,\sand\s[a-z\s,]+\.$|^[a-z\s]+,\sand\s[a-z\s]+\.$|^[a-z\s]+,\sand\s[a-z\s,]+\.$|^[a-z\s,]+\sand\s[a-z\s,]+\.$|^[a-z\s,]+and\s[a-z\s,]+\.$'),
    re.compile(r'^[a-z\s]+,\s[a-z\s]+,\sand\s[a-z\s]+(\.|,)$|^[a-z\s]+(\.|,|&\s)'),
    re.compile(r'^[a-z\s,.]+,\sand\s[a-z\s,.]+$|^[a-z\s,]+,\sand\s[a-z\s,]+$|^[a-z\s,]+,\sand\s[a-z\s,]+,\s[a-z\s,]+$'),
    re.compile(r'^[a-z\s,]+,\sand\s[a-z\s,]+,\s[a-z\s,]+,\s[a-z\s,]+$|^[a-z\s,]+,\sand\s[a-z\s,]+,\s[a-z\s,]+,\s[a-z\s,]+,\s[a-z\s,]+$'),
    re.compile(r'^[a-z\s]+,\s[a-z\s]+,\s[a-z\s]+,\sand\s[a-z\s]+$'),
    re.compile(r'^\d+(\s?,\s?\d+)*\s?\[\d+\]\s?[a-z\s,]+(and|et al.)?[a-z\s,]*\.$|^\[[a-z0-9]+\]\s[a-z\s,]+(and|et al.)?\s[a-z\s,]*\.$|^\d+\s[a-z\s,]+\.$|^\d+(\s?,\s?\d+)*\s?\[[0-9]+\]\s?[a-z\s,]+(et al.)?\.$|^\(cited on\s\d+\)\s[a-z\s,]+(et al.)?\.$|^\d+(\s?,\s?\d+)*\s?\[[0-9]+\]\s?[a-z\s,]+$|^\d+(,\s?\d+)*\s?[a-z\s,]+\.$|^\(cited on page\s\d+(,\s?\d+)*\)$'),
    re.compile(r'^references\s\[\d+\]\s[a-z\s,]+\.$|^references\s\[\d+\]\s[a-z\s,]+,\s[a-z\s,]+\.\s[a-z\s,.]+$'),
    re.compile(r'^in:\s[a-z\s,.()]+(eds.)?\s[a-z\s,]+\spp\.$'),
    re.compile(r'^[a-z\s]+\sclass\s\d+\s\(\d+\)\s[a-z\s,]+$'),
    re.compile(r'^(\d+(\.\d+)?\s?)+[a-z\s,]*$|^(\d+(\.\d+)?\s?)+[a-z\s]+$|^(\d+(\.\d+)?\s?)+fig\.$|^(\d+(\.\d+)?\s?)+[a-z]+$|^-?\d+(\.\d+)?\s(-?\d+(\.\d+)?\s?)*[a-z]+\s\d+(\.\d+)?\s?$'),
    re.compile(r'\bgithub\b'),
    re.compile(r'^code available at:\s?\S+$'),
    re.compile(r'^\d+\s+[a-z\s-]+(\.\s*\.)+$'),
    re.compile(r'^\d+\s+[a-zA-Z\s-]+\s*\.\s*\.\s*\.\s*$')
]

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
    text = re.sub(r'\b"\s', '" ', text)        # Remove misplaced space after an opening quote
    text = re.sub(r'\s"\b', ' "', text)        # Remove misplaced space before a closing quote
    text = text.lower()                        # Convert text to lowercase
    text = re.sub(r'\s+', ' ', text)           # Replace multiple whitespace characters with a single space
    text = re.sub(r',,', ',', text)            # Replace double commas with a single comma
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-UTF-8 characters

    # Separate sentences with spaces after periods and remove extra spaces
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    sentences = [sentence.strip() for sentence in sentences if sentence]

    return sentences

def is_meaningful_sentence(sentence):
    if len(sentence) < 20:
        return False
    for pattern in patterns:
        if pattern.search(sentence):
            return False
    return True

def filter_sentence(sentence):
    words = word_tokenize(sentence)
    if len(words) < 5:  # Filter out very short sentences
        return None
    if all(word.lower() in STOPWORDS for word in words):  # Filter out sentences that are mostly stopwords
        return None
    if is_meaningful_sentence(sentence):  # Apply heuristic rules
        return sentence
    return None

def process_pdf(file_info):
    author, file_path = file_info
    text = extract_text_from_pdf(file_path)
    if not text:
        return [], 0
    cleaned_sentences = clean_text(text)
    
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() * 2) as executor:
        futures = [executor.submit(filter_sentence, sentence) for sentence in cleaned_sentences]
        filtered_sentences = [future.result() for future in as_completed(futures) if future.result() is not None]
    
    pdf_texts = [{'author': author, 'pdf_source': os.path.basename(file_path), 'sentence': sentence} for sentence in filtered_sentences]
    return pdf_texts, len(cleaned_sentences)

def process_chunk(chunk):
    pdf_texts = []
    total_sentences = 0
    for file_info in chunk:
        result, num_sentences = process_pdf(file_info)
        pdf_texts.extend(result)
        total_sentences += num_sentences
    return pdf_texts, total_sentences

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

    total_sentences = 0
    pdf_texts = []

    with ProcessPoolExecutor(max_workers=cpu_count) as executor:  # Increase number of workers
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing PDF chunks"):
            result, num_sentences = future.result()
            pdf_texts.extend(result)
            total_sentences += num_sentences

    return pdf_texts, total_sentences

def save_cleaned_text_to_csv(cleaned_texts, output_file='cleaned_texts.csv'):
    df = pd.DataFrame(cleaned_texts)
    df.to_csv(output_file, index=False, encoding='utf-8')

def print_stats(cleaned_texts, total_sentences):
    num_sentences = len(cleaned_texts)
    print(f"\nTotal number of sentences processed: {total_sentences}")
    print(f"Number of meaningful sentences exported to CSV: {num_sentences}")
    print(f"Number of sentences filtered out: {total_sentences - num_sentences}")

if __name__ == "__main__":
    # Download stopwords and punkt if not already available
    download('stopwords')
    download('punkt')

    STOPWORDS = set(stopwords.words('english')).union(ENGLISH_STOP_WORDS)
    
    directory = 'ref'
    cleaned_texts, total_sentences = load_and_clean_pdfs_parallel(directory)
    save_cleaned_text_to_csv(cleaned_texts)
    print_stats(cleaned_texts, total_sentences)
