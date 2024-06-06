import os
import fitz  # PyMuPDF for PDF handling
import pandas as pd
from collections import defaultdict, Counter
from nltk.util import ngrams
import nltk
from concurrent.futures import ThreadPoolExecutor, as_completed
nltk.download('punkt')

def extract_text_from_pdf(pdf_path):
    """Extract text from a single PDF file."""
    try:
        with fitz.open(pdf_path) as doc:
            text = "".join(page.get_text() for page in doc)
        return text
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return ""

def process_pdfs_concurrently(pdf_files):
    """Process multiple PDFs concurrently to extract text."""
    texts = []
    total = len(pdf_files)
    print(f"Starting the processing of {total} PDFs...")
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(extract_text_from_pdf, pdf): pdf for pdf in pdf_files}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                texts.append(result)
            print(f"Processed {i} of {total} PDFs...")
    return texts

def extract_sentences(text):
    """Tokenize text into sentences."""
    return nltk.sent_tokenize(text)

def compute_ngram_probabilities(sentences, n):
    """Compute n-gram probabilities from a list of sentences."""
    ngram_freq = defaultdict(Counter)
    total_sentences = len(sentences)
    print(f"Computing {n}-gram probabilities for {total_sentences} sentences...")
    for i, sentence in enumerate(sentences, 1):
        tokens = nltk.word_tokenize(sentence.lower())
        for ngram in ngrams(tokens, n + 1, pad_right=True, pad_left=False, right_pad_symbol=None):
            if ngram[:-1] and ngram[-1]:
                ngram_freq[ngram[:-1]][ngram[-1]] += 1
        if i % 100 == 0 or i == total_sentences:
            print(f"Processed {i}/{total_sentences} sentences for {n}-grams...")
    ngram_probabilities = defaultdict(dict)
    for ngram_prefix, suffixes in ngram_freq.items():
        total = sum(suffixes.values())
        for suffix, count in suffixes.items():
            ngram_probabilities[ngram_prefix][suffix] = count / total
    return ngram_probabilities

def sanitize_text(text):
    """Sanitize text to remove characters that are not supported in Excel sheets by openpyxl."""
    allowed_control_chars = {'\x09', '\x0A', '\x0D'}
    return ''.join(char if char.isprintable() or char in allowed_control_chars else '?' for char in text)

def save_ngrams_to_csv(ngram_probabilities, filename='ngram_probabilities.xlsx'):
    """Save n-gram probabilities to different sheets in an Excel file."""
    print(f"Writing n-gram probabilities to {filename}...")
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for n, probabilities in ngram_probabilities.items():
            data = []
            for prefix, suffixes in probabilities.items():
                for word, prob in suffixes.items():
                    sanitized_ngram = ' '.join(sanitize_text(part) for part in prefix)
                    sanitized_word = sanitize_text(word)
                    data.append({'N-gram': sanitized_ngram, 'Continuation': sanitized_word, 'Probability': prob})
            df = pd.DataFrame(data)
            df.to_excel(writer, sheet_name=f'{n}-gram', index=False)
    print("Excel file has been written successfully.")

def scan_directory_for_pdfs(root_dir):
    """Scan a directory recursively for PDF files."""
    pdf_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    print(f"Found {len(pdf_files)} PDF files in {root_dir}.")
    return pdf_files

def main(root_dir='ref'):
    pdf_files = scan_directory_for_pdfs(root_dir)
    texts = process_pdfs_concurrently(pdf_files)
    all_sentences = [sentence for text in texts for sentence in extract_sentences(text)]

    ngram_probabilities = {}
    for n in range(2, 11):  # From 2-grams to 10-grams
        ngram_probabilities[n] = compute_ngram_probabilities(all_sentences, n)

    save_ngrams_to_csv(ngram_probabilities)

if __name__ == "__main__":
    main()
