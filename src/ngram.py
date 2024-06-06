import os
import fitz  # PyMuPDF for PDF handling
import csv
import nltk
from collections import defaultdict, Counter
from nltk.util import ngrams
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
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(extract_text_from_pdf, pdf): pdf for pdf in pdf_files}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                texts.append(result)
            print(f"Processed {i}/{total} PDFs...")
    return texts

def extract_sentences(text):
    """Tokenize text into sentences."""
    return nltk.sent_tokenize(text)

def compute_ngram_probabilities(sentences, n=2):
    """Compute n-gram probabilities from a list of sentences."""
    ngram_freq = defaultdict(Counter)
    total_sentences = len(sentences)
    for i, sentence in enumerate(sentences, 1):
        tokens = nltk.word_tokenize(sentence.lower())
        for ngram in ngrams(tokens, n + 1, pad_right=True, pad_left=False, right_pad_symbol=None):
            if ngram[:-1] and ngram[-1]:
                ngram_freq[ngram[:-1]][ngram[-1]] += 1
        if i % 100 == 0 or i == total_sentences:
            print(f"Processed {i}/{total_sentences} sentences...")
    ngram_probabilities = defaultdict(dict)
    for ngram_prefix, suffixes in ngram_freq.items():
        total = sum(suffixes.values())
        for suffix, count in suffixes.items():
            ngram_probabilities[ngram_prefix][suffix] = count / total
    return ngram_probabilities

def save_probabilities_to_csv(ngram_probabilities, filename='ngram_probabilities.csv'):
    """Save n-gram probabilities to a CSV file using UTF-8 encoding."""
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for ngram_prefix, probabilities in ngram_probabilities.items():
            for word, probability in probabilities.items():
                writer.writerow([' '.join(ngram_prefix), word, f"{probability:.4f}"])

def scan_directory_for_pdfs(root_dir):
    """Scan a directory recursively for PDF files."""
    pdf_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def sort_and_construct_response(last_ngram, ngram_probabilities, max_length=10):
    """Sort continuations by probability and construct a response, showing probabilities."""
    if last_ngram in ngram_probabilities:
        sorted_continuations = sorted(ngram_probabilities[last_ngram].items(), key=lambda x: -x[1])
        response_parts = []
        for word, prob in sorted_continuations[:max_length]:  # Limit to the most probable continuations
            response_parts.append(f"{word} ({prob:.2%})")
        return ' '.join(response_parts)
    return "No continuations found."

def interactive_cli(ngram_probabilities, n=2):
    """Interactive CLI to explore probable word continuations based on n-grams."""
    while True:
        input_text = input("Enter the beginning of a sentence or 'exit' to quit: ")
        if input_text.lower() == 'exit':
            break
        tokens = nltk.word_tokenize(input_text.lower())
        last_ngram = tuple(tokens[-n:])
        response = sort_and_construct_response(last_ngram, ngram_probabilities)
        print(f"Possible continuations: {response}")

def main(root_dir='ref', ngram_size=2):
    pdf_files = scan_directory_for_pdfs(root_dir)
    texts = process_pdfs_concurrently(pdf_files)
    
    all_sentences = [sentence for text in texts for sentence in extract_sentences(text)]
    ngram_probabilities = compute_ngram_probabilities(all_sentences, ngram_size)
    save_probabilities_to_csv(ngram_probabilities)
    interactive_cli(ngram_probabilities, ngram_size)

if __name__ == "__main__":
    main()
