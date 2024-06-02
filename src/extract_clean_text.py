import fitz  # PyMuPDF
import os
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import download
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Download stopwords if not already available
download('stopwords')
download('punkt')

STOPWORDS = set(stopwords.words('english')).union(ENGLISH_STOP_WORDS)

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace characters with a single space
    text = re.sub(r'ISBN\s(?:97[89][-– ])?\d{1,5}[-– ]?\d+[-– ]?\d+[-– ]?[\dX]', '', text, flags=re.IGNORECASE)  # Remove ISBNs
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    sentences = [sentence.strip() for sentence in sentences if sentence]
    return sentences

def is_meaningful_sentence(sentence):
    # Heuristic rules to filter out non-meaningful sentences
    if len(sentence) < 20:  # Filter out very short sentences
        return False
    if sentence.lower().startswith(('figure', 'table', 'definition', 'appendix', 'advances', 'ultra-strong')):
        return False
    if re.match(r'^[A-Za-z]\.?\s', sentence):  # Filter out sentences starting with single letters/abbreviations
        return False
    if re.match(r'^\[?\d+\]?\.$', sentence):  # Filter out numeric and special character sentences
        return False
    if re.search(r'\b(?:[12]\d{3}|vol\.|no\.|pp\.|doi:|ed\.|pages|chapter|cambridge university press|journal|conference|proceedings|in press|forthcoming)\b', sentence, re.IGNORECASE):  # Filter out common citation patterns
        return False
    if re.search(r'\b(?:http|www)\b', sentence):  # Filter out URLs
        return False
    return True

def filter_sentences(sentences):
    filtered_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        if len(words) < 3:  # Filter out very short sentences
            continue
        if all(word.lower() in STOPWORDS for word in words):  # Filter out sentences that are mostly stopwords
            continue
        if is_meaningful_sentence(sentence):  # Apply heuristic rules
            filtered_sentences.append(sentence)
    return filtered_sentences

def load_and_clean_pdfs(directory):
    pdf_texts = []
    for subdir, _, files in os.walk(directory):
        author = os.path.basename(subdir)
        for file in files:
            if file.endswith(".pdf"):
                path = os.path.join(subdir, file)
                text = extract_text_from_pdf(path)
                cleaned_sentences = clean_text(text)
                filtered_sentences = filter_sentences(cleaned_sentences)
                for sentence in filtered_sentences:
                    pdf_texts.append({'author': author, 'pdf_source': file, 'sentence': sentence})
    return pdf_texts

def save_cleaned_text_to_csv(cleaned_texts, output_file='cleaned_texts.csv'):
    df = pd.DataFrame(cleaned_texts)
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    directory = 'ref'
    cleaned_texts = load_and_clean_pdfs(directory)
    save_cleaned_text_to_csv(cleaned_texts)
