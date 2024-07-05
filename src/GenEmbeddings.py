import os
import re
from PyPDF2 import PdfReader
import numpy as np
import pandas as pd
import time
from sentence_transformers import SentenceTransformer

# Function to clean up redundant spacing, remove newlines and tabs, and remove punctuation
def clean_sentence(sentence, seen_sentences):
    sentence = sentence.encode('utf-8', 'ignore').decode('utf-8')
    sentence = sentence.replace('\n', ' ').replace('\t', ' ').replace('  ', ' ')
    if sentence not in seen_sentences:
        seen_sentences.add(sentence)
        return sentence
    return None

# Function to remove unwanted references from text
def remove_references(text):
    text = re.sub(r'\b(Fig|Table|Fig\.|Table\.|Figure)\s?\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(Fig|Table|Figure)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(Sect|Sec|Section|Sect\.|Sec\.)\s?\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(Sect|Sec|Section|Sect\.|Sec\.)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(depicted in|shown in|seen in|illustrated in|presented in|described in)\s?\b.*\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(depicted in|shown in|seen in|illustrated in|presented in|described in)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\(\d+\)', '', text)
    text = text.replace('  ', ' ')
    return text

# Remove sentences which are too short after cleaning and filtering ignoring 1 and 2 letter words
def filter_too_short(text, min_length=5):
    words_in_text = text.split(' ')
    copy = words_in_text.copy()
    for word in words_in_text:
        if len(word) <= 2:
            copy.remove(word)
    return len(copy) >= min_length

# Function to extract sentences from a PDF file
def extract_sentences_from_pdf(file_path):
    reader = PdfReader(file_path)
    sentences = []
    seen_sentences = set()
    for page in reader.pages:
        text = page.extract_text()
        if text:
            cleaned_sentences = [clean_sentence(sentence, seen_sentences) for sentence in text.split('. ')]
            cleaned_sentences = [remove_references(sentence) for sentence in cleaned_sentences if sentence]
            sentences.extend([sentence for sentence in cleaned_sentences if sentence and filter_too_short(sentence)])
    return sentences

# Function to find all PDF files in a directory and its subdirectories
def find_all_pdfs(directory):
    pdf_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

# Load the pre-trained model
# model_name = 'multi-qa-mpnet-base-dot-v1'
model_name = 'google-bert/bert-large-uncased-whole-word-masking-finetuned-squad'
model = SentenceTransformer(model_name)
np.random.seed(42)

# Directory containing the PDF documents
doc_dir = 'ref'

# Find all PDF files in the directory and its subdirectories
pdf_files = find_all_pdfs(doc_dir)

# Extract text from PDFs
docs_sentences = []
doc_references = []

for pdf_file in pdf_files:
    print(f"Processing file: {pdf_file}")
    pdf_name = os.path.basename(pdf_file)
    sentences = extract_sentences_from_pdf(pdf_file)
    if sentences:
        docs_sentences.extend(sentences)
        doc_references.extend([pdf_name] * len(sentences))
    else:
        print(f"No sentences extracted from file: {pdf_file}")

print(f"Total number of sentences: {len(docs_sentences)}")

# Check if any sentences were extracted
if len(docs_sentences) == 0:
    raise ValueError("No sentences were extracted from the PDF files. Please check the content and extraction logic.")

# Create a DataFrame for easy handling
df = pd.DataFrame({'sentence': docs_sentences, 'document': doc_references})

# Generate embeddings for all sentences
print(f"Generating embeddings using {model_name}...")
start_time = time.time()
embeddings = model.encode(df['sentence'].tolist(), show_progress_bar=True)
end_time = time.time()
training_time = end_time - start_time
if len(embeddings) == 0:
    raise ValueError(f"Embeddings for {model_name} are empty. Please check the model and the input data.")

# Save embeddings to CSV with training time in the filename
embeddings_df = pd.DataFrame(embeddings)
embeddings_df['sentence'] = df['sentence']
embeddings_df['document'] = df['document']
filename = f'embeddings_{model_name}_{training_time:.2f}s.csv'
embeddings_df.to_csv(filename, index=False, encoding='utf-8', escapechar='\\')
print(f"Embeddings saved to '{filename}' with training time {training_time:.2f} seconds.")
