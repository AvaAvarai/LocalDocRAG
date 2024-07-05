import os
import re
from PyPDF2 import PdfReader
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

# Main function to process a single PDF file
def process_pdf(file_path):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Extracting sentences from {file_path}")
    sentences = extract_sentences_from_pdf(file_path)
    num_sentences = len(sentences)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Number of sentences extracted: {num_sentences}")
    if not sentences:
        raise ValueError(f"No sentences extracted from file: {file_path}")

    # Load the pre-trained model
    model_name = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)

    # Generate embeddings for all sentences
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Generating embeddings using {model_name}...")
    start_time = time.time()
    embeddings = model.encode(sentences, show_progress_bar=True)
    end_time = time.time()
    training_time = end_time - start_time

    # Save embeddings to CSV
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saving embeddings to CSV")
    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df['sentence'] = sentences
    embeddings_df['document'] = os.path.basename(file_path)
    filename = f'embeddings_{os.path.basename(file_path).rsplit(".", 1)[0]}_{training_time:.2f}s.csv'
    embeddings_df.to_csv(filename, index=False, encoding='utf-8')
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Embeddings saved to '{filename}' with training time {training_time:.2f} seconds.")

if __name__ == '__main__':
    # Example usage
    pdf_file_path = r'ref\Boris_Kovalerchuk\vkd_ml.pdf'  # Replace with your PDF file path
    process_pdf(pdf_file_path)
