import os
import re
from PyPDF2 import PdfReader
import pandas as pd
import time
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline, AutoModelForSeq2SeqLM
from sklearn.neighbors import NearestNeighbors
import numpy as np
import string

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
    # Remove patterns like "Fig. 1", "Table 2", "[3]", "(3)", etc.
    text = re.sub(r'\b(Fig|Table|Fig\.|Table\.|Figure)\s?\d+\b', '', text, flags=re.IGNORECASE)
    # Remove patterns like "Fig.", "Table"
    text = re.sub(r'\b(Fig|Table|Figure)\b', '', text, flags=re.IGNORECASE)
    # Remove patterns like "Section" or "Sect." or "Sec." followed by a number
    text = re.sub(r'\b(Sect|Sec|Section|Sect\.|Sec\.)\s?\d+\b', '', text, flags=re.IGNORECASE)
    # Remove patterns like "Section" or "Sect." or "Sec."
    text = re.sub(r'\b(Sect|Sec|Section|Sect\.|Sec\.)\b', '', text, flags=re.IGNORECASE)
    # remove patterns like "depicted in " or "shown in " or "seen in " or "illustrated in " or "presented in " or "described in " followed by a reference
    text = re.sub(r'\b(depicted in|shown in|seen in|illustrated in|presented in|described in)\s?\b.*\b', '', text, flags=re.IGNORECASE)
    # remove patterns like "depicted in " or "shown in " or "seen in " or "illustrated in " or "presented in " or "described in " not followed by a reference
    text = re.sub(r'\b(depicted in|shown in|seen in|illustrated in|presented in|described in)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\d+\]', '', text)  # Remove patterns like "[3]"
    text = re.sub(r'\(\d+\)', '', text)  # Remove patterns like "(3)"
    text = text.replace('  ', ' ')
    return text

# Remove sentences which are too short after cleaning and filtering ignoring 1 and 2 letter words
def filter_too_short(text, min_length=5):
    words_in_text = text.split(' ')
    copy = words_in_text.copy()
    for word in words_in_text:
        if len(word) <= 2:  # Remove words with 2 or fewer characters
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

# Function to find the most relevant sentences for a query using a specific model
def find_relevant_sentences(query, model, embeddings, top_n=5):
    query_embedding = model.encode([query])[0].reshape(1, -1)
    nbrs = NearestNeighbors(n_neighbors=min(top_n, len(embeddings)), metric='cosine').fit(embeddings)
    distances, indices = nbrs.kneighbors(query_embedding, n_neighbors=min(top_n, len(embeddings)))
    results = []
    added_sentences = set()
    for idx in indices[0]:
        sentence = df.iloc[idx]['sentence']
        if sentence not in added_sentences:
            added_sentences.add(sentence)
            results.append({
                'query': query,
                'sentence': sentence,
                'document': df.iloc[idx]['document'],
                'distance': distances[0][indices[0].tolist().index(idx)]
            })
    return sorted(results, key=lambda x: x['distance'])

# Load the pre-trained models with a fixed seed
models = {
    'SBERT': SentenceTransformer('all-MiniLM-L6-v2'),
    # 'SciBERT': SentenceTransformer('allenai/scibert_scivocab_uncased')
}
np.random.seed(42)

# Directory containing the PDF documents
pdf_dir = 'ref'

# Output directory for processed text files
output_dir = 'output'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Find all PDF files in the directory and its subdirectories
pdf_files = find_all_pdfs(pdf_dir)

# Extract text from PDFs and save to text files
txt_files = []
for pdf_file in pdf_files:
    print(f"Processing file: {pdf_file}")
    relative_path = os.path.relpath(pdf_file, pdf_dir)
    txt_file = os.path.join(output_dir, relative_path.replace('.pdf', '.txt'))

    # Ensure the output subdirectory exists
    os.makedirs(os.path.dirname(txt_file), exist_ok=True)

    sentences = extract_sentences_from_pdf(pdf_file)
    if sentences:
        with open(txt_file, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                f.write(sentence + '\n')
        txt_files.append(txt_file)
    else:
        print(f"No sentences extracted from file: {pdf_file}")

# Load and chunk documents into sentences
docs_sentences = []
doc_references = []

for txt_file in txt_files:
    with open(txt_file, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    if sentences:
        docs_sentences.extend(sentences)
        doc_references.extend([txt_file] * len(sentences))
    else:
        print(f"No sentences found in file: {txt_file}")

print(f"Total number of sentences: {len(docs_sentences)}")

# Check if any sentences were extracted
if len(docs_sentences) == 0:
    raise ValueError("No sentences were extracted from the text files. Please check the content and extraction logic.")

# Create a DataFrame for easy handling
df = pd.DataFrame({'sentence': docs_sentences, 'document': doc_references})

# Generate embeddings for all sentences using all models
embeddings = {}
for model_name, model in models.items():
    print(f"Generating embeddings using {model_name}...")
    start_time = time.time()
    embeddings[model_name] = model.encode(df['sentence'].tolist(), show_progress_bar=True)
    end_time = time.time()
    training_time = end_time - start_time
    if len(embeddings[model_name]) == 0:
        raise ValueError(f"Embeddings for {model_name} are empty. Please check the model and the input data.")
    # Save embeddings to CSV with training time in the filename
    embeddings_df = pd.DataFrame(embeddings[model_name])
    embeddings_df['sentence'] = df['sentence']
    embeddings_df['document'] = df['document']
    filename = f'embeddings_{model_name}_{training_time:.2f}s.csv'
    embeddings_df.to_csv(filename, index=False, encoding='utf-8')
    print(f"Embeddings for {model_name} saved to '{filename}' with training time {training_time:.2f} seconds.")

# Initialize the BERT question-answering model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)

# Initialize the summarization model and tokenizer
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
summarizer_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
summarizer = pipeline('summarization', model=summarizer_model, tokenizer=summarizer_tokenizer)

# List of queries for testing
queries = [
    "What is the purpose of Visual Knowledge Discovery?",
    "How are visual machine learning models built?",
    "What are the challenges of implementing AI in high-risk scenarios?",
    "What are Parallel Coordinates used for?",
    "What are the difficulties with Visual Knowledge Discovery?",
    "What is the VisCanvas software tool used for?",
    "What is the hyper methodology used in VisCanvas?",
    "What is the DCVis software?",
    "How do I use the DCVis software?",
    "What is a hyperblock?",
    "Define hyperblock."
]

# Process each query, compile answers, and save results to a text file
for query in queries:
    all_results = []
    for model_name, model in models.items():
        results = find_relevant_sentences(query, model, embeddings[model_name])
        for result in results:
            result['model'] = model_name
            all_results.append(result)
    results_df = pd.DataFrame(all_results)
    csv_filename = f'query_results_{queries.index(query) + 1}.csv'
    results_df.to_csv(csv_filename, index=False, encoding='utf-8')
    print(f"Results for query '{query}' saved to '{csv_filename}'")

    # Generate an answer for each of the top 5 results using the QA model
    final_answer_parts = []
    citations = []
    for i, result in enumerate(all_results[:5]):
        context = result['sentence']
        answer = qa_pipeline({'question': query, 'context': context})
        final_answer_parts.append(f"Similar sentence: {result['sentence']}Sub-answer: {answer['answer']} [{i+1}]")
        citations.append(f"[{i+1}] {result['document']}")

    # Combine the answers into a single paragraph using the summarization model
    combined_context = " ".join([result['sentence'] for result in all_results[:5]])
    summary = summarizer(combined_context, max_length=150, min_length=50, do_sample=False)[0]['summary_text']

    # Combine the answers, summary, and citations into the final text
    final_answer_text = f"Query: {query}\nFinal answer: {summary}\n\nContext:\n\n" + "\n\n".join(final_answer_parts) + "\n\n" + '\n'.join(citations)

    # Save the final answer with citations to a text file
    final_answer_filename = f'final_answer_{queries.index(query) + 1}.txt'
    with open(final_answer_filename, 'w', encoding='utf-8') as f:
        f.write(final_answer_text)
    print(f"Final answer for query '{query}' saved to '{final_answer_filename}'")
