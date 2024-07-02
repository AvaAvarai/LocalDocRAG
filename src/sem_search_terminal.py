import os
import re
from PyPDF2 import PdfReader
import markdown
import pandas as pd
import time
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline, AutoModelForSeq2SeqLM
from sklearn.neighbors import NearestNeighbors
import numpy as np
from docx import Document

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

# Function to extract sentences from a Markdown file
def extract_sentences_from_markdown(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    html = markdown.markdown(text)
    sentences = re.split(r'\.\s+', html)
    seen_sentences = set()
    cleaned_sentences = [clean_sentence(sentence, seen_sentences) for sentence in sentences]
    cleaned_sentences = [remove_references(sentence) for sentence in cleaned_sentences if sentence]
    return [sentence for sentence in cleaned_sentences if sentence and filter_too_short(sentence)]

# Function to extract sentences from a DOCX file
def extract_sentences_from_docx(file_path):
    doc = Document(file_path)
    sentences = []
    seen_sentences = set()
    for para in doc.paragraphs:
        text = para.text
        if text:
            cleaned_sentences = [clean_sentence(sentence, seen_sentences) for sentence in text.split('. ')]
            cleaned_sentences = [remove_references(sentence) for sentence in cleaned_sentences if sentence]
            sentences.extend([sentence for sentence in cleaned_sentences if sentence and filter_too_short(sentence)])
    return sentences

# Function to find all PDF, Markdown, and DOCX files in a directory and its subdirectories
def find_all_files(directory, extensions=['.pdf', '.md', '.docx']):
    files = []
    for root, _, files_in_dir in os.walk(directory):
        for file in files_in_dir:
            if any(file.endswith(ext) for ext in extensions):
                files.append(os.path.join(root, file))
    return files

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
            similarity_score = 1 - distances[0][indices[0].tolist().index(idx)]
            results.append({
                'query': query,
                'sentence': sentence,
                'document': df.iloc[idx]['document'],
                'distance': distances[0][indices[0].tolist().index(idx)],
                'similarity_score': similarity_score
            })
    return sorted(results, key=lambda x: x['distance'])

# Load the pre-trained model
model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)
np.random.seed(42)

# Directory containing the PDF, Markdown, and DOCX documents
doc_dir = 'ref'

# Output directory for processed text files
output_dir = 'output'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Find all PDF, Markdown, and DOCX files in the directory and its subdirectories
doc_files = find_all_files(doc_dir)

# Extract text from documents and save to text files
txt_files = []
for doc_file in doc_files:
    print(f"Processing file: {doc_file}")
    relative_path = os.path.relpath(doc_file, doc_dir)
    txt_file = os.path.join(output_dir, relative_path.rsplit('.', 1)[0] + '.txt')

    # Ensure the output subdirectory exists
    os.makedirs(os.path.dirname(txt_file), exist_ok=True)

    if doc_file.endswith('.pdf'):
        sentences = extract_sentences_from_pdf(doc_file)
    elif doc_file.endswith('.md'):
        sentences = extract_sentences_from_markdown(doc_file)
    elif doc_file.endswith('.docx'):
        sentences = extract_sentences_from_docx(doc_file)
    else:
        continue

    if sentences:
        with open(txt_file, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                f.write(sentence + '\n')
        txt_files.append(txt_file)
    else:
        print(f"No sentences extracted from file: {doc_file}")

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

# Initialize the BERT question-answering model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
qa_model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
qa_pipeline = pipeline('question-answering', model=qa_model, tokenizer=tokenizer)

# Initialize the summarization model and tokenizer
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
summarizer_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
summarizer = pipeline('summarization', model=summarizer_model, tokenizer=summarizer_tokenizer)

# Interactive chat-like interface
while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break

    threshold = 0.5  # Similarity score threshold
    results = find_relevant_sentences(query, model, embeddings)

    final_answer_parts = []
    citations = []
    combined_context = ""
    for i, result in enumerate([r for r in results if r['similarity_score'] >= threshold][:5]):
        context = result['sentence']
        answer = qa_pipeline({'question': query, 'context': context})
        final_answer_parts.append(f"Context: {result['sentence']} [{i+1}]\nAnswer: {answer['answer']}\nSimilarity score: {result['similarity_score']:.2f}")
        combined_context += f"Since {result['sentence']}, {answer['answer']} [{i+1}]. "
        citations.append(f"[{i+1}] {result['document']}")

    if combined_context == "":
        final_answer_text = "No relevant answers available."
    else:
        # get the number of tokens in the combined context
        length = len(summarizer_tokenizer(combined_context)['input_ids'])
        summarization_percentage = 0.5
        minimum_percentage = 0.25
        
        max_length = int(length * summarization_percentage)
        min_length = int(length * minimum_percentage)
        
        # Generate a summary answer using the summarization model
        summary = summarizer(combined_context, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']

        # Combine the answers, summary, and citations into the final text
        final_answer_text = f"Query: {query}\nFinal answer: {summary}\n\nContext:\n\n" + "\n\n".join(final_answer_parts) + "\n\n" + '\n'.join(citations)

    # Print the final answer
    print(final_answer_text)
