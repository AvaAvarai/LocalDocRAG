import os
import warnings

# Suppress specific FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.")

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Directory where PDF papers are stored
ref_dir = 'ref'

# Function to load and extract text from PDF papers
def load_papers(ref_dir):
    papers = []
    print("Loading papers from directory...")
    for root, _, files in os.walk(ref_dir):
        for filename in files:
            if filename.endswith('.pdf'):
                filepath = os.path.join(root, filename)
                print(f"Processing file: {filepath}")
                reader = PdfReader(filepath)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                papers.append({"title": filename, "content": text})
    print(f"Loaded {len(papers)} papers.")
    return papers

# Load the papers
papers = load_papers(ref_dir)

# Check if any papers were loaded
if not papers:
    print("No papers found in the 'ref' directory.")
    exit()

# Initialize the SciBERT model and tokenizer for QA
print("Initializing SciBERT model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModelForQuestionAnswering.from_pretrained("allenai/scibert_scivocab_uncased")

# Create a QA pipeline
print("Creating QA pipeline...")
nlp = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Initialize the SentenceTransformer model for embeddings
print("Initializing SentenceTransformer model for embeddings...")
embedder = SentenceTransformer('all-mpnet-base-v2')

# Create embeddings for the papers
print("Creating embeddings for the papers...")
paper_texts = [paper["content"] for paper in papers]
embeddings = embedder.encode(paper_texts, convert_to_numpy=True)

# Check embeddings shape
print(f"Embeddings shape: {embeddings.shape}")

# Build FAISS index
print("Building FAISS index...")
d = embeddings.shape[1]  # Dimensionality of the embeddings
index = faiss.IndexFlatL2(d)
index.add(np.array(embeddings))
print("FAISS index built successfully.")

# Function to ask questions
def ask_question(question):
    print(f"Asking question: {question}")
    question_embedding = embedder.encode([question], convert_to_numpy=True)
    _, I = index.search(np.array(question_embedding), k=1)
    closest_paper = papers[I[0][0]]
    context = closest_paper["content"]
    result = nlp(question=question, context=context)
    return result['answer']

# Main function to run the QA interface
def main():
    print("Welcome to PaperQA!")
    print("Ask your questions about the papers in the ref directory.")
    
    while True:
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        answer = ask_question(question)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
