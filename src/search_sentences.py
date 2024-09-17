import csv
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Suppress warnings
warnings.filterwarnings('ignore')

def load_embeddings(csv_file):
    sentences = []
    sources = []
    embeddings = []

    # Read the CSV file
    with open(csv_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sentence = row.get('Sentence')
            source = row.get('Source')
            embedding_str = row.get('Embedding')
            if sentence and source and embedding_str:
                # Convert the JSON string back to a NumPy array
                embedding = np.array(json.loads(embedding_str))
                sentences.append(sentence)
                sources.append(source)
                embeddings.append(embedding)
            else:
                print(f"Skipping a row due to missing data: {row}")

    # Convert lists to NumPy arrays for efficient computation
    if embeddings:
        embeddings = np.vstack(embeddings)
    else:
        embeddings = np.array([])
    return sentences, sources, embeddings

def summarize_context(context, model, tokenizer, device):
    # Create the prompt for summarization
    prompt = f"Summarize the following content to capture the key points:\n\n{context}"
    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    # Generate the summary
    outputs = model.generate(**inputs, max_length=512)
    # Decode the summary
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary.strip()

def generate_answer(question, context, model, tokenizer, device):
    # Create the prompt for answer generation
    prompt = f"""You are an expert assistant for answering questions from a set of provided documents. Based only on the following context, answer the question briefly and avoid repetition.

    Question:
    {question}

    Context:
    {context}

    Answer:"""
    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    # Generate the answer
    outputs = model.generate(**inputs, max_length=512)
    # Decode the answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()

def main():
    # Load the embeddings from the CSV file
    csv_file = 'extracted_sentences.csv'  # Ensure this matches your output file
    print("Loading embeddings from CSV...")
    sentences, sources, embeddings = load_embeddings(csv_file)
    print(f"Loaded {len(sentences)} sentences.")

    if len(sentences) == 0 or embeddings.size == 0:
        print("No data found in the CSV file. Please ensure the preprocessing script ran successfully and generated data.")
        return

    # Initialize the embedding model
    model_name = 'all-MiniLM-L6-v2'  # Ensure this matches the model used previously

    # Correct device assignment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    embedder = SentenceTransformer(model_name, device=device)

    # Initialize the language model and tokenizer
    lm_model_name = 'google/flan-t5-large'  # Use a larger model if resources allow
    lm_tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
    lm_model = AutoModelForSeq2SeqLM.from_pretrained(lm_model_name).to(device)

    while True:
        # Accept user query
        query = input("\nEnter your query (or type 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            break

        # Generate embedding for the query
        print("Generating query embedding...")
        query_embedding = embedder.encode([query])[0]

        # Compute cosine similarities
        print("Computing similarities...")
        similarities = cosine_similarity([query_embedding], embeddings)[0]

        # Apply similarity threshold
        similarity_threshold = 0.5  # Adjust as needed
        relevant_indices = [idx for idx, score in enumerate(similarities) if score >= similarity_threshold]

        if not relevant_indices:
            print("No relevant information found in the documents.")
            continue

        # Collect all similar sentences
        similar_sentences = [sentences[idx] for idx in relevant_indices]
        sources_used = [sources[idx] for idx in relevant_indices]

        # Combine all similar sentences into one context
        combined_context = ' '.join(similar_sentences)

        # Check if the context exceeds the model's max input length
        max_input_length = 2048  # Adjust based on the model's max input length
        context_token_length = len(lm_tokenizer.encode(combined_context))
        if context_token_length > max_input_length:
            # Summarize the context to reduce length
            print("Context is too long, summarizing...")
            combined_context = summarize_context(combined_context, lm_model, lm_tokenizer, device)

        # Generate the final answer
        print("\nGenerating answer...")
        final_answer = generate_answer(query, combined_context, lm_model, lm_tokenizer, device)

        # Display the final answer
        print("\nAnswer:")
        print(final_answer)

        # Display citations
        print("\nRelevant Sources:")
        for source in set(sources_used):
            print(f"- {source}")

if __name__ == "__main__":
    main()
