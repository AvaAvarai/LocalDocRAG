import csv
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
import torch

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
            sentence = row['Sentence']
            source = row['Source']
            embedding_str = row['Embedding']
            # Convert the JSON string back to a NumPy array
            embedding = np.array(json.loads(embedding_str))

            sentences.append(sentence)
            sources.append(source)
            embeddings.append(embedding)

    # Convert lists to NumPy arrays for efficient computation
    embeddings = np.vstack(embeddings)
    return sentences, sources, embeddings

def main():
    # Load the embeddings from the CSV file
    csv_file = 'extracted_sentences.csv'  # Ensure this matches your output file
    print("Loading embeddings from CSV...")
    sentences, sources, embeddings = load_embeddings(csv_file)
    print(f"Loaded {len(sentences)} sentences.")

    # Initialize the embedding model
    model_name = 'all-MiniLM-L6-v2'  # Ensure this matches the model used previously

    # Correct device assignment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    embedder = SentenceTransformer(model_name, device=device)

    # Initialize the LLM for answer generation
    llm_model_name = "gpt2"  # You can change this to a more powerful model if available
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Assign pad_token to eos_token
    llm = AutoModelForCausalLM.from_pretrained(llm_model_name, pad_token_id=tokenizer.eos_token_id).to(device)

    while True:
        # Accept user query
        print("\nEnter your query (or type 'exit' to quit): ", end='', flush=True)
        query = input()
        if query.lower() == 'exit':
            break

        # Generate embedding for the query
        print("Generating query embedding...")
        query_embedding = embedder.encode([query])[0]

        # Compute cosine similarities
        print("Computing similarities...")
        similarities = cosine_similarity([query_embedding], embeddings)[0]

        # Sort sentences by similarity
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Prepare context for LLM
        max_length = 512
        context = ""
        for idx in sorted_indices:
            sentence = sentences[idx]
            if len(tokenizer.encode(context + sentence)) > max_length:  # Adjust this number based on your LLM's max token limit
                break
            context += sentence + " "

        prompt = f"Based on the following information, answer the question: '{query}'\n\nInformation:\n{context}\n\nAnswer:"

        # Generate answer using LLM
        print("\nGenerating answer...")
        input_ids = tokenizer.encode(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

        output = llm.generate(input_ids, attention_mask=attention_mask, max_new_tokens=150, num_return_sequences=1, no_repeat_ngram_size=2)
        answer = tokenizer.decode(output[0], skip_special_tokens=True)

        # Display the generated answer
        print("\nFinal Answer:")
        print(answer)

        # Display citations
        print("\nRelevant Information:")
        for i, idx in enumerate(sorted_indices[:10]):  # Show top 10 most relevant sentences
            print(f"[{i+1}] {sentences[idx]} (Source: {sources[idx]}, Similarity: {similarities[idx]:.4f})")

if __name__ == "__main__":
    main()
