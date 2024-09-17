import csv
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BartTokenizer, BartForConditionalGeneration

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

def generate_sub_answer(question, context, model, tokenizer, device):
    # Create prompt
    prompt = f"Question: {question}\nContext: {context}\nAnswer:"
    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    # Generate the answer
    outputs = model.generate(**inputs, max_length=128)
    # Decode the answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()

def summarize_text(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=1024).to(device)
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=150, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary.strip()

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

    # Initialize the language model and tokenizer for sub-answer generation
    lm_model_name = 'google/flan-t5-base'  # You can use 't5-base', 'flan-t5-base', etc.
    lm_tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
    lm_model = AutoModelForSeq2SeqLM.from_pretrained(lm_model_name).to(device)

    # Initialize the summarization model and tokenizer
    summarizer_model_name = 'facebook/bart-large-cnn'
    summarizer_tokenizer = BartTokenizer.from_pretrained(summarizer_model_name)
    summarizer_model = BartForConditionalGeneration.from_pretrained(summarizer_model_name).to(device)

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

        # Sort indices by similarity
        sorted_indices = sorted(relevant_indices, key=lambda idx: similarities[idx], reverse=True)

        if not sorted_indices:
            print("No relevant information found in the documents.")
            continue

        # Generate sub-answers
        max_subanswers = 5  # Adjust as needed
        sub_answers = []
        sources_used = []
        print("\nGenerating sub-answers...")
        for idx in sorted_indices[:max_subanswers]:
            sentence = sentences[idx]
            source = sources[idx]
            # Generate sub-answer
            sub_answer = generate_sub_answer(query, sentence, lm_model, lm_tokenizer, device)
            sub_answers.append(sub_answer)
            sources_used.append(source)

        # Combine sub-answers
        combined_sub_answers = "\n".join(sub_answers)

        # Generate final answer by summarizing the sub-answers
        print("\nGenerating final answer...")
        final_answer = summarize_text(combined_sub_answers, summarizer_model, summarizer_tokenizer, device)

        # Display the final answer
        print("\nAnswer:")
        print(final_answer)

        # Display citations (optional)
        print("\nRelevant Sources:")
        for source in set(sources_used):
            print(f"- {source}")

if __name__ == "__main__":
    main()
