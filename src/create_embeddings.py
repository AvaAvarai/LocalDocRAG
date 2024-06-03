from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer, models
import pandas as pd
import numpy as np
import warnings

# Suppress specific FutureWarning about `resume_download`
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

# Load SciBERT model and tokenizer from Hugging Face with resume_download set to True
scibert_model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased", resume_download=True)
scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", resume_download=True)
 
# Create a SentenceTransformer model from the SciBERT components
word_embedding_model = models.Transformer(scibert_model.name_or_path)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

# Define the SentenceTransformer model
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

def load_cleaned_texts(file_path):
    df = pd.read_csv(file_path)
    return df

def create_embeddings(df):
    sentences = df['sentence'].tolist()
    embeddings = model.encode(sentences)
    df['embedding'] = embeddings.tolist()
    return df

def save_embeddings_to_csv(df, output_file='embeddings.csv'):
    df.to_csv(output_file, index=False)

def print_stats(df):
    num_sentences = len(df)
    embedding_shape = np.array(df['embedding'].iloc[0]).shape if num_sentences > 0 else (0,)
    print(f"Number of sentences processed: {num_sentences}")
    print(f"Shape of embeddings: {embedding_shape}")
    print("Sample embeddings:")
    print(df[['sentence', 'embedding']].head())

if __name__ == "__main__":
    cleaned_texts_df = load_cleaned_texts('cleaned_texts.csv')
    embeddings_df = create_embeddings(cleaned_texts_df)
    save_embeddings_to_csv(embeddings_df)
    print_stats(embeddings_df)
