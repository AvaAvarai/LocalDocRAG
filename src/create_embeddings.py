from transformers import AutoTokenizer
from sentence_transformers import models, SentenceTransformer
import pandas as pd
import numpy as np
import torch
import warnings
from tqdm import tqdm
import onnxruntime as ort

# Suppress specific FutureWarning about `resume_download`
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

# Set ONNX Runtime logging level to ERROR to suppress warnings
ort.set_default_logger_severity(3)  # 3 corresponds to ERROR level

# Load SciBERT tokenizer
scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", resume_download=True)

# Define the SentenceTransformer model
word_embedding_model = models.Transformer("allenai/scibert_scivocab_uncased")
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Create a wrapper class for the model
class SentenceTransformerWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SentenceTransformerWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        features = {'input_ids': input_ids, 'attention_mask': attention_mask}
        return self.model(features)['sentence_embedding']

wrapped_model = SentenceTransformerWrapper(model)

# Prepare dummy input for ONNX export
dummy_input_text = ["This is a dummy input for ONNX export."]
inputs = scibert_tokenizer(dummy_input_text, return_tensors="pt", padding="max_length", max_length=128, truncation=True)
dummy_input = (inputs["input_ids"].to(torch.int64), inputs["attention_mask"].to(torch.int64))

# Convert the PyTorch model to ONNX format with opset version 14
torch.onnx.export(
    wrapped_model,
    dummy_input,
    "scibert.onnx",
    opset_version=14,
    input_names=["input_ids", "attention_mask"],
    output_names=["output"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "output": {0: "batch_size"}
    }
)

# Set the execution provider to DirectML
providers = [("DmlExecutionProvider", {"device_id": 0})]
session = ort.InferenceSession("scibert.onnx", providers=providers)

def load_cleaned_texts(file_path):
    return pd.read_csv(file_path, encoding='utf-8')

def create_embeddings(df, batch_size=32):
    sentences = df['sentence'].tolist()
    embeddings = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Encoding sentences"):
        batch_sentences = sentences[i:i + batch_size]
        inputs = scibert_tokenizer(batch_sentences, return_tensors='np', padding=True, truncation=True, max_length=128)
        input_ids = inputs['input_ids'].astype(np.int64)
        attention_mask = inputs['attention_mask'].astype(np.int64)
        onnx_inputs = {session.get_inputs()[0].name: input_ids, session.get_inputs()[1].name: attention_mask}
        batch_embeddings = session.run(None, onnx_inputs)[0]
        embeddings.extend(batch_embeddings)
    df['embedding'] = embeddings
    return df

def save_embeddings_to_csv(df, output_file='embeddings.csv'):
    df['embedding'] = df['embedding'].apply(lambda x: x.tolist())
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
    print("Loaded texts:", cleaned_texts_df.head())
    embeddings_df = create_embeddings(cleaned_texts_df)
    save_embeddings_to_csv(embeddings_df)
    print_stats(embeddings_df)
