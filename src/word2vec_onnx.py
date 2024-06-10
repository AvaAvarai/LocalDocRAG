import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import Embedding
from torch.optim import Adam
from gensim.utils import simple_preprocess
from tqdm import tqdm
import numpy as np
import json
import os
import onnx
import onnxruntime as ort

class SentencesDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences
        self.vocab = self.build_vocab(sentences)
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
    
    def build_vocab(self, sentences):
        vocab = set()
        for sentence in sentences:
            for word in sentence:
                vocab.add(word)
        return list(vocab)
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        words = self.sentences[idx]
        indices = [self.word_to_idx[word] for word in words if word in self.word_to_idx]
        return torch.tensor(indices, dtype=torch.long)

def collate_fn(batch):
    batch = [item for item in batch if len(item) > 0]
    max_len = max(len(item) for item in batch)
    padded_batch = [torch.cat([item, torch.zeros(max_len - len(item), dtype=torch.long)]) for item in batch]
    return torch.stack(padded_batch)

class Word2VecModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2VecModel, self).__init__()
        self.embeddings = Embedding(vocab_size, embedding_dim)
    
    def forward(self, inputs):
        return self.embeddings(inputs)

def train_word2vec(sentences, embedding_dim=100, window=5, epochs=5, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = SentencesDataset(sentences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    model = Word2VecModel(len(dataset.vocab), embedding_dim).to(device)
    optimizer = Adam(model.parameters())
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            embeddings = model(batch)
            loss = (embeddings.norm(2, dim=1) ** 2).mean()  # Simplified loss for demonstration
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")
    
    return model, dataset

def export_to_onnx(model, dataset, filename='word2vec.onnx'):
    dummy_input = torch.tensor([[0]], dtype=torch.long).to(next(model.parameters()).device)
    torch.onnx.export(model, dummy_input, filename, input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size', 1: 'seq_length'}, 'output': {0: 'batch_size', 1: 'seq_length'}})

def sentence_vector(sentence, session, word_to_idx):
    words = simple_preprocess(sentence)
    indices = [word_to_idx[word] for word in words if word in word_to_idx]
    if not indices:
        return np.zeros(100).tolist()  # Return a zero vector if no words are found
    inputs = np.array(indices, dtype=np.int64).reshape(1, -1)
    outputs = session.run(None, {'input': inputs})[0]
    return np.mean(outputs, axis=1).tolist()

def main():
    # Load data
    df = pd.read_csv('cleaned_texts.csv')

    # Prepare sentences for Word2Vec training
    sentences = [simple_preprocess(sentence) for sentence in df['sentence']]

    # Train Word2Vec model with GPU support
    model, dataset = train_word2vec(sentences, embedding_dim=100, window=5, epochs=5, batch_size=64)

    # Export model to ONNX
    export_to_onnx(model, dataset, filename='word2vec.onnx')

    # Load ONNX model
    session = ort.InferenceSession('word2vec.onnx')

    # Extract word to index mapping
    word_to_idx = dataset.word_to_idx

    # Process sentences to obtain embeddings
    embeddings = [sentence_vector(sentence, session, word_to_idx) for sentence in df['sentence']]

    # Assign results back to the DataFrame
    df['embedding'] = [json.dumps(embedding) for embedding in embeddings]

    # Save embeddings to a new CSV file
    df.to_csv('embeddings_word2vec.csv', index=False)

if __name__ == '__main__':
    main()
