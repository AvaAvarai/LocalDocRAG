import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import Embedding
from torch.optim import Adam
from gensim.utils import simple_preprocess
from tqdm import tqdm
import numpy as np
import json
import onnx
import onnxruntime as ort
import random

class SentencesDataset(Dataset):
    def __init__(self, sentences, window_size=5):
        self.sentences = sentences
        self.window_size = window_size
        self.vocab = self.build_vocab(sentences)
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.pairs = self.generate_pairs(sentences)

    def build_vocab(self, sentences):
        vocab = set()
        for sentence in sentences:
            for word in sentence:
                vocab.add(word)
        return list(vocab)

    def generate_pairs(self, sentences):
        pairs = []
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(max(0, i - self.window_size), min(len(sentence), i + self.window_size + 1)):
                    if i != j:
                        pairs.append((word, sentence[j]))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        word, context = self.pairs[idx]
        return torch.tensor(self.word_to_idx[word]), torch.tensor(self.word_to_idx[context])

def collate_fn(batch):
    word_batch, context_batch = zip(*batch)
    return torch.stack(word_batch), torch.stack(context_batch)

class Word2VecModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2VecModel, self).__init__()
        self.embeddings = Embedding(vocab_size, embedding_dim)

    def forward(self, inputs):
        return self.embeddings(inputs)

def negative_sampling_loss(model, pos_pairs, neg_samples, device):
    pos_input, pos_context = pos_pairs
    neg_context = neg_samples

    pos_input, pos_context = pos_input.to(device), pos_context.to(device)
    neg_context = neg_context.to(device)

    pos_input_embeds = model(pos_input)
    pos_context_embeds = model(pos_context)
    neg_context_embeds = model(neg_context)

    pos_score = torch.sum(pos_input_embeds * pos_context_embeds, dim=1)
    neg_score = torch.bmm(neg_context_embeds, pos_input_embeds.unsqueeze(2)).squeeze()

    pos_loss = -torch.log(torch.sigmoid(pos_score)).mean()
    neg_loss = -torch.log(torch.sigmoid(-neg_score)).mean()

    return pos_loss + neg_loss

def train_word2vec(sentences, embedding_dim=50, window=5, epochs=5, batch_size=64, learning_rate=0.001, num_neg_samples=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SentencesDataset(sentences, window)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = Word2VecModel(len(dataset.vocab), embedding_dim).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0
        for pos_input, pos_context in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            neg_context = torch.randint(0, len(dataset.vocab), (pos_input.size(0), num_neg_samples), device=device)

            optimizer.zero_grad()
            loss = negative_sampling_loss(model, (pos_input, pos_context), neg_context, device)
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
        return np.zeros(50).tolist()  # Return a zero vector if no words are found
    inputs = np.array(indices, dtype=np.int64).reshape(1, -1)
    onnx_inputs = {session.get_inputs()[0].name: inputs}
    outputs = session.run(None, onnx_inputs)[0]
    return np.mean(outputs, axis=1).tolist()


def main():
    # Load data
    df = pd.read_csv('cleaned_texts.csv')

    # Prepare sentences for Word2Vec training
    sentences = [simple_preprocess(sentence) for sentence in df['sentence']]

    # Initialize the training device to CPU, as PyTorch direct support for AMD GPUs via DirectML is not available
    device = torch.device("cpu")

    # Train Word2Vec model on CPU
    model, dataset = train_word2vec(sentences, embedding_dim=50, window=5, epochs=5, batch_size=64, learning_rate=0.001)
    model.to(device)

    # Export model to ONNX
    export_to_onnx(model, dataset, filename='word2vec.onnx')

    # Set the execution provider to DirectML
    providers = [("DmlExecutionProvider", {"device_id": 0})]
    session = ort.InferenceSession('word2vec.onnx', providers=providers)

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
