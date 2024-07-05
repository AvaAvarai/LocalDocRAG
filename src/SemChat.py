import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import BertForQuestionAnswering, BertTokenizer, BartForConditionalGeneration, BartTokenizer

class QAGui:
    def __init__(self, root):
        self.root = root
        self.root.title("QA Chatbot")

        self.query_label = tk.Label(root, text="Query:")
        self.query_label.pack()

        self.query_entry = tk.Entry(root, width=50)
        self.query_entry.pack()

        self.k_label = tk.Label(root, text="Number of results (K):")
        self.k_label.pack()

        self.k_scale = tk.Scale(root, from_=1, to_=10, orient=tk.HORIZONTAL)
        self.k_scale.pack()

        self.query_button = tk.Button(root, text="Submit Query", command=self.submit_query)
        self.query_button.pack()

        self.result_text = tk.Text(root, height=20, width=80)
        self.result_text.pack()

        self.load_embeddings_button = tk.Button(root, text="Load Embeddings", command=self.load_embeddings)
        self.load_embeddings_button.pack()

        self.model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
        self.qa_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        self.qa_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        self.bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        self.bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        self.embeddings = None
        self.sentences = None
        self.documents = None

        # Set seeds for reproducibility
        self.set_seeds(42)

    def set_seeds(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def load_embeddings(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            data = pd.read_csv(file_path)
            self.embeddings = data.iloc[:, :-2].values
            self.sentences = data['sentence'].values
            self.documents = data['document'].values
            messagebox.showinfo("Info", "Embeddings loaded successfully")

    def submit_query(self):
        query = self.query_entry.get()
        if not query:
            messagebox.showwarning("Warning", "Please enter a query")
            return

        if self.embeddings is None:
            messagebox.showwarning("Warning", "Please load embeddings first")
            return

        k = self.k_scale.get()
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        embeddings_tensor = torch.tensor(self.embeddings, dtype=query_embedding.dtype)  # Ensure dtype match
        similarities = util.pytorch_cos_sim(query_embedding, embeddings_tensor)[0]
        top_k_indices = torch.topk(similarities, k=k).indices

        results = []
        citations = []
        answers = []

        for idx in top_k_indices:
            idx = idx.item()
            sentence = self.sentences[idx]
            document = self.documents[idx]
            score = similarities[idx].item()
            citation = f"[{len(citations) + 1}] {document}"
            answer = self.get_answer(query, sentence)
            results.append(f"Similar sentence: {sentence}\nSub-answer: {answer}\nSimilarity score: {score:.4f}\n")
            citations.append(citation)
            answers.append(answer)

        combined_results = " ".join([f"SEGMENT {i+1}: {self.sentences[idx.item()]} [{i+1}]." for i, idx in enumerate(top_k_indices)])
        summarized_result = self.summarize_with_citations(" ".join(answers))

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"Final answer: {summarized_result}\n\n")
        self.result_text.insert(tk.END, "\n".join(results))
        self.result_text.insert(tk.END, f"\nCitations:\n" + "\n".join(citations))

    def get_answer(self, question, context):
        inputs = self.qa_tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]

        outputs = self.qa_model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1

        answer = self.qa_tokenizer.convert_tokens_to_string(self.qa_tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        return answer

    def summarize_with_citations(self, text):
        inputs = self.bart_tokenizer([text], max_length=1024, return_tensors='pt')
        summary_ids = self.bart_model.generate(inputs['input_ids'], num_beams=4, max_length=150, early_stopping=True, do_sample=False)
        summary = self.bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

if __name__ == "__main__":
    root = tk.Tk()
    app = QAGui(root)
    root.mainloop()
