import os
import warnings

# Suppress TensorFlow oneDNN optimization messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from PyPDF2 import PdfReader
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
from datasets import Dataset
import torch
import onnxruntime as ort
from transformers.onnx import export
import gradio as gr
from tqdm import tqdm

# Constants
MAX_SEQ_LENGTH = 512  # Maximum sequence length for the model
MAX_NEW_TOKENS = 50  # Maximum number of new tokens to generate

# Step 1: Extract Text from PDFs
def extract_text_from_pdfs(ref_folder):
    text_data = []
    for root, dirs, files in os.walk(ref_folder):
        for file in tqdm(files, desc="Extracting text from PDFs"):
            if file.endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                with open(pdf_path, 'rb') as f:
                    reader = PdfReader(f)
                    text = ""
                    for page_num in range(len(reader.pages)):
                        text += reader.pages[page_num].extract_text()
                    if text.strip():
                        text_data.append(text)
                    else:
                        print(f"Warning: No text extracted from {pdf_path}")
    return text_data

# Extract text from the ref folder
pdf_texts = extract_text_from_pdfs('ref')
print(f"\nExtracted text from {len(pdf_texts)} PDFs.")

# Function to split text into chunks
def split_text_into_chunks(text, max_length=MAX_SEQ_LENGTH):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 > max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1
        else:
            current_chunk.append(word)
            current_length += len(word) + 1
            
    if current_chunk:
        chunks.append(' '.join(current_chunk))
        
    return chunks

# Verify chunk splitting
for text in pdf_texts:
    chunks = split_text_into_chunks(text)
    print(f"Text split into {len(chunks)} chunks.")

# Step 2: Generate Synthetic Question-Answer Pairs using GPU
device = 0 if torch.cuda.is_available() else -1  # Use GPU if available, else use CPU
question_generator = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl", device=device)

def generate_qa_pairs(text):
    qas = []
    chunks = split_text_into_chunks(text)
    for i, chunk in enumerate(chunks):
        inputs = "highlight: " + chunk
        results = question_generator(inputs, max_new_tokens=MAX_NEW_TOKENS)
        chunk_qas = []
        for result in results:
            qa_pairs = result['generated_text'].split("<sep>")
            if len(qa_pairs) == 2:
                chunk_qas.append({
                    "question": qa_pairs[0].strip(),
                    "answer": qa_pairs[1].strip()
                })
        if not chunk_qas:
            print(f"Warning: No QA pairs generated for chunk {i} of {len(chunks)}")
        qas.extend(chunk_qas)
    return qas

qa_pairs = []
for text in tqdm(pdf_texts, desc="Generating QA pairs"):
    pairs = generate_qa_pairs(text)
    if not pairs:
        print(f"Warning: No QA pairs generated for text: {text[:100]}...")  # Print first 100 characters for reference
    qa_pairs.extend(pairs)

print(f"Generated {len(qa_pairs)} QA pairs.")

# Step 3: Create a Dataset
df = pd.DataFrame(qa_pairs)
df.to_csv('qa_dataset.csv', index=False)
print(f"Saved QA pairs to 'qa_dataset.csv'.")

# Step 4: Fine-Tune SciBERT Using ONNX Runtime
# Load dataset
dataset = Dataset.from_pandas(pd.read_csv('qa_dataset.csv'))

# Initialize tokenizer and model
model_name = "allenai/scibert_scivocab_uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Tokenize the dataset
def preprocess(examples):
    inputs = tokenizer(
        examples['question'],
        examples['answer'],
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    return inputs

tokenized_dataset = dataset.map(preprocess, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train the model
trainer.train()

# Export the model to ONNX
onnx_path = "scibert_qa.onnx"
export(model, tokenizer, onnx_path)

# Optimize with ONNX Runtime
providers = [("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
ort_session = ort.InferenceSession(onnx_path, providers=providers)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def predict(question, context):
    inputs = tokenizer(question, context, return_tensors='pt')
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inputs['input_ids']),
                  ort_session.get_inputs()[1].name: to_numpy(inputs['attention_mask'])}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs

# Step 5: Create a Chat Interface
def answer_question(question, context):
    answer = predict(question, context)
    return answer

interface = gr.Interface(fn=answer_question, inputs=["text", "text"], outputs="text")
interface.launch()
