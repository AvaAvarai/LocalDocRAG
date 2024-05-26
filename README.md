# EmbedQA

EmbedQA is a semantic embedding-based question-answering system that processes PDFs to generate embeddings for sentences, enabling semantic search and question answering using these embeddings. The system utilizes `sentence-transformers` for generating embeddings and `transformers` for question answering.

## Overview

1. **PDF Extraction**: Load and extract text from PDF files.
2. **Text Cleaning and Splitting**: Clean and split the extracted text into semantically useful sentences.
3. **Embedding Generation**: Generate embeddings for the cleaned sentences using a pre-trained model.
4. **Similarity Search**: Find the top k most similar sentences to a given question.
5. **Contextual Response Generation**: Combine the top k similar sentences and their neighboring sentences to form a context for generating a response using a question-answering model.
6. **Graphical User Interface**: Provide a GUI for user interaction, allowing users to ask questions and get responses based on the processed PDF content.

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/your-username/EmbedQA.git
    cd EmbedQA
    ```

2. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Running the Application

1. **Place your PDF files**:
    - Put your PDF files in the `ref` directory.

2. **Run the main script**:
    ```sh
    python src/embed.py
    ```

### Explanation of the Process

1. **PDF Extraction**:
    - Load PDF files from the specified directory and extract text from each page using `PyPDF2`.

2. **Text Cleaning and Splitting**:
    - Clean the extracted text by removing HTML tags, non-ASCII characters, and multiple spaces.
    - Split the cleaned text into sentences and filter out sentences that are not semantically useful (e.g., too short, contain mostly numbers/symbols).

3. **Embedding Generation**:
    - Generate sentence embeddings using the `sentence-transformers` model `nli-roberta-large`.

4. **Saving and Loading Embeddings**:
    - Save the generated embeddings and their corresponding sentences to a CSV file.
    - Load the embeddings from the CSV file for similarity search.

5. **Similarity Search**:
    - Use the `sentence-transformers` model to find the top k most similar sentences to the user query based on cosine similarity.

6. **Contextual Response Generation**:
    - Extract the neighboring sentences for each of the top k similar sentences.
    - Combine these sentences to form a context.
    - Use the `transformers` model `deepset/roberta-large-squad2` to generate a response based on the combined context.

7. **Graphical User Interface**:
    - Provide a Tkinter-based GUI for users to input their questions and receive responses.
    - Display the top k most similar sentences and the generated response.

### Example Usage

1. **Start the application**:
    ```sh
    python src/embed.py
    ```

2. **Interact with the GUI**:
    - Enter a question in the input field.
    - Click "Send" to get the response generated based on the content of the PDFs in the `ref` directory.

## Dependencies

- `PyPDF2`: For extracting text from PDF files.
- `sentence-transformers`: For generating sentence embeddings.
- `transformers`: For question answering.
- `nltk`: For natural language processing tasks such as tokenization and stopword removal.
- `spacy`: For advanced natural language processing tasks.
- `scikit-learn`: For dimensionality reduction (PCA, t-SNE).
- `matplotlib`: For plotting embeddings.
- `tkinter`: For creating the graphical user interface.
- `beautifulsoup4`: For cleaning HTML tags from text.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes or additions.

## License

This project is licensed under the MIT License.
