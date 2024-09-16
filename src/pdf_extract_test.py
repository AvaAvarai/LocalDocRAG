import os
import random
import csv
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import pandas as pd
import tabula
import io

# Constants
PDF_FOLDER = "pdfs/"
OUTPUT_FOLDER = "output/"
SENTENCES_CSV = os.path.join(OUTPUT_FOLDER, "sentences.csv")
TABLES_CSV = os.path.join(OUTPUT_FOLDER, "tables.csv")
GRAPHICS_CSV = os.path.join(OUTPUT_FOLDER, "graphics.csv")
TABLES_FOLDER = os.path.join(OUTPUT_FOLDER, "tables")
GRAPHICS_FOLDER = os.path.join(OUTPUT_FOLDER, "graphics")

class DocumentProcessor:
    def __init__(self):
        self.pdfs = []
        self.sentences = []
        self.tables = []
        self.graphics = []

    def load_pdfs(self):
        for filename in os.listdir(PDF_FOLDER):
            if filename.endswith(".pdf"):
                self.pdfs.append(os.path.join(PDF_FOLDER, filename))

    def process_pdfs(self):
        for pdf_path in self.pdfs:
            reader = PdfReader(pdf_path)
            for page_num, page in enumerate(reader.pages):
                self.extract_sentences(page)
                self.extract_tables(pdf_path, page_num)
                self.extract_graphics(page, page_num, pdf_path)

    def extract_sentences(self, page):
        text = page.extract_text()
        sentences = text.split('.')
        self.sentences.extend([s.strip() + '.' for s in sentences if s.strip()])

    def extract_tables(self, pdf_path, page_num):
        tables = tabula.read_pdf(pdf_path, pages=page_num+1, multiple_tables=True)
        for i, table in enumerate(tables):
            table_data = {"pdf": os.path.basename(pdf_path), "page": page_num}
            file_location = os.path.join(TABLES_FOLDER, f"table_{len(self.tables)}.csv")
            table_data["file_location"] = file_location
            self.tables.append(table_data)
            table.to_csv(file_location, index=False)

    def extract_graphics(self, page, page_num, pdf_path):
        if '/XObject' in page['/Resources']:
            xObject = page['/Resources']['/XObject'].get_object()
            for obj in xObject:
                if xObject[obj]['/Subtype'] == '/Image':
                    try:
                        size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
                        data = xObject[obj].get_data()
                        if xObject[obj]['/ColorSpace'] == '/DeviceRGB':
                            mode = "RGB"
                        else:
                            mode = "P"
                        
                        img = Image.frombytes(mode, size, data)
                        graphic_data = {"pdf": os.path.basename(pdf_path), "page": page_num}
                        file_location = os.path.join(GRAPHICS_FOLDER, f"graphic_{len(self.graphics)}.png")
                        graphic_data["file_location"] = file_location
                        self.graphics.append(graphic_data)
                        img.save(file_location)
                    except ValueError as e:
                        print(f"Error processing image on page {page_num} of {pdf_path}: {str(e)}")
                        continue

    def save_data(self):
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        os.makedirs(TABLES_FOLDER, exist_ok=True)
        os.makedirs(GRAPHICS_FOLDER, exist_ok=True)

        with open(SENTENCES_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["sentence"])
            for sentence in self.sentences:
                writer.writerow([sentence])

        with open(TABLES_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["pdf", "page", "file_location"])
            writer.writeheader()
            for table in self.tables:
                writer.writerow(table)

        with open(GRAPHICS_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["pdf", "page", "file_location"])
            writer.writeheader()
            for graphic in self.graphics:
                writer.writerow(graphic)

class AssistantChat:
    def __init__(self):
        self.greetings = [
            "Hello! How can I assist you with your documents today?",
            "Welcome! What information would you like to retrieve from your documents?",
            "Greetings! How may I help you analyze your documents?",
            "Hi there! What would you like to know about your documents?"
        ]
        self.responses = {
            "default": [
                "I see. What specific information are you looking for in your documents?",
                "Understood. Can you provide more details about what you're searching for?",
                "Certainly. Which part of your documents would you like me to focus on?",
                "Thank you for your query. Is there a particular topic you'd like to explore further?",
                "I appreciate your question. How else can I assist you with your document analysis?"
            ],
            "goodbye": [
                "It was a pleasure assisting you with your documents. Have a great day!",
                "Thank you for using our document analysis service. Feel free to return if you need more help!",
                "Goodbye! I hope I was able to help you find the information you needed. Take care!"
            ]
        }

    def greet(self):
        return random.choice(self.greetings)

    def respond(self, user_input):
        user_input = user_input.lower()
        
        if "bye" in user_input or "goodbye" in user_input:
            return random.choice(self.responses["goodbye"])
        
        return random.choice(self.responses["default"])

def main():
    processor = DocumentProcessor()
    processor.load_pdfs()
    processor.process_pdfs()
    processor.save_data()

    assistant = AssistantChat()
    print("Assistant:", assistant.greet())

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["bye", "goodbye", "exit", "quit"]:
            print("Assistant:", assistant.respond(user_input))
            break
        print("Assistant:", assistant.respond(user_input))

if __name__ == "__main__":
    main()
