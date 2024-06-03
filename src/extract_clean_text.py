import fitz  # PyMuPDF
import os
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import download
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from concurrent.futures import ProcessPoolExecutor, as_completed

STOPWORDS = []

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text.encode('utf-8', errors='ignore').decode('utf-8')

def clean_text(text):
    text = re.sub(r'-\s*\n', '', text)  # Remove hyphens at line breaks and join the words
    text = text.replace('\n', ' ')  # Replace newlines with spaces
    text = re.sub(r'\b"\s', '" ', text)  # Remove misplaced space after an opening quote
    text = re.sub(r'\s"\b', ' "', text)  # Remove misplaced space before a closing quote
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace characters with a single space
    text = re.sub(r',,', ',', text)  # Replace double commas with a single comma
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-UTF-8 characters
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    sentences = [sentence.strip() for sentence in sentences if sentence]
    return sentences

def is_meaningful_sentence(sentence):
    # Heuristic rules to filter out non-meaningful sentences
    if len(sentence) < 20:  # Filter out very short sentences
        return False
    if sentence.lower().startswith(('figure', 'table', 'definition', 'appendix', 'advances', 'ultra-strong', 'manuscript submitted', 'url', 'available at', 'alternatively', 'abstract')):
        return False
    if re.match(r'^\[\d+\]\s', sentence):  # Filter out sentences starting with "[x] " where x is a number
        return False
    if re.match(r'^[A-Za-z]\.?\s', sentence):  # Filter out sentences starting with single letters/abbreviations
        return False
    if re.match(r'^\[?\d+\]?\.$', sentence):  # Filter out numeric and special character sentences
        return False
    if re.search(r'\b(?:[12]\d{3}|vol\.|no\.|pp\.|doi:|ed\.|pages|chapter|cambridge university press|journal|conference|proceedings|in press|forthcoming)\b', sentence):  # Filter out common citation patterns
        return False
    if re.search(r'\bISBN\s(?:97[89][-– ])?\d{1,5}[-– ]?\d+[-– ]?\d+[-– ]?[\dX]\b', sentence, re.IGNORECASE):  # Filter out ISBNs
        return False
    if re.search(r'https?://\S+|www\.\S+|http\s?:\s?//\S+', sentence):  # Filter out URLs, including those with spaces
        return False
    if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', sentence):  # Filter out emails
        return False
    if re.search(r'\bin fig\b|\bin table\b|\bin diagram\b|\bin screenshot\b', sentence):  # Filter out sentences containing "in fig", "in table", "in diagram", "in screenshot"
        return False
    if re.match(r'^(\d+\s?[,|\.|\s])+[a-z]*\d*$', sentence):  # Filter out sequences of numbers and patterns
        return False
    if re.search(r'\b(?:linear|relu|train loss|test loss|out figure|joint f|case studies|baseline prediction|naive prediction|mix prediction|combined prediction|input: print\(|input:\s|proof of)\b', sentence):  # Filter out specific keywords related to data
        return False
    if re.match(r'^\d+\s*\[.*\]\s*[a-z\s,]*$', sentence):  # Filter out sentences like "12 [62] yuandong tian, xinlei chen, and surya ganguli."
        return False
    if re.match(r'^[A-Za-z\s]+and\s[A-Za-z\s]+(\.|,)$', sentence):  # Filter out sentences like "piotr indyk and rajeev motwani."
        return False
    if re.match(r'^[A-Za-z\s,]+\.$', sentence):  # Filter out sentences like "barrett, and kimberly l."
        return False
    if re.match(r'^[A-Za-z\s,]+,\sand\s[A-Za-z\s,]+\.$', sentence):  # Filter out sentences like "koehn, philipp and knight, kevin."
        return False
    if re.match(r'^[A-Za-z\s]+,\sand\s[A-Za-z\s]+\.$', sentence):  # Filter out sentences like "yuille, and kevin murphy."
        return False
    if re.match(r'^[A-Za-z\s]+,\sand\s[A-Za-z\s,]+\.$', sentence):  # Filter out sentences like "wang, yushi, berant, jonathan, and liang, percy."
        return False
    if re.match(r'^[A-Za-z\s,]+\sand\s[A-Za-z\s,]+\.$', sentence):  # Filter out sentences like "bogdan mu¸ sat and r˘ azvan andonie."
        return False
    if re.match(r'^[A-Za-z\s,]+and\s[A-Za-z\s,]+\.$', sentence):  # Filter out sentences like "zhilu zhang and mert sabuncu."
        return False
    if re.match(r'^[a-z\s]+,\s[a-z\s]+,\sand\s[a-z\s]+(\.|,)$', sentence):  # Filter out sentences like "cuozzo, b., dumay, j., palmaccio, m., & lombardi, r."
        return False
    if re.match(r'^[a-z\s]+(\.|,|&\s)', sentence):  # Filter out sentences like "joan bruna new york university dumitru erhan google inc."
        return False
    if re.match(r'^[a-z\s,.]+,\sand\s[a-z\s,.]+$', sentence):  # Filter out sentences like "tu, z., talebi, h., zhang, h., yang, f., milanfar, p., bovik, a., and li, y."
        return False
    if re.match(r'^[a-z\s,]+,\sand\s[a-z\s,]+$', sentence):  # Filter out sentences like "francesco tonolini, bjørn sand jensen, and roderick murray-smith."
        return False
    if re.match(r'^[a-z\s,]+,\sand\s[a-z\s,]+,\s[a-z\s,]+$', sentence):  # Filter out sentences like "richard yuanzhe pang and he he."
        return False
    if re.match(r'^[a-z\s,]+,\sand\s[a-z\s,]+,\s[a-z\s,]+,\s[a-z\s,]+$', sentence):  # Filter out sentences like "michael a. goodrich, roberto tamassia, and michael h. goldwasser."
        return False
    if re.match(r'^[a-z\s,]+,\sand\s[a-z\s,]+,\s[a-z\s,]+,\s[a-z\s,]+,\s[a-z\s,]+$', sentence):  # Filter out sentences like "michael a. goodrich, roberto tamassia, and michael h. goldwasser."
        return False
    if re.match(r'^[a-z\s]+,\s[a-z\s]+,\s[a-z\s]+,\sand\s[a-z\s]+$', sentence):  # Filter out sentences like "12.2 examples of program evaluation prediction."
        return False
    if re.match(r'^input:\s.*$', sentence):  # Filter out sentences starting with "input: "
        return False
    if re.search(r'\bprint\((.*)\)\b', sentence):  # Filter out sentences containing print statements
        return False
    if re.match(r'.*\|-\s*\(.*\)$', sentence):  # Filter out sentences with |- symbol
        return False
    if re.match(r'.*\bph\b.*\b(ps|ch)\b', sentence):  # Filter out sentences with logical notation
        return False
    if re.match(r'.*\b(nn0|cc|zz)\b.*', sentence):  # Filter out sentences with notation like nn0, cc, zz
        return False
    if re.match(r'.*\b(?:[online]|acm sigkdd int|[A-Za-z]\d+)\b.*', sentence):  # Filter out sentences with specific patterns
        return False
    if re.match(r'.*\[online\].*', sentence):  # Filter out sentences with [online] URLs
        return False
    if re.match(r'^\d+[, ]+\d+\s*\[\d+\]\s*[a-zA-Z\s,]+$', sentence):  # Filter out sentences like "6, 8 [56] jingru tan, changbao wang, buyu li, quanquan li, wanli ouyang, changqing yin, and junjie yan."
        return False
    if re.match(r'^[a-zA-Z\s,]+\(\d+\)[a-z\s]+\.$', sentence):  # Filter out sentences like "mcclel- land, loic matthey, felix hill, and alexander lerchner."
        return False
    if re.match(r'^\d+\s*\([^\)]*\)\s*\[.*\]\s*[a-zA-Z\s,]+$', sentence):  # Filter out sentences like "8 11 [25] xiaowei hu, xi yin, kevin lin, lijuan wang, lei zhang, jianfeng gao, and zicheng liu."
        return False
    if re.match(r'^\([^\)]*\)\s*[a-zA-Z\s,]+$', sentence):  # Filter out sentences like "(cited on 1) yang song, jascha sohl-dickstein, diederik p kingma, abhishek kumar, stefano ermon, and ben poole."
        return False
    if re.match(r'^\[\d+\]\s*[a-zA-Z\s,]+\.$', sentence):  # Filter out sentences like "references [1] harry g barrow, alistair j bray, and julian ml budd."
        return False
    if re.match(r'^\(\d+\)\s*[a-zA-Z\s,]+\.$', sentence):  # Filter out sentences like "(cited on page 5) anna rogers, olga kovaleva, matthew downey, and anna rumshisky."
        return False

    return True

def filter_sentences(sentences):
    filtered_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        if len(words) < 5:  # Filter out very short sentences
            continue
        if all(word.lower() in STOPWORDS for word in words):  # Filter out sentences that are mostly stopwords
            continue
        if is_meaningful_sentence(sentence):  # Apply heuristic rules
            filtered_sentences.append(sentence)
    return filtered_sentences

def process_pdf(file_info):
    author, file_path = file_info
    text = extract_text_from_pdf(file_path)
    cleaned_sentences = clean_text(text)
    filtered_sentences = filter_sentences(cleaned_sentences)
    pdf_texts = [{'author': author, 'pdf_source': os.path.basename(file_path), 'sentence': sentence} for sentence in filtered_sentences]
    return pdf_texts, len(cleaned_sentences)

def load_and_clean_pdfs_parallel(directory):
    pdf_files = []
    for subdir, _, files in os.walk(directory):
        author = os.path.basename(subdir)
        for file in files:
            if file.endswith(".pdf"):
                pdf_files.append((author, os.path.join(subdir, file)))

    total_sentences = 0
    pdf_texts = []

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_pdf, file_info) for file_info in pdf_files]
        for future in as_completed(futures):
            result, num_sentences = future.result()
            pdf_texts.extend(result)
            total_sentences += num_sentences

    return pdf_texts, total_sentences

def save_cleaned_text_to_csv(cleaned_texts, output_file='cleaned_texts.csv'):
    df = pd.DataFrame(cleaned_texts)
    df.to_csv(output_file, index=False, encoding='utf-8')

def print_stats(cleaned_texts, total_sentences):
    num_sentences = len(cleaned_texts)
    print(f"\nTotal number of sentences processed: {total_sentences}")
    print(f"Number of meaningful sentences exported to CSV: {num_sentences}")
    print(f"Number of sentences filtered out: {total_sentences - num_sentences}")

if __name__ == "__main__":
    # Download stopwords and punkt if not already available
    download('stopwords')
    download('punkt')

    STOPWORDS = set(stopwords.words('english')).union(ENGLISH_STOP_WORDS)
    
    directory = 'ref'
    cleaned_texts, total_sentences = load_and_clean_pdfs_parallel(directory)
    save_cleaned_text_to_csv(cleaned_texts)
    print_stats(cleaned_texts, total_sentences)
