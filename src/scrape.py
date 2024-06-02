import requests
from xml.etree import ElementTree
import os

SAVE_DIRECTORY = "ref"
PLACEHOLDER_FILE = "pdfs_go_here"

def fetch_papers_by_author(author_name):
    search_url = f'http://export.arxiv.org/api/query?search_query=au:"{author_name}"&start=0&max_results=50'
    response = requests.get(search_url)
    if response.status_code != 200:
        raise Exception(f"Error fetching papers: {response.status_code}")
    return response.content

def parse_paper_urls(xml_data):
    tree = ElementTree.fromstring(xml_data)
    papers = []
    for entry in tree.findall("{http://www.w3.org/2005/Atom}entry"):
        pdf_url = None
        for link in entry.findall("{http://www.w3.org/2005/Atom}link"):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib["href"]
                break
        if pdf_url:
            papers.append(pdf_url)
    return papers

def download_pdf(url, save_dir):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        content_type = response.headers.get('content-type')
        if 'application/pdf' in content_type:
            filename = url.split('/')[-1] + '.pdf'
            file_path = os.path.join(save_dir, filename)
            with open(file_path, 'wb') as pdf_file:
                pdf_file.write(response.content)
            print(f"Downloaded: {file_path}")
            return file_path
        else:
            print(f"URL does not point to a PDF: {url}")
            return None
    else:
        print(f"Failed to download PDF from {url}")
        return None

def create_placeholder_file(directory, filename):
    placeholder_path = os.path.join(directory, filename)
    with open(placeholder_path, 'w') as placeholder_file:
        placeholder_file.write("")
    print(f"Created placeholder file: {placeholder_path}")

def scrape_arxiv_papers(authors):
    if not os.path.exists(SAVE_DIRECTORY):
        os.makedirs(SAVE_DIRECTORY)
        create_placeholder_file(SAVE_DIRECTORY, PLACEHOLDER_FILE)

    for author in authors:
        author_dir = os.path.join(SAVE_DIRECTORY, author.replace(' ', '_'))
        if not os.path.exists(author_dir):
            os.makedirs(author_dir)
        
        xml_data = fetch_papers_by_author(author)
        paper_urls = parse_paper_urls(xml_data)
        for url in paper_urls:
            download_pdf(url, author_dir)


if __name__ == "__main__":
    authors = ["Boris Kovalerchuk", "Razvan Andonie"]
    scrape_arxiv_papers(authors)
