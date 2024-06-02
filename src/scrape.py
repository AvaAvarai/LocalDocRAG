import arxiv
import os
import requests

# Define the author name
author_name = "Boris Kovalerchuk"  # Change this to the author you want to search for

# Create an arXiv client
client = arxiv.Client()

# Search for papers by the author
search = arxiv.Search(
    query=f'au:"{author_name}"',
    max_results=1000,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

# Create a directory to save the PDFs
if not os.path.exists('ref'):
    os.makedirs('ref')

# Download the PDFs
for result in client.results(search):
    paper_id = result.entry_id.split('/')[-1]
    paper_title = result.title.replace(' ', '_').replace('/', '_')
    pdf_url = result.pdf_url

    print(f"Downloading {paper_title}...")

    response = requests.get(pdf_url)
    if response.status_code == 200:
        with open(f'ref/{paper_id}_{paper_title}.pdf', 'wb') as file:
            file.write(response.content)
    else:
        print(f"Failed to download {paper_title}")

print("Download complete.")
