import os
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import qdrant_client
from urllib.parse import urljoin

os.environ['QDRANT_HOST'] = "https://b9bef3e0-9bba-4f16-a5b3-1788e8b8181b.europe-west3-0.gcp.cloud.qdrant.io"
os.environ['QDRANT_API_KEY'] = "KaDpGXhHnkaAhZLE-yLhNQeqr3fR3kxD9ITb_esiRoBKAM4nqUfsTw"
client = qdrant_client.QdrantClient(
    os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY")
)

os.environ['QDRANT_COLLECTION_NAME'] = 'scrape_data'

vectors_config = qdrant_client.http.models.VectorParams(
    size=384,  
    distance=qdrant_client.http.models.Distance.COSINE
)

client.recreate_collection(
    collection_name=os.getenv('QDRANT_COLLECTION_NAME'),
    vectors_config=vectors_config
)

model = SentenceTransformer('all-MiniLM-L6-v2')

def scrape_url(url, visited):
    if url in visited:
        return []
    visited.add(url)
    
    print(f"Scraping URL: {url}")
    
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    texts = []
    paragraphs = soup.find_all('p')
    for p in paragraphs:
        text = p.get_text().strip()
        if text:
            texts.append(text)
    
    # Find and follow links to subpages
    links = soup.find_all('a', href=True)
    for link in links:
        href = link['href']
        if href.startswith('/') or href.startswith(url):
            full_url = urljoin(url, href)
            if full_url.startswith("https://u.ae/en/information-and-services"):
                texts.extend(scrape_url(full_url, visited))
    
    return texts

main_url = "https://u.ae/en/information-and-services#/"
visited_urls = set()
all_texts = scrape_url(main_url, visited_urls)

for i, text in enumerate(all_texts):
    vector = model.encode(text).tolist() 
    client.upsert(
        collection_name=os.getenv('QDRANT_COLLECTION_NAME'),
        points=[
            qdrant_client.http.models.PointStruct(
                id=i,  
                vector=vector,
                payload={"text": text}
            )
        ]
    )

print("Text successfully stored in Qdrant")
