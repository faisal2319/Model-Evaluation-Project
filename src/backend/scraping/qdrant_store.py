import os
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import qdrant_client
from urllib.parse import urljoin
from dotenv import load_dotenv

load_dotenv()

QDRANT_HOST = os.getenv('QDRANT_HOST')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
QDRANT_COLLECTION_NAME = os.getenv('QDRANT_COLLECTION_NAME')

client = qdrant_client.QdrantClient(
    url=QDRANT_HOST,
    api_key=QDRANT_API_KEY
)

vectors_config = qdrant_client.http.models.VectorParams(
    size=384,  
    distance=qdrant_client.http.models.Distance.COSINE
)

client.recreate_collection(
    collection_name=QDRANT_COLLECTION_NAME,
    vectors_config=vectors_config
)

# Initialize Sentence Transformer model
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
        collection_name=QDRANT_COLLECTION_NAME,
        points=[
            qdrant_client.http.models.PointStruct(
                id=i,  
                vector=vector,
                payload={"text": text}
            )
        ]
    )

print("Text successfully stored in Qdrant")
