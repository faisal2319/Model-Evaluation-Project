import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import qdrant_client
import openai
from sentence_transformers import SentenceTransformer
import asyncio
import replicate
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv('QDRANT_COLLECTION_NAME', 'scrape_data')
client = qdrant_client.QdrantClient(
    url=QDRANT_HOST,
    api_key=QDRANT_API_KEY
)

openai.api_key = os.getenv("OPENAI_API_KEY")

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

replicate_client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

class UserQuery(BaseModel):
    query: str

class LLMResponse(BaseModel):
    model: str
    response: str
    bleu_score: float
    cosine_similarity: float

class QueryResponse(BaseModel):
    query: str
    results: List[LLMResponse]
    best_response: LLMResponse

async def query_openai_model(model_name: str, prompt: str) -> str:
    response = await openai.ChatCompletion.acreate(
        model=model_name,
        messages=[{"role": "system", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

async def query_llama_model(prompt: str) -> str:
    model = "meta/llama-2-70b-chat"
    prediction = replicate_client.run(model, input={"prompt": prompt})
    print(f"Llama-2-70b-chat prediction: {prediction}")  
    return ''.join(prediction)

async def query_falcon_model(prompt: str) -> str:
    model = "joehoover/falcon-40b-instruct"
    prediction = replicate_client.run(model, input={"prompt": prompt})
    print(f"Falcon-40b-instruct prediction: {prediction}")  
    return ''.join(prediction)

def compute_bleu(reference, hypothesis):
    reference = [reference.split()]
    hypothesis = hypothesis.split()
    smoothie = SmoothingFunction().method4
    score = sentence_bleu(reference, hypothesis, smoothing_function=smoothie)
    return score

def compute_cosine_similarity(embedding_model, ref_text, hyp_text):
    ref_vector = embedding_model.encode(ref_text)
    hyp_vector = embedding_model.encode(hyp_text)
    score = cosine_similarity([ref_vector], [hyp_vector])[0][0]
    return score

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            user_query = UserQuery(query=data)
            query_text = user_query.query
            query_vector = embedding_model.encode(query_text).tolist()

            search_results = client.search(
                collection_name=QDRANT_COLLECTION_NAME,
                query_vector=query_vector,
                limit=5
            )

            search_texts = [result.payload['text'] for result in search_results]
            combined_search_results = "\n".join(search_texts)

            await websocket.send_json({"type": "search_results", "results": search_texts})

            combined_prompt = f"Search results:\n{combined_search_results}\n\nUser query: {query_text}\nAnswer based on the search results only:"

            models = {
                "gpt-3.5-turbo": query_openai_model,
                "gpt-4": query_openai_model,
                "Llama-2-70b-chat": query_llama_model,
                # "Falcon-40b-instruct": query_falcon_model
                # Permission Issues
            }

            tasks = [
                models[model](model, combined_prompt) if 'gpt' in model else models[model](combined_prompt)
                for model in models
            ]

            llm_responses = []
            combined_scores = []

            for i, model in enumerate(models):
                response = await tasks[i]
                bleu_score = float(compute_bleu(combined_search_results, response))
                cosine_score = float(compute_cosine_similarity(embedding_model, combined_search_results, response))
                combined_score = (bleu_score + cosine_score) / 2
                llm_responses.append(LLMResponse(model=model, response=response, bleu_score=bleu_score, cosine_similarity=cosine_score))
                combined_scores.append(combined_score)
                
                await websocket.send_json({"type": "llm_response", "model": model, "response": response, "bleu_score": bleu_score, "cosine_similarity": cosine_score})

            best_index = combined_scores.index(max(combined_scores))
            best_response = llm_responses[best_index]

            await websocket.send_json({"type": "best_response", "best_response": best_response.dict()})

    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
