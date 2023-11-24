from fastapi import FastAPI,Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
import datetime
import torch
from sentence_transformers import SentenceTransformer
import pinecone
from transformers import GPTJForCausalLM

app = FastAPI()
url_bucket = "https://storage.googleapis.com/zakony-data/"

pinecone.init(      
	api_key='f102e74e-a772-42c8-b0b6-a29211469df6',      
	environment='asia-southeast1-gcp'      
)
index = pinecone.Index("prvni")
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1', device='cuda')

device = "cuda"
# model = GPTJForCausalLM.from_pretrained(
#     "EleutherAI/gpt-j-6B",
#     revision="float16",
#     torch_dtype=torch.float16,
# ).to(device)


def search_similar_documents(query, top_k=500):

    # Convert the query to a vector
    print(query)
    query_vector = model.encode([query]).tolist()

    # Search the index
    results = index.query(queries=query_vector, top_k=top_k, include_metadata=True)

    # Process results
    return results

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to a specific origin in a production environment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def filter_docs(docs):
    filenames = {}
    out = []
    for doc in docs['results'][0]['matches']:
        source = str(doc["metadata"]["source"])
        text = str(doc["metadata"]["text"])
        if source in filenames or len(text) < 100:
            continue

        filenames[source] = True
        out.append((source,text))
    return out
    
class User(BaseModel):
    input_data: str

@app.post("/generate-pdfs")
async def generate_pdfs(user: User):
    input = user.input_data

    similar_docs = search_similar_documents(input )
    similar_docs = filter_docs(similar_docs )
    pdf_files = []

    for result in similar_docs:
        url = url_bucket + '/'.join(str(result[0]).split("/")[-2:])
        pdf_files.append({
            # "title": result["metadata"],
            "title": result[0],
            "url": url,
            "text": result[1],
            "summary": "Shrnuti rozsudku, zatim nedodano, dolor sit amet, uga buga je smyslem sveta",
        })

    # return pdf_files
    # pdf_files = [
    #     {"title": "PDF 1", "url": "pdf1.pdf", "summary": "Summary of PDF 1"},
    #     {"title": "PDF 2", "url": "pdf2.pdf", "summary": "Summary of PDF 2"},
    #     {"title": "PDF 3", "url": "pdf3.pdf", "summary": "Summary of PDF 3"},
    # ]
    return pdf_files


