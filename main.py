import os
from fastapi import FastAPI, Query
from sentence_transformers import SentenceTransformer
from supabase import create_client
from typing import List
from dotenv import load_dotenv

# Load env variables from .env file
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise Exception("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in environment variables.")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
model = SentenceTransformer('all-MiniLM-L6-v2')

app = FastAPI()

@app.get("/recommend")
async def recommend_songs(query: str = Query(..., min_length=1), top_k: int = 10):
    query_emb = model.encode(query).tolist()

    try:
        response = supabase.rpc("match_songs", {
            "query_embedding": query_emb,
            "match_count": top_k
        }).execute()
    except Exception as e:
        return {"error": str(e)}

    results = getattr(response, "data", None)
    if results is None:
        return {"error": "No data found or unexpected response from Supabase"}

    return {"query": query, "results": results}
