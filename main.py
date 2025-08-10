import os
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from supabase import create_client, Client
from fastapi.middleware.cors import CORSMiddleware
import uvicorn # Ensure uvicorn is imported for local run, if needed

# --- Supabase Configuration ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# Initialize Supabase client
if not SUPABASE_URL or not SUPABASE_KEY:
    print("WARNING: Supabase URL or Key not found in environment variables. Supabase features will not work.")
    supabase = None
else:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- NEW: RPC Name for Supabase Vector Search Function ---
RPC_NAME_SIMILAR_SONGS = "match_songs_by_embedding"

# --- FastAPI App & Model Loading ---
os.environ["TRANSFORMERS_CACHE"] = "/app/cache"
app = FastAPI()

# --- CORS Configuration ---
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models to None. They will be assigned in the try block.
model = None # For embeddings
text_generator = None # For chat

try:
    # Sentence Transformer for text embeddings (for similarity search)
    model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    # Text generation model (for chat responses only, no longer for playlist descriptions)
    text_generator = pipeline("text-generation", model="sshleifer/tiny-distilgpt2")
except Exception as e:
    print(f"Error loading AI models: {e}")
    print("AI models are not loaded. Some features will be unavailable.")

# --- Pydantic Models for Request/Response Validation ---
class ChatRequest(BaseModel):
    message: str

class PlaylistRequest(BaseModel):
    user_id: str
    mood: str
    liked_songs: str = "" # Comma-separated string of liked songs

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "Amplify AI API is running"}

@app.get("/recommend")
async def recommend_songs(query: str = Query(..., description="Text query for song recommendation"), top_k: int = 5):
    if not model:
        raise HTTPException(status_code=503, detail="AI embedding model not loaded. Please check server logs.")
    try:
        emb = model.encode(query).tolist()
        return {"query": query, "embedding": emb[:top_k], "status": "success"}
    except Exception as e:
        return {"query": query, "error": str(e), "status": "failed"}

@app.post("/chat")
async def chat(request: ChatRequest):
    if not text_generator:
        raise HTTPException(status_code=503, detail="AI text generation model not loaded. Please check server logs.")
    prompt = request.message
    try:
        result = text_generator(prompt, max_length=50, num_return_sequences=1)
        generated_text = result[0]['generated_text']
        return {"reply": generated_text, "status": "success"}
    except Exception as e:
        return {"reply": f"Error generating response: {e}", "status": "failed"}

@app.get("/search_song_db")
async def search_song_db(query: str = Query(..., description="Query to search song database")):
    """
    Searches a Supabase 'songs' table for titles matching the query (case-insensitive LIKE).
    """
    if not supabase:
        raise HTTPException(status_code=503, detail="Supabase client not initialized.")
    try:
        response = supabase.table('songs').select('*').ilike('title', f'%{query}%').execute()
        
        if response.data:
            return {"results": response.data, "status": "success"}
        else:
            return {"results": [], "message": "No songs found matching your query.", "status": "success"}
    except Exception as e:
        print(f"Error searching Supabase: {e}")
        return {"error": f"Failed to search database: {e}", "status": "failed"}

@app.post("/recommend_playlist")
async def recommend_playlist(request: PlaylistRequest):
    """
    Recommends actual songs based on user mood/liked songs by performing a vector similarity search in Supabase.
    
    Args:
        request (PlaylistRequest): Contains user_id, mood, and liked_songs.
    """
    if not supabase:
        raise HTTPException(status_code=503, detail="Supabase client not initialized.")
    if not model: # We need the embedding model for this
        raise HTTPException(status_code=503, detail="AI embedding model not loaded for similarity search.")

    user_id = request.user_id
    mood = request.mood
    liked_songs_list = request.liked_songs.split(',') if request.liked_songs else []

    try:
        # 1. Store/Update User Profile in Supabase
        response = supabase.table('user_profiles').select('*', count='exact').eq('user_id', user_id).execute()
        user_data = response.data
        
        user_profile_data = {
            "last_mood": mood,
            "liked_songs": liked_songs_list, # Store as array in Supabase
            "last_active": "now()"
        }

        if user_data and len(user_data) > 0:
            supabase.table('user_profiles').update(user_profile_data).eq('user_id', user_id).execute()
        else:
            user_profile_data["user_id"] = user_id
            user_profile_data["created_at"] = "now()"
            supabase.table('user_profiles').insert(user_profile_data).execute()

        # 2. Generate Embedding for the Recommendation Query
        # Combine mood and liked songs for a richer query embedding
        combined_query = f"{mood} songs"
        if liked_songs_list:
            combined_query += f" similar to {', '.join(liked_songs_list)}"
        
        query_embedding = model.encode(combined_query).tolist()

        # 3. Call Supabase RPC for Similar Songs (Vector Search)
        # This calls the SQL function created in Supabase (match_songs_by_embedding)
        rpc_response = supabase.rpc(
            RPC_NAME_SIMILAR_SONGS, 
            {
                "query_embedding": query_embedding,
                "match_threshold": 0.5, # Adjust this value (0 to 1, higher for more similar)
                "match_count": 10       # Number of top similar songs to return
            }
        ).execute()

        recommended_songs = rpc_response.data
        
        return {
            "user_id": user_id,
            "mood": mood,
            "liked_songs": liked_songs_list,
            "recommended_songs": recommended_songs, # List of song objects found
            "message": "User profile updated and similar songs recommended.",
            "status": "success"
        }
    except Exception as e:
        print(f"Error recommending playlist: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate playlist: {e}")

# --- Local Development Server (Optional for Deployment) ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

