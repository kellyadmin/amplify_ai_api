import os
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
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

# --- RPC Name for Supabase Vector Search Function ---
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

# Initialize embedding model.
model = None 

try:
    # Sentence Transformer for text embeddings (for similarity search)
    model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    print("SentenceTransformer model loaded successfully.")
except Exception as e:
    print(f"Error loading AI embedding model: {e}")
    print("AI embedding model is not loaded. Playlist recommendation features will be unavailable.")


# --- Pydantic Models for Request/Response Validation ---
class PlaylistRequest(BaseModel):
    user_id: str
    mood: str
    liked_songs: str = "" # Comma-separated string of liked songs
    match_threshold: float = 0.05 # NEW: Configurable threshold from frontend
    match_count: int = 10 # NEW: Configurable match count

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "Amplify AI API is running"}

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
        request (PlaylistRequest): Contains user_id, mood, liked_songs, match_threshold, and match_count.
    """
    if not supabase:
        raise HTTPException(status_code=503, detail="Supabase client not initialized.")
    if not model: # We need the embedding model for this
        raise HTTPException(status_code=503, detail="AI embedding model not loaded for similarity search.")

    user_id = request.user_id
    mood = request.mood
    liked_songs_list = [s.strip() for s in request.liked_songs.split(',') if s.strip()] # Clean and split
    match_threshold = request.match_threshold # Use threshold from request
    match_count = request.match_count # Use count from request

    # Add logging for debugging
    print(f"Received playlist request for User ID: {user_id}, Mood: '{mood}', Liked Songs: {liked_songs_list}")
    print(f"Using match_threshold: {match_threshold}, match_count: {match_count}")

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
        combined_query = f"{mood} songs"
        if liked_songs_list:
            combined_query += f" similar to {', '.join(liked_songs_list)}"
        
        query_embedding = model.encode(combined_query).tolist()
        # Add logging for debugging
        print(f"Generated query embedding (first 5 dims): {query_embedding[:5]}...")

        # 3. Call Supabase RPC for Similar Songs (Vector Search)
        rpc_response = supabase.rpc(
            RPC_NAME_SIMILAR_SONGS, 
            {
                "query_embedding": query_embedding,
                "match_threshold": match_threshold, # Use dynamic threshold
                "match_count": match_count          # Use dynamic count
            }
        ).execute()

        recommended_songs = rpc_response.data
        # Add logging for debugging
        print(f"Raw RPC Response Data: {recommended_songs}")
        
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

