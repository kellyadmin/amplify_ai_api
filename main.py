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
    # Log an error if keys are missing but don't stop app for immediate testing
    print("WARNING: Supabase URL or Key not found in environment variables. Supabase features will not work.")
    supabase = None # Set to None to allow app startup even without Supabase
else:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


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

try:
    model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    text_generator = pipeline("text-generation", model="sshleifer/tiny-distilgpt2")
except Exception as e:
    print(f"Error loading AI models: {e}")
    # In a production app, you might want to raise an exception or
    # have fallback logic if models fail to load.

# --- Pydantic Models for Request/Response Validation ---
class ChatRequest(BaseModel):
    message: str

# New Pydantic model for playlist recommendation request
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
    try:
        emb = model.encode(query).tolist()
        return {"query": query, "embedding": emb[:top_k], "status": "success"}
    except Exception as e:
        return {"query": query, "error": str(e), "status": "failed"}

@app.post("/chat")
async def chat(request: ChatRequest):
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
    Searches a Supabase 'songs' table for titles matching the query.
    Assumes you have a table named 'songs' in Supabase with a 'title' column.
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
    Generates a playlist description based on user mood and stores/retrieves user data.
    
    Args:
        request (PlaylistRequest): Contains user_id, mood, and liked_songs.
    """
    if not supabase:
        raise HTTPException(status_code=503, detail="Supabase client not initialized.")

    user_id = request.user_id
    mood = request.mood
    liked_songs = request.liked_songs.split(',') if request.liked_songs else []

    try:
        # 1. Store/Update User Profile in Supabase
        # Check if user exists
        user_data, count = supabase.table('user_profiles').select('*').eq('user_id', user_id).execute()
        
        if user_data and user_data[0]:
            # Update existing user profile
            updated_data = {
                "last_mood": mood,
                "liked_songs": liked_songs, # Overwrite or merge as per your logic
                "last_active": "now()" # Update last active timestamp
            }
            supabase.table('user_profiles').update(updated_data).eq('user_id', user_id).execute()
        else:
            # Create new user profile
            new_user_data = {
                "user_id": user_id,
                "last_mood": mood,
                "liked_songs": liked_songs,
                "created_at": "now()",
                "last_active": "now()"
            }
            supabase.table('user_profiles').insert(new_user_data).execute()

        # 2. Generate Playlist Description using AI
        # This is a descriptive recommendation, not actual song matching yet.
        ai_prompt = f"Create a playlist description for someone feeling '{mood}'. Consider popular songs related to this mood."
        if liked_songs:
            ai_prompt += f" They also like songs such as {', '.join(liked_songs)}."

        generated_result = text_generator(ai_prompt, max_length=150, num_return_sequences=1)
        playlist_description = generated_result[0]['generated_text']

        return {
            "user_id": user_id,
            "mood": mood,
            "liked_songs": liked_songs,
            "playlist_description": playlist_description,
            "message": "User profile updated and playlist description generated.",
            "status": "success"
        }
    except Exception as e:
        print(f"Error recommending playlist: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate playlist: {e}")

