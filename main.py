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
RPC_NAME_SIMILAR_SONGS = "get_similar_songs" 

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
    similarity_threshold: float = 0.05 
    num_matches: int = 10 

# NEW: Pydantic model for adding a new song
class SongAddRequest(BaseModel):
    title: str
    artist: str = "Unknown Artist"
    genre: str = "Unknown Genre"
    mood: str = "" # e.g., "energetic", "calm"
    tempo: str = "" # NEW: e.g., "fast", "slow", "moderate"
    # You can add other fields from your Supabase 'songs' table here (e.g., audio_url, album_art_url)


# Helper function to generate embedding text
def generate_song_embedding_text(song_data: dict) -> str:
    """Generates a descriptive string for embedding based on song metadata."""
    title = song_data.get('title', '')
    artist = song_data.get('artist', 'Unknown Artist')
    genre = song_data.get('genre', 'Unknown Genre')
    mood = song_data.get('mood', '')
    tempo = song_data.get('tempo', '') # NEW: Include tempo

    text_to_embed = f"{title} by {artist} ({genre})"
    if mood:
        text_to_embed += f". Mood: {mood}."
    if tempo:
        text_to_embed += f" Tempo: {tempo}."
    return text_to_embed


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
            return {"results": [], "message": f"No songs found matching \"{query}\".", "status": "success"}
    except Exception as e:
        print(f"Error searching Supabase: {e}")
        return {"error": f"Failed to search database: {e}", "status": "failed"}


# NEW: Endpoint to add a new song with auto-generated embedding
@app.post("/add_song")
async def add_song(request: SongAddRequest):
    """
    Adds a new song to the Supabase 'songs' table, automatically generating its embedding.
    """
    if not supabase:
        raise HTTPException(status_code=503, detail="Supabase client not initialized.")
    if not model:
        raise HTTPException(status_code=503, detail="AI embedding model not loaded. Cannot generate embedding.")
    
    try:
        # Convert request Pydantic model to a dictionary
        song_data_to_insert = request.dict()

        # Generate embedding for the new song based on its metadata
        embedding_text = generate_song_embedding_text(song_data_to_insert)
        song_embedding = model.encode(embedding_text).tolist()
        
        # Add the generated embedding to the song data
        song_data_to_insert['embedding'] = song_embedding

        # Insert the new song data into Supabase
        response = supabase.table('songs').insert(song_data_to_insert).execute()
        
        if response.data:
            print(f"Successfully added song '{request.title}' with auto-generated embedding.")
            return {"message": "Song added successfully with embedding!", "song": response.data[0], "status": "success"}
        else:
            print(f"Failed to add song '{request.title}'. Response: {response.status_code}, {response.data}")
            raise HTTPException(status_code=500, detail="Failed to add song to database.")
            
    except Exception as e:
        print(f"Error adding song: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add song: {e}")


@app.post("/recommend_playlist")
async def recommend_playlist(request: PlaylistRequest):
    """
    Recommends actual songs based on user mood/liked songs by performing a vector similarity search in Supabase.
    Handles liked songs that may not be in the database.
    
    Args:
        request (PlaylistRequest): Contains user_id, mood, liked_songs, similarity_threshold, and num_matches.
    """
    if not supabase:
        raise HTTPException(status_code=503, detail="Supabase client not initialized.")
    if not model: # We need the embedding model for this
        raise HTTPException(status_code=503, detail="AI embedding model not loaded for similarity search.")

    user_id = request.user_id
    mood = request.mood
    raw_liked_songs_list = [s.strip() for s in request.liked_songs.split(',') if s.strip()] 
    similarity_threshold = request.similarity_threshold 
    num_matches = request.num_matches 

    # Debugging logs
    print(f"Received playlist request for User ID: {user_id}, Mood: '{mood}', Raw Liked Songs: {raw_liked_songs_list}")
    print(f"Using similarity_threshold: {similarity_threshold}, num_matches: {num_matches}")

    found_liked_songs = []
    unfound_liked_songs_messages = []

    try:
        # 1. Store/Update User Profile in Supabase
        response = supabase.table('user_profiles').select('*', count='exact').eq('user_id', user_id).execute()
        user_data = response.data
        
        user_profile_data = {
            "last_mood": mood,
            "liked_songs": raw_liked_songs_list, 
            "last_active": "now()"
        }

        if user_data and len(user_data) > 0:
            supabase.table('user_profiles').update(user_profile_data).eq('user_id', user_id).execute()
        else:
            user_profile_data["user_id"] = user_id
            user_profile_data["created_at"] = "now()"
            supabase.table('user_profiles').insert(user_profile_data).execute()

        # 2. Process Liked Songs: Check if they exist and contribute to query embedding
        query_components_for_embedding = [mood] 
        
        if raw_liked_songs_list:
            for liked_song_title in raw_liked_songs_list:
                # Select additional metadata for richer embedding if found
                song_search_response = supabase.table('songs').select('title, artist, genre, mood, tempo').ilike('title', liked_song_title).limit(1).execute()
                
                if song_search_response.data and len(song_search_response.data) > 0:
                    found_song = song_search_response.data[0]
                    found_liked_songs.append(found_song['title']) 
                    # Use the helper function for consistency
                    query_components_for_embedding.append(generate_song_embedding_text(found_song))
                    print(f"  Found liked song '{liked_song_title}' in DB.")
                else:
                    unfound_liked_songs_messages.append(liked_song_title)
                    query_components_for_embedding.append(liked_song_title)
                    print(f"  Liked song '{liked_song_title}' not found in DB or has no embedding. Still using for query embedding.")

        # Construct the final query string for embedding
        combined_query_text = f"playlist for {', '.join(query_components_for_embedding)}"
        print(f"Final combined query text for embedding: '{combined_query_text}'")
        
        query_embedding = model.encode(combined_query_text).tolist()
        print(f"Generated query embedding (first 5 dims): {query_embedding[:5]}...")

        # 3. Call Supabase RPC for Similar Songs (Vector Search)
        rpc_response = supabase.rpc(
            RPC_NAME_SIMILAR_SONGS, 
            {
                "query_embedding": query_embedding,
                "similarity_threshold": similarity_threshold, 
                "num_matches": num_matches                    
            }
        ).execute()

        recommended_songs = rpc_response.data
        print(f"Raw RPC Response Data: {recommended_songs}")
        
        return {
            "user_id": user_id,
            "mood": mood,
            "liked_songs_input": raw_liked_songs_list, 
            "found_liked_songs": found_liked_songs,     
            "unfound_liked_songs_messages": unfound_liked_songs_messages, 
            "recommended_songs": recommended_songs, 
            "message": "User profile updated and similar songs recommended.",
            "status": "success"
        }
    except Exception as e:
        print(f"Error recommending playlist: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate playlist: {e}")

# --- Local Development Server (Optional for Deployment) ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

