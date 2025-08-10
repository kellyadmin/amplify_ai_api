import os
from fastapi import FastAPI, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from supabase import create_client, Client # Import Supabase client

# --- Supabase Configuration ---
# Get Supabase URL and Key from environment variables
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# Initialize Supabase client. This will be used for database interactions.
# Ensure SUPABASE_URL and SUPABASE_KEY are set in Render environment variables.
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- FastAPI App & Model Loading ---
# This environment variable attempts to cache the downloaded models within the '/app/cache' directory
# This helps if Render's build process supports persistent caching of this directory.
os.environ["TRANSFORMERS_CACHE"] = "/app/cache"

app = FastAPI()

# Load models once on startup
# These models will be downloaded to /app/cache/ (or a default path) on the first run/build.
# Consider using smaller or quantized versions if you hit deployment size limits.
try:
    # Sentence Transformer for text embeddings (e.g., for song query similarity)
    # This model is still paraphrase-MiniLM-L3-v2, ensure it fits memory limits.
    model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    # Text generation model (e.g., for chat responses)
    # Using 'sshleifer/tiny-distilgpt2' for memory efficiency on Render.
    text_generator = pipeline("text-generation", model="sshleifer/tiny-distilgpt2")
except Exception as e:
    print(f"Error loading AI models: {e}")
    # In a production app, you might want to log this error and gracefully
    # handle cases where models fail to load (e.g., return a 500 error).

# --- Pydantic Models for Request/Response Validation ---
class ChatRequest(BaseModel):
    message: str

class RecommendationRequest(BaseModel):
    prompt: str

# --- API Endpoints ---
@app.get("/")
async def root():
    """
    Root endpoint to confirm the API is running.
    """
    return {"message": "Amplify AI API is running"}

@app.get("/recommend")
async def recommend_songs(query: str = Query(..., description="Text query for song recommendation"), top_k: int = 5):
    """
    Recommends songs based on a text query by generating a sentence embedding.
    
    Args:
        query (str): The text description or song name for recommendation.
        top_k (int): The number of embedding dimensions to return for demonstration.
                     (Note: For actual song recommendation, you'd use this embedding
                     to search a vector database of song embeddings, not just return it.)
    """
    try:
        # Encode the query text into a numerical vector (embedding)
        emb = model.encode(query).tolist()
        # Returning a slice of the embedding for demonstration purposes.
        # In a real app, you'd search a database for similar song embeddings.
        return {"query": query, "embedding": emb[:top_k], "status": "success"}
    except Exception as e:
        return {"query": query, "error": str(e), "status": "failed"}

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Generates a chat response based on the user's message using a text generation model.
    
    Args:
        request (ChatRequest): A Pydantic model containing the user's message.
    """
    prompt = request.message
    try:
        # Generate text based on the prompt
        # max_length: Controls the length of the generated response.
        # num_return_sequences: How many different responses to generate (we pick the first).
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
    try:
        # Perform a case-insensitive search (ilike) on the 'title' column in the 'songs' table
        response = supabase.table('songs').select('*').ilike('title', f'%{query}%').execute()
        
        # Check if the query was successful and data exists
        if response.data:
            return {"results": response.data, "status": "success"}
        else:
            return {"results": [], "message": "No songs found matching your query.", "status": "success"}
    except Exception as e:
        # Log the full error for debugging
        print(f"Error searching Supabase: {e}")
        return {"error": f"Failed to search database: {e}", "status": "failed"}


# --- Local Development Server (Optional for Deployment) ---
# This block is for running your app locally. Render uses the Procfile.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

