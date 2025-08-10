from fastapi import FastAPI, Query
from pydantic import BaseModel
# Ensure these libraries are in your requirements.txt
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import os

# --- Model Loading Strategy for Deployment ---
# This environment variable attempts to cache the downloaded models within the '/app/cache' directory
# This helps if Render's build process supports persistent caching of this directory.
# However, if the combined size of the models + Python environment exceeds 4GB,
# you will still hit the Render free tier limit.
os.environ["TRANSFORMERS_CACHE"] = "/app/cache"

app = FastAPI()

# Load models once on startup
# These models will be downloaded to /app/cache/ (or a default path) on the first run/build.
# Consider using smaller or quantized versions if you hit deployment size limits.
try:
    # Sentence Transformer for text embeddings (e.g., for song query similarity)
    model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    # Text generation model (e.g., for chat responses)
    # GPT-2 is quite large. Consider 'distilgpt2' for a smaller alternative, or
    # quantize this model.
    text_generator = pipeline("text-generation", model="sshleifer/tiny-distilgpt2")
except Exception as e:
    print(f"Error loading AI models: {e}")
    # Depending on your use case, you might want to raise an exception or
    # have fallback logic if models fail to load.

# --- Pydantic Models for Request/Response Validation ---
class ChatRequest(BaseModel):
    message: str

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

# --- Local Development Server (Optional for Deployment) ---
# This block is for running your app locally. Render uses the Procfile.
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
