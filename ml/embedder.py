from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from backend.core.config import settings
import base64
import httpx

_embedding_model = None

def get_embedding_model() -> SentenceTransformer:
    """Lazy load embedding model (singleton)."""
    global _embedding_model
    if _embedding_model is None:
        print(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _embedding_model


def get_pinecone_index():
    """Get Pinecone index."""
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    return pc.Index(settings.PINECONE_INDEX_NAME)


def embed_text(text: str) -> list[float]:
    """Embed a single text string."""
    model = get_embedding_model()
    return model.encode(text, normalize_embeddings=True).tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts efficiently."""
    model = get_embedding_model()
    return model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()


def describe_frame_with_clip(frame_path: str) -> str:
    """
    Generate a text description of a frame using OpenAI Vision.
    Returns a text description we can embed.
    """
    from openai import OpenAI
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    with open(frame_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # cheap + fast vision model
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}",
                        "detail": "low"  # cheaper
                    }
                },
                {
                    "type": "text",
                    "text": "Describe what is shown in this video frame in 1-2 sentences. Focus on: people, text on screen, diagrams, actions, and setting."
                }
            ]
        }],
        max_tokens=150
    )
    return response.choices[0].message.content


def upsert_chunks_to_pinecone(video_id: str, chunks: list[dict]) -> int:
    """
    Embed chunks and upsert to Pinecone.
    Each chunk must have: text, start_time, end_time, type
    """
    index = get_pinecone_index()

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_batch(texts)

    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vector_id = f"{video_id}_{chunk['type']}_{i}"
        vectors.append({
            "id": vector_id,
            "values": [float(v) for v in embedding],  # convert numpy -> python float
            "metadata": {
                "video_id": str(video_id),
                "text": chunk["text"][:500],
                "start_time": float(chunk["start_time"]),  # convert here too
                "end_time": float(chunk["end_time"]),
                "type": str(chunk["type"]),
            }
        })

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)

    print(f"Upserted {len(vectors)} vectors to Pinecone")
    return len(vectors)