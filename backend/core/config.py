from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # App
    APP_ENV: str = "development"
    SECRET_KEY: str = "change-me-in-production"
    
    # OpenAI
    OPENAI_API_KEY: str
    
    # Pinecone
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "semantic-video-search"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # PostgreSQL
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/videosearch"
    
    # Processing
    FRAME_EXTRACTION_INTERVAL: int = 5  # seconds
    WHISPER_MODEL_SIZE: str = "base"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 30  # seconds per transcript chunk
    
    class Config:
        env_file = ".env"
        extra = "ignore"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()