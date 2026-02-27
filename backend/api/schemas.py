from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from backend.models.video import ProcessingStatus

class VideoSubmitRequest(BaseModel):
    url: str
    use_vision: bool = False

class VideoResponse(BaseModel):
    video_id: str = Field(alias="id")
    title: Optional[str] = None
    url: str
    status: ProcessingStatus
    duration: Optional[float] = None
    thumbnail_url: Optional[str] = None
    audio_chunks: int = 0
    visual_chunks: int = 0
    total_chunks: int = 0
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True
        populate_by_name = True

class SearchRequest(BaseModel):
    query: str
    video_id: Optional[str] = None
    top_k: int = 5
    search_type: str = "all"

class SearchResult(BaseModel):
    score: float
    video_id: str
    text: str
    start_time: float
    end_time: float
    type: str
    timestamp_formatted: str
    youtube_url: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total_results: int
    latency_ms: float

class JobStatusResponse(BaseModel):
    job_id: str
    video_id: str
    status: str
    message: Optional[str] = None