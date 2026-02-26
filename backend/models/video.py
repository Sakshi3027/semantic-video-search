from sqlalchemy import Column, String, Integer, Float, DateTime, Enum, Text, JSON
from sqlalchemy.sql import func
import enum
import uuid
from backend.core.database import Base

class ProcessingStatus(str, enum.Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    TRANSCRIBING = "transcribing"
    EXTRACTING_FRAMES = "extracting_frames"
    EMBEDDING = "embedding"
    COMPLETED = "completed"
    FAILED = "failed"

class Video(Base):
    __tablename__ = "videos"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=True)
    url = Column(String, nullable=False)
    duration = Column(Float, nullable=True)  # seconds
    status = Column(Enum(ProcessingStatus), default=ProcessingStatus.PENDING)
    error_message = Column(Text, nullable=True)
    thumbnail_url = Column(String, nullable=True)
    
    # Stats
    total_chunks = Column(Integer, default=0)
    audio_chunks = Column(Integer, default=0)
    visual_chunks = Column(Integer, default=0)
    processing_time = Column(Float, nullable=True)  # seconds
    
    # Metadata
    video_metadata = Column(JSON, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class SearchLog(Base):
    __tablename__ = "search_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    query = Column(String, nullable=False)
    video_id = Column(String, nullable=True)  # None = search all videos
    results_count = Column(Integer, default=0)
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())