import time
import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.core.database import get_db
from backend.models.video import Video, ProcessingStatus, SearchLog
from backend.api.schemas import (
    VideoSubmitRequest, VideoResponse,
    SearchRequest, SearchResponse, SearchResult,
    JobStatusResponse
)
from backend.workers.tasks import process_video_task
from ml.pipeline import search_video

router = APIRouter()

@router.post("/videos", response_model=JobStatusResponse, status_code=202)
def submit_video(request: VideoSubmitRequest, db: Session = Depends(get_db)):
    """Submit a video URL for processing."""

    # Create video record in DB
    video_id = str(uuid.uuid4())[:8]
    video = Video(
        id=video_id,
        url=request.url,
        status=ProcessingStatus.PENDING
    )
    db.add(video)
    db.commit()

    # Queue the Celery task
    job = process_video_task.delay(
        video_id=video_id,
        url=request.url,
        use_vision=request.use_vision
    )

    return JobStatusResponse(
        job_id=job.id,
        video_id=video_id,
        status="queued",
        message="Video queued for processing. Poll /videos/{video_id} for status."
    )


@router.get("/videos/{video_id}", response_model=VideoResponse)
def get_video(video_id: str, db: Session = Depends(get_db)):
    """Get video processing status and metadata."""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return video


@router.get("/videos", response_model=list[VideoResponse])
def list_videos(db: Session = Depends(get_db)):
    """List all processed videos."""
    return db.query(Video).order_by(Video.created_at.desc()).all()


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest, db: Session = Depends(get_db)):
    """Search across video content using natural language."""
    start_time = time.time()

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Validate video_id if provided
    if request.video_id:
        video = db.query(Video).filter(
            Video.id == request.video_id,
            Video.status == ProcessingStatus.COMPLETED
        ).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found or not yet processed")

    # Run search
    raw_results = search_video(
        query=request.query,
        video_id=request.video_id,
        top_k=request.top_k
    )

    # Format results + add YouTube deep links
    results = []
    for r in raw_results:
        # Build YouTube timestamp URL
        youtube_url = None
        video_record = db.query(Video).filter(Video.id == r["video_id"]).first()
        if video_record and "youtube.com" in video_record.url:
            t = int(r["start_time"])
            youtube_url = f"{video_record.url}&t={t}s"

        results.append(SearchResult(
            score=r["score"],
            video_id=r["video_id"],
            text=r["text"],
            start_time=r["start_time"],
            end_time=r["end_time"],
            type=r["type"],
            timestamp_formatted=r["timestamp_formatted"],
            youtube_url=youtube_url
        ))

    latency_ms = round((time.time() - start_time) * 1000, 2)

    # Log the search
    log = SearchLog(
        query=request.query,
        video_id=request.video_id,
        results_count=len(results),
        latency_ms=latency_ms
    )
    db.add(log)
    db.commit()

    return SearchResponse(
        query=request.query,
        results=results,
        total_results=len(results),
        latency_ms=latency_ms
    )


@router.delete("/videos/{video_id}", status_code=204)
def delete_video(video_id: str, db: Session = Depends(get_db)):
    """Delete a video and its vectors from Pinecone."""
    from pinecone import Pinecone
    from backend.core.config import settings

    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Delete from Pinecone
    try:
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        index = pc.Index(settings.PINECONE_INDEX_NAME)
        index.delete(filter={"video_id": {"$eq": video_id}})
    except Exception as e:
        print(f"Warning: Pinecone delete failed: {e}")

    db.delete(video)
    db.commit()

@router.post("/search/smart", response_model=SearchResponse)
def smart_search_endpoint(request: SearchRequest, db: Session = Depends(get_db)):
    """Enhanced search with query expansion and re-ranking."""
    import time
    start_time = time.time()

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    from ml.intelligence import smart_search
    result = smart_search(
        query=request.query,
        video_id=request.video_id,
        top_k=request.top_k
    )

    raw_results = result["results"]
    results = []
    for r in raw_results:
        youtube_url = None
        video_record = db.query(Video).filter(Video.id == r["video_id"]).first()
        if video_record and "youtube.com" in video_record.url:
            t = int(r["start_time"])
            youtube_url = f"{video_record.url}&t={t}s"

        results.append(SearchResult(
            score=r["score"],
            video_id=r["video_id"],
            text=r["text"],
            start_time=r["start_time"],
            end_time=r["end_time"],
            type=r["type"],
            timestamp_formatted=r["timestamp_formatted"],
            youtube_url=youtube_url
        ))

    latency_ms = round((time.time() - start_time) * 1000, 2)

    # Log search
    log = SearchLog(
        query=request.query,
        video_id=request.video_id,
        results_count=len(results),
        latency_ms=latency_ms
    )
    db.add(log)
    db.commit()

    return SearchResponse(
        query=f"{request.query} (smart)",
        results=results,
        total_results=len(results),
        latency_ms=latency_ms
    )


@router.get("/videos/{video_id}/chapters")
def get_video_chapters(video_id: str, db: Session = Depends(get_db)):
    """Auto-generate chapters for a video."""
    from ml.intelligence import generate_chapters
    from ml.transcriber import transcribe_audio
    from ml.video_processor import download_video
    from pathlib import Path

    video = db.query(Video).filter(
        Video.id == video_id,
        Video.status == ProcessingStatus.COMPLETED
    ).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found or not processed")

    audio_path = f"downloads/{video_id}/audio.wav"

    # If audio doesn't exist, re-download it
    if not Path(audio_path).exists():
        print(f"Audio not found, re-downloading video {video_id}...")
        try:
            video_info = download_video(video.url, video_id)
            audio_path = video_info["audio_path"]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not download video: {str(e)}")

    try:
        chunks = transcribe_audio(audio_path)
        chapters = generate_chapters(chunks, video.title or "Video")
        return {"video_id": video_id, "title": video.title, "chapters": chapters}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chapter generation failed: {str(e)}")