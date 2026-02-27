from backend.workers.celery_app import celery_app
from backend.core.database import SessionLocal
from backend.models.video import Video, ProcessingStatus
from ml.pipeline import process_video
import traceback

def update_video_status(video_id: str, status: ProcessingStatus, error: str = None):
    """Helper to update video status in DB."""
    db = SessionLocal()
    try:
        video = db.query(Video).filter(Video.id == video_id).first()
        if video:
            video.status = status
            if error:
                video.error_message = error
            db.commit()
    finally:
        db.close()

@celery_app.task(bind=True, max_retries=3)
def process_video_task(self, video_id: str, url: str, use_vision: bool = False):
    """
    Async Celery task to process a video end-to-end.
    Updates DB status at each stage.
    """
    db = SessionLocal()
    try:
        # Mark as started
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise ValueError(f"Video {video_id} not found in DB")

        video.status = ProcessingStatus.DOWNLOADING
        db.commit()

        # Run the full pipeline
        result = process_video(video_id, url, use_vision=use_vision)

        # Update DB with results
        video.status = ProcessingStatus.COMPLETED
        video.title = result["title"]
        video.duration = result["duration"]
        video.thumbnail_url = result["thumbnail"]
        video.audio_chunks = result["audio_chunks"]
        video.visual_chunks = result["visual_chunks"]
        video.total_chunks = result["total_vectors"]
        video.processing_time = result["processing_time"]
        db.commit()

        return result

    except Exception as exc:
        db.rollback()
        # Update DB with error
        try:
            video = db.query(Video).filter(Video.id == video_id).first()
            if video:
                video.status = ProcessingStatus.FAILED
                video.error_message = str(exc)[:500]
                db.commit()
        except:
            pass

        # Retry up to 3 times
        raise self.retry(exc=exc, countdown=60)

    finally:
        db.close()