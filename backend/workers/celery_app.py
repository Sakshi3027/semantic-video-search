from celery import Celery
from backend.core.config import settings

celery_app = Celery(
    "semantic_video_search",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["backend.workers.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=3600,
    # Fix for macOS PyTorch/MPS fork crash
    worker_pool="solo",
)