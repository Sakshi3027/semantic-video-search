import os
import subprocess
import cv2
import yt_dlp
from pathlib import Path
from backend.core.config import settings

DOWNLOAD_DIR = Path("downloads")
DOWNLOAD_DIR.mkdir(exist_ok=True)

def download_video(url: str, video_id: str) -> dict:
    """Download video and extract audio using yt-dlp."""
    video_dir = DOWNLOAD_DIR / video_id
    video_dir.mkdir(exist_ok=True)

    ydl_opts = {
        "outtmpl": str(video_dir / "video.%(ext)s"),
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "Unknown")
        duration = info.get("duration", 0)
        thumbnail = info.get("thumbnail", None)

    # Find the downloaded video file
    video_files = list(video_dir.glob("video.*"))
    if not video_files:
        raise FileNotFoundError("Video download failed")
    
    video_path = str(video_files[0])

    # Extract audio as wav for Whisper
    audio_path = str(video_dir / "audio.wav")
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-ar", "16000",  # 16kHz required by Whisper
        "-ac", "1",      # mono
        "-y", audio_path
    ], check=True, capture_output=True)

    return {
        "video_path": video_path,
        "audio_path": audio_path,
        "title": title,
        "duration": duration,
        "thumbnail": thumbnail
    }


def extract_frames(video_path: str, video_id: str, interval: int = None) -> list[dict]:
    """Extract frames every N seconds from video."""
    interval = interval or settings.FRAME_EXTRACTION_INTERVAL
    frames_dir = DOWNLOAD_DIR / video_id / "frames"
    frames_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    frames = []
    frame_interval = int(fps * interval)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            frame_path = str(frames_dir / f"frame_{int(timestamp):06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frames.append({
                "timestamp": round(timestamp, 2),
                "frame_path": frame_path
            })

        frame_count += 1

    cap.release()
    print(f"Extracted {len(frames)} frames from {duration:.1f}s video")
    return frames