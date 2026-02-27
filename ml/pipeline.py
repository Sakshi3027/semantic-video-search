import time
from pathlib import Path
from ml.video_processor import download_video, extract_frames
from ml.transcriber import transcribe_audio
from ml.embedder import describe_frame_with_clip, upsert_chunks_to_pinecone

def process_video(video_id: str, url: str, use_vision: bool = True) -> dict:
    """
    Full pipeline: download → transcribe → extract frames → embed → store
    Returns a summary of what was processed.
    """
    start_time = time.time()
    print(f"\n{'='*50}")
    print(f"Starting pipeline for video: {video_id}")
    print(f"URL: {url}")
    print(f"{'='*50}\n")

    # Step 1: Download video
    print("📥 Step 1: Downloading video...")
    video_info = download_video(url, video_id)
    print(f"✅ Downloaded: {video_info['title']} ({video_info['duration']}s)\n")

    # Step 2: Transcribe audio
    print("🎙️ Step 2: Transcribing audio...")
    audio_chunks = transcribe_audio(video_info["audio_path"])
    print(f"✅ Got {len(audio_chunks)} transcript chunks\n")

    # Step 3: Extract frames
    print("🎞️ Step 3: Extracting frames...")
    frames = extract_frames(video_info["video_path"], video_id)
    print(f"✅ Extracted {len(frames)} frames\n")

    # Step 4: Generate visual descriptions (optional - costs API credits)
    visual_chunks = []
    if use_vision and frames:
        print("👁️ Step 4: Generating visual descriptions with GPT-4o Vision...")
        # Sample every 3rd frame to reduce API costs
        sampled_frames = frames[::3]
        print(f"Processing {len(sampled_frames)} sampled frames...")

        for i, frame in enumerate(sampled_frames):
            try:
                description = describe_frame_with_clip(frame["frame_path"])
                visual_chunks.append({
                    "text": description,
                    "start_time": frame["timestamp"],
                    "end_time": frame["timestamp"] + 15,
                    "type": "visual"
                })
                if (i + 1) % 5 == 0:
                    print(f"  Processed {i+1}/{len(sampled_frames)} frames...")
            except Exception as e:
                print(f"  Warning: Frame {i} failed: {e}")
                continue

        print(f"✅ Generated {len(visual_chunks)} visual descriptions\n")
    else:
        print("⏭️ Step 4: Skipping vision (use_vision=False)\n")

    # Step 5: Embed and store everything
    print("🔢 Step 5: Embedding and storing in Pinecone...")
    all_chunks = audio_chunks + visual_chunks

    # Add video_id to each chunk
    for chunk in all_chunks:
        chunk["video_id"] = video_id

    total_vectors = upsert_chunks_to_pinecone(video_id, all_chunks)
    print(f"✅ Stored {total_vectors} vectors in Pinecone\n")

    processing_time = time.time() - start_time

    result = {
        "video_id": video_id,
        "title": video_info["title"],
        "duration": video_info["duration"],
        "thumbnail": video_info["thumbnail"],
        "audio_chunks": len(audio_chunks),
        "visual_chunks": len(visual_chunks),
        "total_vectors": total_vectors,
        "processing_time": round(processing_time, 2)
    }

    print(f"{'='*50}")
    print(f"✅ Pipeline complete in {processing_time:.1f}s")
    print(f"📊 Summary: {result}")
    print(f"{'='*50}\n")

    return result


def search_video(query: str, video_id: str = None, top_k: int = 5) -> list[dict]:
    """
    Search across one or all videos using natural language.
    """
    from ml.embedder import embed_text, get_pinecone_index

    index = get_pinecone_index()
    query_embedding = embed_text(query)

    # Build filter
    filter_dict = {}
    if video_id:
        filter_dict["video_id"] = {"$eq": video_id}

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict if filter_dict else None
    )

    formatted = []
    for match in results.matches:
        formatted.append({
            "score": round(match.score, 4),
            "video_id": match.metadata.get("video_id"),
            "text": match.metadata.get("text"),
            "start_time": match.metadata.get("start_time"),
            "end_time": match.metadata.get("end_time"),
            "type": match.metadata.get("type"),
            "timestamp_formatted": format_timestamp(match.metadata.get("start_time", 0))
        })

    return formatted


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    if seconds is None:
        return "00:00"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"