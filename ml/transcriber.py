from faster_whisper import WhisperModel
from backend.core.config import settings

_model = None

def get_whisper_model() -> WhisperModel:
    """Lazy load whisper model (singleton)."""
    global _model
    if _model is None:
        print(f"Loading Whisper model: {settings.WHISPER_MODEL_SIZE}")
        _model = WhisperModel(
            settings.WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type="int8"
        )
    return _model


def transcribe_audio(audio_path: str) -> list[dict]:
    """
    Transcribe audio and return timestamped chunks.
    Each chunk = ~30 seconds of speech merged together.
    """
    model = get_whisper_model()
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,  # removes silence
    )

    print(f"Detected language: {info.language} (confidence: {info.language_probability:.2f})")

    # Merge segments into ~30s chunks
    chunks = []
    current_chunk = {"text": "", "start": 0, "end": 0, "words": []}
    chunk_duration = settings.CHUNK_SIZE

    for segment in segments:
        if (segment.end - current_chunk["start"]) > chunk_duration and current_chunk["text"]:
            chunks.append({
                "text": current_chunk["text"].strip(),
                "start_time": round(current_chunk["start"], 2),
                "end_time": round(current_chunk["end"], 2),
                "type": "audio"
            })
            current_chunk = {"text": "", "start": segment.start, "end": segment.end, "words": []}

        if not current_chunk["text"]:
            current_chunk["start"] = segment.start

        current_chunk["text"] += " " + segment.text
        current_chunk["end"] = segment.end

    # Don't forget the last chunk
    if current_chunk["text"].strip():
        chunks.append({
            "text": current_chunk["text"].strip(),
            "start_time": round(current_chunk["start"], 2),
            "end_time": round(current_chunk["end"], 2),
            "type": "audio"
        })

    print(f"Created {len(chunks)} transcript chunks")
    return chunks