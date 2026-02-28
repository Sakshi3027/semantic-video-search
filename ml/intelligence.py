import json
from sentence_transformers import CrossEncoder
from transformers import pipeline

# ── Lazy loaded models ───────────────────────────────────────────────────────
_reranker = None
_summarizer = None

def get_reranker():
    global _reranker
    if _reranker is None:
        print("Loading re-ranking model...")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

def get_summarizer():
    global _summarizer
    if _summarizer is None:
        print("Loading summarization model...")
        _summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            max_length=60,
            min_length=10
        )
    return _summarizer


def expand_query(query: str) -> str:
    """
    Free query expansion using keyword repetition and simple NLP.
    No API needed — works offline.
    """
    # Simple but effective: repeat key terms with variations
    words = query.lower().split()
    
    # Common semantic expansions for video search
    expansions = {
        "why": "reason cause explanation",
        "how": "method process steps technique",
        "what": "definition meaning explanation",
        "when": "time moment timing",
        "who": "person people speaker",
        "procrastination": "procrastinate delay avoid postpone distraction",
        "deadline": "due date time pressure urgency",
        "strategy": "plan approach method tactics",
        "problem": "issue challenge difficulty obstacle",
        "solution": "fix answer resolve approach",
        "learn": "understand study education knowledge",
        "work": "job career productivity task",
        "money": "finance revenue cost pricing budget",
        "success": "achievement goal accomplish result",
        "fail": "failure mistake error wrong",
        "avoid": "escape ignore resist reluctant",
        "hard": "difficult challenging tough complex",
        "easy": "simple straightforward basic",
        "important": "critical essential key significant",
        "start": "begin initiate launch kickoff",
    }
    
    extra_terms = []
    for word in words:
        if word in expansions:
            extra_terms.append(expansions[word])
    
    expanded = query
    if extra_terms:
        expanded = f"{query} {' '.join(extra_terms)}"
    
    print(f"Expanded: {expanded}")
    return expanded


def rerank_results(query: str, results: list, top_k: int = 5) -> list:
    """
    Re-rank results using a free HuggingFace cross-encoder.
    Much more accurate than vector similarity alone.
    """
    if len(results) <= 1:
        return results

    try:
        reranker = get_reranker()

        # Create query-passage pairs
        pairs = [[query, r["text"]] for r in results]

        # Score each pair
        scores = reranker.predict(pairs)

        # Sort by score descending
        scored_results = list(zip(scores, results))
        scored_results.sort(key=lambda x: x[0], reverse=True)

        reranked = [r for _, r in scored_results[:top_k]]
        print(f"Re-ranked {len(results)} → {len(reranked)} results")
        return reranked

    except Exception as e:
        print(f"Reranking failed: {e}")
        return results[:top_k]


def generate_chapters(transcript_chunks: list, video_title: str) -> list:
    """
    Generate chapters by grouping transcript chunks into topic sections.
    Uses simple sliding window — no API needed.
    """
    if not transcript_chunks:
        return []

    try:
        # Group chunks into sections of ~3 chunks each
        section_size = max(3, len(transcript_chunks) // 5)
        chapters = []

        for i in range(0, len(transcript_chunks), section_size):
            section = transcript_chunks[i:i + section_size]
            if not section:
                continue

            # Combine text for this section
            combined_text = " ".join([c["text"] for c in section])[:500]
            start_time = section[0]["start_time"]
            end_time = section[-1]["end_time"]

            # Generate a title using first meaningful sentence
            sentences = combined_text.split(".")
            title = "Section"
            for s in sentences:
                s = s.strip()
                if len(s) > 20 and len(s) < 100:
                    # Truncate to make a good chapter title
                    title = s[:60].strip()
                    if len(s) > 60:
                        title += "..."
                    break

            m, s = divmod(int(start_time), 60)
            chapters.append({
                "title": title,
                "start_time": start_time,
                "end_time": end_time,
                "timestamp_formatted": f"{m:02d}:{s:02d}"
            })

        print(f"Generated {len(chapters)} chapters")
        return chapters

    except Exception as e:
        print(f"Chapter generation failed: {e}")
        return []


def smart_search(query: str, video_id: str = None, top_k: int = 5) -> dict:
    """
    Enhanced search with free query expansion + cross-encoder re-ranking.
    100% offline — no API costs.
    """
    from ml.pipeline import search_video

    print(f"Original query: {query}")
    expanded = expand_query(query)

    # Get more results for re-ranking
    raw_results = search_video(
        query=expanded,
        video_id=video_id,
        top_k=top_k * 2
    )

    if not raw_results:
        return {"results": [], "expanded_query": expanded}

    # Re-rank with cross-encoder
    final_results = rerank_results(query, raw_results, top_k=top_k)

    return {
        "results": final_results,
        "expanded_query": expanded,
        "original_query": query
    }