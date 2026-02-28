import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ── Config ──────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="Semantic Video Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Dark Theme CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0e1117; color: #fafafa; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    
    /* Cards */
    .card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        transition: border-color 0.2s;
    }
    .card:hover { border-color: #58a6ff; }
    
    /* Search result card */
    .result-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-left: 4px solid #58a6ff;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }
    
    /* Score badge */
    .score-badge {
        background: #1f6feb;
        color: white;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }
    
    /* Timestamp badge */
    .timestamp-badge {
        background: #238636;
        color: white;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }

    /* Type badge */
    .type-badge {
        background: #6e40c9;
        color: white;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 12px;
    }

    /* Metric cards */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value { font-size: 36px; font-weight: bold; color: #58a6ff; }
    .metric-label { font-size: 14px; color: #8b949e; margin-top: 4px; }

    /* Status badges */
    .status-completed { color: #3fb950; font-weight: bold; }
    .status-failed { color: #f85149; font-weight: bold; }
    .status-pending { color: #d29922; font-weight: bold; }
    .status-downloading { color: #58a6ff; font-weight: bold; }

    /* Input fields */
    .stTextInput input, .stTextArea textarea {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #fafafa !important;
        border-radius: 8px !important;
    }

    /* Buttons */
    .stButton button {
        background: #1f6feb;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: bold;
        transition: background 0.2s;
    }
    .stButton button:hover { background: #388bfd; }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Divider */
    hr { border-color: #30363d; }
</style>
""", unsafe_allow_html=True)


# ── API Helpers ──────────────────────────────────────────────────────────────
def api_get(endpoint):
    try:
        r = requests.get(f"{API_BASE}{endpoint}", timeout=30)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def api_post(endpoint, data):
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=data, timeout=120)
        return r.json(), r.status_code
    except Exception as e:
        return {"error": str(e)}, 500

def api_delete(endpoint):
    try:
        r = requests.delete(f"{API_BASE}{endpoint}", timeout=10)
        return r.status_code == 204
    except:
        return False

def format_duration(seconds):
    if not seconds:
        return "N/A"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s" if h else f"{m}m {s}s"

def status_color(status):
    colors = {
        "completed": "status-completed",
        "failed": "status-failed",
        "pending": "status-pending",
        "downloading": "status-downloading",
        "transcribing": "status-downloading",
        "embedding": "status-downloading",
    }
    return colors.get(status, "")


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### VideoSearch AI")
    st.markdown("*Search video content with natural language*")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🔍 Search", "📚 Video Library", "📊 Analytics", "⚙️ Admin"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Health check
    health = api_get("/health" .replace("/api/v1", ""))
    try:
        health = requests.get("http://localhost:8000/health", timeout=5).json()
        st.success("🟢 API Online")
    except:
        st.error("🔴 API Offline")

    st.markdown("---")
    st.markdown("### Quick Stats")
    videos = api_get("/videos") or []
    completed = [v for v in videos if v.get("status") == "completed"]
    total_chunks = sum(v.get("total_chunks", 0) for v in completed)
    st.metric("Videos Indexed", len(completed))
    st.metric("Searchable Chunks", total_chunks)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SEARCH
# ════════════════════════════════════════════════════════════════════════════
if page == "🔍 Search":
    st.markdown("# 🔍 Semantic Video Search")
    st.markdown("Search across video content using natural language — find exact moments instantly.")
    st.markdown("---")

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Search query",
            placeholder='e.g. "where was pricing strategy discussed" or "show me the demo"',
            label_visibility="collapsed"
        )
    with col2:
        top_k = st.selectbox("Results", [3, 5, 10], index=1, label_visibility="collapsed")

    # Video filter
    videos = api_get("/videos") or []
    completed_videos = [v for v in videos if v.get("status") == "completed"]
    video_options = {"All Videos": None}
    for v in completed_videos:
        title = v.get("title", v.get("video_id", "Unknown"))[:50]
        video_options[title] = v.get("video_id") or v.get("id")

    selected_video_label = st.selectbox("Filter by video", list(video_options.keys()))
    selected_video_id = video_options[selected_video_label]

    search_clicked = st.button("🔍 Search", width="stretch")

    if search_clicked and query:
        with st.spinner("Searching..."):
            payload = {"query": query, "top_k": top_k}
            if selected_video_id:
                payload["video_id"] = selected_video_id

            results, status = api_post("/search", payload)

        if status == 200 and results.get("results"):
            st.markdown(f"### Found {results['total_results']} results in {results['latency_ms']:.0f}ms")
            st.markdown("---")

            for i, r in enumerate(results["results"]):
                score_pct = int(r["score"] * 100)
                youtube_url = r.get("youtube_url", "#")

                st.markdown(f"""
                <div class="result-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px">
                        <div>
                            <span class="score-badge">⚡ {score_pct}% match</span>
                            &nbsp;
                            <span class="timestamp-badge">⏱ {r['timestamp_formatted']}</span>
                            &nbsp;
                            <span class="type-badge">{'🎙 Audio' if r['type'] == 'audio' else '👁 Visual'}</span>
                        </div>
                        <small style="color:#8b949e">Video: {r['video_id']}</small>
                    </div>
                    <p style="color:#e6edf3; margin:8px 0; line-height:1.6">{r['text'][:300]}...</p>
                    <a href="{youtube_url}" target="_blank" style="color:#58a6ff; text-decoration:none">
                        ▶ Jump to {r['timestamp_formatted']} on YouTube →
                    </a>
                </div>
                """, unsafe_allow_html=True)
        elif status == 200:
            st.info("No results found. Try a different query.")
        else:
            st.error(f"Search failed: {results}")

    elif search_clicked and not query:
        st.warning("Please enter a search query.")

    # Example queries
    st.markdown("---")
    st.markdown("### 💡 Try these example queries")
    examples = [
        "What causes procrastination?",
        "How does the panic monster work?",
        "What happens without deadlines?",
        "Life advice and taking action",
    ]
    cols = st.columns(len(examples))
    for col, example in zip(cols, examples):
        with col:
            if st.button(example, width="stretch"):
                st.session_state["example_query"] = example
                st.rerun()

    if "example_query" in st.session_state:
        st.info(f'Click the search box and type: "{st.session_state.example_query}"')


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — VIDEO LIBRARY
# ════════════════════════════════════════════════════════════════════════════
elif page == "📚 Video Library":
    st.markdown("# 📚 Video Library")
    st.markdown("---")

    # Submit new video
    with st.expander("➕ Add New Video", expanded=False):
        new_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
        use_vision = st.checkbox("Enable Vision Analysis (uses GPT-4o, costs more)", value=False)
        if st.button("🚀 Process Video"):
            if new_url:
                with st.spinner("Submitting video for processing..."):
                    result, status = api_post("/videos", {"url": new_url, "use_vision": use_vision})
                if status == 202:
                    st.success(f"✅ Video queued! ID: `{result['video_id']}` — Processing in background...")
                    st.rerun()
                else:
                    st.error(f"Failed: {result}")
            else:
                st.warning("Please enter a YouTube URL")

    st.markdown("---")

    # Video list
    videos = api_get("/videos") or []
    if not videos:
        st.info("No videos yet. Add your first video above!")
    else:
        st.markdown(f"### {len(videos)} Videos")
        for v in videos:
            status = v.get("status", "unknown")
            vid_id = v.get("video_id") or v.get("id", "N/A")
            title = v.get("title") or "Processing..."
            duration = format_duration(v.get("duration"))
            chunks = v.get("total_chunks", 0)
            proc_time = v.get("processing_time")

            with st.container():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                with col1:
                    st.markdown(f"**{title[:60]}**")
                    st.caption(f"ID: `{vid_id}` · Duration: {duration}")
                with col2:
                    st.markdown(f"<span class='{status_color(status)}'>{status.upper()}</span>",
                               unsafe_allow_html=True)
                with col3:
                    st.metric("Chunks", chunks)
                with col4:
                    if proc_time:
                        st.metric("Process Time", f"{proc_time:.0f}s")

                # Thumbnail if available
                thumb = v.get("thumbnail_url")
                if thumb and status == "completed":
                    st.image(thumb, width=200)

                st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ANALYTICS
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics":
    st.markdown("# 📊 Analytics Dashboard")
    st.markdown("---")

    videos = api_get("/videos") or []
    completed = [v for v in videos if v.get("status") == "completed"]
    failed = [v for v in videos if v.get("status") == "failed"]

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(videos)}</div>
            <div class="metric-label">Total Videos</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        total_chunks = sum(v.get("total_chunks", 0) for v in completed)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_chunks}</div>
            <div class="metric-label">Vectors in Pinecone</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        avg_time = sum(v.get("processing_time", 0) for v in completed) / len(completed) if completed else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_time:.0f}s</div>
            <div class="metric-label">Avg Processing Time</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        success_rate = (len(completed) / len(videos) * 100) if videos else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{success_rate:.0f}%</div>
            <div class="metric-label">Success Rate</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    if completed:
        col1, col2 = st.columns(2)

        with col1:
            # Status distribution
            status_counts = {}
            for v in videos:
                s = v.get("status", "unknown")
                status_counts[s] = status_counts.get(s, 0) + 1

            fig = px.pie(
                values=list(status_counts.values()),
                names=list(status_counts.keys()),
                title="Video Status Distribution",
                color_discrete_map={
                    "completed": "#3fb950",
                    "failed": "#f85149",
                    "pending": "#d29922",
                    "downloading": "#58a6ff"
                },
                hole=0.4
            )
            fig.update_layout(
                paper_bgcolor="#161b22",
                plot_bgcolor="#161b22",
                font_color="#fafafa",
                title_font_color="#fafafa"
            )
            st.plotly_chart(fig, width="stretch")

        with col2:
            # Processing time bar chart
            if len(completed) > 1:
                df = pd.DataFrame([{
                    "title": (v.get("title") or "Unknown")[:30],
                    "processing_time": v.get("processing_time", 0),
                    "chunks": v.get("total_chunks", 0)
                } for v in completed])

                fig2 = px.bar(
                    df, x="title", y="processing_time",
                    title="Processing Time per Video (seconds)",
                    color="chunks",
                    color_continuous_scale="blues"
                )
                fig2.update_layout(
                    paper_bgcolor="#161b22",
                    plot_bgcolor="#161b22",
                    font_color="#fafafa",
                    title_font_color="#fafafa",
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig2, width="stretch")

        # Video details table
        st.markdown("### Video Details")
        df_table = pd.DataFrame([{
            "Title": (v.get("title") or "N/A")[:40],
            "Status": v.get("status"),
            "Duration": format_duration(v.get("duration")),
            "Audio Chunks": v.get("audio_chunks", 0),
            "Visual Chunks": v.get("visual_chunks", 0),
            "Total Vectors": v.get("total_chunks", 0),
            "Process Time": f"{v.get('processing_time') or 0:.0f}s"
        } for v in videos])
        st.dataframe(df_table, width="stretch", hide_index=True)
    else:
        st.info("Process some videos to see analytics!")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ADMIN
# ════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Admin":
    st.markdown("# ⚙️ Admin Panel")
    st.markdown("---")

    videos = api_get("/videos") or []

    st.markdown("### 🗑️ Delete Videos")
    st.warning("Deleting a video removes it from the database and Pinecone. This cannot be undone.")

    if not videos:
        st.info("No videos to manage.")
    else:
        for v in videos:
            vid_id = v.get("video_id") or v.get("id")
            title = v.get("title") or "Processing..."
            status = v.get("status", "unknown")

            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"**{title[:50]}**")
                st.caption(f"ID: `{vid_id}`")
            with col2:
                st.markdown(f"<span class='{status_color(status)}'>{status.upper()}</span>",
                           unsafe_allow_html=True)
            with col3:
                if st.button(f"🗑️ Delete", key=f"del_{vid_id}"):
                    if api_delete(f"/videos/{vid_id}"):
                        st.success("Deleted!")
                        st.rerun()
                    else:
                        st.error("Delete failed")
            st.markdown("---")

    st.markdown("### 🔧 System Info")
    col1, col2 = st.columns(2)
    with col1:
        try:
            health = requests.get("http://localhost:8000/health", timeout=5).json()
            st.success(f"✅ API: {health.get('status')} ({health.get('env')})")
        except:
            st.error("❌ API Offline")
    with col2:
        st.info(f"📊 Total videos in system: {len(videos)}")